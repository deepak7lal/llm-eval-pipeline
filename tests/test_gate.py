"""Gate logic tests - the part that decides whether CI goes red."""

import dataclasses
import json

import pytest

from evals import config
from evals.config import GATE
from evals.cost import Spend
from evals.gate import ContaminatedBaseline, _BaselineEntry, evaluate, write_baseline
from evals.runner import SuiteResult, TrialResult


def suite(name, passed, total, repeats=1):
    """A suite result with `passed` of `total` trials green."""
    trials = [
        TrialResult(case_id=f"c{i // repeats}", repeat=i % repeats, passed=i < passed, score=float(i < passed))
        for i in range(total)
    ]
    return SuiteResult(name=name, trials=trials)


def base(rate, trials):
    return {"s": _BaselineEntry(pass_rate=rate, trials=trials)}


def errored_suite(name, total, message="TypeError: could not resolve authentication method"):
    """A suite where every trial died before the model answered."""
    trials = [
        TrialResult(case_id=f"c{i}", repeat=0, passed=False, score=0.0, error=message)
        for i in range(total)
    ]
    return SuiteResult(name=name, trials=trials)


class TestFloor:
    def test_passes_above_floor_with_no_baseline(self):
        run = evaluate([suite("s", 9, 10)], baseline={})
        [verdict] = run.suites
        assert not run.failed
        assert verdict.status == "PASS"
        assert verdict.baseline is None

    def test_fails_below_absolute_floor(self):
        run = evaluate([suite("s", 5, 10)], baseline={})
        assert run.failed
        assert "below the floor" in run.suites[0].reasons[0]

    def test_tiny_suite_reports_but_does_not_gate(self):
        small = suite("s", 0, GATE.min_cases_to_gate - 1)
        run = evaluate([small], baseline={})
        assert not run.failed
        assert any("not gating" in r for r in run.suites[0].reasons)


class TestRegression:
    def test_significant_regression_fails(self):
        # 85/100 against a 98% baseline over 100 trials: well outside noise.
        run = evaluate([suite("s", 85, 100)], baseline=base(0.98, 100))
        assert run.failed
        assert any("regressed" in r for r in run.suites[0].reasons)
        assert run.suites[0].p_value < 0.05

    def test_large_drop_on_a_tiny_sample_does_not_fail(self):
        # 6/10 vs a 90% baseline of 10 trials looks awful but cannot be
        # distinguished from noise - this is the false alarm the gate avoids.
        run = evaluate([suite("s", 6, 10)], baseline=base(0.90, 10))
        [verdict] = run.suites
        assert any("sampling noise" in r for r in verdict.reasons)
        # It still fails, but on the floor, not on the regression.
        assert all("regressed" not in r for r in verdict.reasons)

    def test_small_regression_within_allowance_only_warns(self):
        run = evaluate([suite("s", 96, 100)], baseline=base(0.98, 100))
        assert not run.failed
        assert run.suites[0].status == "WARN"

    def test_improvement_passes(self):
        run = evaluate([suite("s", 99, 100)], baseline=base(0.90, 100))
        assert not run.failed
        assert run.suites[0].delta > 0

    def test_significance_can_be_disabled(self, monkeypatch):
        # Gate is frozen by design - swap the whole object, not a field.
        monkeypatch.setattr(config, "GATE", dataclasses.replace(GATE, require_significance=False))
        run = evaluate([suite("s", 6, 10)], baseline=base(0.90, 10))
        assert any("regressed" in r for r in run.suites[0].reasons)


class TestFlaky:
    def _flaky_suite(self):
        # Case "a" passes twice and fails once; every other case is clean.
        trials = [
            TrialResult(case_id="a", repeat=0, passed=True, score=1.0),
            TrialResult(case_id="a", repeat=1, passed=True, score=1.0),
            TrialResult(case_id="a", repeat=2, passed=False, score=0.0),
        ] + [
            TrialResult(case_id=f"c{i}", repeat=r, passed=True, score=1.0)
            for i in range(4)
            for r in range(3)
        ]
        return SuiteResult(name="s", trials=trials)

    def test_flakiness_is_detected_and_reported(self):
        run = evaluate([self._flaky_suite()], baseline={})
        [verdict] = run.suites
        assert verdict.flaky == ["a"]
        assert verdict.status == "FLAKY"
        assert not verdict.failed  # reported, not gated, by default

    def test_flakiness_gates_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "GATE", dataclasses.replace(GATE, fail_on_flaky=True))
        run = evaluate([self._flaky_suite()], baseline={})
        assert run.failed
        assert any("flaky" in r for r in run.suites[0].reasons)


class TestBudget:
    def test_expensive_run_fails_even_when_scores_are_perfect(self):
        spend = Spend(input_tokens=10_000_000, output_tokens=10_000_000, api_calls=100)
        run = evaluate([suite("s", 10, 10)], baseline={}, spend=spend)
        assert run.over_budget
        assert run.failed
        assert any("budget" in r for r in run.reasons)

    def test_cheap_run_passes(self):
        spend = Spend(input_tokens=1000, output_tokens=200, api_calls=5)
        run = evaluate([suite("s", 10, 10)], baseline={}, spend=spend)
        assert not run.over_budget
        assert not run.failed
        assert run.usd == pytest.approx(0.01, abs=0.01)


class TestInfrastructureFailures:
    """A run that never reached the model must not be reported as a quality score.

    Auth, network, and rate-limit failures make every trial fail, which is
    arithmetically identical to a 0% pass rate. Reporting that as a regression
    blames the prompt for a broken pipe, and a gate that cries wolf gets muted.
    """

    def test_all_trials_errored_is_reported_as_error_not_fail(self):
        verdict = evaluate([errored_suite("s", 6)], base(1.0, 60))
        assert verdict.suites[0].status == "ERROR"
        assert verdict.suites[0].errored is True

    def test_an_errored_suite_still_fails_the_build(self):
        verdict = evaluate([errored_suite("s", 6)], base(1.0, 60))
        assert verdict.failed is True

    def test_the_reason_names_the_underlying_error(self):
        verdict = evaluate([errored_suite("s", 6)], base(1.0, 60))
        reason = " ".join(verdict.suites[0].reasons)
        assert "authentication" in reason
        assert "6 trials errored" in reason

    def test_an_errored_suite_is_not_blamed_for_regressing(self):
        verdict = evaluate([errored_suite("s", 6)], base(1.0, 60))
        reasons = " ".join(verdict.suites[0].reasons)
        assert "regressed" not in reasons
        assert "below the floor" not in reasons
        assert verdict.suites[0].delta is None

    def test_a_run_of_only_errored_suites_is_flagged_as_errored(self):
        verdict = evaluate([errored_suite("s", 6)], base(1.0, 60))
        assert verdict.errored is True

    def test_a_genuine_zero_score_is_still_a_normal_failure(self):
        verdict = evaluate([suite("s", 0, 6)], base(1.0, 60))
        assert verdict.suites[0].status == "FAIL"
        assert verdict.suites[0].errored is False
        assert verdict.errored is False

    def test_a_partly_errored_suite_is_still_scored(self):
        trials = errored_suite("s", 3).trials + suite("s", 3, 3).trials
        result = SuiteResult(name="s", trials=trials)
        assert result.errored is False
        assert result.n_errors == 3
        assert evaluate([result], base(1.0, 60)).suites[0].status != "ERROR"


class TestBaselineContamination:
    """A baseline measured through a broken pipe is worse than no new baseline.

    Partial errors still count as failed trials, so a run that lost two thirds
    of its calls to a 401 looks like a collapse in quality. Recording that would
    lower the bar every later run is compared against.
    """

    def test_errored_trials_block_the_write(self, tmp_path):
        results = [suite("s", 4, 6), errored_suite("e", 3)]
        with pytest.raises(ContaminatedBaseline):
            write_baseline(results, tmp_path / "main.json")

    def test_a_partly_errored_suite_also_blocks_the_write(self, tmp_path):
        mixed = SuiteResult(name="s", trials=errored_suite("s", 2).trials + suite("s", 4, 4).trials)
        with pytest.raises(ContaminatedBaseline):
            write_baseline([mixed], tmp_path / "main.json")

    def test_the_refusal_names_the_suite_and_the_error(self, tmp_path):
        with pytest.raises(ContaminatedBaseline) as exc:
            write_baseline([errored_suite("e", 3)], tmp_path / "main.json")
        message = str(exc.value)
        assert "e 3/3" in message
        assert "authentication" in message

    def test_nothing_is_written_when_it_refuses(self, tmp_path):
        path = tmp_path / "main.json"
        with pytest.raises(ContaminatedBaseline):
            write_baseline([errored_suite("e", 3)], path)
        assert not path.exists()

    def test_a_clean_run_still_records(self, tmp_path):
        path = write_baseline([suite("s", 5, 6)], tmp_path / "main.json")
        recorded = json.loads(path.read_text())
        assert recorded["suites"]["s"]["successes"] == 5
        assert recorded["suites"]["s"]["trials"] == 6
