"""Gate logic tests - the part that decides whether CI goes red."""

import dataclasses

import pytest

from evals import config
from evals.config import GATE
from evals.cost import Spend
from evals.gate import _BaselineEntry, evaluate
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
