"""End-to-end tests with a stubbed model.

These exercise the real runner, gate, report, and CLI against the real dataset
files - everything except the network. A change that breaks the pipeline wiring
fails here, in CI, for free, instead of after a paid run.
"""

import json

import pytest

from evals import cli, providers
from evals import compare as compare_mod
from evals import config
from evals.client import Completion
from evals.runner import discover_suites, run_suite


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Route every model call to a scripted stub and isolate all output paths."""
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "BASELINE_PATH", tmp_path / "baseline.json")

    def stub(*, system, user, model=None, effort=None, max_tokens=None, repeat=0, use_cache=True):
        # The classification suite's expected labels, keyed by a phrase in the
        # input, so most cases pass and one fails deterministically.
        table = {
            "card was declined": "billing",
            "crashes every time": "bug",
            "dark mode": "feature_request",
            "locked out": "account_access",
            "looks great": "other",
            "invoice": "wrong_label",  # deliberate miss
        }
        answer = next((v for k, v in table.items() if k in user), "other")
        return Completion(
            text=answer,
            input_tokens=100,
            output_tokens=5,
            cache_read_tokens=0,
            latency_ms=42,
            stop_reason="end_turn",
        )

    monkeypatch.setattr("evals.runner.complete", stub)
    return tmp_path


@pytest.fixture
def classification():
    return next(s for s in discover_suites() if s["name"] == "classification")


class TestRunSuite:
    def test_scores_the_real_dataset(self, offline, classification):
        result = run_suite(classification, repeats=1, use_cache=False)
        assert result.n_trials == len(classification["cases"])
        # Five of six labels are right in the stub table.
        assert result.successes == result.n_trials - 1
        assert "billing-disguised-as-bug" in [c.id for c in result.cases if not c.passed]

    def test_repeats_multiply_trials_without_changing_the_rate(self, offline, classification):
        one = run_suite(classification, repeats=1, use_cache=False)
        three = run_suite(classification, repeats=3, use_cache=False)
        assert three.n_trials == one.n_trials * 3
        assert three.pass_rate == pytest.approx(one.pass_rate)
        # A deterministic stub cannot be flaky.
        assert three.flaky_cases == []

    def test_more_trials_narrow_the_interval(self, offline, classification):
        one = run_suite(classification, repeats=1, use_cache=False)
        five = run_suite(classification, repeats=5, use_cache=False)
        assert five.interval.width < one.interval.width

    def test_spend_is_accumulated(self, offline, classification):
        result = run_suite(classification, repeats=2, use_cache=False)
        assert result.spend.api_calls == result.n_trials
        assert result.spend.input_tokens == 100 * result.n_trials
        assert result.spend.usd("claude-opus-5") > 0


class TestCli:
    def test_run_writes_both_reports(self, offline, monkeypatch):
        monkeypatch.setattr(config, "MIN_PASS_RATE", 0.0, raising=False)
        code = cli.main(["run", "--suite", "classification", "--no-gate", "--no-cache"])
        assert code == 0

        scorecard = (offline / "reports" / "scorecard.md").read_text(encoding="utf-8")
        assert "LLM eval scorecard" in scorecard
        assert "classification" in scorecard
        assert "95% CI" in scorecard

        payload = json.loads((offline / "reports" / "results.json").read_text(encoding="utf-8"))
        assert payload["suites"][0]["name"] == "classification"
        assert "ci_low" in payload["verdicts"][0]

    def test_gate_fails_the_process_on_a_bad_score(self, offline, monkeypatch):
        # Stub every answer wrong, so the floor is definitely breached.
        monkeypatch.setattr(
            "evals.runner.complete",
            lambda **kw: Completion("nonsense", 10, 2, 0, 1, "end_turn"),
        )
        assert cli.main(["run", "--suite", "classification", "--no-cache"]) == 1

    def test_no_gate_always_exits_zero(self, offline, monkeypatch):
        monkeypatch.setattr(
            "evals.runner.complete",
            lambda **kw: Completion("nonsense", 10, 2, 0, 1, "end_turn"),
        )
        assert cli.main(["run", "--suite", "classification", "--no-cache", "--no-gate"]) == 0

    def test_bare_invocation_defaults_to_run(self, offline):
        assert cli.main(["--suite", "classification", "--no-gate", "--no-cache"]) == 0

    def test_update_baseline_records_trials(self, offline):
        cli.main(["run", "--suite", "classification", "--no-gate", "--no-cache", "--update-baseline"])
        payload = json.loads((offline / "baseline.json").read_text(encoding="utf-8"))
        entry = payload["suites"]["classification"]
        assert entry["trials"] == 6
        assert entry["successes"] == 5

    def test_unknown_suite_is_an_error(self, offline):
        with pytest.raises(ValueError, match="no such suite"):
            cli.main(["run", "--suite", "does-not-exist"])

    def test_power_reports_required_trials(self, offline, capsys):
        assert cli.main(["power", "--effect", "0.10", "--suite", "classification"]) == 0
        out = capsys.readouterr().out
        assert "trials per arm" in out
        assert "classification" in out

    def test_compare_skips_suites_without_the_variant(self, offline, capsys):
        assert cli.main(["compare", "--variant", "nope"]) == 2
        assert "no suite declares a variant" in capsys.readouterr().err


class TestCompare:
    def test_identical_arms_are_not_significant(self, offline, classification):
        result = compare_mod.compare(classification, treatment="terse", repeats=3, use_cache=False)
        # The stub ignores the system prompt, so both arms score identically.
        assert result.delta == pytest.approx(0.0)
        assert not result.significant
        assert result.verdict == "no significant difference"

    def test_markdown_names_the_arms(self, offline, classification):
        result = compare_mod.compare(classification, treatment="terse", repeats=2, use_cache=False)
        md = compare_mod.to_markdown([result])
        assert "terse" in md
        assert "baseline" in md
        assert "Cost:" in md

    def test_to_dict_is_serialisable(self, offline, classification):
        result = compare_mod.compare(classification, treatment="terse", repeats=2, use_cache=False)
        assert json.loads(json.dumps(result.to_dict()))["suite"] == "classification"


def test_no_provider_was_ever_constructed(offline):
    """The stub must be what ran - no accidental live client construction."""
    assert providers._cached is None
