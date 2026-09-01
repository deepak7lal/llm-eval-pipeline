"""Runner aggregation and suite-loading tests. No API calls."""

import pytest
import yaml

from evals.runner import SuiteResult, TrialResult, load_suite, system_for


def trial(case_id, repeat, passed, score=None):
    return TrialResult(
        case_id=case_id,
        repeat=repeat,
        passed=passed,
        score=float(passed) if score is None else score,
    )


class TestAggregation:
    def test_pass_rate_counts_trials_not_cases(self):
        # One case sampled 4 times, green 3 of them: 75%, not 0% or 100%.
        result = SuiteResult(
            name="s",
            trials=[trial("a", i, i < 3) for i in range(4)],
        )
        assert result.n_trials == 4
        assert result.successes == 3
        assert result.pass_rate == pytest.approx(0.75)
        assert len(result.cases) == 1

    def test_case_rollup_preserves_first_seen_order(self):
        result = SuiteResult(
            name="s",
            trials=[trial("b", 0, True), trial("a", 0, True), trial("b", 1, True)],
        )
        assert [c.id for c in result.cases] == ["b", "a"]

    def test_flaky_case_is_neither_passed_nor_cleanly_failed(self):
        result = SuiteResult(name="s", trials=[trial("a", 0, True), trial("a", 1, False)])
        [case] = result.cases
        assert case.flaky
        assert not case.passed
        assert result.flaky_cases == [case]

    def test_consistent_case_is_not_flaky(self):
        for passed in (True, False):
            result = SuiteResult(name="s", trials=[trial("a", i, passed) for i in range(3)])
            assert not result.cases[0].flaky

    def test_empty_suite_does_not_divide_by_zero(self):
        result = SuiteResult(name="s", trials=[])
        assert result.pass_rate == 0.0
        assert result.mean_score == 0.0
        assert result.interval.high == 1.0

    def test_to_dict_is_json_serialisable(self):
        import json

        result = SuiteResult(name="s", trials=[trial("a", 0, True), trial("a", 1, False)])
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["flaky_cases"] == ["a"]
        assert payload["trials"] == 2


class TestLoadSuite:
    def _write(self, tmp_path, data):
        path = tmp_path / "suite.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    def _valid(self):
        return {
            "name": "s",
            "system": "do the thing",
            "cases": [{"id": "a", "input": "x", "graders": [{"type": "exact", "value": "x"}]}],
        }

    def test_loads_a_valid_suite(self, tmp_path):
        assert load_suite(self._write(tmp_path, self._valid()))["name"] == "s"

    def test_rejects_missing_top_level_key(self, tmp_path):
        data = self._valid()
        del data["system"]
        with pytest.raises(ValueError, match="missing required key"):
            load_suite(self._write(tmp_path, data))

    def test_rejects_duplicate_case_ids(self, tmp_path):
        data = self._valid()
        data["cases"] = data["cases"] * 2
        with pytest.raises(ValueError, match="duplicate case id"):
            load_suite(self._write(tmp_path, data))

    def test_rejects_case_without_graders_key(self, tmp_path):
        data = self._valid()
        del data["cases"][0]["graders"]
        with pytest.raises(ValueError, match="missing 'graders'"):
            load_suite(self._write(tmp_path, data))

    def test_rejects_reserved_variant_name(self, tmp_path):
        data = self._valid()
        data["variants"] = {"baseline": "nope"}
        with pytest.raises(ValueError, match="reserved"):
            load_suite(self._write(tmp_path, data))

    def test_rejects_non_mapping_variants(self, tmp_path):
        data = self._valid()
        data["variants"] = ["terse"]
        with pytest.raises(ValueError, match="must be a mapping"):
            load_suite(self._write(tmp_path, data))


class TestSystemFor:
    def test_baseline_uses_the_suite_prompt(self):
        suite = {"name": "s", "system": "base", "variants": {"terse": "short"}}
        assert system_for(suite, "baseline") == "base"

    def test_named_variant_overrides_it(self):
        suite = {"name": "s", "system": "base", "variants": {"terse": "short"}}
        assert system_for(suite, "terse") == "short"

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="no variant"):
            system_for({"name": "s", "system": "base"}, "nope")
