"""Grader unit tests. These never call the API, so they run on every push."""

import pytest

from evals.graders import grade


def _case(text="task text"):
    return {"id": "t", "input": text, "graders": []}


class TestExact:
    def test_matches_ignoring_case_and_whitespace(self):
        assert grade("  Billing \n", {"type": "exact", "value": "billing"}, _case()).passed

    def test_rejects_extra_words(self):
        result = grade("billing issue", {"type": "exact", "value": "billing"}, _case())
        assert not result.passed
        assert "expected" in result.detail

    def test_case_sensitive_when_asked(self):
        spec = {"type": "exact", "value": "Billing", "ignore_case": False}
        assert not grade("billing", spec, _case()).passed


class TestContains:
    def test_all_terms_present(self):
        spec = {"type": "contains", "all_of": ["checkout", "failover"]}
        assert grade("Checkout broke during failover.", spec, _case()).passed

    def test_partial_credit_for_missing_term(self):
        spec = {"type": "contains", "all_of": ["checkout", "failover"]}
        result = grade("Checkout broke.", spec, _case())
        assert not result.passed
        assert result.score == pytest.approx(0.5)

    def test_forbidden_term_zeroes_the_score(self):
        spec = {"type": "contains", "all_of": ["checkout"], "none_of": ["timestamp"]}
        result = grade("Checkout broke at timestamp 02:14.", spec, _case())
        assert not result.passed
        assert result.score == 0.0


class TestRegex:
    def test_match(self):
        assert grade('{"a": 1}', {"type": "regex", "pattern": r"^\s*\{"}, _case()).passed

    def test_no_match(self):
        assert not grade("nope", {"type": "regex", "pattern": r"^\s*\{"}, _case()).passed


class TestJsonFields:
    def test_all_fields_match(self):
        spec = {"type": "json_fields", "equals": {"order_id": "A-1", "quantity": 3}}
        assert grade('{"order_id": "A-1", "quantity": 3}', spec, _case()).passed

    def test_strips_code_fence(self):
        spec = {"type": "json_fields", "equals": {"quantity": 1}}
        assert grade('```json\n{"quantity": 1}\n```', spec, _case()).passed

    def test_partial_score_on_one_bad_field(self):
        spec = {"type": "json_fields", "equals": {"order_id": "A-1", "quantity": 3}}
        result = grade('{"order_id": "A-1", "quantity": 9}', spec, _case())
        assert not result.passed
        assert result.score == pytest.approx(0.5)
        assert "quantity" in result.detail

    def test_invalid_json_scores_zero(self):
        spec = {"type": "json_fields", "equals": {"a": 1}}
        result = grade("not json at all", spec, _case())
        assert not result.passed
        assert result.score == 0.0
        assert "invalid JSON" in result.detail


def test_unknown_grader_raises():
    with pytest.raises(ValueError, match="unknown grader"):
        grade("x", {"type": "vibes"}, _case())
