"""Dataset files must stay loadable and internally consistent."""

import pytest

from evals.graders import GRADERS
from evals.runner import discover_suites


@pytest.fixture(scope="module")
def suites():
    found = discover_suites()
    assert found, "no suites discovered under evals/datasets"
    return found


def test_every_grader_type_is_known(suites):
    for suite in suites:
        for case in suite["cases"]:
            for spec in case["graders"]:
                assert spec["type"] in GRADERS, f"{suite['name']}/{case['id']}: {spec['type']}"


def test_suite_names_are_unique(suites):
    names = [s["name"] for s in suites]
    assert len(names) == len(set(names))


def test_every_case_has_at_least_one_grader(suites):
    for suite in suites:
        for case in suite["cases"]:
            assert case["graders"], f"{suite['name']}/{case['id']} has no graders"
