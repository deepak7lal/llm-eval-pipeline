"""Statistics tests. These protect the gate from both false alarms and blind spots."""

import pytest

from evals.stats import (
    is_significant_drop,
    required_trials,
    two_proportion_p_value,
    wilson,
)


class TestWilson:
    def test_point_estimate_is_the_raw_rate(self):
        assert wilson(9, 10).point == pytest.approx(0.9)

    def test_interval_brackets_the_estimate(self):
        interval = wilson(45, 50)
        assert interval.low < interval.point < interval.high

    def test_perfect_score_still_has_a_lower_bound_below_one(self):
        # The reason we use Wilson: 10/10 is not proof of 100%.
        interval = wilson(10, 10)
        assert interval.high == pytest.approx(1.0, abs=1e-9)
        assert interval.low < 0.8

    def test_interval_never_escapes_zero_to_one(self):
        for successes, trials in [(0, 3), (1, 3), (3, 3), (0, 1)]:
            interval = wilson(successes, trials)
            assert 0.0 <= interval.low <= interval.high <= 1.0

    def test_more_trials_narrow_the_interval(self):
        assert wilson(90, 100).width < wilson(9, 10).width

    def test_zero_trials_is_maximally_uncertain(self):
        interval = wilson(0, 0)
        assert (interval.low, interval.high) == (0.0, 1.0)


class TestTwoProportion:
    def test_identical_rates_are_not_significant(self):
        assert two_proportion_p_value(90, 100, 90, 100) == pytest.approx(1.0, abs=1e-6)

    def test_large_difference_on_large_samples_is_significant(self):
        assert two_proportion_p_value(70, 100, 95, 100) < 0.01

    def test_same_difference_on_tiny_samples_is_not(self):
        assert two_proportion_p_value(7, 10, 9, 10) > 0.05

    def test_both_perfect_is_not_a_difference(self):
        assert two_proportion_p_value(10, 10, 10, 10) == 1.0

    def test_empty_sample_is_not_a_difference(self):
        assert two_proportion_p_value(0, 0, 5, 10) == 1.0


class TestSignificantDrop:
    def test_improvement_is_never_a_regression(self):
        significant, p = is_significant_drop(
            current_successes=99, current_trials=100, baseline_rate=0.80, baseline_trials=100
        )
        assert not significant
        assert p == 1.0

    def test_real_drop_on_a_large_sample_is_caught(self):
        significant, p = is_significant_drop(
            current_successes=70, current_trials=100, baseline_rate=0.95, baseline_trials=100
        )
        assert significant
        assert p < 0.05

    def test_noisy_drop_on_a_small_sample_is_not(self):
        significant, _ = is_significant_drop(
            current_successes=6, current_trials=10, baseline_rate=0.90, baseline_trials=10
        )
        assert not significant


class TestPower:
    def test_smaller_effects_need_more_trials(self):
        assert required_trials(0.05, 0.90) > required_trials(0.20, 0.90)

    def test_rejects_an_impossible_effect(self):
        with pytest.raises(ValueError):
            required_trials(effect=0.95, base_rate=0.90)

    def test_rejects_unsupported_power(self):
        with pytest.raises(ValueError):
            required_trials(0.1, 0.9, power=0.5)
