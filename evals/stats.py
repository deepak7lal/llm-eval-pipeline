"""Statistics for eval scores.

LLM evals are noisy: the same suite run twice will not give the same number.
A gate that fires on any downward movement cries wolf, and a gate that ignores
movement misses real regressions. Both problems are the same problem - a pass
rate is a sample, and samples need error bars.

Everything here is closed-form, so the harness stays dependency-free (no scipy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided normal critical values. Enough resolution for a CI gate; the exact
# inverse normal CDF is not worth a dependency here.
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}


def z_for(alpha: float) -> float:
    """Critical value for a two-sided test at significance `alpha`."""
    if alpha not in _Z:
        raise ValueError(f"unsupported alpha {alpha}; choose one of {sorted(_Z)}")
    return _Z[alpha]


@dataclass(frozen=True)
class Interval:
    """A confidence interval on a proportion."""

    point: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"


def wilson(successes: int, trials: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because eval suites are small and
    pass rates sit near 1.0, where the naive interval runs past 100% or collapses
    to zero width at a perfect score.
    """
    if trials <= 0:
        return Interval(0.0, 0.0, 1.0)

    z = z_for(alpha)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_p_value(s1: int, n1: int, s2: int, n2: int) -> float:
    """Two-sided p-value for H0: the two pass rates come from the same process.

    Pooled two-proportion z-test. Used to ask whether a drop against the
    baseline is real or just sampling noise.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0

    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    if pooled in (0.0, 1.0):
        # Both runs perfect or both zero - no evidence of a difference.
        return 1.0

    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0

    z = abs(p1 - p2) / se
    # Two-sided tail of the standard normal, via the error function.
    return math.erfc(z / math.sqrt(2))


def is_significant_drop(
    *,
    current_successes: int,
    current_trials: int,
    baseline_rate: float,
    baseline_trials: int,
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Did the score drop by more than noise explains?

    Returns `(significant, p_value)`. Only downward movement counts - an
    improvement is never a regression, however large.
    """
    if current_trials <= 0 or baseline_trials <= 0:
        return False, 1.0
    if current_successes / current_trials >= baseline_rate:
        return False, 1.0

    baseline_successes = round(baseline_rate * baseline_trials)
    p = two_proportion_p_value(
        current_successes, current_trials, baseline_successes, baseline_trials
    )
    return p < alpha, p


def required_trials(effect: float, base_rate: float = 0.9, alpha: float = 0.05, power: float = 0.80) -> int:
    """Trials per arm needed to detect a drop of `effect` from `base_rate`.

    Reported by `evals power` so suite size is a decision, not an accident.
    """
    if not 0 < effect < base_rate:
        raise ValueError("effect must be positive and smaller than the base rate")

    z_alpha = z_for(alpha)
    z_beta = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}.get(power)
    if z_beta is None:
        raise ValueError("power must be one of 0.80, 0.90, 0.95")

    p1, p2 = base_rate, base_rate - effect
    pbar = (p1 + p2) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (effect**2))
