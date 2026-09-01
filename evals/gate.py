"""The quality gate.

Three independent ways a run can fail:

  1. **Floor** - a suite's pass rate falls below the absolute minimum.
  2. **Regression** - a suite drops against the baseline by more than the
     allowance, *and* the drop is statistically significant. Requiring both is
     what stops the gate firing on noise; see `stats.is_significant_drop`.
  3. **Budget** - the run cost more than the ceiling, whatever the scores did.

Flakiness (a case that passes on some repeats and fails on others) is always
reported and optionally gated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config
from .cost import BUDGET_USD, Spend, over_budget
from .runner import SuiteResult
from .stats import Interval, is_significant_drop


@dataclass
class SuiteVerdict:
    suite: str
    pass_rate: float
    interval: Interval
    baseline: float | None
    delta: float | None
    p_value: float | None
    flaky: list[str]
    failed: bool
    reasons: list[str]

    @property
    def status(self) -> str:
        if self.failed:
            return "FAIL"
        if self.flaky:
            return "FLAKY"
        if self.delta is not None and self.delta < 0:
            return "WARN"
        return "PASS"


@dataclass
class RunVerdict:
    """The whole run: per-suite verdicts plus the budget check."""

    suites: list[SuiteVerdict]
    usd: float
    budget: float
    over_budget: bool

    @property
    def failed(self) -> bool:
        return self.over_budget or any(s.failed for s in self.suites)

    @property
    def reasons(self) -> list[str]:
        out = [f"`{s.suite}`: {r}" for s in self.suites if s.failed for r in s.reasons]
        if self.over_budget:
            out.append(f"run cost ${self.usd:.2f} exceeded the ${self.budget:.2f} budget")
        return out


@dataclass
class _BaselineEntry:
    pass_rate: float
    trials: int


def load_baseline(path: Path | None = None) -> dict[str, _BaselineEntry]:
    """Baseline pass rates and sample sizes; empty when none is committed.

    The trial count matters: comparing 90% of 10 against 90% of 1000 is not the
    same comparison, and the significance test needs both sample sizes.
    """
    path = path or config.BASELINE_PATH
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, _BaselineEntry] = {}
    for name, entry in data.get("suites", {}).items():
        out[name] = _BaselineEntry(
            pass_rate=float(entry["pass_rate"]),
            # Older baselines recorded case counts only; treat those as trials.
            trials=int(entry.get("trials", entry.get("cases", 0))),
        )
    return out


def evaluate(
    results: list[SuiteResult],
    baseline: dict[str, _BaselineEntry] | None = None,
    *,
    spend: Spend | None = None,
) -> RunVerdict:
    """Apply the gate to every suite and to the run's total spend."""
    baseline = load_baseline() if baseline is None else baseline
    verdicts: list[SuiteVerdict] = []

    for result in results:
        gate = config.gate_for(result.name)
        prior = baseline.get(result.name)
        delta = None if prior is None else result.pass_rate - prior.pass_rate
        reasons: list[str] = []
        p_value: float | None = None
        gating = result.n_trials >= gate.min_cases_to_gate

        if result.pass_rate < gate.min_pass_rate:
            reasons.append(
                f"pass rate {result.pass_rate:.1%} is below the floor of {gate.min_pass_rate:.0%}"
            )

        if prior is not None and delta is not None and -delta > gate.max_regression:
            significant, p_value = is_significant_drop(
                current_successes=result.successes,
                current_trials=result.n_trials,
                baseline_rate=prior.pass_rate,
                baseline_trials=prior.trials,
                alpha=config.ALPHA,
            )
            if significant or not gate.require_significance:
                reasons.append(
                    f"regressed {-delta:.1%} against baseline {prior.pass_rate:.1%} "
                    f"(allowance {gate.max_regression:.0%}, p={p_value:.3f})"
                )
            else:
                reasons.append(
                    f"drop of {-delta:.1%} is within sampling noise (p={p_value:.3f}) - "
                    f"not gating, but worth a look"
                )

        flaky = [c.id for c in result.flaky_cases]
        if flaky and gate.fail_on_flaky:
            reasons.append(f"{len(flaky)} flaky case(s): {', '.join(flaky)}")

        blocking = [
            r for r in reasons if "within sampling noise" not in r and "worth a look" not in r
        ]
        if blocking and not gating:
            reasons.append(
                f"not gating: only {result.n_trials} trial(s), "
                f"minimum is {gate.min_cases_to_gate}"
            )

        verdicts.append(
            SuiteVerdict(
                suite=result.name,
                pass_rate=result.pass_rate,
                interval=result.interval,
                baseline=None if prior is None else prior.pass_rate,
                delta=delta,
                p_value=p_value,
                flaky=flaky,
                failed=bool(blocking) and gating,
                reasons=reasons,
            )
        )

    total = spend if spend is not None else _total_spend(results)
    exceeded, usd = over_budget(total, config.MODEL)
    return RunVerdict(suites=verdicts, usd=usd, budget=BUDGET_USD, over_budget=exceeded)


def _total_spend(results: list[SuiteResult]) -> Spend:
    total = Spend()
    for result in results:
        total = total + result.spend
    return total


def write_baseline(results: list[SuiteResult], path: Path | None = None) -> Path:
    """Persist current scores as the new baseline (run on main, not on PRs)."""
    path = path or config.BASELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": config.MODEL,
        "effort": config.EFFORT,
        "alpha": config.ALPHA,
        "suites": {
            r.name: {
                "pass_rate": round(r.pass_rate, 4),
                "mean_score": round(r.mean_score, 4),
                "cases": len(r.cases),
                "trials": r.n_trials,
                "successes": r.successes,
            }
            for r in results
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
