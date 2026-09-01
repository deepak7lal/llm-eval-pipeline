"""Renders results as a Markdown scorecard and a machine-readable JSON artifact.

The scorecard is written for someone skimming a PR, so it leads with the verdict
and the numbers that could change a decision: score with its confidence
interval, movement against the baseline, flakiness, and cost.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .cost import Spend, unknown_models
from .gate import RunVerdict
from .runner import SuiteResult

_ICON = {"PASS": "[PASS]", "WARN": "[WARN]", "FLAKY": "[FLAKY]", "FAIL": "[FAIL]"}


def _delta(delta: float | None) -> str:
    return "n/a" if delta is None else f"{delta:+.1%}"


def to_markdown(results: list[SuiteResult], run: RunVerdict) -> str:
    """A scorecard compact enough to post as a PR comment."""
    by_name = {v.suite: v for v in run.suites}
    overall = "FAIL" if run.failed else "PASS"

    total_trials = sum(r.n_trials for r in results)
    total_successes = sum(r.successes for r in results)
    spend = Spend()
    for result in results:
        spend = spend + result.spend

    lines = [
        f"## LLM eval scorecard - {_ICON[overall]}",
        "",
        f"`{config.MODEL}` at effort `{config.EFFORT}` - "
        f"**{total_successes}/{total_trials} trials passed** "
        f"across {len(results)} suite(s). Cost: **${run.usd:.4f}** "
        f"of a ${run.budget:.2f} budget.",
        "",
        "| Suite | Pass rate | 95% CI | Baseline | Delta | Trials | Flaky | Status |",
        "|---|---:|:--|---:|---:|---:|---:|:--|",
    ]
    for result in results:
        v = by_name[result.name]
        interval = result.interval
        baseline = "n/a" if v.baseline is None else f"{v.baseline:.1%}"
        lines.append(
            f"| `{result.name}` | {result.pass_rate:.1%} "
            f"| [{interval.low:.0%}, {interval.high:.0%}] "
            f"| {baseline} | {_delta(v.delta)} "
            f"| {result.successes}/{result.n_trials} "
            f"| {len(v.flaky) or '-'} | {_ICON[v.status]} |"
        )

    if run.failed:
        lines += ["", "### Why the build failed", ""]
        lines += [f"- {reason}" for reason in run.reasons]

    # A drop that did not clear the significance bar still deserves a mention -
    # two of those in a row is a trend, not noise.
    noted = [
        (v.suite, r)
        for v in run.suites
        if not v.failed
        for r in v.reasons
        if "sampling noise" in r
    ]
    if noted:
        lines += ["", "### Movement within noise", ""]
        lines += [f"- `{suite}`: {reason}" for suite, reason in noted]

    flaky = [(r.name, c) for r in results for c in r.flaky_cases]
    if flaky:
        lines += ["", "### Flaky cases", "", "Passed on some repeats and failed on others:", ""]
        lines += [
            f"- **`{suite}` / `{c.id}`** - {c.passes}/{c.trials} passed" for suite, c in flaky
        ]

    failures = [(r.name, c) for r in results for c in r.cases if not c.passed and not c.flaky]
    if failures:
        lines += ["", f"<details><summary>Failed cases ({len(failures)})</summary>", ""]
        for suite_name, case in failures:
            reason = case.error or "; ".join(case.details) or "no detail"
            lines.append(f"- **`{suite_name}` / `{case.id}`** - {reason}")
        lines += ["", "</details>"]

    lines += ["", f"<sub>{spend.summary(config.MODEL)}</sub>"]

    missing = unknown_models([config.MODEL, config.JUDGE_MODEL])
    if missing:
        lines += [
            "",
            f"> Cost shown excludes {', '.join(missing)} - no price on file in `evals/cost.py`.",
        ]

    return "\n".join(lines) + "\n"


def write_reports(results: list[SuiteResult], run: RunVerdict) -> tuple[Path, Path]:
    """Write scorecard.md and results.json under reports/."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    md_path = config.REPORTS_DIR / "scorecard.md"
    md_path.write_text(to_markdown(results, run), encoding="utf-8")

    json_path = config.REPORTS_DIR / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "model": config.MODEL,
                "effort": config.EFFORT,
                "repeats": config.REPEATS,
                "alpha": config.ALPHA,
                "failed": run.failed,
                "usd": round(run.usd, 5),
                "budget_usd": run.budget,
                "over_budget": run.over_budget,
                "suites": [r.to_dict() for r in results],
                "verdicts": [
                    {
                        "suite": v.suite,
                        "pass_rate": v.pass_rate,
                        "ci_low": v.interval.low,
                        "ci_high": v.interval.high,
                        "baseline": v.baseline,
                        "delta": v.delta,
                        "p_value": v.p_value,
                        "flaky": v.flaky,
                        "status": v.status,
                        "reasons": v.reasons,
                    }
                    for v in run.suites
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return md_path, json_path
