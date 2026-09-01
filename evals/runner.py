"""Loads suites from YAML, runs every case concurrently, collects results.

Each case can be sampled more than once (`repeats`). A sample is a *trial*, and
a suite's score is successes over trials - which is what makes the confidence
intervals in `stats.py` mean anything. Sampling also exposes flakiness: a case
that passes 2 of 3 times is not the same signal as one that passes every time,
and the scorecard says so.
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from . import config
from .client import complete
from .cost import Spend
from .graders import grade
from .stats import Interval, wilson


@dataclass
class TrialResult:
    """One sample of one case."""

    case_id: str
    repeat: int
    passed: bool
    score: float
    details: list[str] = field(default_factory=list)
    output: str = ""
    error: str | None = None
    latency_ms: int = 0
    from_cache: bool = False
    spend: Spend = field(default_factory=Spend)


@dataclass
class CaseSummary:
    """All samples of one case, collapsed."""

    id: str
    passes: int
    trials: int
    mean_score: float
    details: list[str]
    error: str | None

    @property
    def passed(self) -> bool:
        return self.passes == self.trials

    @property
    def flaky(self) -> bool:
        """Passed sometimes and failed sometimes - worse than a clean failure."""
        return 0 < self.passes < self.trials


@dataclass
class SuiteResult:
    name: str
    trials: list[TrialResult]
    variant: str = "baseline"

    @property
    def successes(self) -> int:
        return sum(t.passed for t in self.trials)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def pass_rate(self) -> float:
        return self.successes / self.n_trials if self.trials else 0.0

    @property
    def interval(self) -> Interval:
        return wilson(self.successes, self.n_trials, alpha=config.ALPHA)

    @property
    def mean_score(self) -> float:
        return statistics.fmean(t.score for t in self.trials) if self.trials else 0.0

    @property
    def spend(self) -> Spend:
        total = Spend()
        for trial in self.trials:
            total = total + trial.spend
        return total

    @property
    def cases(self) -> list[CaseSummary]:
        """Per-case rollup, in first-seen order."""
        order: list[str] = []
        buckets: dict[str, list[TrialResult]] = {}
        for trial in self.trials:
            if trial.case_id not in buckets:
                buckets[trial.case_id] = []
                order.append(trial.case_id)
            buckets[trial.case_id].append(trial)

        summaries = []
        for case_id in order:
            group = buckets[case_id]
            details: list[str] = []
            for trial in group:
                for detail in trial.details:
                    if detail not in details:
                        details.append(detail)
            summaries.append(
                CaseSummary(
                    id=case_id,
                    passes=sum(t.passed for t in group),
                    trials=len(group),
                    mean_score=statistics.fmean(t.score for t in group),
                    details=details,
                    error=next((t.error for t in group if t.error), None),
                )
            )
        return summaries

    @property
    def flaky_cases(self) -> list[CaseSummary]:
        return [c for c in self.cases if c.flaky]

    def to_dict(self) -> dict:
        interval = self.interval
        return {
            "name": self.name,
            "variant": self.variant,
            "pass_rate": round(self.pass_rate, 4),
            "ci_low": round(interval.low, 4),
            "ci_high": round(interval.high, 4),
            "mean_score": round(self.mean_score, 4),
            "cases_total": len(self.cases),
            "cases_passed": sum(c.passed for c in self.cases),
            "trials": self.n_trials,
            "successes": self.successes,
            "flaky_cases": [c.id for c in self.flaky_cases],
            "spend": asdict(self.spend),
            "usd": round(self.spend.usd(config.MODEL), 5),
            "cases": [
                {
                    "id": c.id,
                    "passes": c.passes,
                    "trials": c.trials,
                    "mean_score": round(c.mean_score, 4),
                    "flaky": c.flaky,
                    "details": c.details,
                    "error": c.error,
                }
                for c in self.cases
            ],
        }


def load_suite(path: Path) -> dict:
    """Read and validate one suite YAML file."""
    suite = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("name", "system", "cases"):
        if key not in suite:
            raise ValueError(f"{path.name}: missing required key {key!r}")

    seen: set[str] = set()
    for case in suite["cases"]:
        for key in ("id", "input", "graders"):
            if key not in case:
                raise ValueError(f"{path.name}: case {case.get('id', '?')} missing {key!r}")
        if case["id"] in seen:
            raise ValueError(f"{path.name}: duplicate case id {case['id']!r}")
        seen.add(case["id"])

    variants = suite.get("variants") or {}
    if not isinstance(variants, dict):
        raise ValueError(f"{path.name}: 'variants' must be a mapping of name -> system prompt")
    if "baseline" in variants:
        raise ValueError(f"{path.name}: 'baseline' is reserved for the suite's own system prompt")
    return suite


def discover_suites(directory: Path | None = None, only: list[str] | None = None) -> list[dict]:
    directory = directory or config.DATASETS_DIR
    suites = [load_suite(p) for p in sorted(directory.glob("*.yaml"))]
    if only:
        suites = [s for s in suites if s["name"] in only]
        missing = set(only) - {s["name"] for s in suites}
        if missing:
            raise ValueError(f"no such suite(s): {sorted(missing)}")
    return suites


def system_for(suite: dict, variant: str) -> str:
    """The system prompt a variant runs with. 'baseline' is the suite's own."""
    if variant == "baseline":
        return suite["system"]
    variants = suite.get("variants") or {}
    if variant not in variants:
        raise ValueError(f"suite {suite['name']!r} has no variant {variant!r}")
    return variants[variant]


def run_trial(suite: dict, case: dict, repeat: int, variant: str, use_cache: bool) -> TrialResult:
    """One sample: one model call, then every grader attached to the case."""
    try:
        completion = complete(
            system=system_for(suite, variant),
            user=case["input"],
            effort=suite.get("effort"),
            max_tokens=suite.get("max_tokens"),
            repeat=repeat,
            use_cache=use_cache,
        )
    except Exception as exc:  # noqa: BLE001 - an API failure is a failed trial, not a crash
        return TrialResult(
            case_id=case["id"],
            repeat=repeat,
            passed=False,
            score=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )

    spend = Spend(
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cache_read_tokens=completion.cache_read_tokens,
        cache_write_tokens=completion.cache_write_tokens,
        api_calls=0 if completion.from_cache else 1,
        cached_calls=1 if completion.from_cache else 0,
    )

    if completion.stop_reason == "refusal":
        # Deliberately *not* routed to a fallback model: a refusal is a real
        # result about the model under test, and silently swapping models would
        # make the score incomparable to the baseline.
        return TrialResult(
            case_id=case["id"],
            repeat=repeat,
            passed=False,
            score=0.0,
            error="model refused the request (stop_reason=refusal)",
            latency_ms=completion.latency_ms,
            spend=spend,
        )

    results = [grade(completion.text, spec, case) for spec in case["graders"]]
    return TrialResult(
        case_id=case["id"],
        repeat=repeat,
        passed=all(r.passed for r in results),
        score=statistics.fmean(r.score for r in results),
        details=[r.detail for r in results if r.detail],
        output=completion.text[:2000],
        latency_ms=completion.latency_ms,
        from_cache=completion.from_cache,
        spend=spend,
    )


def run_suite(
    suite: dict,
    *,
    repeats: int | None = None,
    variant: str = "baseline",
    use_cache: bool = True,
) -> SuiteResult:
    """Run every case in a suite `repeats` times, bounded by EVAL_CONCURRENCY."""
    repeats = repeats or suite.get("repeats") or config.REPEATS
    work = [(case, r) for case in suite["cases"] for r in range(repeats)]

    with ThreadPoolExecutor(max_workers=config.CONCURRENCY) as pool:
        trials = list(
            pool.map(lambda item: run_trial(suite, item[0], item[1], variant, use_cache), work)
        )
    return SuiteResult(name=suite["name"], trials=trials, variant=variant)


def run_all(
    only: list[str] | None = None,
    *,
    repeats: int | None = None,
    variant: str = "baseline",
    use_cache: bool = True,
) -> list[SuiteResult]:
    return [
        run_suite(s, repeats=repeats, variant=variant, use_cache=use_cache)
        for s in discover_suites(only=only)
    ]
