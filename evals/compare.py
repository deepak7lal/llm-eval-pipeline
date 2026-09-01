"""A/B comparison between two prompt variants.

Prompt engineering without a comparison harness is guesswork: a change that
"looks better" on three hand-picked examples routinely loses on the suite. This
runs both arms over the same cases and reports whether the difference is real or
noise, using the same two-proportion test the regression gate uses.

Define variants in the suite YAML:

    variants:
      terse: |
        You classify support messages. Reply with one label.

Then: `python -m evals compare --suite classification --variant terse`
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .cost import Spend
from .runner import SuiteResult, run_suite
from .stats import Interval, two_proportion_p_value, wilson


@dataclass
class Comparison:
    suite: str
    control_name: str
    treatment_name: str
    control: SuiteResult
    treatment: SuiteResult
    p_value: float

    @property
    def delta(self) -> float:
        return self.treatment.pass_rate - self.control.pass_rate

    @property
    def significant(self) -> bool:
        return self.p_value < config.ALPHA

    @property
    def verdict(self) -> str:
        """Plain-language call, so nobody has to interpret a p-value at 2am."""
        if not self.significant:
            return "no significant difference"
        return "treatment wins" if self.delta > 0 else "control wins"

    @property
    def delta_interval(self) -> Interval:
        """Rough interval on the difference, via the two arms' own intervals.

        Wide by construction - a conservative read is the right one when the
        decision is 'ship this prompt or not'.
        """
        c, t = self.control.interval, self.treatment.interval
        return Interval(self.delta, t.low - c.high, t.high - c.low)

    @property
    def spend(self) -> Spend:
        return self.control.spend + self.treatment.spend

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "control": {
                "name": self.control_name,
                "pass_rate": round(self.control.pass_rate, 4),
                "trials": self.control.n_trials,
            },
            "treatment": {
                "name": self.treatment_name,
                "pass_rate": round(self.treatment.pass_rate, 4),
                "trials": self.treatment.n_trials,
            },
            "delta": round(self.delta, 4),
            "p_value": round(self.p_value, 5),
            "significant": self.significant,
            "verdict": self.verdict,
            "usd": round(self.spend.usd(config.MODEL), 5),
        }


def compare(
    suite: dict,
    treatment: str,
    *,
    control: str = "baseline",
    repeats: int | None = None,
    use_cache: bool = True,
) -> Comparison:
    """Run both arms over the same cases and test the difference.

    Repeats default higher than a normal run: with one sample per case, a
    six-case suite cannot distinguish a real 10-point gain from a coin flip.
    """
    repeats = repeats or max(config.REPEATS, 3)

    control_result = run_suite(suite, repeats=repeats, variant=control, use_cache=use_cache)
    treatment_result = run_suite(suite, repeats=repeats, variant=treatment, use_cache=use_cache)

    p = two_proportion_p_value(
        control_result.successes,
        control_result.n_trials,
        treatment_result.successes,
        treatment_result.n_trials,
    )
    return Comparison(
        suite=suite["name"],
        control_name=control,
        treatment_name=treatment,
        control=control_result,
        treatment=treatment_result,
        p_value=p,
    )


def to_markdown(comparisons: list[Comparison]) -> str:
    """Scorecard for a comparison run."""
    lines = [
        "## Prompt variant comparison",
        "",
        f"`{config.MODEL}` - significance at alpha = {config.ALPHA}.",
        "",
        "| Suite | Control | Treatment | Delta | p | Verdict |",
        "|---|---:|---:|---:|---:|:--|",
    ]
    for c in comparisons:
        lines.append(
            f"| `{c.suite}` | {c.control_name} {c.control.pass_rate:.1%} "
            f"({c.control.successes}/{c.control.n_trials}) "
            f"| {c.treatment_name} {c.treatment.pass_rate:.1%} "
            f"({c.treatment.successes}/{c.treatment.n_trials}) "
            f"| {c.delta:+.1%} | {c.p_value:.3f} | **{c.verdict}** |"
        )

    inconclusive = [c for c in comparisons if not c.significant]
    if inconclusive:
        lines += ["", "### Underpowered comparisons", ""]
        for c in inconclusive:
            needed = _suggested_trials(c)
            lines.append(
                f"- `{c.suite}`: {c.delta:+.1%} could be noise. "
                f"About {needed} trials per arm would resolve a difference this size."
            )

    total = Spend()
    for c in comparisons:
        total = total + c.spend
    lines += ["", f"Cost: {total.summary(config.MODEL)}", ""]
    return "\n".join(lines)


def _suggested_trials(c: Comparison) -> int:
    """Trials per arm needed to call a difference of the observed size."""
    from .stats import required_trials

    effect = abs(c.delta)
    base = max(c.control.pass_rate, 0.05)
    if effect < 0.01:
        # Below ~1 point, the sample size needed is enormous and the difference
        # almost certainly does not matter - say so rather than print a number.
        return 10_000
    try:
        return required_trials(effect=min(effect, base - 0.01), base_rate=base, alpha=config.ALPHA)
    except ValueError:
        return 10_000


def unused_variants(suite: dict) -> list[str]:
    """Variants declared in YAML but never compared - surfaced by `evals list`."""
    return sorted((suite.get("variants") or {}).keys())


__all__ = ["Comparison", "compare", "to_markdown", "unused_variants", "wilson"]
