"""Offline walkthrough of the pipeline. No API key, no cost, no network.

Every scenario below drives the real runner, gate, and report - only the model
is simulated, so what you see is what CI would print for the same scores.

    python scripts/demo.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import compare as compare_mod  # noqa: E402
from evals import config, report, runner  # noqa: E402
from evals.client import Completion  # noqa: E402
from evals.gate import _BaselineEntry, evaluate  # noqa: E402
from evals.stats import required_trials, wilson  # noqa: E402

# The right answer for each classification case, keyed by a phrase in the input.
TRUTH = {
    "card was declined": "billing",
    "crashes every time": "bug",
    "dark mode": "feature_request",
    "locked out": "account_access",
    "looks great": "other",
    "invoice": "billing",
}

WRONG = {"billing": "bug", "bug": "other", "feature_request": "other",
         "account_access": "billing", "other": "bug"}


def fake_model(accuracy: float, *, flaky_on: str | None = None, seed: int = 0):
    """A stand-in model that is right `accuracy` of the time.

    `flaky_on` names a phrase whose case answers inconsistently - the pattern a
    real model shows on a genuinely ambiguous input.
    """
    rng = random.Random(seed)

    def stub(*, system, user, model=None, effort=None, max_tokens=None, repeat=0, use_cache=True):
        correct = next((v for k, v in TRUTH.items() if k in user), "other")
        if flaky_on and flaky_on in user:
            answer = correct if rng.random() < 0.4 else WRONG[correct]
        else:
            answer = correct if rng.random() < accuracy else WRONG[correct]
        # Token counts in the range a real short-prompt call produces.
        return Completion(answer, 620, 6, 540, 180, "end_turn")

    return stub


def banner(n: int, title: str, subtitle: str) -> None:
    print(f"\n{'=' * 78}\n  {n}. {title}\n     {subtitle}\n{'=' * 78}\n")


def load(name: str) -> dict:
    return next(s for s in runner.discover_suites() if s["name"] == name)


def run(suite, *, accuracy, repeats, baseline=None, flaky_on=None, seed=0):
    """Score a suite against a simulated model and apply the real gate."""
    runner.complete = fake_model(accuracy, flaky_on=flaky_on, seed=seed)
    result = runner.run_suite(suite, repeats=repeats, use_cache=False)
    verdict = evaluate([result], baseline=baseline or {})
    return result, verdict


def main() -> int:
    suite = load("classification")
    cases = len(suite["cases"])
    print(f"\nDemo suite: `classification` - {cases} cases, model simulated offline.")

    # ---------------------------------------------------------------- 1
    banner(1, "A healthy run", "Good prompt, no baseline yet. This is what green looks like.")
    result, verdict = run(suite, accuracy=1.0, repeats=3)
    print(report.to_markdown([result], verdict))

    # ---------------------------------------------------------------- 2
    banner(2, "A real regression",
           "Someone edits the prompt and accuracy collapses. The gate blocks the merge.")
    baseline = {"classification": _BaselineEntry(pass_rate=0.98, trials=90)}
    result, verdict = run(suite, accuracy=0.55, repeats=15, baseline=baseline, seed=1)
    print(report.to_markdown([result], verdict))

    # ---------------------------------------------------------------- 3
    banner(3, "The same-looking drop, but on a small sample",
           "This is the false alarm most eval setups fire on. Watch it not fire.")
    small_baseline = {"classification": _BaselineEntry(pass_rate=0.95, trials=6)}
    result, verdict = run(suite, accuracy=0.72, repeats=1, baseline=small_baseline, seed=6)
    v = verdict.suites[0]
    print(f"  score      : {result.pass_rate:.0%}  ({result.successes}/{result.n_trials} trials)")
    print(f"  baseline   : {v.baseline:.0%}")
    print(f"  delta      : {v.delta:+.0%}   <- looks like a disaster")
    print(f"  95% CI     : [{v.interval.low:.0%}, {v.interval.high:.0%}]  <- but the error bars are huge")
    print(f"  gate says  : {v.status}")
    for reason in v.reasons:
        print(f"               - {reason}")
    print("\n  The same 67% measured over 90 trials instead of 6:")
    big = wilson(60, 90)
    print(f"    95% CI narrows to [{big.low:.0%}, {big.high:.0%}] - now the drop is callable.")

    # ---------------------------------------------------------------- 4
    banner(4, "A flaky case",
           "One input the model cannot answer consistently. Not the same as a clean failure.")
    result, verdict = run(suite, accuracy=1.0, repeats=5, flaky_on="invoice", seed=2)
    for case in result.cases:
        mark = "FLAKY" if case.flaky else ("ok" if case.passed else "FAIL")
        print(f"  {case.passes}/{case.trials}  {case.id:<28} {mark}")
    print(f"\n  gate status: {verdict.suites[0].status}"
          f"  (reported, not blocking - see EVAL_FAIL_ON_FLAKY)")

    # ---------------------------------------------------------------- 5
    banner(5, "Cost control", "Every run is priced, and a budget failure is a build failure.")
    spend = result.spend
    print(f"  {spend.summary(config.MODEL)}")
    print(f"  budget     : ${verdict.budget:.2f}   over budget: {verdict.over_budget}")
    print(f"\n  Without prompt caching those {spend.cache_read_tokens:,} cached tokens")
    print(f"  would bill at full rate - roughly {1 / 0.10:.0f}x more for the prefix.")

    # ---------------------------------------------------------------- 6
    banner(6, "Prompt A/B testing", "Is the new prompt actually better, or did you get lucky?")
    runner.complete = fake_model(0.95, seed=5)
    comparison = compare_mod.compare(suite, treatment="terse", repeats=4, use_cache=False)
    print(compare_mod.to_markdown([comparison]))

    # ---------------------------------------------------------------- 7
    banner(7, "Power analysis", "How many trials would you need to trust a result?")
    for effect in (0.20, 0.10, 0.05):
        n = required_trials(effect=effect, base_rate=0.90)
        print(f"  to detect a {effect:.0%} drop from 90%: {n:>4} trials per arm"
              f"  ({-(-n // cases):>3} repeats over {cases} cases)")
    print("\n  This is why the demo above uses repeats=15 to call a real regression,")
    print("  and why a 6-trial run cannot call anything at all.")

    print(f"\n{'=' * 78}")
    print("  Everything above ran offline against a simulated model.")
    print("  With ANTHROPIC_API_KEY set, `python -m evals run` does the same")
    print("  against the real model - same gate, same scorecard, real scores.")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
