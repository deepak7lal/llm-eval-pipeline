"""Central configuration. Everything tunable lives here or in an env override."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "evals" / "datasets"
REPORTS_DIR = ROOT / "reports"
BASELINE_PATH = ROOT / "baselines" / "main.json"

# The model under test. Do not downgrade for cost without changing the baseline
# too - scores are not comparable across models.
MODEL = os.environ.get("EVAL_MODEL", "claude-opus-5")

# Model used by the llm_judge grader. Kept separate so the judge can stay fixed
# while the model under test changes.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "claude-opus-5")

# output_config.effort for the model under test: low | medium | high | xhigh | max
EFFORT = os.environ.get("EVAL_EFFORT", "medium")
JUDGE_EFFORT = os.environ.get("EVAL_JUDGE_EFFORT", "low")

MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "4096"))

# How many times each case is sampled. >1 buys real confidence intervals and
# exposes flaky cases, at a linear cost in dollars.
REPEATS = int(os.environ.get("EVAL_REPEATS", "1"))

# Significance level for confidence intervals and the regression test.
ALPHA = float(os.environ.get("EVAL_ALPHA", "0.05"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "8"))

# Per-case retries for transient API failures. The SDK already retries
# connection errors and 429/5xx twice; this is the outer envelope.
MAX_ATTEMPTS = int(os.environ.get("EVAL_MAX_ATTEMPTS", "2"))


@dataclass(frozen=True)
class Gate:
    """Quality gate: an absolute floor plus a regression allowance."""

    # Fail the build if a suite's pass rate drops below this.
    min_pass_rate: float = 0.80
    # Fail the build if a suite regresses more than this against the baseline,
    # even when it still clears min_pass_rate.
    max_regression: float = 0.05
    # Suites with fewer trials than this are reported but never gate the build -
    # small samples are too noisy to fail on.
    min_cases_to_gate: int = 5
    # Require a regression to be statistically significant before failing. Keeps
    # the gate from firing on sampling noise; set False for a purely
    # threshold-based gate.
    require_significance: bool = True
    # Fail if any case flips between pass and fail across repeats. Off by
    # default: flakiness is reported first, gated once a suite is stable.
    fail_on_flaky: bool = False
    # Per-suite overrides, e.g. {"extraction": Gate(min_pass_rate=0.9)}
    overrides: dict[str, "Gate"] = field(default_factory=dict)


GATE = Gate(
    min_pass_rate=float(os.environ.get("EVAL_MIN_PASS_RATE", "0.80")),
    max_regression=float(os.environ.get("EVAL_MAX_REGRESSION", "0.05")),
    require_significance=os.environ.get("EVAL_REQUIRE_SIGNIFICANCE", "1") not in ("0", "false"),
    fail_on_flaky=os.environ.get("EVAL_FAIL_ON_FLAKY", "0") not in ("0", "false"),
)


def gate_for(suite: str) -> Gate:
    """Resolve the gate that applies to a suite, honouring per-suite overrides."""
    return GATE.overrides.get(suite, GATE)
