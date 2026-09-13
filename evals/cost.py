"""Token accounting in dollars, plus a spend ceiling for CI.

Prices are per million tokens, first-party Claude API rates. Cached reads bill
at 10% of the input rate; cache writes at 125%. Keep this table in sync with
https://claude.com/pricing - it is the one place in the repo where a stale
number costs real money.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# model id -> (input $/MTok, output $/MTok)
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Free tiers. Priced at zero so the scorecard reads $0.00 rather than
    # flagging them as unknown - they genuinely cost nothing within quota.
    "gemini-2.0-flash": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "openai/gpt-oss-20b": (0.0, 0.0),
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "llama3.2": (0.0, 0.0),
}

CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

# Hard ceiling for one CI run. Exceeding it fails the build even if every suite
# passed - a prompt change that quietly triples cost is a regression too.
BUDGET_USD = float(os.environ.get("EVAL_BUDGET_USD", "5.00"))


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    api_calls: int = 0
    cached_calls: int = 0

    def __add__(self, other: "Spend") -> "Spend":
        return Spend(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            api_calls=self.api_calls + other.api_calls,
            cached_calls=self.cached_calls + other.cached_calls,
        )

    def usd(self, model: str) -> float:
        """Cost of this spend on `model`. Unknown models price at zero."""
        rates = PRICING.get(model)
        if rates is None:
            return 0.0
        in_rate, out_rate = rates
        return (
            self.input_tokens * in_rate
            + self.cache_read_tokens * in_rate * CACHE_READ_MULTIPLIER
            + self.cache_write_tokens * in_rate * CACHE_WRITE_MULTIPLIER
            + self.output_tokens * out_rate
        ) / 1_000_000

    def summary(self, model: str) -> str:
        saved = f", {self.cached_calls} served from cache" if self.cached_calls else ""
        return (
            f"${self.usd(model):.4f} over {self.api_calls} API call(s){saved} - "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out, "
            f"{self.cache_read_tokens:,} cache read"
        )


def over_budget(spend: Spend, model: str, budget: float | None = None) -> tuple[bool, float]:
    """Is this run over the ceiling? Returns `(exceeded, usd)`."""
    limit = BUDGET_USD if budget is None else budget
    usd = spend.usd(model)
    return usd > limit, usd


def unknown_models(models: list[str]) -> list[str]:
    """Models with no price on file - reported so cost never silently reads $0."""
    return sorted({m for m in models if m not in PRICING})
