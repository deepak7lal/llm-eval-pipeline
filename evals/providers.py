"""Model providers.

The harness measures prompts, not vendors, so the model call sits behind one
small interface. Everything downstream - graders, statistics, the gate, the
scorecard - works the same whichever provider is selected.

Two implementations:

  * ``anthropic``      - the official Anthropic SDK, with adaptive thinking,
                         effort, and prompt caching. The default.
  * ``openai_compat``  - the OpenAI chat-completions shape, which Gemini, Groq,
                         OpenRouter, Ollama, and OpenAI itself all speak. One
                         adapter, four free options.

Select with ``EVAL_PROVIDER``. Presets fill in the endpoint and key variable:

    EVAL_PROVIDER=gemini   GEMINI_API_KEY=...    # free tier, no card
    EVAL_PROVIDER=groq     GROQ_API_KEY=...      # free tier, very fast
    EVAL_PROVIDER=ollama                         # local, no key at all

Scores are not comparable across providers or models. Switching either means
regenerating ``baselines/main.json``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Response:
    """What every provider returns. Deliberately smaller than any SDK's type."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    refused: bool = False


class Provider(Protocol):
    """The whole contract. A provider is one method."""

    name: str
    default_model: str

    def complete(
        self, *, system: str, user: str, model: str, effort: str, max_tokens: int
    ) -> Response: ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


class AnthropicProvider:
    """Claude via the official SDK.

    Uses the features the other providers have no equivalent for: adaptive
    thinking, an effort level, and a cache breakpoint on the system prompt.
    """

    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(self) -> None:
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.Anthropic(max_retries=6)

    def complete(self, *, system: str, user: str, model: str, effort: str, max_tokens: int) -> Response:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": user}],
        )
        usage = response.usage
        return Response(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            stop_reason=response.stop_reason,
            refused=response.stop_reason == "refusal",
        )

    def retryable(self) -> tuple[type[Exception], ...]:
        return (
            self._sdk.RateLimitError,
            self._sdk.APIConnectionError,
            self._sdk.InternalServerError,
        )


# --------------------------------------------------------------------------
# Anything speaking the OpenAI chat-completions shape
# --------------------------------------------------------------------------

# base_url, key env var, default model - for providers with a free tier.
PRESETS: dict[str, tuple[str, str, str]] = {
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-2.0-flash",
    ),
    "groq": (
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "openai/gpt-oss-20b",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free",
    ),
    "ollama": (
        "http://localhost:11434/v1",
        "",  # local server wants no key
        "llama3.2",
    ),
    "openai": (
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
    ),
}


class OpenAICompatProvider:
    """One adapter for every OpenAI-shaped endpoint.

    `effort` is accepted and ignored - it is an Anthropic concept, and silently
    dropping it is better than failing, since the suites declare it.
    """

    name = "openai_compat"

    def __init__(self, preset: str) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown provider {preset!r}; known: {sorted(PRESETS)}")

        base_url, key_var, default_model = PRESETS[preset]
        self.name = preset
        self.default_model = os.environ.get("EVAL_MODEL", default_model)

        base_url = os.environ.get("EVAL_BASE_URL", base_url)
        api_key = os.environ.get(key_var, "") if key_var else "not-needed"
        if key_var and not api_key:
            raise RuntimeError(
                f"provider {preset!r} needs {key_var} in the environment. "
                f"Get a free key and export it, or use EVAL_PROVIDER=ollama to run locally."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise RuntimeError(
                f"provider {preset!r} needs the openai package: pip install -e '.[compat]'"
            ) from exc

        self._client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, max_retries=3)

    def complete(self, *, system: str, user: str, model: str, effort: str, max_tokens: int) -> Response:
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        finish = choice.finish_reason
        return Response(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=finish,
            # These APIs report a filtered response as a finish reason rather
            # than a distinct status, so map it onto the same concept.
            refused=finish == "content_filter",
        )

    def retryable(self) -> tuple[type[Exception], ...]:
        import openai

        return (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

_cached: Provider | None = None


def get_provider(name: str | None = None) -> Provider:
    """Build (once) the provider named by `EVAL_PROVIDER`, default anthropic."""
    global _cached
    if _cached is not None and name is None:
        return _cached

    selected = (name or os.environ.get("EVAL_PROVIDER", "anthropic")).lower()
    provider: Provider
    if selected == "anthropic":
        provider = AnthropicProvider()
    else:
        provider = OpenAICompatProvider(selected)

    if name is None:
        _cached = provider
    return provider


def reset() -> None:
    """Drop the cached provider. Used by tests and by CLI overrides."""
    global _cached
    _cached = None


def available() -> list[str]:
    return ["anthropic", *sorted(PRESETS)]
