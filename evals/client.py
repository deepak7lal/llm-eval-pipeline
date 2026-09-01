"""Thin wrapper around the Anthropic SDK.

Three rules the rest of the harness relies on:
  * the system prompt carries a cache breakpoint, so repeated runs over a suite
    pay for the prefix once;
  * every call reports its own token usage, so the scorecard can price the run;
  * identical calls are served from the on-disk cache, so an unrelated PR does
    not re-buy answers it already has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import anthropic

from . import cache, config

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Lazily build a shared client.

    Credentials resolve from the environment (ANTHROPIC_API_KEY, then
    ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile) - never hardcode one.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=3)
    return _client


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: int
    stop_reason: str | None
    cache_write_tokens: int = 0
    from_cache: bool = False


def complete(
    *,
    system: str,
    user: str,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
    repeat: int = 0,
    use_cache: bool = True,
) -> Completion:
    """Run one prompt against the model under test.

    The system prompt carries a cache breakpoint and is passed first, so the
    whole prefix is reused across every case in a suite. `repeat` distinguishes
    repeated samples of the same case - it is part of the cache key, so each
    sample is stored separately and variance survives a cached run.
    """
    model = model or config.MODEL
    effort = effort or config.EFFORT
    max_tokens = max_tokens or config.MAX_TOKENS

    digest = cache.key(
        model=model, effort=effort, system=system, user=user,
        max_tokens=max_tokens, repeat=repeat,
    )
    if use_cache:
        hit = cache.get(digest)
        if hit is not None:
            return Completion(
                text=hit.text,
                input_tokens=hit.input_tokens,
                output_tokens=hit.output_tokens,
                cache_read_tokens=hit.cache_read_tokens,
                latency_ms=hit.latency_ms,
                stop_reason=hit.stop_reason,
                from_cache=True,
            )

    client = get_client()
    started = time.perf_counter()

    last_error: Exception | None = None
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": user}],
            )
            break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            # Retryable: back off and try again within our own envelope.
            last_error = exc
            if attempt == config.MAX_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
        except anthropic.APIStatusError:
            # 400/404 and friends are bugs in the request - surface immediately.
            raise
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError("exhausted attempts") from last_error

    text = "".join(b.text for b in response.content if b.type == "text")
    usage = response.usage
    completion = Completion(
        text=text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        latency_ms=int((time.perf_counter() - started) * 1000),
        stop_reason=response.stop_reason,
    )

    if use_cache and completion.stop_reason != "refusal":
        # Refusals are not cached: they are the least stable outcome, and a
        # stale one would keep failing a case the model would now answer.
        cache.put(
            digest,
            cache.CachedCompletion(
                text=completion.text,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cache_read_tokens=completion.cache_read_tokens,
                latency_ms=completion.latency_ms,
                stop_reason=completion.stop_reason,
                stored_at=time.time(),
            ),
        )
    return completion
