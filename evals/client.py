"""The single entry point for calling a model.

Three rules the rest of the harness relies on:
  * the provider is pluggable (see `providers.py`), so the suites, graders,
    statistics, and gate never learn which vendor answered;
  * every call reports its own token usage, so the scorecard can price the run;
  * identical calls are served from the on-disk cache, so an unrelated PR does
    not re-buy answers it already has.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import cache, config, providers


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


def get_client():
    """The active provider. Named for the call site it replaced."""
    return providers.get_provider()


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

    `repeat` distinguishes repeated samples of the same case - it is part of the
    cache key, so each sample is stored separately and variance survives a
    cached run.
    """
    provider = providers.get_provider()
    model = model or config.MODEL or provider.default_model
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

    started = time.perf_counter()
    retryable = getattr(provider, "retryable", lambda: ())()

    last_error: Exception | None = None
    for attempt in range(config.MAX_ATTEMPTS):
        try:
            response = provider.complete(
                system=system, user=user, model=model, effort=effort, max_tokens=max_tokens
            )
            break
        except retryable as exc:  # type: ignore[misc]
            # Rate limits and transient network faults: back off and retry
            # within our own envelope. The SDKs already retry internally.
            last_error = exc
            if attempt == config.MAX_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover - loop always breaks or raises
        raise RuntimeError("exhausted attempts") from last_error

    completion = Completion(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        latency_ms=int((time.perf_counter() - started) * 1000),
        stop_reason="refusal" if response.refused else response.stop_reason,
    )

    if use_cache and not response.refused:
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
