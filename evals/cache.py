"""Content-addressed response cache.

A pull request that touches one prompt should not pay to re-run the suites it
did not touch. The cache key is everything that can change an output - model,
effort, system prompt, user input, max_tokens, and a repeat index - so a hit is
always a legitimate hit, and any edit to a prompt misses by construction.

Sampling caveat: with `repeats > 1` the repeat index is part of the key, so N
distinct samples are stored and replayed. The cache preserves the variance a run
measured; it does not collapse it to one answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE_DIR = Path(os.environ.get("EVAL_CACHE_DIR", Path(__file__).resolve().parent.parent / ".eval-cache"))
ENABLED = os.environ.get("EVAL_CACHE", "1") not in ("0", "false", "no")
TTL_SECONDS = int(os.environ.get("EVAL_CACHE_TTL", str(14 * 24 * 3600)))


@dataclass
class CachedCompletion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: int
    stop_reason: str | None
    stored_at: float


def key(
    *,
    model: str,
    effort: str,
    system: str,
    user: str,
    max_tokens: int,
    repeat: int,
) -> str:
    """Stable hash of everything that determines the response."""
    payload = json.dumps(
        {
            "model": model,
            "effort": effort,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "repeat": repeat,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(digest: str) -> Path:
    # Shard by first two chars so the directory stays navigable at scale.
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def get(digest: str) -> CachedCompletion | None:
    """Return a live cache entry, or None on miss, staleness, or corruption."""
    if not ENABLED:
        return None
    path = _path(digest)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A half-written entry is a miss, never a crash.
        return None
    if time.time() - data.get("stored_at", 0) > TTL_SECONDS:
        return None
    return CachedCompletion(**data)


def put(digest: str, completion: CachedCompletion) -> None:
    """Store an entry. Failures are non-fatal - the cache is an optimisation."""
    if not ENABLED:
        return
    path = _path(digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a killed run never leaves a torn file behind.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(completion)), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def clear() -> int:
    """Delete every entry. Returns how many were removed."""
    if not CACHE_DIR.exists():
        return 0
    removed = 0
    for entry in CACHE_DIR.rglob("*.json"):
        entry.unlink(missing_ok=True)
        removed += 1
    return removed


def stats() -> dict[str, int]:
    entries = list(CACHE_DIR.rglob("*.json")) if CACHE_DIR.exists() else []
    return {
        "entries": len(entries),
        "bytes": sum(e.stat().st_size for e in entries),
    }
