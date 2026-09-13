"""Minimal .env loader.

Small enough not to warrant a dependency, and deliberately narrow: real
environment variables always win, so an export or a CI secret overrides the
file rather than the other way round.

config.py skips this under pytest: the suite asserts against the defaults
declared there, and a developer's local .env naming a different provider would
quietly change what those tests measure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


def load(path: Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines into the environment without clobbering it.

    Returns the names taken from the file, which is what callers want to log.
    """
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line: Final = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
