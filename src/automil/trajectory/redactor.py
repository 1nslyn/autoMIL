"""Redaction-on-capture for trajectory events (TRJ-03 / D-82, D-83).

Compiled regex set applied to every string field before append.
Redaction is mandatory — no opt-out flag (Pitfall 5a defence).
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Compiled at module-import time — one-time cost, applied to every event (D-82)
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                               "sk-[REDACTED]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"),                                  "hf_[REDACTED]"),
    (re.compile(r"ghp_[A-Za-z0-9]{30,}"),                                 "ghp_[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                 "AKIA[REDACTED]"),
    # WR-01 (Phase 3 review): `\S+` matched any non-whitespace value, which
    # over-redacted legitimate ENV-style strings with very short values like
    # `CACHE_KEY=abc`, `NO_API_KEY=true`, `GIT_TOKEN=on`. Real secrets are at
    # least ~8 chars; require `\S{8,}` to filter the obvious false positives.
    # Path-like values (`PUBLIC_KEY=/path/to/pub`) still match — over-redaction
    # is the safer side per Pitfall 5a (catastrophic if leaked).
    (re.compile(r"([A-Z][A-Z0-9_]{1,40}_API_KEY)\s*[:=]\s*\S{8,}"),        r"\1=[REDACTED]"),
    (re.compile(r"([A-Z][A-Z0-9_]{1,40}_TOKEN)\s*[:=]\s*\S{8,}"),          r"\1=[REDACTED]"),
    (re.compile(r"([A-Z][A-Z0-9_]{1,40}_KEY)\s*[:=]\s*\S{8,}"),            r"\1=[REDACTED]"),
]

_SIZE_CAP_BYTES = 8192  # D-83: 8 KB per-event cap

# D-139: regex to match node IDs that may be held-out gate-eval children
_NODE_ID_RE = re.compile(r"\bnode_\d{4,}\b")


@functools.lru_cache(maxsize=1)
def _held_out_ids_cached(graph_mtime: float) -> frozenset:
    """Mtime-keyed lookup of held-out node IDs.

    The lru_cache key is the graph.json mtime; modifying the file invalidates
    the cache automatically (Pitfall 2 mitigation). Returns frozenset() on any
    read error — soft-fail discipline (Pitfall 5a parallel).
    """
    from automil.cli._helpers import _find_automil_dir
    try:
        adir = _find_automil_dir()
        data = json.loads((adir / "graph.json").read_text())
        return frozenset(
            nid for nid, n in data.get("nodes", {}).items()
            if isinstance(n, dict)
            and n.get("metadata", {}).get("held_out", False)
        )
    except Exception:
        return frozenset()


def _held_out_ids() -> frozenset:
    """Return current held-out node IDs, cached by graph.json mtime."""
    from automil.cli._helpers import _find_automil_dir
    try:
        adir = _find_automil_dir()
        mtime = (adir / "graph.json").stat().st_mtime
    except Exception:
        mtime = 0.0
    return _held_out_ids_cached(mtime)


def redact_secrets(s: str) -> str:
    """Mask credential-shaped substrings. Pure: no project lookup, no cwd.

    The dashboard record builders run outside any project cwd and must not
    consult graph.json through ``_find_automil_dir``; they need only the
    pattern half of :func:`redact`.
    """
    for pattern, replacement in _PATTERNS:
        s = pattern.sub(replacement, s)
    return s


def redact(s: str) -> str:
    """Apply all compiled redaction patterns to a string. Returns a new string."""
    s = redact_secrets(s)
    # D-139 dynamic: redact held-out node IDs to <HELD_OUT> placeholder
    held_out = _held_out_ids()
    if held_out:
        s = _NODE_ID_RE.sub(
            lambda m: "<HELD_OUT>" if m.group(0) in held_out else m.group(0),
            s,
        )
    return s


def redact_event(d: dict) -> dict:
    """Walk event dict recursively and redact all string leaves.

    Returns a NEW dict — original is not mutated.
    Handles: dict (recurse values), list/tuple (recurse elements),
    str (apply patterns), int/float/bool/None (pass through).
    Soft-fail discipline: exceptions are caught and logged — never raises.
    """
    try:
        return _walk(d)
    except Exception as exc:
        logger.warning("redact_event failed: %s; returning sentinel event", exc)
        return {"gen_ai.event.name": "redaction_error", "_error": str(exc)}


def _walk(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        walked = [_walk(v) for v in obj]
        return type(obj)(walked)
    if isinstance(obj, str):
        return redact(obj)
    return obj


def apply_size_cap(event: dict) -> dict:
    """Enforce the 8 KB per-event cap (D-83) post-redaction.

    If event exceeds _SIZE_CAP_BYTES:
      1. Truncate gen_ai.tool.call.arguments and gen_ai.tool.call.result first.
      2. If still over cap, replace with a minimal sentinel event.
    Soft-fail: exceptions logged, sentinel returned.
    """
    try:
        encoded = json.dumps(event, ensure_ascii=False).encode("utf-8")
        if len(encoded) <= _SIZE_CAP_BYTES:
            return event

        # Attempt field truncation
        truncated = dict(event)
        original_size = len(encoded)
        for field in ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result"):
            if field in truncated and isinstance(truncated[field], str):
                truncated[field] = truncated[field][:512] + f"...[truncated:{original_size}B]"

        encoded2 = json.dumps(truncated, ensure_ascii=False).encode("utf-8")
        if len(encoded2) <= _SIZE_CAP_BYTES:
            return truncated

        # Pathological case: metadata bloat — replace with sentinel
        logger.warning(
            "Event still %d bytes after field truncation (cap %d); dropping to sentinel",
            len(encoded2), _SIZE_CAP_BYTES,
        )
        return {
            "gen_ai.event.name": "truncated",
            "gen_ai.event.timestamp": event.get("gen_ai.event.timestamp", ""),
            "_dropped_size": original_size,
        }
    except Exception as exc:
        logger.warning("apply_size_cap failed: %s", exc)
        return event
