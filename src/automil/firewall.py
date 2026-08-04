"""Held-out redaction for agent-facing text surfaces (H-1).

The val-firewall seals ``held_out`` out of ``result.json`` and quarantines the
test-bearing artifacts under ``archive/<node>/certify/``. But the training
subprocess's raw stdout is captured to ``archive/<node>/run.log``, which is an
agent-visible debugging surface — so any script that *prints* a test metric leaks
the sealed quantity, and the daemon additionally copies the tail of that log into
the agent-facing ``result["error"]`` on failure.

Rather than trusting every training script (and every vendored dependency) to
self-censor, redact at the orchestrator boundary. This is defence-in-depth: the
per-run gating in the training code still applies; this catches regressions such
as a re-vendored upstream that reintroduces a test-metric print.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

REDACTION = "[REDACTED: held-out metric — val-firewall]"

#: Substrings that mark a line as carrying test/held-out numbers. Case-insensitive.
#: Deliberately broad: a false positive costs one redacted debug line, a false
#: negative costs the firewall.
_HELD_OUT_MARKERS = (
    "test_auc", "test_bacc", "test_acc", "test_c_index", "test_cindex",
    "test_f1", "test_auprc", "test_loss", "test_error", "test error",
    "held_out", "heldout", "held out",
)


def _line_leaks(line: str, extra_keys: tuple[str, ...]) -> bool:
    low = line.lower()
    if any(m in low for m in _HELD_OUT_MARKERS):
        return True
    return any(k.lower() in low for k in extra_keys)


def redact_held_out(text: str, extra_keys: tuple[str, ...] = ()) -> str:
    """Replace every line that mentions a held-out/test metric.

    ``extra_keys`` lets the caller add the run's actual ``held_out`` metric names
    (from result.json), so a consumer-specific key is redacted even when it does
    not match the generic markers.
    """
    if not text:
        return text
    out = []
    for line in text.splitlines():
        out.append(REDACTION if _line_leaks(line, extra_keys) else line)
    return "\n".join(out)


def held_out_keys(result: Mapping[str, Any]) -> tuple[str, ...]:
    """Metric names declared in the result's sealed ``held_out`` block."""
    block = (result or {}).get("held_out")
    if not isinstance(block, dict):
        return ()
    return tuple(str(k) for k in block)


def redact_log_file(path: Path, extra_keys: tuple[str, ...] = ()) -> int:
    """Rewrite ``path`` in place with held-out lines redacted.

    Returns the number of redacted lines. Best-effort: a missing/unreadable file
    is a no-op (the caller must never fail a completion over log hygiene).
    """
    try:
        original = path.read_text(errors="replace")
    except (OSError, AttributeError):
        return 0
    redacted = redact_held_out(original, extra_keys)
    if redacted == original:
        return 0
    try:
        path.write_text(redacted + ("\n" if original.endswith("\n") else ""))
    except OSError:
        return 0
    return len(re.findall(re.escape(REDACTION), redacted))
