"""Scrape Claude Code's localhost Prometheus active-time counter."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from automil.activity_hooks import ACTIVITY_METRICS_URL
from automil.cells.activity import ActivityError, ingest_prometheus_metrics

_MAX_PAYLOAD_BYTES = 1_000_000


def refresh_activity_metrics(
    automil_dir: Path | str,
    *,
    timeout: float = 1.0,
) -> tuple[str, ...] | None:
    """Persist a native counter snapshot; return ``None`` if Claude is absent."""

    try:
        with urlopen(ACTIVITY_METRICS_URL, timeout=timeout) as response:
            payload = response.read(_MAX_PAYLOAD_BYTES + 1)
    except OSError:
        return None
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ActivityError("Claude metrics payload is too large")
    try:
        exposition = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ActivityError("Claude metrics payload is not UTF-8") from exc
    return ingest_prometheus_metrics(automil_dir, exposition)
