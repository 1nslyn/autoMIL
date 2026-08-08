"""Scrape Claude Code's localhost Prometheus active-time counter."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from automil.activity_hooks import ACTIVITY_METRICS_URL
from automil.cells.activity import (
    ActivityError,
    ActivityObservation,
    ingest_prometheus_metrics,
)

_MAX_PAYLOAD_BYTES = 1_000_000


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: object, **_kwargs: object) -> None:
        return None


_NO_PROXY_HANDLER = ProxyHandler({})
_DIRECT_OPENER = build_opener(_NO_PROXY_HANDLER, _NoRedirects())


def observe_activity_metrics(
    automil_dir: Path | str,
    *,
    timeout: float = 1.0,
    open_url: Callable[..., BinaryIO] | None = None,
    observed_at: float | None = None,
) -> ActivityObservation:
    """Fetch, validate, and persist one live loopback observation."""

    timestamp = time.time() if observed_at is None else observed_at
    opener = _DIRECT_OPENER.open if open_url is None else open_url
    try:
        with opener(ACTIVITY_METRICS_URL, timeout=timeout) as response:
            payload = response.read(_MAX_PAYLOAD_BYTES + 1)
    except OSError as exc:
        return ActivityObservation(
            available=False, observed_at=timestamp, error=str(exc),
        )
    try:
        if len(payload) > _MAX_PAYLOAD_BYTES:
            raise ActivityError("Claude metrics payload is too large")
        exposition = payload.decode("utf-8")
        sessions = ingest_prometheus_metrics(
            automil_dir, exposition, observed_at=timestamp
        )
    except UnicodeError:
        return ActivityObservation(
            available=False,
            observed_at=timestamp,
            error="Claude metrics payload is not UTF-8",
        )
    except ActivityError as exc:
        # Invalid live telemetry is an admission-health failure, not a daemon
        # failure and never a reason to fabricate consumed budget.
        return ActivityObservation(
            available=False, observed_at=timestamp, error=str(exc),
        )
    return ActivityObservation(
        available=True,
        sessions=sessions,
        observed_at=timestamp,
    )


def refresh_activity_metrics(
    automil_dir: Path | str,
    *,
    timeout: float = 1.0,
) -> tuple[str, ...] | None:
    """Persist a native counter snapshot; return ``None`` if Claude is absent."""

    observation = observe_activity_metrics(automil_dir, timeout=timeout)
    return observation.sessions if observation.available else None


__all__ = ["observe_activity_metrics", "refresh_activity_metrics"]
