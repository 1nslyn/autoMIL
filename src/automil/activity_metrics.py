"""Scrape Claude Code's localhost Prometheus active-time counter."""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from automil.activity_hooks import (
    ACTIVITY_METRICS_URL,
    activity_metrics_url,
    project_exporter_port,
)
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


def fetch_activity_exposition(
    *,
    url: str = ACTIVITY_METRICS_URL,
    timeout: float = 1.0,
    open_url: Callable[..., BinaryIO] | None = None,
) -> str:
    """Fetch the raw Prometheus exposition text, or raise ``ActivityError``."""

    opener = _DIRECT_OPENER.open if open_url is None else open_url
    try:
        with opener(url, timeout=timeout) as response:
            payload = response.read(_MAX_PAYLOAD_BYTES + 1)
    except OSError as exc:
        raise ActivityError(f"Claude metrics endpoint unavailable: {exc}") from exc
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ActivityError("Claude metrics payload is too large")
    try:
        return payload.decode("utf-8")
    except UnicodeError as exc:
        raise ActivityError("Claude metrics payload is not UTF-8") from exc


def observe_activity_metrics(
    automil_dir: Path | str,
    *,
    timeout: float = 1.0,
    open_url: Callable[..., BinaryIO] | None = None,
    observed_at: float | None = None,
) -> ActivityObservation:
    """Fetch, validate, and persist one live loopback observation."""

    timestamp = time.time() if observed_at is None else observed_at
    try:
        try:
            url = activity_metrics_url(project_exporter_port(automil_dir))
        except ValueError as exc:
            raise ActivityError(str(exc)) from exc
        exposition = fetch_activity_exposition(
            url=url, timeout=timeout, open_url=open_url
        )
        sessions = ingest_prometheus_metrics(
            automil_dir, exposition, observed_at=timestamp
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


__all__ = [
    "fetch_activity_exposition",
    "observe_activity_metrics",
    "refresh_activity_metrics",
]
