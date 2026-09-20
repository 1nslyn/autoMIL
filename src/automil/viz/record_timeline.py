"""When each node was submitted, launched and finished, and the run's event list.

The orchestrator log is the only record of launch times; ``spec.json`` holds
the submission time and ``completed/<node>.json`` the finish time. Every stamp
leaves here in UTC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from automil.viz.clock import HostClock, parse_log_asctime, to_utc_iso

_LAUNCHED = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[INFO\] Launched "
    r"(?P<node>node_\d{4,}) on (?P<slot>.+?) \(PID"
)
_COMPLETED = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[INFO\] Completed "
    r"(?P<node>node_\d{4,}): status=(?P<status>[A-Za-z_]+), .*?, (?P<slot>[^,]+)$"
)
_GPU_SLOT = re.compile(r"^(?P<kind>[A-Za-z]+) GPU (?P<index>\d+)$")


@dataclass(frozen=True)
class LogEvent:
    at: str
    kind: str
    node_id: str
    slot: str | None
    status: str | None = None


def normalize_slot(text: str | None) -> str | None:
    """``CUDA GPU 0`` (log form) and ``cuda:0`` (state form) become ``cuda:0``."""
    if not text:
        return None
    match = _GPU_SLOT.match(text.strip())
    if match:
        return f"{match['kind'].lower()}:{match['index']}"
    return text.strip()


def parse_orchestrator_log(lines: Iterable[str], clock: HostClock) -> tuple[LogEvent, ...]:
    """Launch and completion events from the daemon's log, in file order."""
    events = []
    for line in lines:
        launched = _LAUNCHED.match(line)
        if launched:
            stamp = parse_log_asctime(launched["stamp"])
            at = to_utc_iso(stamp.isoformat(), clock) if stamp else None
            if at:
                events.append(LogEvent(at, "launched", launched["node"], normalize_slot(launched["slot"])))
            continue
        completed = _COMPLETED.match(line)
        if completed:
            stamp = parse_log_asctime(completed["stamp"])
            at = to_utc_iso(stamp.isoformat(), clock) if stamp else None
            if at:
                events.append(LogEvent(at, "completed", completed["node"], normalize_slot(completed["slot"]), completed["status"]))
    return tuple(events)


def _seconds_between(start: str | None, end: str | None) -> float | None:
    from datetime import datetime

    if not start or not end:
        return None
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((b - a).total_seconds(), 3)


def node_timing(
    node_id: str,
    *,
    spec: Mapping[str, Any] | None,
    completed: Mapping[str, Any] | None,
    log_events: Iterable[LogEvent],
    clock: HostClock,
) -> dict[str, Any]:
    """Submission, launch and finish stamps for one node, with derived waits."""
    submitted_at = to_utc_iso(spec.get("submitted_at"), clock) if isinstance(spec, Mapping) else None
    launched_at = None
    finished_at = None
    slot = None
    for event in log_events:
        if event.node_id != node_id:
            continue
        if event.kind == "launched" and launched_at is None:
            launched_at, slot = event.at, event.slot
        elif event.kind == "completed":
            finished_at = event.at
            slot = slot or event.slot
    if isinstance(completed, Mapping):
        completed_at = to_utc_iso(completed.get("completed_at"), clock)
        if completed_at:
            finished_at = completed_at
        gpu = completed.get("gpu")
        accelerator = completed.get("accelerator")
        if slot is None and isinstance(gpu, int) and gpu >= 0 and isinstance(accelerator, str):
            slot = f"{accelerator}:{gpu}"
    return {
        "submitted_at": submitted_at,
        "launched_at": launched_at,
        "completed_at": finished_at,
        "slot": slot,
        "queue_wait_s": _seconds_between(submitted_at, launched_at),
        "run_s": _seconds_between(launched_at, finished_at),
    }


def build_timeline(
    *,
    run_id: str,
    clock: HostClock,
    nodes: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """The record's timeline file: nodes, session spans and dated events."""
    ordered = sorted(events, key=lambda e: (e.get("at") or "", e.get("kind") or ""))
    return {
        "schema": 1,
        "run_id": run_id,
        "clock": clock.as_dict(),
        "nodes": list(nodes),
        "sessions": list(sessions),
        "events": ordered,
        "warnings": list(warnings),
    }
