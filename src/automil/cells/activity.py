"""Durable accounting of Claude Code's native active-time metric.

Lifecycle hooks are immutable facts in a short JSONL journal. Claude's
cumulative ``claude_code.active_time.total`` value is a snapshot, so only its
latest value per session is stored in an atomically replaced state file. This
keeps repeated daemon scrapes O(1) instead of growing and replaying a 12-hour
sample log.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

ACTIVITY_JOURNAL_FILENAME = ".activity.jsonl"
ACTIVITY_SAMPLES_FILENAME = ".activity.samples.json"
ACTIVE_TIME_METRIC = "claude_code.active_time.total"

_LOCK_FILENAME = ".activity.lock"
_SAMPLES_SCHEMA_VERSION = 1
_PROMETHEUS_METRIC_NAMES = {
    ACTIVE_TIME_METRIC,
    ACTIVE_TIME_METRIC.replace(".", "_"),
}
_PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[^\s{]+)\{(?P<labels>.*)\}\s+"
    r"(?P<value>\S+)(?:\s+\d+)?$"
)
_PROMETHEUS_LABEL = re.compile(
    r'\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*"((?:\\.|[^"\\])*)"\s*(?:,|$)'
)
_EVENT_KEYS = {
    "session_open": frozenset({"event", "cell_id", "session_id", "observed_at"}),
    "session_end": frozenset(
        {
            "event",
            "cell_id",
            "session_id",
            "final_sample_observed_at",
            "observed_at",
        }
    ),
    "session_bind": frozenset(
        {"event", "cell_id", "session_id", "binding_sha256", "observed_at"}
    ),
}


class ActivityError(ValueError):
    """The activity state or an incoming activity event is invalid."""


class ActivityHealth(str, Enum):
    """Whether stored evidence currently permits agent-active work."""

    OPEN_HEALTHY = "open-healthy"
    ENDED_COMPLETE = "ended-complete"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class ActivityObservation:
    """One attempt to observe the runtime's live cumulative counters."""

    available: bool
    sessions: tuple[str, ...] = ()
    observed_at: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.available:
            if self.error is not None:
                raise ActivityError(
                    "an available activity observation cannot carry an error"
                )
            if self.observed_at is None:
                raise ActivityError(
                    "an available activity observation requires observed_at"
                )
        elif self.sessions:
            raise ActivityError(
                "an unavailable activity observation cannot carry sessions"
            )


@dataclass(frozen=True)
class ActivityReport:
    """Immutable accounting summary for one cell."""

    active_seconds: float
    sessions: tuple[str, ...]
    metered_sessions: tuple[str, ...]
    complete: bool
    event_count: int
    sha256: str
    bindings: tuple[tuple[str, str], ...]
    open_sessions: tuple[str, ...]
    ended_sessions: tuple[str, ...]


@dataclass(frozen=True)
class ActivityAssessment:
    """Admission-facing interpretation of authentic stored activity evidence."""

    active_seconds: float
    health: ActivityHealth
    reason: str | None
    sessions: tuple[str, ...]
    open_sessions: tuple[str, ...]
    ended_sessions: tuple[str, ...]
    admissible: bool
    complete: bool


@dataclass
class _Session:
    cell_id: str | None
    session_id: str
    opened_at: float
    open_event: dict[str, Any]
    latest_at: float
    ended_at: float | None = None
    end_event: dict[str, Any] | None = None
    binding_sha256: str | None = None
    active_seconds: float | None = None
    sample_observed_at: float | None = None


@dataclass
class _Replay:
    sessions: dict[str, _Session] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


def record_hook_event(
    automil_dir: Path | str,
    cell_id: str | None,
    payload: Mapping[str, Any],
    observed_at: float | None = None,
    final_sample_observed_at: float | None = None,
) -> None:
    """Normalize and append one supported Claude session hook event."""

    timestamp = _timestamp(time.time() if observed_at is None else observed_at)
    cell = _optional_identifier(cell_id, "cell_id")
    if not isinstance(payload, Mapping):
        raise ActivityError("hook payload must be an object")
    hook_name = _identifier(payload.get("hook_event_name"), "hook_event_name")
    session_id = _identifier(payload.get("session_id"), "session_id")

    if hook_name == "SessionStart":
        source = payload.get("source")
        if source is not None and source != "startup":
            raise ActivityError(
                f"unsupported SessionStart source {source!r}; start a fresh session"
            )
        event = _event("session_open", cell, session_id, timestamp)
    elif hook_name == "SessionEnd":
        if final_sample_observed_at is None:
            raise ActivityError(
                "SessionEnd requires the exact final active-time observation"
            )
        event = _event(
            "session_end",
            cell,
            session_id,
            timestamp,
            final_sample_observed_at=_timestamp(final_sample_observed_at),
        )
    else:
        raise ActivityError(f"unsupported hook event {hook_name!r}")

    _append_event(Path(automil_dir), event)


def bind_activity_session(
    automil_dir: Path | str,
    cell_id: str,
    session_id: str,
    binding_sha256: str | None = None,
    observed_at: float | None = None,
) -> None:
    """Attach an immutable launch-binding digest to an existing session."""

    event = _event(
        "session_bind",
        _identifier(cell_id, "cell_id"),
        _identifier(session_id, "session_id"),
        _timestamp(time.time() if observed_at is None else observed_at),
        binding_sha256=(
            None
            if binding_sha256 is None
            else _sha256(binding_sha256, "binding_sha256")
        ),
    )
    _append_event(Path(automil_dir), event)


def ingest_prometheus_metrics(
    automil_dir: Path | str,
    exposition: str,
    observed_at: float | None = None,
) -> tuple[str, ...]:
    """Persist latest cumulative active time from one Prometheus scrape.

    Returns the session IDs present in the scrape. Samples for sessions not yet
    opened by the startup hook are ignored because the cumulative value remains
    available on the next scrape.
    """

    timestamp = _timestamp(time.time() if observed_at is None else observed_at)
    incoming = _parse_active_samples(exposition)
    observed_sessions = tuple(sorted(incoming))
    root = Path(automil_dir)
    if not (root / ACTIVITY_JOURNAL_FILENAME).exists():
        return observed_sessions

    with _activity_lock(root, exclusive=True):
        replay = _read_journal_unlocked(root)
        samples = _read_samples_unlocked(root)
        advanced = 0
        for session_id, active_seconds in sorted(incoming.items()):
            session = replay.sessions.get(session_id)
            if session is None:
                continue
            prior = samples.get(session_id)
            if prior is not None:
                if active_seconds < prior["active_seconds"]:
                    raise ActivityError(
                        f"active-time regression for session {session_id!r}"
                    )
                if timestamp < prior["observed_at"]:
                    raise ActivityError(
                        f"timestamp regression for session {session_id!r}"
                    )
            if timestamp < session.opened_at:
                raise ActivityError(
                    f"timestamp regression for session {session_id!r}"
                )
            if session.ended_at is not None:
                if prior is None:
                    raise ActivityError(
                        f"ended session {session_id!r} has no final sample"
                    )
                if active_seconds != prior["active_seconds"]:
                    raise ActivityError(
                        f"active-time changed for ended session {session_id!r}"
                    )
                continue
            next_sample = {
                "active_seconds": active_seconds,
                "observed_at": timestamp,
            }
            if prior != next_sample:
                samples[session_id] = next_sample
                advanced += 1
        if advanced:
            _write_samples_atomic(root, samples)
        return observed_sessions


def read_activity_report(
    automil_dir: Path | str,
    cell_id: str,
) -> ActivityReport:
    """Read lifecycle facts and latest samples for one budget cell."""

    root = Path(automil_dir)
    cell = _identifier(cell_id, "cell_id")
    journal_path = root / ACTIVITY_JOURNAL_FILENAME
    samples_path = root / ACTIVITY_SAMPLES_FILENAME
    if not journal_path.exists() and not samples_path.exists():
        return _empty_report()

    with _activity_lock(root, exclusive=False):
        replay = _read_journal_unlocked(root)
        samples = _read_samples_unlocked(root)

    _attach_samples(replay, samples)

    sessions = [session for session in replay.sessions.values() if session.cell_id == cell]
    return _activity_report(sessions, replay.events)


def read_unbound_activity_report(automil_dir: Path | str) -> ActivityReport:
    """Read project-local sessions which have not yet been assigned to a cell."""

    root = Path(automil_dir)
    journal_path = root / ACTIVITY_JOURNAL_FILENAME
    samples_path = root / ACTIVITY_SAMPLES_FILENAME
    if not journal_path.exists() and not samples_path.exists():
        return _empty_report()
    with _activity_lock(root, exclusive=False):
        replay = _read_journal_unlocked(root)
        samples = _read_samples_unlocked(root)
    _attach_samples(replay, samples)
    sessions = [
        session for session in replay.sessions.values() if session.cell_id is None
    ]
    return _activity_report(sessions, replay.events)


def assess_activity(
    report: ActivityReport,
    observation: ActivityObservation | None,
) -> ActivityAssessment:
    """Assess admission without inventing time or changing the cell cap state."""

    if report.complete:
        return ActivityAssessment(
            active_seconds=report.active_seconds,
            health=ActivityHealth.ENDED_COMPLETE,
            reason=None,
            sessions=report.sessions,
            open_sessions=report.open_sessions,
            ended_sessions=report.ended_sessions,
            admissible=True,
            complete=True,
        )

    reason: str | None = None
    if not report.sessions:
        reason = "no bound activity session has been recorded"
    elif report.metered_sessions != report.sessions:
        reason = "one or more activity sessions have no authentic sample"
    elif not report.open_sessions:
        reason = "activity session evidence is incomplete"
    elif observation is None or not observation.available:
        reason = (
            observation.error
            if observation is not None and observation.error
            else "live activity telemetry is unavailable"
        )
    elif observation.sessions != report.open_sessions:
        reason = (
            "observed activity session set does not match the bound open sessions: "
            f"expected {report.open_sessions!r}, observed {observation.sessions!r}"
        )

    health = ActivityHealth.OPEN_HEALTHY if reason is None else ActivityHealth.DEGRADED
    return ActivityAssessment(
        active_seconds=report.active_seconds,
        health=health,
        reason=reason,
        sessions=report.sessions,
        open_sessions=report.open_sessions,
        ended_sessions=report.ended_sessions,
        admissible=health is ActivityHealth.OPEN_HEALTHY,
        complete=False,
    )


def _attach_samples(
    replay: _Replay,
    samples: Mapping[str, Mapping[str, float]],
) -> None:
    for session_id, sample in samples.items():
        session = replay.sessions.get(session_id)
        if session is None:
            raise ActivityError(
                f"active-time sample belongs to unopened session {session_id!r}"
            )
        if sample["observed_at"] < session.opened_at:
            raise ActivityError(
                f"active-time sample predates session {session_id!r}"
            )
        session.active_seconds = sample["active_seconds"]
        session.sample_observed_at = sample["observed_at"]
    for session in replay.sessions.values():
        if session.end_event is None:
            continue
        if session.sample_observed_at is None:
            raise ActivityError(
                f"ended session {session.session_id!r} has no final sample"
            )
        if (
            session.sample_observed_at
            != session.end_event["final_sample_observed_at"]
        ):
            raise ActivityError(
                f"ended session {session.session_id!r} final sample is stale"
            )


def _activity_report(
    sessions: list[_Session],
    all_events: list[dict[str, Any]],
) -> ActivityReport:
    metered = [
        session.session_id
        for session in sessions
        if session.active_seconds is not None
    ]
    session_ids = {session.session_id for session in sessions}
    evidence = [event for event in all_events if event["session_id"] in session_ids]
    evidence.extend(
        _event(
            "active_sample",
            session.cell_id,
            session.session_id,
            session.sample_observed_at,
            active_seconds=session.active_seconds,
        )
        for session in sorted(sessions, key=lambda item: item.session_id)
        if session.active_seconds is not None
        and session.sample_observed_at is not None
    )
    evidence_bytes = b"".join(_canonical_line(event) for event in evidence)
    return ActivityReport(
        active_seconds=sum(session.active_seconds or 0.0 for session in sessions),
        sessions=tuple(sorted(session.session_id for session in sessions)),
        metered_sessions=tuple(sorted(metered)),
        complete=bool(sessions) and all(
            session.ended_at is not None and session.active_seconds is not None
            for session in sessions
        ),
        event_count=len(evidence),
        sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        bindings=tuple(
            sorted(
                (session.session_id, session.binding_sha256)
                for session in sessions
                if session.binding_sha256 is not None
            )
        ),
        open_sessions=tuple(
            sorted(
                session.session_id
                for session in sessions
                if session.ended_at is None
            )
        ),
        ended_sessions=tuple(
            sorted(
                session.session_id
                for session in sessions
                if session.ended_at is not None
            )
        ),
    )


def _empty_report() -> ActivityReport:
    return ActivityReport(
        active_seconds=0.0,
        sessions=(),
        metered_sessions=(),
        complete=False,
        event_count=0,
        sha256=hashlib.sha256(b"").hexdigest(),
        bindings=(),
        open_sessions=(),
        ended_sessions=(),
    )


def _parse_active_samples(exposition: str) -> dict[str, float]:
    if not isinstance(exposition, str):
        raise ActivityError("Prometheus metrics payload must be text")
    by_series: dict[tuple[str, str], float] = {}
    for raw_line in exposition.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0]
        if name not in _PROMETHEUS_METRIC_NAMES:
            continue
        match = _PROMETHEUS_LINE.fullmatch(line)
        if match is None:
            raise ActivityError("active-time metric has invalid exposition syntax")
        attributes = _prometheus_labels(match.group("labels"))
        session_id = _identifier(
            attributes.get("session.id", attributes.get("session_id")),
            "session.id",
        )
        activity_type = _identifier(attributes.get("type"), "type")
        if activity_type not in {"cli", "user"}:
            raise ActivityError(f"unsupported active-time type {activity_type!r}")
        key = (session_id, activity_type)
        if key in by_series:
            raise ActivityError(f"duplicate active-time series {key!r}")
        by_series[key] = _prometheus_number(match.group("value"))

    samples: dict[str, float] = {}
    for (session_id, _), value in by_series.items():
        samples[session_id] = samples.get(session_id, 0.0) + value
    return samples


def _prometheus_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _PROMETHEUS_LABEL.match(raw, position)
        if match is None:
            raise ActivityError("active-time metric has invalid labels")
        key, escaped_value = match.groups()
        if key in labels:
            raise ActivityError(f"duplicate Prometheus label {key!r}")
        try:
            labels[key] = json.loads(f'"{escaped_value}"')
        except json.JSONDecodeError as exc:
            raise ActivityError(f"invalid Prometheus label {key!r}") from exc
        position = match.end()
    return labels


def _prometheus_number(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ActivityError("active-time value must be numeric") from exc
    if not math.isfinite(value) or value < 0:
        raise ActivityError("active-time value must be finite and non-negative")
    return value


def _event(
    kind: str,
    cell_id: str | None,
    session_id: str,
    observed_at: float,
    **extra: object,
) -> dict[str, Any]:
    return {
        "event": kind,
        "cell_id": cell_id,
        "session_id": session_id,
        **extra,
        "observed_at": observed_at,
    }


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ActivityError(f"{name} must be a non-empty trimmed string")
    return value


def _optional_identifier(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _identifier(value, name)


def _timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActivityError("observed_at must be a finite non-negative number")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ActivityError("observed_at must be a finite non-negative number")
    return timestamp


def _seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActivityError("active_seconds must be a finite non-negative number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ActivityError("active_seconds must be a finite non-negative number")
    return seconds


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ActivityError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_line(event: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


@contextmanager
def _activity_lock(automil_dir: Path, *, exclusive: bool) -> Iterator[None]:
    path = automil_dir / _LOCK_FILENAME
    try:
        lock = path.open("a+")
    except OSError as exc:
        raise ActivityError(f"cannot open activity lock {path}: {exc}") from exc
    with lock:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(lock.fileno(), operation)
        except OSError as exc:
            raise ActivityError(f"cannot acquire activity lock {path}: {exc}") from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _append_event(automil_dir: Path, event: dict[str, Any]) -> None:
    path = automil_dir / ACTIVITY_JOURNAL_FILENAME
    try:
        with _activity_lock(automil_dir, exclusive=True):
            replay = _read_journal_unlocked(automil_dir)
            if not _apply_event(replay, event):
                return
            if event["event"] == "session_end":
                sample = _read_samples_unlocked(automil_dir).get(event["session_id"])
                if sample is None:
                    raise ActivityError(
                        f"session {event['session_id']!r} has no final "
                        "active-time sample"
                    )
                if sample["observed_at"] != event["final_sample_observed_at"]:
                    raise ActivityError(
                        f"session {event['session_id']!r} final active-time "
                        "sample is stale"
                    )
                if sample["observed_at"] > event["observed_at"]:
                    raise ActivityError(
                        f"final active-time sample for session "
                        f"{event['session_id']!r} is newer than SessionEnd"
                    )
            with path.open("ab") as journal:
                journal.write(_canonical_line(event))
                journal.flush()
                os.fsync(journal.fileno())
    except ActivityError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ActivityError(f"cannot append activity journal {path}: {exc}") from exc


def _read_journal_unlocked(automil_dir: Path) -> _Replay:
    path = automil_dir / ACTIVITY_JOURNAL_FILENAME
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _Replay()
    except (OSError, UnicodeError) as exc:
        raise ActivityError(f"cannot read activity journal {path}: {exc}") from exc
    return _replay_text(content, path)


def _replay_text(content: str, path: Path) -> _Replay:
    replay = _Replay()
    if not content:
        return replay
    if not content.endswith("\n"):
        raise ActivityError(f"activity journal {path} has a truncated final line")
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ActivityError(f"activity journal {path}:{line_number} is blank")
        try:
            event = _validate_journal_event(json.loads(line))
            _apply_event(replay, event)
        except ActivityError as exc:
            raise ActivityError(
                f"activity journal {path}:{line_number}: {exc}"
            ) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ActivityError(
                f"activity journal {path}:{line_number} is corrupt: {exc}"
            ) from exc
    return replay


def _validate_journal_event(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ActivityError("event must be a JSON object")
    kind = raw.get("event")
    if kind not in _EVENT_KEYS:
        raise ActivityError(f"unknown journal event {kind!r}")
    if frozenset(raw) != _EVENT_KEYS[kind]:
        raise ActivityError(f"invalid fields for {kind!r}")
    event = _event(
        kind,
        _optional_identifier(raw.get("cell_id"), "cell_id"),
        _identifier(raw.get("session_id"), "session_id"),
        _timestamp(raw.get("observed_at")),
    )
    if kind == "session_bind":
        digest = raw.get("binding_sha256")
        event["binding_sha256"] = (
            None if digest is None else _sha256(digest, "binding_sha256")
        )
    elif kind == "session_end":
        event["final_sample_observed_at"] = _timestamp(
            raw.get("final_sample_observed_at")
        )
        if event["final_sample_observed_at"] > event["observed_at"]:
            raise ActivityError("final active-time sample is newer than SessionEnd")
    return event


def _apply_event(replay: _Replay, event: dict[str, Any]) -> bool:
    kind = event["event"]
    cell_id = event["cell_id"]
    session_id = event["session_id"]
    observed_at = event["observed_at"]
    session = replay.sessions.get(session_id)

    if kind == "session_open":
        if session is not None:
            if session.open_event == event:
                return False
            if session.cell_id != cell_id:
                raise ActivityError(f"session {session_id!r} belongs to another cell")
            raise ActivityError(
                f"conflicting duplicate session_open for {session_id!r}"
            )
        replay.sessions[session_id] = _Session(
            cell_id=cell_id,
            session_id=session_id,
            opened_at=observed_at,
            open_event=event,
            latest_at=observed_at,
        )
        replay.events.append(event)
        return True

    if session is None:
        if kind == "session_end":
            raise ActivityError(f"unmatched session_end for session {session_id!r}")
        raise ActivityError(f"event for unopened session {session_id!r}")
    if kind == "session_end" and cell_id is not None and session.cell_id != cell_id:
        raise ActivityError(f"session {session_id!r} belongs to another cell")

    if kind == "session_bind":
        digest = event["binding_sha256"]
        if session.ended_at is not None:
            raise ActivityError(f"session_bind after session {session_id!r} ended")
        if session.cell_id is not None:
            if session.cell_id != cell_id:
                raise ActivityError(f"conflicting binding for session {session_id!r}")
            if session.binding_sha256 == digest:
                return False
            if session.binding_sha256 is not None:
                raise ActivityError(f"conflicting binding for session {session_id!r}")
        _require_monotonic(session, observed_at)
        session.cell_id = cell_id
        session.binding_sha256 = digest
    elif kind == "session_end":
        if session.end_event is not None:
            if session.end_event == event:
                return False
            raise ActivityError(f"conflicting duplicate session_end for {session_id!r}")
        _require_monotonic(session, observed_at)
        session.ended_at = observed_at
        session.end_event = event
    else:  # pragma: no cover - journal validation makes this unreachable
        raise ActivityError(f"unknown event {kind!r}")

    session.latest_at = observed_at
    replay.events.append(event)
    return True


def _require_monotonic(session: _Session, observed_at: float) -> None:
    if observed_at < session.latest_at:
        raise ActivityError(f"timestamp regression for session {session.session_id!r}")


def _read_samples_unlocked(automil_dir: Path) -> dict[str, dict[str, float]]:
    path = automil_dir / ACTIVITY_SAMPLES_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ActivityError(f"cannot read activity samples {path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "sessions"}:
        raise ActivityError(f"activity samples {path} have invalid fields")
    if raw["schema_version"] != _SAMPLES_SCHEMA_VERSION:
        raise ActivityError(f"activity samples {path} have unsupported schema")
    sessions = raw["sessions"]
    if not isinstance(sessions, dict):
        raise ActivityError(f"activity samples {path} sessions must be an object")

    normalized: dict[str, dict[str, float]] = {}
    for raw_session_id, raw_sample in sessions.items():
        session_id = _identifier(raw_session_id, "sample session_id")
        if not isinstance(raw_sample, dict) or set(raw_sample) != {
            "active_seconds",
            "observed_at",
        }:
            raise ActivityError(
                f"activity sample for {session_id!r} has invalid fields"
            )
        normalized[session_id] = {
            "active_seconds": _seconds(raw_sample["active_seconds"]),
            "observed_at": _timestamp(raw_sample["observed_at"]),
        }
    return normalized


def _write_samples_atomic(
    automil_dir: Path,
    samples: Mapping[str, Mapping[str, float]],
) -> None:
    destination = automil_dir / ACTIVITY_SAMPLES_FILENAME
    payload = {
        "schema_version": _SAMPLES_SCHEMA_VERSION,
        "sessions": samples,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=".activity.samples.", suffix=".tmp", dir=automil_dir
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise ActivityError(f"cannot write activity samples {destination}: {exc}") from exc
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
