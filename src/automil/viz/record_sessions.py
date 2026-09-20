"""Which sessions a run has, their turns in chunks, and node-to-turn links.

A run's sessions are the journaled ones (``.activity.jsonl``) joined with the
stored records under ``automil/sessions``. A session still open in the journal
is read from the runtime's own file so the dashboard can follow it live.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from automil.cells.activity import ActivityError, JournalSession, journal_sessions
from automil.session_record import locate_home_transcript, read_stored_sessions
from automil.viz.clock import HostClock, to_utc_iso
from automil.viz.record_files import RunPaths
from automil.viz.subagents import SubagentTranscript
from automil.viz.transcript import ParsedTranscript

TURN_CHUNK_SIZE = 100


@dataclass(frozen=True)
class SessionSource:
    session_id: str
    transcript: Path | None
    sidecar: Path | None
    origin: Literal["stored", "home", "missing"]
    live: bool
    journal: JournalSession | None


def discover_sessions(paths: RunPaths, *, config_dir: Path | None = None) -> tuple[tuple[SessionSource, ...], str | None]:
    """Journaled sessions joined with stored records; returns (sources, journal error)."""
    error = None
    try:
        journaled = journal_sessions(paths.automil_dir)
    except ActivityError as exc:
        journaled = ()
        error = str(exc)
    stored = {record.session_id: record for record in read_stored_sessions(paths.automil_dir)}
    sources: list[SessionSource] = []
    seen: set[str] = set()
    for session in journaled:
        seen.add(session.session_id)
        live = session.ended_at is None
        record = stored.get(session.session_id)
        home = locate_home_transcript(session.session_id, config_dir=config_dir) if live or record is None else None
        if live and home is not None:
            sources.append(SessionSource(session.session_id, home, home.with_suffix(""), "home", True, session))
        elif record is not None:
            sources.append(SessionSource(session.session_id, record.transcript, record.sidecar, "stored", live, session))
        elif home is not None:
            sources.append(SessionSource(session.session_id, home, home.with_suffix(""), "home", live, session))
        else:
            sources.append(SessionSource(session.session_id, None, None, "missing", live, session))
    for session_id, record in stored.items():
        if session_id not in seen:
            sources.append(SessionSource(session_id, record.transcript, record.sidecar, "stored", False, None))
    return tuple(sources), error


def chunk_turns(turns: Iterable[Mapping[str, Any]], size: int = TURN_CHUNK_SIZE) -> list[list[Mapping[str, Any]]]:
    items = list(turns)
    return [items[i:i + size] for i in range(0, len(items), size)] or []


def _usage_total(turns: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = {"input": 0, "output": 0, "cache_read": 0, "cache_create": 0}
    for turn in turns:
        usage = turn.get("usage") or {}
        for key in total:
            value = usage.get(key)
            if isinstance(value, int):
                total[key] += value
    return total


def _epoch_iso(value: float | None, clock: HostClock) -> str | None:
    return to_utc_iso(value, clock) if value is not None else None


def session_summary(
    source: SessionSource,
    parsed: ParsedTranscript | None,
    subagents: tuple[SubagentTranscript, ...],
    clock: HostClock,
    *,
    active_seconds: float | None = None,
) -> dict[str, Any]:
    """One entry of ``sessions.json``."""
    turns = parsed.turns if parsed else ()
    calls = [call for turn in turns for call in turn.get("tool_calls", ())]
    journal = source.journal
    stamps = [t["at"] for t in turns if t.get("at")]
    return {
        "session_id": source.session_id,
        "cell_id": journal.cell_id if journal else None,
        "opened_at": _epoch_iso(journal.opened_at, clock) if journal else None,
        "ended_at": _epoch_iso(journal.ended_at, clock) if journal else None,
        "ended_by": journal.ended_by if journal else None,
        "source": source.origin,
        "live": source.live,
        "n_turns": len(turns),
        "n_chunks": len(chunk_turns(turns)),
        "first_at": min(stamps) if stamps else None,
        "last_at": max(stamps) if stamps else None,
        "models": list(parsed.models) if parsed else [],
        "usage": _usage_total(turns),
        "n_tool_calls": len(calls),
        "n_automil_commands": sum(1 for call in calls if call.get("automil")),
        "n_prompts": sum(1 for t in turns if t.get("kind") == "human"),
        "active_seconds": active_seconds,
        "cwd": parsed.cwd if parsed else None,
        "subagents": [
            {
                "agent_id": s.agent_id,
                "tool_use_id": s.tool_use_id,
                "agent_type": s.agent_type,
                "description": s.description,
                "n_turns": len(s.turns),
            }
            for s in subagents
        ],
        "stats": parsed.stats.as_dict() if parsed else None,
    }


def _parse_stamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _call_window(turn: Mapping[str, Any], call: Mapping[str, Any], next_turn_at: str | None) -> tuple[datetime | None, datetime | None]:
    start = _parse_stamp(turn.get("at"))
    result = call.get("result") or {}
    end = _parse_stamp(result.get("at")) or _parse_stamp(next_turn_at) or _parse_stamp(turn.get("at_end"))
    return start, end


def build_links(
    session_id: str,
    turns: Iterable[Mapping[str, Any]],
    submitted_at: Mapping[str, str | None] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """Node -> turn links for one session, and the reverse map.

    Kinds: ``propose``/``submit``/``resubmit`` when a call's output reported
    the node created; ``submit`` also when the id is on the command line or
    when the node's ``submitted_at`` falls inside a submit call's window (the
    agent often submits in shell loops whose ids never appear literally);
    ``mention`` for ids on any other subcommand's command line.
    """
    ordered = list(turns)
    by_node: dict[str, list[dict[str, Any]]] = {}
    by_turn: dict[str, list[str]] = {}
    submitted = dict(submitted_at or {})

    def add(node_id: str, turn_index: int, kind: str, at: str | None) -> None:
        entry = {"session_id": session_id, "turn": turn_index, "kind": kind, "at": at}
        bucket = by_node.setdefault(node_id, [])
        if not any(e["turn"] == turn_index and e["kind"] == kind for e in bucket):
            bucket.append(entry)
        key = f"{session_id}:{turn_index}"
        if node_id not in by_turn.setdefault(key, []):
            by_turn[key].append(node_id)

    for position, turn in enumerate(ordered):
        next_at = ordered[position + 1].get("at") if position + 1 < len(ordered) else None
        for call in turn.get("tool_calls", ()):
            automil = call.get("automil")
            if not automil:
                continue
            at = turn.get("at")
            for created in automil.get("created", ()):
                add(created["node_id"], turn["index"], created["sub"], at)
            subs = {c.get("sub") for c in automil.get("calls", ())}
            for sub_call in automil.get("calls", ()):
                kind = "submit" if sub_call.get("sub") == "submit" else "mention"
                for node_id in sub_call.get("node_ids", ()):
                    add(node_id, turn["index"], kind, at)
            if "submit" in subs and submitted:
                start, end = _call_window(turn, call, next_at)
                if start and end:
                    for node_id, stamp in submitted.items():
                        moment = _parse_stamp(stamp)
                        if moment and start <= moment <= end:
                            add(node_id, turn["index"], "submit", at)
    for bucket in by_node.values():
        bucket.sort(key=lambda e: (e["turn"], e["kind"]))
    for ids in by_turn.values():
        ids.sort()
    return by_node, by_turn


def agent_events(session_id: str, turns: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Dated timeline events from one session: creations, prompts, notifications, compactions."""
    events: list[dict[str, Any]] = []
    for turn in turns:
        at = turn.get("at")
        kind = turn.get("kind")
        if kind == "human":
            events.append({"at": at, "kind": "prompt", "session_id": session_id, "turn": turn["index"], "node_id": None, "label": (turn.get("text") or "")[:80]})
        elif kind == "notification":
            note = turn.get("notification") or {}
            events.append({"at": at, "kind": "notification", "session_id": session_id, "turn": turn["index"], "node_id": None, "label": note.get("title") or ""})
        elif kind == "system" and (turn.get("system") or {}).get("subtype") == "compact_boundary":
            events.append({"at": at, "kind": "compact", "session_id": session_id, "turn": turn["index"], "node_id": None, "label": "context compacted"})
        for call in turn.get("tool_calls", ()):
            automil = call.get("automil")
            if not automil:
                continue
            for created in automil.get("created", ()):
                events.append({
                    "at": at, "kind": created["sub"], "session_id": session_id, "turn": turn["index"],
                    "node_id": created["node_id"], "label": created["node_id"],
                })
            for sub_call in automil.get("calls", ()):
                if sub_call.get("sub") in ("reconcile", "rank"):
                    events.append({"at": at, "kind": sub_call["sub"], "session_id": session_id, "turn": turn["index"], "node_id": None, "label": sub_call["sub"]})
    return events


def session_spans(sources: Iterable[SessionSource], clock: HostClock) -> list[dict[str, Any]]:
    spans = []
    for source in sources:
        journal = source.journal
        spans.append({
            "session_id": source.session_id,
            "cell_id": journal.cell_id if journal else None,
            "opened_at": _epoch_iso(journal.opened_at, clock) if journal else None,
            "ended_at": _epoch_iso(journal.ended_at, clock) if journal else None,
            "live": source.live,
        })
    return spans


def session_journal_events(sources: Iterable[SessionSource], clock: HostClock) -> list[dict[str, Any]]:
    events = []
    for source in sources:
        journal = source.journal
        if journal is None:
            continue
        events.append({"at": _epoch_iso(journal.opened_at, clock), "kind": "session_open", "session_id": source.session_id, "turn": None, "node_id": None, "label": "session opened"})
        if journal.ended_at is not None:
            events.append({"at": _epoch_iso(journal.ended_at, clock), "kind": "session_end", "session_id": source.session_id, "turn": None, "node_id": None, "label": f"session ended ({journal.ended_by})"})
    return events
