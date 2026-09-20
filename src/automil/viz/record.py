"""One run's record: the payloads the dashboard reads, built from the project.

``RunSource`` is the only composition point. The live server calls its
builders per request; ``automil viz export`` calls them once and writes the
results to disk. Both get the same dicts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from automil.cells.registry import scan_cells
from automil.viz.clock import HostClock, offset_cross_check, to_utc_iso
from automil.viz.record_files import (
    RunPaths,
    load_campaign_state,
    load_certification,
    load_completed,
    load_config,
    load_gpu_state,
    load_graph,
    load_notes,
    load_orchestrator_log,
    load_overlay_file,
    load_result,
    load_run_log_tail,
    load_spec,
    overlay_files,
)
from automil.viz.record_graph import (
    RECORD_SCHEMA,
    node_verdict,
    project_graph,
    project_node,
    project_result,
    running_node_ids,
    verdict_unavailable,
)
from automil.viz.record_sessions import (
    SessionSource,
    agent_events,
    build_links,
    chunk_turns,
    discover_sessions,
    session_journal_events,
    session_spans,
    session_summary,
)
from automil.viz.record_timeline import build_timeline, node_timing, parse_orchestrator_log
from automil.viz.subagents import SubagentTranscript, attach_subagents, read_subagents
from automil.viz.transcript import (
    FullResultRef,
    ParsedTranscript,
    parse_transcript,
    read_full_result,
)

logger = logging.getLogger(__name__)

_RUN_ID_CLEAN = re.compile(r"[^a-z0-9._-]+")
_TERMINAL = frozenset({"keep", "discard", "crash", "partial", "cancelled", "candidate", "registered", "oom", "timeout"})


def run_id_from_config(config: Mapping[str, Any], fallback: str = "project") -> str:
    name = (config.get("project") or {}).get("name") if isinstance(config.get("project"), Mapping) else None
    slug = _RUN_ID_CLEAN.sub("-", str(name).strip().lower()).strip("-.") if name else ""
    return slug or fallback


@dataclass
class _SessionCache:
    source: SessionSource
    parsed: ParsedTranscript
    subagents: tuple[SubagentTranscript, ...]
    size: int
    mtime_ns: int


@dataclass
class RunSource:
    """Everything needed to build one run's record, with light caching."""

    automil_dir: Path
    clock: HostClock
    run_id: str | None = None
    title: str | None = None
    config_dir: Path | None = None
    mode: str = "live"
    _sessions: dict[str, _SessionCache] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.automil_dir = Path(self.automil_dir)
        self.paths = RunPaths(self.automil_dir)
        self.config = load_config(self.paths)
        if self.run_id is None:
            self.run_id = run_id_from_config(self.config)
        if self.title is None:
            project = self.config.get("project") if isinstance(self.config.get("project"), Mapping) else {}
            self.title = str(project.get("description") or project.get("name") or self.run_id)

    # -- graph ------------------------------------------------------------

    def raw_graph(self) -> dict[str, Any]:
        return load_graph(self.paths) or {"nodes": {}, "meta": {}, "technique_stats": {}}

    def running_ids(self) -> tuple[str, ...]:
        return running_node_ids(load_gpu_state(self.paths))

    def build_graph(self) -> dict[str, Any]:
        graph = project_graph(self.raw_graph(), self.clock, self.running_ids())
        graph["run_id"] = self.run_id
        return graph

    # -- sessions ---------------------------------------------------------

    def sessions(self) -> tuple[tuple[SessionSource, ...], str | None]:
        return discover_sessions(self.paths, config_dir=self.config_dir)

    def _parsed(self, source: SessionSource) -> tuple[ParsedTranscript | None, tuple[SubagentTranscript, ...]]:
        if source.transcript is None:
            return None, ()
        try:
            stat = source.transcript.stat()
        except OSError:
            return None, ()
        cached = self._sessions.get(source.session_id)
        if cached and cached.size == stat.st_size and cached.mtime_ns == stat.st_mtime_ns and cached.source.transcript == source.transcript:
            return cached.parsed, cached.subagents
        parsed = parse_transcript(source.transcript, ended=not source.live)
        subagents = read_subagents(source.sidecar)
        parsed = attach_subagents(parsed, subagents)
        self._sessions[source.session_id] = _SessionCache(source, parsed, subagents, stat.st_size, stat.st_mtime_ns)
        return parsed, subagents

    def _active_seconds(self) -> dict[str, float]:
        from automil.viz.record_files import read_json

        payload = read_json(self.automil_dir / ".activity.samples.json")
        sessions = payload.get("sessions") if isinstance(payload, dict) else None
        if not isinstance(sessions, dict):
            return {}
        out = {}
        for session_id, sample in sessions.items():
            if isinstance(sample, dict) and isinstance(sample.get("active_seconds"), (int, float)):
                out[str(session_id)] = float(sample["active_seconds"])
        return out

    def build_sessions(self) -> dict[str, Any]:
        sources, error = self.sessions()
        active = self._active_seconds()
        entries = []
        for source in sources:
            parsed, subagents = self._parsed(source)
            entries.append(session_summary(source, parsed, subagents, self.clock, active_seconds=active.get(source.session_id)))
        scan = scan_cells(self.paths.cells_dir)
        cells = [
            {
                "cell_id": c.cell_id, "dataset": c.dataset, "encoder": c.encoder, "mil_model": c.mil_model,
                "task": getattr(c, "task", None), "status": getattr(c.status, "value", str(c.status)),
                "mode": c.mode, "started_at": to_utc_iso(c.started_at, self.clock),
                "budget_seconds": c.budget_seconds, "eval_budget": c.eval_budget,
                "consumed_evals": c.consumed_evals, "completed_evals": c.completed_evals,
            }
            for c in scan.cells
        ]
        return {
            "schema": RECORD_SCHEMA,
            "run_id": self.run_id,
            "chunk_size": 100,
            "journal_error": error,
            "sessions": entries,
            "cells": cells,
            "cell_errors": [f"{e.path}: {e.message}" for e in scan.errors],
        }

    def build_chunk(self, session_id: str, chunk: int) -> dict[str, Any] | None:
        source = self._source(session_id)
        if source is None:
            return None
        parsed, _ = self._parsed(source)
        if parsed is None:
            return None
        chunks = chunk_turns(parsed.turns)
        if chunk < 0 or chunk >= max(len(chunks), 1):
            return None
        turns = chunks[chunk] if chunks else []
        return {
            "schema": RECORD_SCHEMA,
            "session_id": session_id,
            "chunk": chunk,
            "from": chunk * 100,
            "turns": list(turns),
            "n_turns": len(parsed.turns),
            "open_turn": parsed.open_turn if chunk == max(len(chunks) - 1, 0) else None,
        }

    def build_full_result(self, session_id: str, tool_use_id: str) -> dict[str, Any] | None:
        source = self._source(session_id)
        if source is None or source.transcript is None:
            return None
        parsed, subagents = self._parsed(source)
        if parsed is None:
            return None
        ref: FullResultRef | None = parsed.full_refs.get(tool_use_id)
        path = source.transcript
        if ref is None:
            for agent in subagents:
                if tool_use_id in agent.full_refs:
                    ref, path = agent.full_refs[tool_use_id], agent.path
                    break
        if ref is None:
            return None
        return dict(read_full_result(path, ref), session_id=session_id)

    def build_agent(self, session_id: str, agent_id: str) -> dict[str, Any] | None:
        source = self._source(session_id)
        if source is None:
            return None
        parsed, subagents = self._parsed(source)
        for agent in subagents:
            if agent.agent_id == agent_id:
                return {
                    "schema": RECORD_SCHEMA,
                    "session_id": session_id,
                    "agent_id": agent.agent_id,
                    "tool_use_id": agent.tool_use_id,
                    "agent_type": agent.agent_type,
                    "description": agent.description,
                    "turns": list(agent.turns),
                    "models": list(agent.models),
                }
        if parsed is not None and agent_id in parsed.inline_sidechains:
            return {
                "schema": RECORD_SCHEMA, "session_id": session_id, "agent_id": agent_id,
                "tool_use_id": None, "agent_type": None, "description": None,
                "turns": list(parsed.inline_sidechains[agent_id]), "models": [],
            }
        return None

    def _source(self, session_id: str) -> SessionSource | None:
        sources, _ = self.sessions()
        for source in sources:
            if source.session_id == session_id:
                return source
        return None

    # -- links and timeline ----------------------------------------------

    def _executed_ids(self, graph: Mapping[str, Any]) -> list[str]:
        nodes = graph.get("nodes") or {}
        return [nid for nid, node in nodes.items() if isinstance(node, Mapping) and node.get("type") == "executed"]

    def _timings(self, graph: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        events = parse_orchestrator_log(load_orchestrator_log(self.paths), self.clock)
        timings = {}
        for node_id in self._executed_ids(graph):
            timings[node_id] = node_timing(
                node_id, spec=load_spec(self.paths, node_id), completed=load_completed(self.paths, node_id),
                log_events=events, clock=self.clock,
            )
        return timings

    def build_links(self) -> dict[str, Any]:
        graph = self.raw_graph()
        submitted = {nid: t.get("submitted_at") for nid, t in self._timings(graph).items()}
        sources, _ = self.sessions()
        nodes: dict[str, list[dict[str, Any]]] = {}
        turns: dict[str, list[str]] = {}
        for source in sources:
            parsed, _ = self._parsed(source)
            if parsed is None:
                continue
            by_node, by_turn = build_links(source.session_id, parsed.turns, submitted)
            for node_id, entries in by_node.items():
                nodes.setdefault(node_id, []).extend(entries)
            turns.update(by_turn)
        return {"schema": RECORD_SCHEMA, "run_id": self.run_id, "nodes": nodes, "turns": turns}

    def build_timeline(self) -> dict[str, Any]:
        graph = self.raw_graph()
        timings = self._timings(graph)
        raw_nodes = graph.get("nodes") or {}
        running = set(self.running_ids())
        nodes = []
        for node_id, timing in timings.items():
            node = raw_nodes[node_id]
            nodes.append({
                "node_id": node_id,
                "parent_id": node.get("parent_id"),
                "status": "running" if node_id in running else node.get("status"),
                "primary_value": node.get("primary_value"),
                "created_at": to_utc_iso(node.get("created_at"), self.clock),
                **{k: timing.get(k) for k in ("submitted_at", "launched_at", "completed_at", "slot")},
            })
        sources, _ = self.sessions()
        events = session_journal_events(sources, self.clock)
        pairs = []
        for source in sources:
            parsed, _ = self._parsed(source)
            if parsed is None:
                continue
            events.extend(agent_events(source.session_id, parsed.turns))
            for turn in parsed.turns:
                for call in turn.get("tool_calls", ()):
                    for created in (call.get("automil") or {}).get("created", ()):
                        raw_created = (raw_nodes.get(created["node_id"]) or {}).get("created_at")
                        if isinstance(raw_created, str) and turn.get("at"):
                            pairs.append((raw_created, turn["at"]))
        for node_id, timing in timings.items():
            for key, kind in (("launched_at", "launched"), ("completed_at", "completed")):
                if timing.get(key):
                    events.append({"at": timing[key], "kind": kind, "session_id": None, "turn": None, "node_id": node_id, "label": node_id})
        warnings = []
        inferred = offset_cross_check(pairs)
        if inferred is not None and inferred != self.clock.utc_offset_s:
            warnings.append(
                f"the transcript implies a host offset of {inferred // 3600:+d} h; the record used {self.clock.utc_offset_s // 3600:+d} h"
            )
        return build_timeline(
            run_id=self.run_id, clock=self.clock, nodes=nodes,
            sessions=session_spans(sources, self.clock), events=events, warnings=warnings,
        )

    # -- nodes ------------------------------------------------------------

    def build_node(self, node_id: str) -> dict[str, Any] | None:
        graph = self.raw_graph()
        raw = (graph.get("nodes") or {}).get(node_id)
        if not isinstance(raw, Mapping):
            return None
        running = set(self.running_ids())
        node = project_node(raw, self.clock)
        if node_id in running:
            node["status"] = "running"
        parent = (graph.get("nodes") or {}).get(raw.get("parent_id")) if raw.get("parent_id") else None
        verdict = node_verdict(graph.get("meta"), parent, raw) if node_id not in running else None
        reason = "not terminal" if node_id in running else verdict_unavailable(raw)
        completed = load_completed(self.paths, node_id) if raw.get("type") == "executed" else None
        result_raw = completed or (load_result(self.paths, node_id) if raw.get("type") == "executed" else None)
        terminal = node_id not in running and (completed is not None or raw.get("status") in _TERMINAL)
        spec = load_spec(self.paths, node_id) if raw.get("type") == "executed" else None
        run_log = load_run_log_tail(self.paths, node_id) if (terminal and raw.get("type") == "executed" and not raw.get("bootstrapped")) else None
        if run_log is None:
            run_log = {"available": False, "reason": "not terminal" if not terminal else "no run log", "tail": []}
        events = parse_orchestrator_log(load_orchestrator_log(self.paths), self.clock)
        timing = node_timing(node_id, spec=spec, completed=completed, log_events=events, clock=self.clock)
        links = self.build_links()["nodes"].get(node_id, [])
        return {
            "schema": RECORD_SCHEMA,
            "run_id": self.run_id,
            "node_id": node_id,
            "node": node,
            "parent": project_node(parent, self.clock) if isinstance(parent, Mapping) else None,
            "result": project_result(result_raw, self.clock),
            "verdict": verdict,
            "verdict_unavailable": reason,
            "overlay": {
                "base_commit": spec.get("base_commit") if spec else None,
                "files": overlay_files(spec),
                "deletions": list(spec.get("deletions") or []) if spec else [],
                "run_command_override": spec.get("run_command_override") if spec else None,
            },
            "run_log": run_log,
            "timing": timing,
            "agent": links,
        }

    def build_overlay_file(self, node_id: str, rel_path: str) -> dict[str, Any] | None:
        return load_overlay_file(self.paths, node_id, rel_path)

    # -- notes, certification, index --------------------------------------

    def build_notes(self) -> dict[str, Any]:
        notes = load_notes(self.paths)
        return {"schema": RECORD_SCHEMA, "run_id": self.run_id, **notes}

    def build_certified(self) -> dict[str, Any] | None:
        """The cell's own certification bundle, when the campaign produced one."""
        bundle = load_certification(self.paths)
        if bundle is None:
            return None
        return {"schema": RECORD_SCHEMA, "run_id": self.run_id, "revealed": True, "bundle": bundle}

    def index_entry(self) -> dict[str, Any]:
        graph = self.raw_graph()
        nodes = graph.get("nodes") or {}
        meta = graph.get("meta") or {}
        stamps = sorted(s for s in (to_utc_iso(n.get("created_at"), self.clock) for n in nodes.values() if isinstance(n, Mapping)) if s)
        sources, error = self.sessions()
        state = load_campaign_state(self.paths)
        executed = [n for n in nodes.values() if isinstance(n, Mapping) and n.get("type") == "executed"]
        best_id = meta.get("best_node_id")
        return {
            "run_id": self.run_id,
            "title": self.title,
            "project": self.config.get("project") if isinstance(self.config.get("project"), Mapping) else {},
            "task": (self.config.get("task") or {}).get("name") if isinstance(self.config.get("task"), Mapping) else None,
            "encoder": (self.config.get("encoders") or {}).get("primary") if isinstance(self.config.get("encoders"), Mapping) else None,
            "mil_model": (self.config.get("run") or {}).get("mil_model") if isinstance(self.config.get("run"), Mapping) else None,
            "n_nodes": len(nodes),
            "n_executed": len(executed),
            "n_running": len(self.running_ids()),
            "best_primary_value": meta.get("best_primary_value"),
            "best_node_id": best_id,
            "baseline_primary_value": meta.get("baseline_primary_value"),
            "primary_metric": (meta.get("scoring") or {}).get("formula"),
            "started_at": stamps[0] if stamps else None,
            "ended_at": stamps[-1] if stamps else None,
            "cell_ids": sorted({str(n.get("cell_id")) for n in executed if n.get("cell_id")}),
            "n_sessions": len(sources),
            "live_sessions": sum(1 for s in sources if s.live),
            "journal_error": error,
            "campaign_phase": state.get("phase") if state else None,
            "certified": load_certification(self.paths) is not None,
            "clock": self.clock.as_dict(),
        }


def build_index(sources: list[RunSource], mode: str) -> dict[str, Any]:
    return {
        "schema": RECORD_SCHEMA,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "runs": [source.index_entry() for source in sources],
    }
