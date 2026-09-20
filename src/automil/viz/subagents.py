"""Subagent transcripts from a session's sidecar directory.

Claude Code keeps each subagent's own transcript at
``<session>/subagents/agent-<id>.jsonl`` with an ``agent-<id>.meta.json``
naming the parent ``toolUseId``. Reading them is the same as reading the main
file; attaching them links the parent turn's ``Agent`` call to the agent id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from automil.viz.transcript import FullResultRef, ParsedTranscript, ReadStats, parse_transcript

SUBAGENTS_DIRNAME = "subagents"
_PREFIX = "agent-"


@dataclass(frozen=True)
class SubagentTranscript:
    agent_id: str
    tool_use_id: str | None
    agent_type: str | None
    description: str | None
    spawn_depth: int | None
    path: Path
    turns: tuple[dict[str, Any], ...]
    full_refs: Mapping[str, FullResultRef]
    stats: ReadStats
    models: tuple[str, ...]


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_subagents(sidecar_dir: Path | None) -> tuple[SubagentTranscript, ...]:
    """Every subagent under ``sidecar_dir/subagents``, by agent id.

    A transcript without its meta file is still read (with unknown parent);
    a meta file without its transcript yields an empty subagent, so the parent
    link is not lost.
    """
    if sidecar_dir is None:
        return ()
    folder = sidecar_dir / SUBAGENTS_DIRNAME
    if not folder.is_dir():
        return ()
    ids: set[str] = set()
    for entry in folder.iterdir():
        name = entry.name
        if not name.startswith(_PREFIX):
            continue
        if name.endswith(".meta.json"):
            ids.add(name[len(_PREFIX):-len(".meta.json")])
        elif name.endswith(".jsonl"):
            ids.add(name[len(_PREFIX):-len(".jsonl")])
    out = []
    for agent_id in sorted(ids):
        meta = _read_meta(folder / f"{_PREFIX}{agent_id}.meta.json")
        transcript = folder / f"{_PREFIX}{agent_id}.jsonl"
        if transcript.is_file():
            parsed = parse_transcript(transcript, sidechain_is_main=True)
            turns, refs, stats, models = parsed.turns, parsed.full_refs, parsed.stats, parsed.models
        else:
            turns, refs, stats, models = (), {}, ReadStats(), ()
        depth = meta.get("spawnDepth")
        out.append(
            SubagentTranscript(
                agent_id=agent_id,
                tool_use_id=meta.get("toolUseId") if isinstance(meta.get("toolUseId"), str) else None,
                agent_type=meta.get("agentType") if isinstance(meta.get("agentType"), str) else None,
                description=meta.get("description") if isinstance(meta.get("description"), str) else None,
                spawn_depth=depth if isinstance(depth, int) else None,
                path=transcript,
                turns=turns,
                full_refs=refs,
                stats=stats,
                models=models,
            )
        )
    return tuple(out)


def attach_subagents(parsed: ParsedTranscript, subagents: tuple[SubagentTranscript, ...]) -> ParsedTranscript:
    """Return a transcript whose ``Agent`` calls name their subagent's id."""
    by_tool: dict[str, str] = {s.tool_use_id: s.agent_id for s in subagents if s.tool_use_id}
    if not by_tool:
        return parsed
    turns = []
    for turn in parsed.turns:
        calls = list(turn["tool_calls"])
        changed = False
        for index, call in enumerate(calls):
            agent_id = by_tool.get(call["tool_use_id"])
            if agent_id is None:
                continue
            result = call["result"] or {"status": "missing"}
            if result.get("agent_id") == agent_id:
                continue
            calls[index] = dict(call, result=dict(result, agent_id=agent_id))
            changed = True
        turns.append(dict(turn, tool_calls=calls) if changed else turn)
    return replace(parsed, turns=tuple(turns))
