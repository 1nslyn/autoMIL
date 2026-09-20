"""Claude Code transcript lines become turns.

A transcript is one JSON object per line. One model response is written as
several ``assistant`` lines sharing a ``requestId``; tool results arrive later
as ``user`` lines. :class:`TurnBuilder` folds that stream into turns and keeps
enough state (pending tool calls, byte offsets) to be fed a growing file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from automil.trajectory.redactor import redact_secrets
from automil.viz.transcript_commands import detect_automil, summarize_calls

RESULT_PREVIEW_CHARS = 2048
_PREVIEW_HEAD = 1536
_PREVIEW_TAIL = 512
RESULT_FULL_MAX_BYTES = 1 << 20
THINKING_MAX_CHARS = 32 * 1024
META_MAX_CHARS = 2048

_IGNORED_TYPES = frozenset({
    "attachment", "queue-operation", "mode", "permission-mode", "ai-title",
    "custom-title", "last-prompt", "file-history-snapshot", "file-history-delta",
    "atis-latch", "bridge-session", "summary",
})
_USAGE_FIELDS = (
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_read", "cache_read_input_tokens"),
    ("cache_create", "cache_creation_input_tokens"),
)
_TAG = {
    name: re.compile(rf"<{name}>(.*?)</{name}>", re.DOTALL)
    for name in ("task-id", "tool-use-id", "status", "summary", "result")
}


@dataclass(frozen=True)
class Record:
    """One decodable transcript line and where it sits in the file."""

    payload: dict[str, Any]
    offset: int
    length: int
    line_no: int


@dataclass(frozen=True)
class RecordBatch:
    records: tuple[Record, ...]
    garbled: int
    partial_tail: bool
    resume_offset: int
    next_line_no: int


def _decode(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_records(path: Path, *, start_offset: int = 0, line_no: int = 1) -> RecordBatch:
    """Read complete lines from ``start_offset``; a partial last line waits.

    Lines that are not UTF-8, not JSON or not an object are skipped and
    counted; blank lines are skipped silently. Never raises on content.
    """
    with open(path, "rb") as fh:
        fh.seek(start_offset)
        data = fh.read()
    records: list[Record] = []
    garbled = 0
    offset = start_offset
    number = line_no
    pos = 0
    while True:
        newline = data.find(b"\n", pos)
        if newline < 0:
            break
        raw = data[pos:newline]
        length = newline - pos + 1
        if raw.strip():
            payload = _decode(raw)
            if payload is None:
                garbled += 1
            else:
                records.append(Record(payload, offset, length, number))
        offset += length
        pos = newline + 1
        number += 1
    return RecordBatch(
        records=tuple(records),
        garbled=garbled,
        partial_tail=pos < len(data),
        resume_offset=offset,
        next_line_no=number,
    )


@dataclass(frozen=True)
class FullResultRef:
    """Where the untruncated text of one tool result lives in the file."""

    tool_use_id: str
    offset: int
    length: int


@dataclass(frozen=True)
class ReadStats:
    lines_total: int = 0
    lines_garbled: int = 0
    lines_unknown_type: int = 0
    lines_ignored: int = 0
    lines_sidechain: int = 0
    orphan_results: int = 0
    partial_tail: bool = False
    resume_offset: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class BuildDelta:
    """What one fed record changed: newly closed turns, late results, the open turn."""

    closed: tuple[dict[str, Any], ...] = ()
    patches: tuple[tuple[int, int, dict[str, Any]], ...] = ()
    open_changed: bool = False


@dataclass
class _Counters:
    total: int = 0
    unknown: int = 0
    ignored: int = 0
    sidechain: int = 0
    orphans: int = 0


@dataclass
class _Open:
    """The turn being assembled; becomes a plain dict when closed."""

    kind: str
    at: str | None
    uuids: list[str] = field(default_factory=list)
    request_id: str | None = None
    at_end: str | None = None
    model: str | None = None
    texts: list[str] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    images: int = 0
    notification: dict[str, Any] | None = None
    system: dict[str, Any] | None = None
    n_lines: int = 0


def _result_text(block: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[str, int]:
    """The textual content of a ``tool_result`` block and its image count."""
    content = block.get("content")
    images = 0
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif kind == "image":
                images += 1
            elif kind == "tool_reference":
                parts.append("[tool_reference]")
        text = "\n".join(parts)
    else:
        text = ""
    if not text:
        structured = payload.get("toolUseResult")
        if isinstance(structured, dict) and isinstance(structured.get("stdout"), str):
            text = structured["stdout"]
            if isinstance(structured.get("stderr"), str) and structured["stderr"]:
                text = f"{text}\n{structured['stderr']}" if text else structured["stderr"]
    return text, images


def _preview(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    omitted = len(text) - _PREVIEW_HEAD - _PREVIEW_TAIL
    return f"{text[:_PREVIEW_HEAD]}\n…[{omitted} chars omitted]…\n{text[-_PREVIEW_TAIL:]}", True


def _parse_notification(text: str) -> dict[str, Any]:
    """The task notification's fields; the ``<summary>`` tag becomes ``title``."""
    fields = {}
    for name, key in (("task-id", "task_id"), ("tool-use-id", "tool_use_id"), ("status", "status"), ("summary", "title")):
        match = _TAG[name].search(text)
        fields[key] = match.group(1).strip() if match else None
    return fields


def _notification_body(text: str) -> str:
    match = _TAG["result"].search(text)
    return match.group(1).strip() if match else text.strip()


class TurnBuilder:
    """Fold transcript records into turns; safe to feed incrementally."""

    def __init__(
        self,
        *,
        preview_chars: int = RESULT_PREVIEW_CHARS,
        redact: Callable[[str], str] = redact_secrets,
        sidechain_is_main: bool = False,
    ) -> None:
        self._preview_chars = preview_chars
        self._redact = redact
        self._sidechain_is_main = sidechain_is_main
        self._turns: list[dict[str, Any]] = []
        self._open: _Open | None = None
        self._pending: dict[str, tuple[int | None, int]] = {}
        self._uuid_turn: dict[str, int | None] = {}
        self._counters = _Counters()
        self._models: set[str] = set()
        self.full_refs: dict[str, FullResultRef] = {}
        self.inline: dict[str, TurnBuilder] = {}
        self.session_id: str | None = None
        self.cwd: str | None = None
        self.git_branch: str | None = None

    # -- public state ----------------------------------------------------

    @property
    def turns(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._turns)

    @property
    def open_turn(self) -> dict[str, Any] | None:
        return self._finish(self._open, len(self._turns)) if self._open else None

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def stats(self, *, garbled: int = 0, partial_tail: bool = False, resume_offset: int = 0) -> ReadStats:
        c = self._counters
        return ReadStats(
            lines_total=c.total + garbled,
            lines_garbled=garbled,
            lines_unknown_type=c.unknown,
            lines_ignored=c.ignored,
            lines_sidechain=c.sidechain,
            orphan_results=c.orphans,
            partial_tail=partial_tail,
            resume_offset=resume_offset,
        )

    # -- feeding ---------------------------------------------------------

    def feed(self, record: Record) -> BuildDelta:
        payload = record.payload
        self._counters.total += 1
        self._note_session(payload)
        if payload.get("isSidechain") and not self._sidechain_is_main:
            self._counters.sidechain += 1
            agent_id = str(payload.get("agentId") or "inline")
            self.inline.setdefault(agent_id, TurnBuilder(preview_chars=self._preview_chars, redact=self._redact))
            self.inline[agent_id].feed(Record(dict(payload, isSidechain=False), record.offset, record.length, record.line_no))
            return BuildDelta()
        kind = payload.get("type")
        if kind == "assistant":
            return self._feed_assistant(record)
        if kind == "user":
            return self._feed_user(record)
        if kind == "system":
            return self._feed_system(record)
        if kind in _IGNORED_TYPES:
            self._counters.ignored += 1
        else:
            self._counters.unknown += 1
        return BuildDelta()

    def flush(self) -> tuple[dict[str, Any], ...]:
        """Close the open turn and mark every unanswered call as missing."""
        closed = self._close_open()
        missing = {"status": "missing"}
        patched: list[int] = []
        for tool_use_id, (turn_index, call_index) in list(self._pending.items()):
            if turn_index is None:
                continue
            self._turns[turn_index] = _with_result(self._turns[turn_index], call_index, missing)
            patched.append(turn_index)
            del self._pending[tool_use_id]
        for builder in self.inline.values():
            builder.flush()
        return tuple(closed) + tuple(self._turns[i] for i in sorted(set(patched)) if i >= len(self._turns) - len(closed))

    # -- assistant -------------------------------------------------------

    def _feed_assistant(self, record: Record) -> BuildDelta:
        payload = record.payload
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        request_id = payload.get("requestId") or message.get("id")
        closed: tuple[dict[str, Any], ...] = ()
        joins = (
            self._open is not None
            and self._open.kind == "assistant"
            and (request_id is None or self._open.request_id is None or self._open.request_id == request_id)
        )
        if not joins:
            closed = self._close_open()
            self._open = _Open(kind="assistant", at=payload.get("timestamp"), request_id=request_id)
        turn = self._open
        assert turn is not None
        turn.n_lines += 1
        turn.at_end = payload.get("timestamp") or turn.at_end
        if request_id and turn.request_id is None:
            turn.request_id = request_id
        uuid = payload.get("uuid")
        if isinstance(uuid, str):
            turn.uuids.append(uuid)
            self._uuid_turn[uuid] = None
        model = message.get("model")
        if isinstance(model, str) and model:
            turn.model = model
            self._models.add(model)
        turn.usage = _merge_usage(turn.usage, message.get("usage"))
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                turn.texts.append(block["text"])
            elif kind == "thinking" and isinstance(block.get("thinking"), str) and block["thinking"]:
                turn.thoughts.append(block["thinking"])
            elif kind == "tool_use":
                self._add_call(turn, block)
        return BuildDelta(closed=closed, open_changed=True)

    def _add_call(self, turn: _Open, block: Mapping[str, Any]) -> None:
        tool_use_id = str(block.get("id") or f"call-{len(turn.calls)}")
        name = str(block.get("name") or "?")
        raw_input = block.get("input") if isinstance(block.get("input"), dict) else {}
        automil = None
        if name == "Bash" and isinstance(raw_input.get("command"), str):
            calls = detect_automil(raw_input["command"])
            if calls:
                automil = summarize_calls(calls)
        turn.calls.append({
            "tool_use_id": tool_use_id,
            "name": name,
            "input": raw_input,
            "automil": automil,
            "result": None,
        })
        self._pending[tool_use_id] = (None, len(turn.calls) - 1)

    # -- user ------------------------------------------------------------

    def _feed_user(self, record: Record) -> BuildDelta:
        payload = record.payload
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if results:
                patches: list[tuple[int, int, dict[str, Any]]] = []
                open_changed = False
                for block in results:
                    outcome = self._attach(record, block)
                    if outcome is None:
                        continue
                    if outcome[0] is None:
                        open_changed = True
                    else:
                        patches.append(outcome)  # type: ignore[arg-type]
                return BuildDelta(patches=tuple(patches), open_changed=open_changed)
            text = "\n".join(b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
            images = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image")
            return self._simple_turn(record, "human", text, images=images)
        if not isinstance(content, str):
            self._counters.ignored += 1
            return BuildDelta()
        origin = payload.get("origin") if isinstance(payload.get("origin"), dict) else {}
        stripped = content.lstrip()
        if payload.get("isCompactSummary"):
            return self._simple_turn(record, "system", content, system={"subtype": "compact_summary"})
        if payload.get("isMeta") or stripped.startswith(("<local-command", "<command-name>", "<command-message>")):
            return self._simple_turn(record, "system", content[:META_MAX_CHARS], system={"subtype": "meta"})
        if origin.get("kind") == "task-notification" or stripped.startswith("<task-notification>"):
            return self._simple_turn(
                record, "notification", _notification_body(content), notification=_parse_notification(content)
            )
        return self._simple_turn(record, "human", content)

    def _simple_turn(
        self,
        record: Record,
        kind: str,
        text: str,
        *,
        images: int = 0,
        notification: dict[str, Any] | None = None,
        system: dict[str, Any] | None = None,
    ) -> BuildDelta:
        closed = self._close_open()
        turn = _Open(kind=kind, at=record.payload.get("timestamp"), notification=notification, system=system)
        turn.n_lines = 1
        turn.images = images
        uuid = record.payload.get("uuid")
        if isinstance(uuid, str):
            turn.uuids.append(uuid)
        turn.texts.append(text)
        self._turns.append(self._finish(turn, len(self._turns)))
        return BuildDelta(closed=closed + (self._turns[-1],))

    def _attach(self, record: Record, block: Mapping[str, Any]) -> tuple[int | None, int, dict[str, Any]] | None:
        tool_use_id = block.get("tool_use_id")
        location = self._pending.pop(tool_use_id, None) if isinstance(tool_use_id, str) else None
        if location is None:
            source = record.payload.get("sourceToolAssistantUUID")
            location = self._first_unanswered(source) if isinstance(source, str) else None
        if location is None:
            self._counters.orphans += 1
            return None
        turn_index, call_index = location
        text, images = _result_text(block, record.payload)
        text = self._redact(text)
        preview, truncated = _preview(text, self._preview_chars)
        structured = record.payload.get("toolUseResult")
        structured = structured if isinstance(structured, dict) else {}
        status = "error" if block.get("is_error") else "interrupted" if structured.get("interrupted") else "ok"
        call_id = self._call_at(turn_index, call_index)["tool_use_id"]
        if truncated:
            self.full_refs[call_id] = FullResultRef(call_id, record.offset, record.length)
        result = {
            "status": status,
            "preview": preview,
            "bytes": len(text.encode("utf-8")),
            "truncated": truncated,
            "full_ref": f"results/{call_id}.json" if truncated else None,
            "images": images,
            "at": record.payload.get("timestamp"),
            "agent_id": structured.get("agentId") if isinstance(structured.get("agentId"), str) else None,
        }
        self._set_result(turn_index, call_index, result, text)
        return (turn_index, call_index, result)

    def _first_unanswered(self, source_uuid: str) -> tuple[int | None, int] | None:
        if source_uuid not in self._uuid_turn:
            return None
        turn_index = self._uuid_turn[source_uuid]
        calls = self._open.calls if turn_index is None and self._open else self._turns[turn_index]["tool_calls"] if turn_index is not None else []
        for call_index, call in enumerate(calls):
            if call["result"] is None:
                self._pending.pop(call["tool_use_id"], None)
                return (turn_index, call_index)
        return None

    def _call_at(self, turn_index: int | None, call_index: int) -> dict[str, Any]:
        if turn_index is None:
            assert self._open is not None
            return self._open.calls[call_index]
        return self._turns[turn_index]["tool_calls"][call_index]

    def _set_result(self, turn_index: int | None, call_index: int, result: dict[str, Any], text: str) -> None:
        call = self._call_at(turn_index, call_index)
        automil = call["automil"]
        if automil is not None and result["status"] != "interrupted":
            automil = summarize_calls(detect_automil(call["input"].get("command", "")), text)
        updated = dict(call, result=result, automil=automil)
        if turn_index is None:
            assert self._open is not None
            self._open.calls[call_index] = updated
        else:
            turn = dict(self._turns[turn_index])
            calls = list(turn["tool_calls"])
            calls[call_index] = updated
            turn["tool_calls"] = calls
            turn["node_ids"] = _node_ids(calls)
            self._turns[turn_index] = turn

    # -- system ----------------------------------------------------------

    def _feed_system(self, record: Record) -> BuildDelta:
        payload = record.payload
        if payload.get("subtype") != "compact_boundary":
            self._counters.ignored += 1
            return BuildDelta()
        meta = payload.get("compactMetadata") if isinstance(payload.get("compactMetadata"), dict) else {}
        system = {
            "subtype": "compact_boundary",
            "pre_tokens": meta.get("preTokens"),
            "post_tokens": meta.get("postTokens"),
        }
        return self._simple_turn(record, "system", "", system=system)

    # -- closing ---------------------------------------------------------

    def _close_open(self) -> tuple[dict[str, Any], ...]:
        if self._open is None:
            return ()
        index = len(self._turns)
        turn = self._finish(self._open, index)
        self._turns.append(turn)
        for uuid in self._open.uuids:
            self._uuid_turn[uuid] = index
        for call_index, call in enumerate(self._open.calls):
            if call["result"] is None:
                self._pending[call["tool_use_id"]] = (index, call_index)
        self._open = None
        return (turn,)

    @staticmethod
    def _finish(turn: _Open, index: int) -> dict[str, Any]:
        thinking = "\n\n".join(turn.thoughts)
        if len(thinking) > THINKING_MAX_CHARS:
            thinking = thinking[:THINKING_MAX_CHARS] + "\n…[thinking truncated]…"
        return {
            "index": index,
            "kind": turn.kind,
            "uuid": turn.uuids[0] if turn.uuids else None,
            "request_id": turn.request_id,
            "at": turn.at,
            "at_end": turn.at_end,
            "model": turn.model,
            "text": "\n\n".join(turn.texts),
            "thinking": thinking or None,
            "tool_calls": list(turn.calls),
            "usage": turn.usage,
            "images": turn.images,
            "node_ids": _node_ids(turn.calls),
            "notification": turn.notification,
            "system": turn.system,
            "n_lines": turn.n_lines,
        }

    def _note_session(self, payload: Mapping[str, Any]) -> None:
        if self.session_id is None:
            sid = payload.get("sessionId") or payload.get("session_id")
            if isinstance(sid, str):
                self.session_id = sid
        if self.cwd is None and isinstance(payload.get("cwd"), str):
            self.cwd = payload["cwd"]
        if self.git_branch is None and isinstance(payload.get("gitBranch"), str):
            self.git_branch = payload["gitBranch"]


def _with_result(turn: dict[str, Any], call_index: int, result: dict[str, Any]) -> dict[str, Any]:
    calls = list(turn["tool_calls"])
    calls[call_index] = dict(calls[call_index], result=result)
    return dict(turn, tool_calls=calls)


def _node_ids(calls: Iterable[Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for call in calls:
        automil = call.get("automil")
        if automil:
            ids.update(automil.get("node_ids") or ())
    return sorted(ids)


def _merge_usage(current: dict[str, int] | None, raw: object) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return current
    merged = dict(current or {})
    for name, source in _USAGE_FIELDS:
        value = raw.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            merged[name] = max(int(value), merged.get(name, 0))
    return merged or current


@dataclass(frozen=True)
class ParsedTranscript:
    session_id: str | None
    cwd: str | None
    git_branch: str | None
    models: tuple[str, ...]
    turns: tuple[dict[str, Any], ...]
    open_turn: dict[str, Any] | None
    full_refs: Mapping[str, FullResultRef]
    stats: ReadStats
    inline_sidechains: Mapping[str, tuple[dict[str, Any], ...]]


def parse_transcript(
    path: Path, *, ended: bool = True, sidechain_is_main: bool = False
) -> ParsedTranscript:
    """Parse a whole file. ``ended=False`` keeps the trailing turn open.

    A subagent's own file is all sidechain lines; ``sidechain_is_main`` reads
    it as the main stream.
    """
    builder = TurnBuilder(sidechain_is_main=sidechain_is_main)
    batch = read_records(path)
    for record in batch.records:
        builder.feed(record)
    if ended:
        builder.flush()
    inline = {agent_id: b.turns for agent_id, b in builder.inline.items()}
    return ParsedTranscript(
        session_id=builder.session_id,
        cwd=builder.cwd,
        git_branch=builder.git_branch,
        models=builder.models,
        turns=builder.turns,
        open_turn=None if ended else builder.open_turn,
        full_refs=dict(builder.full_refs),
        stats=builder.stats(garbled=batch.garbled, partial_tail=batch.partial_tail, resume_offset=batch.resume_offset),
        inline_sidechains=inline,
    )


def read_full_result(
    path: Path,
    ref: FullResultRef,
    *,
    max_bytes: int = RESULT_FULL_MAX_BYTES,
    redact: Callable[[str], str] = redact_secrets,
) -> dict[str, Any]:
    """Re-read one result line by offset; the text is redacted and capped."""
    with open(path, "rb") as fh:
        fh.seek(ref.offset)
        raw = fh.read(ref.length)
    payload = _decode(raw.rstrip(b"\n"))
    if payload is None:
        return {"tool_use_id": ref.tool_use_id, "text": "", "bytes": 0, "truncated": False, "images": 0, "error": "line unreadable"}
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), list) else []
    block = next((b for b in content if isinstance(b, dict) and b.get("tool_use_id") == ref.tool_use_id), None)
    if block is None:
        return {"tool_use_id": ref.tool_use_id, "text": "", "bytes": 0, "truncated": False, "images": 0, "error": "result not on line"}
    text, images = _result_text(block, payload)
    text = redact(text)
    encoded = text.encode("utf-8")
    truncated = len(encoded) > max_bytes
    if truncated:
        text = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return {
        "tool_use_id": ref.tool_use_id,
        "text": text,
        "bytes": len(encoded),
        "truncated": truncated,
        "images": images,
    }
