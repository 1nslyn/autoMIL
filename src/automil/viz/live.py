"""Follow open sessions while they run and push their new turns over SSE.

The runtime appends to its own transcript under ``~/.claude/projects``,
outside the project, so this polls the file's size every second (a watchdog
observer there would wake on every runtime write) and feeds the new complete
lines to a :class:`TurnBuilder` kept per session. Frames go into the same
subscriber queues as ``graph_update``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from automil.viz.record import RunSource
from automil.viz.record_sessions import TURN_CHUNK_SIZE, SessionSource, build_links
from automil.viz.transcript import TurnBuilder, read_records

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0
RESCAN_INTERVAL_S = 5.0
INVALIDATE_GAP_TURNS = 50


@dataclass
class _Tail:
    source: SessionSource
    builder: TurnBuilder = field(default_factory=TurnBuilder)
    offset: int = 0
    line_no: int = 1
    emitted: int = 0


class LiveRecord:
    """Per-server tailer for the run's open sessions."""

    def __init__(
        self,
        source: RunSource,
        subscribers: list[asyncio.Queue],
        *,
        interval_s: float = POLL_INTERVAL_S,
        rescan_s: float = RESCAN_INTERVAL_S,
    ) -> None:
        self._source = source
        self._subscribers = subscribers
        self._interval = interval_s
        self._rescan = rescan_s
        self._tails: dict[str, _Tail] = {}
        self._journal_sig: tuple[int, int] | None = None
        self._sessions_sig: tuple[str, ...] | None = None
        self._since_rescan = 0.0

    # -- loop --------------------------------------------------------------

    async def run(self) -> None:
        while True:
            try:
                for event in self.poll_once():
                    self.broadcast(event)
            except Exception:  # noqa: BLE001 - the loop must survive a bad file
                logger.exception("live: poll failed")
            await asyncio.sleep(self._interval)

    def broadcast(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event)
        dead = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(queue)
        for queue in dead:
            self._subscribers.remove(queue)

    # -- one pass ----------------------------------------------------------

    def _journal_signature(self) -> tuple[int, int] | None:
        try:
            stat = os.stat(self._source.automil_dir / ".activity.jsonl")
        except OSError:
            return None
        return (stat.st_size, stat.st_mtime_ns)

    def poll_once(self, *, elapsed_s: float | None = None) -> list[dict[str, Any]]:
        """Read what changed since the last pass and return the frames to send."""
        events: list[dict[str, Any]] = []
        self._since_rescan += self._interval if elapsed_s is None else elapsed_s
        journal_sig = self._journal_signature()
        if journal_sig != self._journal_sig or self._sessions_sig is None or self._since_rescan >= self._rescan:
            self._journal_sig = journal_sig
            self._since_rescan = 0.0
            sessions_event = self._rediscover()
            if sessions_event is not None:
                events.append(sessions_event)
        for tail in list(self._tails.values()):
            event = self._advance(tail)
            if event is not None:
                events.append(event)
        return events

    def _rediscover(self) -> dict[str, Any] | None:
        sources, _ = self._source.sessions()
        live = {s.session_id: s for s in sources if s.live and s.transcript is not None}
        for session_id in list(self._tails):
            if session_id not in live:
                del self._tails[session_id]
        for session_id, source in live.items():
            tail = self._tails.get(session_id)
            if tail is None or tail.source.transcript != source.transcript:
                self._tails[session_id] = _Tail(source=source)
        signature = tuple(sorted(f"{s.session_id}:{s.origin}:{s.live}" for s in sources))
        if signature == self._sessions_sig:
            return None
        self._sessions_sig = signature
        return {"type": "sessions_update", "run_id": self._source.run_id, "sessions": self._source.build_sessions()}

    def _advance(self, tail: _Tail) -> dict[str, Any] | None:
        path: Path = tail.source.transcript  # type: ignore[assignment]
        try:
            size = os.stat(path).st_size
        except OSError:
            return None
        if size < tail.offset:
            self._tails[tail.source.session_id] = _Tail(source=tail.source)
            return {
                "type": "transcript_invalidate",
                "run_id": self._source.run_id,
                "session_id": tail.source.session_id,
                "reason": "file shrank",
            }
        if size == tail.offset:
            return None
        batch = read_records(path, start_offset=tail.offset, line_no=tail.line_no)
        tail.offset = batch.resume_offset
        tail.line_no = batch.next_line_no
        closed: list[dict[str, Any]] = []
        patches: list[dict[str, Any]] = []
        open_changed = False
        for record in batch.records:
            delta = tail.builder.feed(record)
            closed.extend(delta.closed)
            patches.extend({"turn": t, "call": c, "result": r} for t, c, r in delta.patches)
            open_changed = open_changed or delta.open_changed
        if not closed and not patches and not open_changed:
            return None
        from_turn = tail.emitted
        n_turns = len(tail.builder.turns)
        tail.emitted = n_turns
        if n_turns - from_turn > INVALIDATE_GAP_TURNS:
            return {
                "type": "transcript_invalidate",
                "run_id": self._source.run_id,
                "session_id": tail.source.session_id,
                "reason": f"{n_turns - from_turn} turns arrived at once",
                "n_turns": n_turns,
            }
        by_node, _ = build_links(tail.source.session_id, closed)
        return {
            "type": "transcript_delta",
            "run_id": self._source.run_id,
            "session_id": tail.source.session_id,
            "from_turn": from_turn,
            "turns": closed,
            "patches": patches,
            "open_turn": tail.builder.open_turn,
            "n_turns": n_turns,
            "n_chunks": (n_turns + TURN_CHUNK_SIZE - 1) // TURN_CHUNK_SIZE,
            "links": by_node,
        }
