"""viz.live: open sessions stream their turns as the runtime writes them."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from automil.viz.clock import host_clock
from automil.viz.live import LiveRecord
from automil.viz.record import RunSource

from tests.viz.conftest import (
    SID,
    assistant,
    bash_result,
    mini_session_lines,
    stamp,
    text,
    tool_result,
    tool_use,
    write_lines,
    write_project,
)


def _setup(tmp_path: Path):
    automil = write_project(tmp_path / "proj")  # the fixture journal has SID open
    home = tmp_path / "claude"
    path = write_lines(home / "projects" / "-data-project" / f"{SID}.jsonl", mini_session_lines())
    source = RunSource(automil, host_clock(tz_name="UTC"), run_id="demo", config_dir=home)
    return automil, path, source


def test_first_poll_announces_sessions_and_all_closed_turns(tmp_path: Path):
    _, path, source = _setup(tmp_path)
    live = LiveRecord(source, [])
    events = live.poll_once()
    kinds = [e["type"] for e in events]
    assert kinds == ["sessions_update", "transcript_delta"]
    delta = events[1]
    assert delta["session_id"] == SID and delta["from_turn"] == 0
    assert delta["turns"][0]["kind"] == "human"
    assert delta["open_turn"]["text"] == "Done for now."  # the last assistant turn stays open
    assert delta["n_turns"] == len(delta["turns"]) and delta["n_chunks"] == 1
    assert "node_0002" in delta["links"]
    assert live.poll_once() == []  # nothing new


def test_appended_lines_arrive_as_a_delta_and_late_results_as_patches(tmp_path: Path):
    _, path, source = _setup(tmp_path)
    live = LiveRecord(source, [])
    first = live.poll_once()[1]
    n = first["n_turns"]
    with path.open("ab") as fh:
        fh.write(json.dumps(assistant("b1", "a10", stamp(90), [tool_use("t9", "Bash", command="uv run automil rank")], request_id="req10")).encode() + b"\n")
    (delta,) = live.poll_once()
    assert delta["type"] == "transcript_delta" and delta["from_turn"] == n
    assert [t["text"] for t in delta["turns"]] == ["Done for now."]  # closed by the new request
    assert delta["open_turn"]["tool_calls"][0]["result"] is None
    with path.open("ab") as fh:
        fh.write(json.dumps(assistant("b2", "b1", stamp(91), [text("ranked")], request_id="req11")).encode() + b"\n")
        fh.write(json.dumps(tool_result("rr", "b1", stamp(92), "t9", "1. [node_0003]", tool_use_result=bash_result("1. [node_0003]"))).encode() + b"\n")
    (delta,) = live.poll_once()
    assert delta["from_turn"] == n + 1 and len(delta["turns"]) == 1
    assert delta["patches"] == [{"turn": n + 1, "call": 0, "result": delta["patches"][0]["result"]}]
    assert delta["patches"][0]["result"]["preview"] == "1. [node_0003]"


def test_partial_lines_wait_and_a_shrunken_file_invalidates(tmp_path: Path):
    _, path, source = _setup(tmp_path)
    live = LiveRecord(source, [])
    live.poll_once()
    with path.open("ab") as fh:
        fh.write(b'{"type": "assistant", "requestId": "r-partial"')
    assert live.poll_once() == []
    with path.open("ab") as fh:
        fh.write(b', "message": {"content": [{"type": "text", "text": "late"}]}}\n')
    (delta,) = live.poll_once()
    assert delta["open_turn"]["text"] == "late"
    path.write_bytes(b"")
    (event,) = live.poll_once()
    assert event["type"] == "transcript_invalidate" and event["reason"] == "file shrank"


def test_broadcast_drops_a_full_queue_and_session_end_stops_the_tail(tmp_path: Path):
    automil, path, source = _setup(tmp_path)
    full: asyncio.Queue = asyncio.Queue(maxsize=1)
    full.put_nowait("stale")
    open_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    live = LiveRecord(source, [full, open_queue])
    for event in live.poll_once():
        live.broadcast(event)
    assert full not in live._subscribers and open_queue.qsize() == 2
    (automil / ".activity.samples.json").write_text(
        '{"schema_version":1,"sessions":{"%s":{"active_seconds":1.0,"observed_at":1788766022.0}}}' % SID
    )
    with (automil / ".activity.jsonl").open("a") as fh:
        fh.write('{"cell_id":null,"event":"session_end","final_sample_observed_at":1788766022.0,"observed_at":1788766112.0,"session_id":"%s"}\n' % SID)
    events = live.poll_once()
    assert [e["type"] for e in events] == ["sessions_update"]
    assert events[0]["sessions"]["sessions"][0]["live"] is False
    assert live._tails == {}
