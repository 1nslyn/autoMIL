"""viz.transcript: Claude Code JSONL lines become turns."""
from __future__ import annotations

import json
from pathlib import Path

from automil.viz.transcript import (
    RESULT_PREVIEW_CHARS,
    TurnBuilder,
    parse_transcript,
    read_full_result,
    read_records,
)

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
    write_mini_session,
)


# --- reader -------------------------------------------------------------


def test_reader_yields_offsets_and_counts_bad_lines(tmp_path: Path):
    path = write_lines(tmp_path / "t.jsonl", [{"type": "mode"}, b"\xff\xfe", b"nope", b"[1, 2]", {"type": "user"}])
    batch = read_records(path)
    assert [r.payload["type"] for r in batch.records] == ["mode", "user"]
    assert batch.garbled == 3
    assert batch.partial_tail is False
    first, second = batch.records
    assert first.offset == 0 and first.length == len(json.dumps({"type": "mode"})) + 1
    assert second.line_no == 5
    assert batch.resume_offset == path.stat().st_size


def test_reader_leaves_a_partial_last_line_for_later(tmp_path: Path):
    path = write_lines(tmp_path / "t.jsonl", [{"type": "mode"}], partial_tail='{"type": "us')
    batch = read_records(path)
    assert [r.payload["type"] for r in batch.records] == ["mode"]
    assert batch.partial_tail is True
    assert batch.resume_offset == len(json.dumps({"type": "mode"})) + 1

    with path.open("ab") as fh:
        fh.write(b'er"}\n')
    resumed = read_records(path, start_offset=batch.resume_offset, line_no=batch.next_line_no)
    assert [r.payload["type"] for r in resumed.records] == ["user"]
    assert resumed.records[0].line_no == 2
    assert resumed.partial_tail is False


def test_reader_handles_an_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    batch = read_records(path)
    assert batch.records == () and batch.resume_offset == 0 and batch.partial_tail is False


# --- turns --------------------------------------------------------------


def _parsed(tmp_path: Path):
    path = write_mini_session(tmp_path)
    return path, parse_transcript(path)


def test_turns_group_one_request_and_count_usage_once(tmp_path: Path):
    _, parsed = _parsed(tmp_path)
    kinds = [t["kind"] for t in parsed.turns]
    assert kinds[:3] == ["human", "assistant", "assistant"]
    first_response = parsed.turns[1]
    assert first_response["request_id"] == "req1"
    assert first_response["text"] == "I will read the state."
    assert first_response["thinking"] == "think first"
    assert first_response["at"] == stamp(5)
    assert first_response["usage"] == {"input": 2, "output": 50, "cache_read": 1000, "cache_create": 0}
    assert [c["name"] for c in first_response["tool_calls"]] == ["Bash"]
    assert first_response["index"] == 1
    assert parsed.session_id == SID and parsed.cwd == "/data/project"
    assert parsed.models == ("claude-opus-5",)


def test_results_attach_with_status_and_automil_commands_are_detected(tmp_path: Path):
    _, parsed = _parsed(tmp_path)
    propose = parsed.turns[1]["tool_calls"][0]
    assert propose["result"]["status"] == "ok"
    assert propose["result"]["preview"] == "Added proposal node_0002 [hp]: lr down"
    assert propose["result"]["at"] == stamp(8)
    assert propose["automil"] == {
        "calls": [{
            "sub": "propose",
            "argv": ["propose", "--parent", "node_0001", "--desc", "lr down", "--kind", "hp"],
            "node_ids": ["node_0001"],
        }],
        "node_ids": ["node_0001", "node_0002"],
        "created": [{"node_id": "node_0002", "sub": "propose"}],
    }
    submit = parsed.turns[2]["tool_calls"][0]
    assert submit["automil"]["calls"][0]["sub"] == "submit"
    assert submit["automil"]["created"] == [{"node_id": "node_0002", "sub": "submit"}]
    assert parsed.turns[2]["node_ids"] == ["node_0002"]
    # a plain Read is not an automil command
    read_call = next(c for t in parsed.turns for c in t["tool_calls"] if c["name"] == "Read")
    assert read_call["automil"] is None
    assert read_call["result"]["preview"] == "line 1\nline 2"


def test_error_image_and_interrupted_results(tmp_path: Path):
    _, parsed = _parsed(tmp_path)
    calls = {c["tool_use_id"]: c for t in parsed.turns for c in t["tool_calls"]}
    assert calls["t5"]["result"]["status"] == "error"
    assert calls["t5"]["result"]["preview"] == "Exit code 1"
    assert calls["t6"]["result"]["images"] == 1
    assert calls["t6"]["result"]["preview"] == ""
    # the interrupted sleep never got a result; the session ended, so it is missing
    assert calls["t8"]["result"] == {"status": "missing"}
    # the Agent call knows its subagent id from toolUseResult
    assert calls["t3"]["result"]["agent_id"] == "agent1"


def test_large_results_get_a_preview_a_full_ref_and_redaction(tmp_path: Path):
    path, parsed = _parsed(tmp_path)
    big = next(c for t in parsed.turns for c in t["tool_calls"] if c["tool_use_id"] == "t7")
    result = big["result"]
    assert result["truncated"] is True
    assert len(result["preview"]) <= RESULT_PREVIEW_CHARS + 80
    assert result["preview"].startswith("HF_TOKEN=[REDACTED]")
    assert result["preview"].endswith("TAIL-MARKER")
    assert "chars omitted" in result["preview"]
    assert result["bytes"] > 5000
    ref = parsed.full_refs["t7"]
    full = read_full_result(path, ref)
    assert full["text"].startswith("HF_TOKEN=[REDACTED]\n")
    assert full["text"].endswith("TAIL-MARKER")
    assert full["truncated"] is False and full["images"] == 0
    assert "hf_abcdefghijklmnopqrstuvwxyz1234" not in full["text"]
    # small results have no ref
    assert "t1" not in parsed.full_refs


def test_notifications_system_lines_and_junk(tmp_path: Path):
    _, parsed = _parsed(tmp_path)
    note = next(t for t in parsed.turns if t["kind"] == "notification")
    assert note["notification"] == {
        "task_id": "agent1",
        "tool_use_id": "t3",
        "status": "completed",
        "title": 'Agent "Research MIL recipes" finished',
    }
    assert note["text"] == "# Report\nUse a lower lr."
    boundary = next(t for t in parsed.turns if t["kind"] == "system")
    assert boundary["system"] == {"subtype": "compact_boundary", "pre_tokens": 150000, "post_tokens": 20000}
    stats = parsed.stats
    assert stats.lines_garbled == 2
    assert stats.lines_unknown_type == 1
    assert stats.lines_ignored >= 3  # mode, permission-mode, attachment, queue-operation
    assert stats.partial_tail is False
    # queue-operation duplicates never become turns
    assert sum(1 for t in parsed.turns if t["kind"] == "notification") == 1
    assert parsed.turns[-1]["text"] == "Done for now."
    assert [t["index"] for t in parsed.turns] == list(range(len(parsed.turns)))


def test_meta_and_compact_summary_user_lines_are_system_turns(tmp_path: Path):
    lines = [
        dict(assistant("a1", None, stamp(1), [text("hi")], request_id="r1")),
        {"type": "user", "uuid": "m1", "parentUuid": "a1", "timestamp": stamp(2), "isMeta": True,
         "message": {"role": "user", "content": "<local-command-stdout>Bye!</local-command-stdout>"}},
        {"type": "user", "uuid": "c1", "parentUuid": "m1", "timestamp": stamp(3), "isCompactSummary": True,
         "message": {"role": "user", "content": "Summary of the conversation so far"}},
    ]
    parsed = parse_transcript(write_lines(tmp_path / "t.jsonl", lines))
    assert [t["kind"] for t in parsed.turns] == ["assistant", "system", "system"]
    assert parsed.turns[1]["system"]["subtype"] == "meta"
    assert parsed.turns[2]["system"]["subtype"] == "compact_summary"
    assert parsed.turns[2]["text"] == "Summary of the conversation so far"


def test_sidechain_lines_in_the_main_file_stay_out_of_the_main_stream(tmp_path: Path):
    lines = [
        assistant("a1", None, stamp(1), [text("main")], request_id="r1"),
        dict(assistant("sa1", None, stamp(2), [text("side")], request_id="r2"), isSidechain=True, agentId="agentX"),
    ]
    parsed = parse_transcript(write_lines(tmp_path / "t.jsonl", lines))
    assert [t["text"] for t in parsed.turns] == ["main"]
    assert parsed.stats.lines_sidechain == 1
    assert "agentX" in parsed.inline_sidechains
    assert parsed.inline_sidechains["agentX"][0]["text"] == "side"


def test_result_by_source_uuid_when_the_tool_id_is_unknown(tmp_path: Path):
    lines = [
        assistant("a1", None, stamp(1), [tool_use("t1", "Bash", command="ls")], request_id="r1"),
        tool_result("r1", "a1", stamp(2), "t-other", "listing", source_assistant="a1"),
    ]
    parsed = parse_transcript(write_lines(tmp_path / "t.jsonl", lines))
    call = parsed.turns[0]["tool_calls"][0]
    assert call["result"]["preview"] == "listing"


# --- streaming ----------------------------------------------------------


def test_builder_streams_closed_turns_and_patches_late_results(tmp_path: Path):
    builder = TurnBuilder()
    a1 = assistant("a1", None, stamp(1), [tool_use("t1", "Bash", command="ls")], request_id="r1")
    path = write_lines(tmp_path / "t.jsonl", [a1])
    batch = read_records(path)
    delta = builder.feed(batch.records[0])
    assert delta.closed == () and delta.open_changed is True
    assert builder.open_turn is not None and builder.open_turn["tool_calls"][0]["result"] is None

    # a new request closes the first turn even though its result is still pending
    a2 = assistant("a2", "a1", stamp(3), [text("next")], request_id="r2")
    path = write_lines(tmp_path / "t.jsonl", [a1, a2])
    batch = read_records(path, start_offset=batch.resume_offset, line_no=batch.next_line_no)
    delta = builder.feed(batch.records[0])
    assert [t["index"] for t in delta.closed] == [0]
    assert delta.closed[0]["tool_calls"][0]["result"] is None

    # the late result lands on the closed turn as a patch
    r1 = tool_result("r1", "a1", stamp(4), "t1", "files", tool_use_result=bash_result("files"))
    path = write_lines(tmp_path / "t.jsonl", [a1, a2, r1])
    batch = read_records(path, start_offset=batch.resume_offset, line_no=batch.next_line_no)
    delta = builder.feed(batch.records[0])
    assert delta.closed == ()
    assert len(delta.patches) == 1
    turn_index, call_index, result = delta.patches[0]
    assert (turn_index, call_index, result["preview"]) == (0, 0, "files")
    assert builder.turns[0]["tool_calls"][0]["result"]["preview"] == "files"

    closed = builder.flush()
    assert [t["text"] for t in closed] == ["next"]
    assert builder.open_turn is None


def test_parse_of_an_open_session_keeps_the_last_turn_open(tmp_path: Path):
    path = write_lines(tmp_path / "t.jsonl", mini_session_lines(), partial_tail='{"type": "assis')
    parsed = parse_transcript(path, ended=False)
    assert parsed.open_turn is not None and parsed.open_turn["text"] == "Done for now."
    assert parsed.stats.partial_tail is True
    assert parsed.turns[-1]["text"] != "Done for now."
