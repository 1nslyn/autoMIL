"""viz.subagents: sidecar subagent transcripts attach to their parent call."""
from __future__ import annotations

import json
from pathlib import Path

from automil.viz.subagents import attach_subagents, read_subagents
from automil.viz.transcript import parse_transcript

from tests.viz.conftest import SID, assistant, stamp, text, tool_use, write_lines, write_mini_session


def test_reads_the_sidecar_and_links_the_agent_call(tmp_path: Path):
    path = write_mini_session(tmp_path)
    subagents = read_subagents(tmp_path / SID)
    assert [s.agent_id for s in subagents] == ["agent1"]
    (agent,) = subagents
    assert (agent.tool_use_id, agent.agent_type, agent.description, agent.spawn_depth) == (
        "t3", "general-purpose", "Research MIL recipes", 1,
    )
    assert [t["kind"] for t in agent.turns] == ["human", "assistant"]
    assert agent.turns[1]["text"].startswith("# Report")

    parsed = attach_subagents(parse_transcript(path), subagents)
    call = next(c for t in parsed.turns for c in t["tool_calls"] if c["tool_use_id"] == "t3")
    assert call["result"]["agent_id"] == "agent1"


def test_missing_meta_or_missing_transcript_are_tolerated(tmp_path: Path):
    folder = tmp_path / "subagents"
    folder.mkdir()
    write_lines(folder / "agent-orphan.jsonl", [assistant("x", None, stamp(1), [text("hi")], request_id="r")])
    (folder / "agent-empty.meta.json").write_text(json.dumps({"toolUseId": "t9", "agentType": "Explore"}))
    (folder / "notes.txt").write_text("ignored")
    subagents = {s.agent_id: s for s in read_subagents(tmp_path)}
    assert set(subagents) == {"empty", "orphan"}
    assert subagents["orphan"].tool_use_id is None and len(subagents["orphan"].turns) == 1
    assert subagents["empty"].turns == () and subagents["empty"].tool_use_id == "t9"


def test_attach_fills_a_missing_result_and_leaves_other_calls_alone(tmp_path: Path):
    lines = [assistant("a1", None, stamp(1), [tool_use("t9", "Agent", prompt="go"), tool_use("t2", "Bash", command="ls")], request_id="r1")]
    parsed = parse_transcript(write_lines(tmp_path / "t.jsonl", lines))
    folder = tmp_path / "side" / "subagents"
    folder.mkdir(parents=True)
    (folder / "agent-z.meta.json").write_text(json.dumps({"toolUseId": "t9"}))
    attached = attach_subagents(parsed, read_subagents(tmp_path / "side"))
    agent_call, bash_call = attached.turns[0]["tool_calls"]
    assert agent_call["result"] == {"status": "missing", "agent_id": "z"}
    assert bash_call["result"] == {"status": "missing"}
    assert read_subagents(None) == () and read_subagents(tmp_path / "nowhere") == ()
