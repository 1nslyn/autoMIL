"""viz.record_sessions: session discovery, chunks, links and events."""
from __future__ import annotations

from pathlib import Path

from automil.session_record import store_session_record
from automil.viz.clock import host_clock
from automil.viz.record_files import RunPaths
from automil.viz.record_sessions import (
    agent_events,
    build_links,
    chunk_turns,
    discover_sessions,
    session_summary,
)
from automil.viz.subagents import read_subagents
from automil.viz.transcript import parse_transcript

from tests.viz.conftest import SID, stamp, write_mini_session, write_project

CLOCK = host_clock(tz_name="UTC")
LIVE = "2222bbbb-0000-4000-8000-000000000000"


def test_discover_joins_journal_stored_and_live_home_files(tmp_path: Path):
    automil = write_project(tmp_path / "proj")
    home = tmp_path / "claude"
    slug = home / "projects" / "-data-project"
    source = write_mini_session(slug)
    store_session_record(automil, SID, source)
    # a second, still-open session that only exists in the runtime's home
    live_source = slug / f"{LIVE}.jsonl"
    live_source.write_bytes(source.read_bytes())
    with (automil / ".activity.jsonl").open("a") as fh:
        fh.write('{"cell_id":null,"event":"session_open","observed_at":1788740600.0,"session_id":"%s"}\n' % LIVE)
    # a stored record with no journal entry (a resumed session)
    orphan = "3333cccc-0000-4000-8000-000000000000"
    orphan_source = slug / f"{orphan}.jsonl"
    orphan_source.write_bytes(source.read_bytes())
    store_session_record(automil, orphan, orphan_source)

    sources, error = discover_sessions(RunPaths(automil), config_dir=home)
    assert error is None
    by_id = {s.session_id: s for s in sources}
    assert [s.session_id for s in sources] == [SID, LIVE, orphan]
    # SID is journaled as open (the fixture journal has no end) so it is live and read from home
    assert by_id[SID].origin == "home" and by_id[SID].live is True
    assert by_id[LIVE].origin == "home" and by_id[LIVE].transcript == live_source
    assert by_id[orphan].origin == "stored" and by_id[orphan].live is False and by_id[orphan].journal is None


def test_discover_reports_a_corrupt_journal_and_missing_sources(tmp_path: Path):
    automil = write_project(tmp_path / "proj")
    (automil / ".activity.jsonl").write_text("not json\n")
    sources, error = discover_sessions(RunPaths(automil), config_dir=tmp_path / "nohome")
    assert sources == () and "corrupt" in error
    (automil / ".activity.jsonl").write_text(
        '{"cell_id":null,"event":"session_open","observed_at":1.0,"session_id":"%s"}\n' % SID
    )
    sources, error = discover_sessions(RunPaths(automil), config_dir=tmp_path / "nohome")
    assert error is None and sources[0].origin == "missing" and sources[0].transcript is None


def test_chunks_and_session_summary(tmp_path: Path):
    path = write_mini_session(tmp_path)
    parsed = parse_transcript(path)
    subagents = read_subagents(tmp_path / SID)
    chunks = chunk_turns(parsed.turns, size=4)
    assert sum(len(c) for c in chunks) == len(parsed.turns) and len(chunks[0]) == 4
    assert chunk_turns([], size=4) == []
    sources, _ = discover_sessions(RunPaths(write_project(tmp_path / "proj")), config_dir=tmp_path / "nohome")
    summary = session_summary(sources[0], parsed, subagents, CLOCK, active_seconds=12.5)
    assert summary["session_id"] == SID and summary["source"] == "missing"
    assert summary["n_turns"] == len(parsed.turns) and summary["n_chunks"] == 1
    assert summary["n_tool_calls"] == 8 and summary["n_automil_commands"] == 2 and summary["n_prompts"] == 1
    assert summary["usage"]["output"] == 50 * 9  # nine assistant requests
    assert summary["models"] == ["claude-opus-5"] and summary["active_seconds"] == 12.5
    assert summary["subagents"][0]["agent_id"] == "agent1" and summary["subagents"][0]["n_turns"] == 2
    assert summary["opened_at"] == "2026-09-07T00:21:53.917Z"
    assert summary["first_at"] == stamp(0)


def test_links_cover_created_argv_window_and_mentions(tmp_path: Path):
    parsed = parse_transcript(write_mini_session(tmp_path))
    # node_0009 was submitted in a loop: not in any argv, but inside the submit call's window
    by_node, by_turn = build_links(SID, parsed.turns, submitted_at={"node_0009": stamp(11), "node_0050": stamp(300)})
    assert by_node["node_0002"] == [
        {"session_id": SID, "turn": 1, "kind": "propose", "at": stamp(5)},
        {"session_id": SID, "turn": 2, "kind": "submit", "at": stamp(10)},
    ]
    assert by_node["node_0001"] == [{"session_id": SID, "turn": 1, "kind": "mention", "at": stamp(5)}]
    assert by_node["node_0009"] == [{"session_id": SID, "turn": 2, "kind": "submit", "at": stamp(10)}]
    assert "node_0050" not in by_node
    assert by_turn[f"{SID}:1"] == ["node_0001", "node_0002"] and by_turn[f"{SID}:2"] == ["node_0002", "node_0009"]


def test_agent_events_name_creations_prompts_notifications_and_compactions(tmp_path: Path):
    parsed = parse_transcript(write_mini_session(tmp_path))
    events = agent_events(SID, parsed.turns)
    kinds = [(e["kind"], e["node_id"]) for e in events]
    assert kinds == [
        ("prompt", None), ("propose", "node_0002"), ("submit", "node_0002"),
        ("notification", None), ("compact", None),
    ]
    assert events[0]["label"].startswith("Session is bound")
    assert events[3]["label"] == 'Agent "Research MIL recipes" finished'
