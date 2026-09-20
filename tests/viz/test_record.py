"""viz.record: the run record end to end on a small project with a stored session."""
from __future__ import annotations

from pathlib import Path

import pytest

from automil.session_record import store_session_record
from automil.viz.clock import host_clock
from automil.viz.record import RunSource, build_index, run_id_from_config

from tests.viz.conftest import SID, stamp, write_mini_session, write_project


@pytest.fixture
def source(tmp_path: Path) -> RunSource:
    automil = write_project(tmp_path / "proj")
    home = tmp_path / "claude"
    transcript = write_mini_session(home / "projects" / "-data-project")
    store_session_record(automil, SID, transcript)
    # the fixture journal leaves SID open; close it so the stored copy is used
    (automil / ".activity.samples.json").write_text(
        '{"schema_version":1,"sessions":{"%s":{"active_seconds":3047.0,"observed_at":1788766022.0}}}' % SID
    )
    with (automil / ".activity.jsonl").open("a") as fh:
        fh.write(
            '{"cell_id":null,"event":"session_end","final_sample_observed_at":1788766022.0,'
            '"observed_at":1788766112.0,"session_id":"%s"}\n' % SID
        )
    return RunSource(automil, host_clock(tz_name="America/Vancouver"), config_dir=tmp_path / "nohome")


def test_run_id_and_index_entry(source: RunSource):
    assert run_id_from_config({"project": {"name": "TCGA LUAD / kras"}}) == "tcga-luad-kras"
    assert run_id_from_config({}) == "project"
    assert source.run_id == "demo_project" and source.title == "demo"
    entry = source.index_entry()
    assert entry["n_nodes"] == 5 and entry["n_executed"] == 4 and entry["best_node_id"] == "node_0003"
    assert entry["n_sessions"] == 1 and entry["live_sessions"] == 0 and entry["certified"] is False
    assert entry["task"] == "kras" and entry["mil_model"] == "abmil" and entry["primary_metric"] == "val_auc"
    assert entry["started_at"] == "2026-09-07T00:00:00.000Z"
    index = build_index([source], "static")
    assert index["mode"] == "static" and index["runs"][0]["run_id"] == "demo_project"


def test_graph_sessions_chunks_and_links(source: RunSource):
    graph = source.build_graph()
    assert graph["run_id"] == "demo_project" and set(graph["nodes"]) == {f"node_000{i}" for i in range(1, 6)}
    sessions = source.build_sessions()
    (entry,) = sessions["sessions"]
    assert entry["source"] == "stored" and entry["live"] is False and entry["ended_by"] == "hook"
    assert entry["active_seconds"] == 3047.0 and entry["n_chunks"] == 1
    assert sessions["cells"] == [] and sessions["journal_error"] is None
    chunk = source.build_chunk(SID, 0)
    assert chunk["n_turns"] == entry["n_turns"] and chunk["turns"][0]["kind"] == "human"
    assert source.build_chunk(SID, 1) is None and source.build_chunk("nope", 0) is None
    links = source.build_links()
    assert links["nodes"]["node_0002"][0] == {"session_id": SID, "turn": 1, "kind": "propose", "at": stamp(5)}
    assert links["turns"][f"{SID}:1"] == ["node_0001", "node_0002"]
    full = source.build_full_result(SID, "t7")
    assert full["session_id"] == SID and full["text"].endswith("TAIL-MARKER")
    assert source.build_full_result(SID, "t1") is None
    agent = source.build_agent(SID, "agent1")
    assert agent["tool_use_id"] == "t3" and len(agent["turns"]) == 2
    assert source.build_agent(SID, "ghost") is None


def test_node_detail_carries_verdict_overlay_log_timing_and_links(source: RunSource):
    detail = source.build_node("node_0003")
    assert detail["node"]["status"] == "keep" and detail["verdict"]["decision"] == "keep"
    assert detail["verdict_unavailable"] is None
    assert detail["overlay"]["files"] == [{"path": "automil/variants/_policies/dropout.py", "sha256": "ab" * 32}]
    assert detail["overlay"]["base_commit"].startswith("1660b0d8")
    assert detail["run_log"]["available"] is True and detail["run_log"]["tail"][-1] == "Experiment complete"
    assert detail["timing"]["launched_at"] == "2026-09-07T00:36:41.000Z"
    assert detail["result"]["metrics"] == {"val_auc": pytest.approx(0.66), "val_bacc": 0.60}
    assert detail["parent"]["id"] == "node_0001"
    assert source.build_overlay_file("node_0003", "automil/variants/_policies/dropout.py")["text"] == "DROPOUT = 0.5\n"

    crash = source.build_node("node_0004")
    assert crash["verdict"] is None and crash["verdict_unavailable"] == "crash"
    assert "0.91" not in crash["node"]["error"]
    proposal = source.build_node("node_0005")
    assert proposal["result"] is None and proposal["run_log"]["available"] is False
    assert source.build_node("node_9999") is None


def test_running_node_has_no_verdict_and_no_log(tmp_path: Path):
    automil = write_project(tmp_path / "proj", running=("node_0002",))
    source = RunSource(automil, host_clock(tz_name="UTC"), config_dir=tmp_path / "nohome")
    detail = source.build_node("node_0002")
    assert detail["node"]["status"] == "running"
    assert detail["verdict"] is None and detail["verdict_unavailable"] == "not terminal"
    assert detail["run_log"]["available"] is False
    assert source.build_graph()["running"] == ["node_0002"]


def test_timeline_and_notes(source: RunSource):
    timeline = source.build_timeline()
    by_id = {n["node_id"]: n for n in timeline["nodes"]}
    assert by_id["node_0002"]["launched_at"] == "2026-09-07T00:36:40.814Z"
    assert by_id["node_0002"]["completed_at"] == "2026-09-07T02:12:51.003Z"
    kinds = [e["kind"] for e in timeline["events"]]
    assert kinds[0] == "session_open" and "propose" in kinds and "launched" in kinds and "session_end" in kinds
    assert timeline["sessions"][0]["session_id"] == SID
    # the fixture transcript is in UTC while node stamps are Pacific: the cross-check warns
    assert timeline["warnings"] and "host offset" in timeline["warnings"][0]
    notes = source.build_notes()
    assert notes["plan_md"].startswith("# Plan") and notes["learnings_md"].startswith("# Learnings")
    assert source.build_certified() is None
