"""viz.record_timeline: node timing from the log, spec and completion files."""
from __future__ import annotations

from pathlib import Path

from automil.viz.clock import host_clock
from automil.viz.record_files import RunPaths, load_completed, load_orchestrator_log, load_spec
from automil.viz.record_timeline import (
    build_timeline,
    node_timing,
    normalize_slot,
    parse_orchestrator_log,
)

from tests.viz.conftest import write_project

CLOCK = host_clock(tz_name="America/Vancouver")


def test_log_lines_parse_on_the_exact_daemon_formats():
    lines = [
        "2026-09-06 17:36:40,814 [INFO] Launched node_0002 on CUDA GPU 0 (PID 2470654, est. 0.5GB, timeout 600min)",
        "2026-09-06 19:12:58,109 [INFO] Completed node_0002: status=completed, primary_value=0.6157, elapsed=96.2min, CUDA GPU 0",
        "2026-09-06 19:25:08,401 [INFO] Completed node_0015: status=crash, primary_value=0.0, elapsed=0.8min, CUDA GPU 1",
        "2026-09-06 17:36:40,900 [WARNING] Holding new work for cell c40d until telemetry recovers",
        "garbage line",
    ]
    events = parse_orchestrator_log(lines, CLOCK)
    assert [(e.kind, e.node_id, e.slot, e.status) for e in events] == [
        ("launched", "node_0002", "cuda:0", None),
        ("completed", "node_0002", "cuda:0", "completed"),
        ("completed", "node_0015", "cuda:1", "crash"),
    ]
    assert events[0].at == "2026-09-07T00:36:40.814Z"
    assert normalize_slot("cuda:1") == "cuda:1" and normalize_slot("ROCm GPU 2") == "rocm:2" and normalize_slot("") is None


def test_node_timing_prefers_completion_file_and_falls_back_to_the_log(tmp_path: Path):
    paths = RunPaths(write_project(tmp_path))
    events = parse_orchestrator_log(load_orchestrator_log(paths), CLOCK)
    timing = node_timing(
        "node_0002", spec=load_spec(paths, "node_0002"), completed=load_completed(paths, "node_0002"),
        log_events=events, clock=CLOCK,
    )
    assert timing["submitted_at"] == "2026-09-07T00:32:23.214Z"
    assert timing["launched_at"] == "2026-09-07T00:36:40.814Z"
    assert timing["completed_at"] == "2026-09-07T02:12:51.003Z"  # from completed/<node>.json
    assert timing["slot"] == "cuda:0"
    assert timing["queue_wait_s"] == 257.6
    assert round(timing["run_s"]) == 5770

    without_file = node_timing("node_0002", spec=None, completed=None, log_events=events, clock=CLOCK)
    assert without_file["submitted_at"] is None
    assert without_file["completed_at"] == "2026-09-07T02:12:58.109Z"  # from the log
    assert without_file["queue_wait_s"] is None
    assert node_timing("node_0099", spec=None, completed=None, log_events=events, clock=CLOCK)["launched_at"] is None


def test_build_timeline_sorts_events_and_records_the_clock():
    timeline = build_timeline(
        run_id="demo",
        clock=CLOCK,
        nodes=[{"node_id": "node_0002"}],
        sessions=[{"session_id": "s"}],
        events=[
            {"at": "2026-09-07T00:40:00.000Z", "kind": "launched"},
            {"at": "2026-09-07T00:30:00.000Z", "kind": "propose"},
            {"at": None, "kind": "session_open"},
        ],
        warnings=["clock inferred"],
    )
    assert [e["kind"] for e in timeline["events"]] == ["session_open", "propose", "launched"]
    assert timeline["clock"]["tz_name"] == "America/Vancouver"
    assert timeline["warnings"] == ["clock inferred"] and timeline["schema"] == 1
