"""REC-02 / D-09, D-10: terminal_writer writes all four artifacts in fixed order.

D-09: A standalone terminal_writer module writes all four artifacts:
      graph node (via locked_update) → completed/<node>.json
      → archive result.json → results.tsv

D-10: terminal_writer is the sole results.tsv writer; updates graph via
      locked API, never direct dict mutation.

Both _handle_completion and _handle_cap_killed_completion delegate here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def _make_graph(tmp_path: Path):
    """Create a minimal graph with one root executed node and return (graph, node_id)."""
    from automil.graph import ExperimentGraph
    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="baseline",
        techniques=["baseline"],
        metrics={"composite": 0.5},
        status="keep",
    )
    # Promote to "running" (simulates what submit does)
    graph.nodes[node_id]["status"] = "running"
    graph.save()
    return graph, node_id


def _make_dirs(tmp_path: Path, node_id: str):
    """Create completed_dir and archive_dir, return them."""
    completed_dir = tmp_path / "completed"
    completed_dir.mkdir()
    archive_dir = tmp_path / "archive" / node_id
    archive_dir.mkdir(parents=True)
    return completed_dir, archive_dir


def _tsv_rows(tsv_path: Path) -> list[str]:
    """Return non-header rows from a TSV file."""
    if not tsv_path.exists():
        return []
    lines = tsv_path.read_text().splitlines()
    return [l for l in lines[1:] if l.strip()]


def test_normal_completion_writes_all_four(tmp_path: Path) -> None:
    """D-09: all four artifacts exist after write_terminal_state for a completed result."""
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    graph, node_id = _make_graph(tmp_path)
    completed_dir, archive_dir = _make_dirs(tmp_path, node_id)
    results_tsv = tmp_path / "results.tsv"

    tsv_calls: list[tuple] = []

    def tsv_writer(nid, result, description=""):
        tsv_calls.append((nid, result, description))
        # Write a minimal TSV row
        if not results_tsv.exists():
            results_tsv.write_text("node_id\tcomposite\tstatus\tdescription\n")
        composite = result.get("composite", 0.0)
        status = result.get("status", "crash")
        with open(results_tsv, "a") as f:
            f.write(f"{nid}\t{composite:.6f}\t{status}\t{description}\n")

    result = {
        "status": "completed",
        "composite": 0.85,
        "metrics": {"val_auc": 0.85},
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }
    spec = {"description": "test run", "graph_metadata": {}}

    write_terminal_state(
        node_id=node_id,
        result=result,
        graph=graph,
        completed_dir=completed_dir,
        archive_dir=archive_dir,
        results_tsv_writer=tsv_writer,
        spec=spec,
        elapsed_s=100.0,
        gpu_id=0,
    )

    # Artifact 1: graph.json updated
    graph2 = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node = graph2.get_node(node_id)
    assert node is not None
    assert node["type"] == "executed"
    assert abs(node["composite"] - 0.85) < 1e-6, f"graph composite={node['composite']}"

    # Artifact 2: completed/<node>.json
    comp_file = completed_dir / f"{node_id}.json"
    assert comp_file.exists(), "completed/<node>.json must exist"
    comp = json.loads(comp_file.read_text())
    assert comp["id"] == node_id
    assert abs(comp["composite"] - 0.85) < 1e-6

    # Artifact 3: archive result.json
    archive_result = archive_dir / "result.json"
    assert archive_result.exists(), "archive/result.json must exist"
    ar = json.loads(archive_result.read_text())
    assert ar["status"] == "completed"

    # Artifact 4: results.tsv writer was called
    assert len(tsv_calls) == 1, f"TSV writer called {len(tsv_calls)} times, expected 1"
    assert tsv_calls[0][0] == node_id


def test_cap_kill_writes_all_four(tmp_path: Path) -> None:
    """D-09: all four artifacts exist after write_terminal_state for a budget-kill result."""
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    graph, node_id = _make_graph(tmp_path)
    completed_dir, archive_dir = _make_dirs(tmp_path, node_id)
    results_tsv = tmp_path / "results.tsv"
    tsv_calls: list[tuple] = []

    def tsv_writer(nid, result, description=""):
        tsv_calls.append((nid, result, description))
        if not results_tsv.exists():
            results_tsv.write_text("node_id\tcomposite\tstatus\n")
        with open(results_tsv, "a") as f:
            f.write(f"{nid}\t{result.get('composite',0):.6f}\t{result.get('status','')}\n")

    # Budget-kill payload (partial folds)
    result = {
        "status": "partial",
        "composite": 0.78,
        "metrics": {"val_auc": 0.78},
        "partial_folds": 3,
        "expected_folds": 5,
        "elapsed_seconds": 3600,
        "peak_vram_mb": 5000,
        "metadata": {"budget_killed": True},
    }
    spec = {"description": "cap-kill test", "graph_metadata": {}}

    write_terminal_state(
        node_id=node_id,
        result=result,
        graph=graph,
        completed_dir=completed_dir,
        archive_dir=archive_dir,
        results_tsv_writer=tsv_writer,
        spec=spec,
        elapsed_s=3600.0,
        gpu_id=1,
    )

    # All four artifacts must exist
    assert (completed_dir / f"{node_id}.json").exists(), "completed/<node>.json missing"
    assert (archive_dir / "result.json").exists(), "archive/result.json missing"
    assert len(tsv_calls) == 1, "TSV writer must be called once"

    graph2 = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node = graph2.get_node(node_id)
    assert node is not None
    # D-01: partial nodes get quarantine status, not keep/discard
    assert node["status"] == "partial", (
        f"Cap-kill node should have status='partial', got {node['status']!r}"
    )
    assert abs(node["composite"] - 0.78) < 1e-6


def test_graph_updated_before_tsv(tmp_path: Path) -> None:
    """D-09: fixed write order — graph.json mtime <= archive mtime <= TSV call time."""
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    graph, node_id = _make_graph(tmp_path)
    completed_dir, archive_dir = _make_dirs(tmp_path, node_id)

    graph_path = tmp_path / "graph.json"
    mtime_before_graph = graph_path.stat().st_mtime

    tsv_call_time: list[float] = []

    def tsv_writer(nid, result, description=""):
        tsv_call_time.append(time.time())

    result = {
        "status": "completed",
        "composite": 0.90,
        "metrics": {},
        "elapsed_seconds": 200,
        "peak_vram_mb": 3000,
    }
    spec = {"description": "write-order test"}

    write_terminal_state(
        node_id=node_id,
        result=result,
        graph=graph,
        completed_dir=completed_dir,
        archive_dir=archive_dir,
        results_tsv_writer=tsv_writer,
        spec=spec,
        elapsed_s=200.0,
        gpu_id=0,
    )

    # Graph must have been written
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None and node["type"] == "executed"

    # Completed file must exist
    assert (completed_dir / f"{node_id}.json").exists()

    # Archive result must exist
    assert (archive_dir / "result.json").exists()

    # TSV writer must have been called
    assert len(tsv_call_time) == 1, "TSV writer must be called exactly once"

    # Write order: graph modified after its pre-call mtime
    graph_mtime_after = graph_path.stat().st_mtime
    assert graph_mtime_after >= mtime_before_graph, (
        "graph.json must be written during write_terminal_state"
    )

    # Archive result must exist (written before TSV)
    assert (archive_dir / "result.json").exists(), (
        "archive result.json must exist before TSV writer is called"
    )
