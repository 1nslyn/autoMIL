"""REC-02: automil rank and results.tsv agree after terminal_writer completes.

After write_terminal_state:
  - Graph primary_value must equal the primary_value written to results.tsv.
  - TSV and graph primary_value values must be identical (no split-write drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rank_and_tsv_agree_after_terminal_write(tmp_path: Path) -> None:
    """D-09/D-10: graph primary_value == results.tsv primary_value after write_terminal_state."""
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    # Create graph with a running node
    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="consistency test baseline",
        techniques=["baseline"],
        metrics={"primary_value": 0.0},
        status="keep",
    )
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    completed_dir = tmp_path / "completed"
    completed_dir.mkdir()
    archive_dir = tmp_path / "archive" / node_id
    archive_dir.mkdir(parents=True)
    results_tsv = tmp_path / "results.tsv"

    # Track what TSV writer receives
    tsv_received: list[dict] = []

    def tsv_writer(nid, result, description=""):
        tsv_received.append({"nid": nid, "primary_value": result.get("primary_value", 0.0)})
        # Write a real TSV file
        if not results_tsv.exists():
            results_tsv.write_text("node_id\tprimary_value\tstatus\tdescription\n")
        primary_value = result.get("primary_value", 0.0)
        status = result.get("status", "crash")
        with open(results_tsv, "a") as f:
            f.write(f"{nid}\t{primary_value:.6f}\t{status}\t{description}\n")

    # CR-1b: the primary_value is derived from the val metrics, so the fixture must be
    # internally consistent — (0.90 + 0.84) / 2 == 0.87. (It previously declared
    # 0.87 alongside metrics averaging 0.84; the firewall now overrides such a
    # mismatch to the val-derived value, which is the point of CR-1b.)
    expected_primary_value = 0.87
    result = {
        "status": "completed",
        "primary_value": expected_primary_value,
        "metrics": {"val_auc": 0.90, "val_bacc": 0.84},
        "elapsed_seconds": 4098,
        "peak_vram_mb": 4500,
    }
    spec = {"description": "consistency test run", "graph_metadata": {}}

    write_terminal_state(
        node_id=node_id,
        result=result,
        graph=graph,
        completed_dir=completed_dir,
        archive_dir=archive_dir,
        results_tsv_writer=tsv_writer,
        spec=spec,
        elapsed_s=4098.0,
        gpu_id=0,
    )

    # Check graph primary_value
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None, "node must exist in graph after write_terminal_state"
    graph_primary_value = node.get("primary_value", None)
    assert graph_primary_value is not None, "graph node must have primary_value field"
    assert abs(graph_primary_value - expected_primary_value) < 1e-6, (
        f"graph primary_value={graph_primary_value} != expected {expected_primary_value}"
    )

    # Check TSV primary_value matches graph
    assert len(tsv_received) == 1, f"TSV writer called {len(tsv_received)} times"
    tsv_primary_value = tsv_received[0]["primary_value"]
    assert abs(tsv_primary_value - graph_primary_value) < 1e-6, (
        f"TSV primary_value={tsv_primary_value} != graph primary_value={graph_primary_value} "
        "(rank/TSV drift — D-09/D-10 broken)"
    )

    # Verify the TSV file content agrees
    assert results_tsv.exists(), "results.tsv must exist"
    lines = results_tsv.read_text().splitlines()
    data_rows = [l for l in lines[1:] if l.strip()]
    assert len(data_rows) >= 1, "results.tsv must have at least one data row"
    first_row = data_rows[0].split("\t")
    tsv_file_primary_value = float(first_row[1])
    assert abs(tsv_file_primary_value - expected_primary_value) < 1e-4, (
        f"results.tsv file primary_value={tsv_file_primary_value} != expected {expected_primary_value}"
    )
