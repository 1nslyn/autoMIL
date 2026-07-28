"""REC-02: automil rank and results.tsv agree after terminal_writer completes.

After write_terminal_state:
  - Graph composite must equal the composite written to results.tsv.
  - TSV and graph composite values must be identical (no split-write drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rank_and_tsv_agree_after_terminal_write(tmp_path: Path) -> None:
    """D-09/D-10: graph composite == results.tsv composite after write_terminal_state."""
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    # Create graph with a running node
    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="consistency test baseline",
        techniques=["baseline"],
        metrics={"composite": 0.0},
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
        tsv_received.append({"nid": nid, "composite": result.get("composite", 0.0)})
        # Write a real TSV file
        if not results_tsv.exists():
            results_tsv.write_text("node_id\tcomposite\tstatus\tdescription\n")
        composite = result.get("composite", 0.0)
        status = result.get("status", "crash")
        with open(results_tsv, "a") as f:
            f.write(f"{nid}\t{composite:.6f}\t{status}\t{description}\n")

    # CR-1b: the composite is derived from the val metrics, so the fixture must be
    # internally consistent — (0.90 + 0.84) / 2 == 0.87. (It previously declared
    # 0.87 alongside metrics averaging 0.84; the firewall now overrides such a
    # mismatch to the val-derived value, which is the point of CR-1b.)
    expected_composite = 0.87
    result = {
        "status": "completed",
        "composite": expected_composite,
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

    # Check graph composite
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None, "node must exist in graph after write_terminal_state"
    graph_composite = node.get("composite", None)
    assert graph_composite is not None, "graph node must have composite field"
    assert abs(graph_composite - expected_composite) < 1e-6, (
        f"graph composite={graph_composite} != expected {expected_composite}"
    )

    # Check TSV composite matches graph
    assert len(tsv_received) == 1, f"TSV writer called {len(tsv_received)} times"
    tsv_composite = tsv_received[0]["composite"]
    assert abs(tsv_composite - graph_composite) < 1e-6, (
        f"TSV composite={tsv_composite} != graph composite={graph_composite} "
        "(rank/TSV drift — D-09/D-10 broken)"
    )

    # Verify the TSV file content agrees
    assert results_tsv.exists(), "results.tsv must exist"
    lines = results_tsv.read_text().splitlines()
    data_rows = [l for l in lines[1:] if l.strip()]
    assert len(data_rows) >= 1, "results.tsv must have at least one data row"
    first_row = data_rows[0].split("\t")
    tsv_file_composite = float(first_row[1])
    assert abs(tsv_file_composite - expected_composite) < 1e-4, (
        f"results.tsv file composite={tsv_file_composite} != expected {expected_composite}"
    )
