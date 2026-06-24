"""REC-01 end-to-end kill simulation + D-01 quarantine of partial nodes.

D-01: Partial results are quarantined — excluded from keep/discard and best_node
      selection, but remain visible in rank/dashboard/TSV.

Tests:
  - Existing aggregate_folds behavior (confirm partial status works) — GREEN
  - Partial nodes excluded from graph.best_node — RED until D-01 quarantine ships
  - Partial nodes not assigned keep/discard status — RED until D-01 quarantine ships
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automil.cells.reconcile import aggregate_folds


def _write_fold(archive: Path, idx: int, composite: float = 0.80) -> None:
    """Write a well-formed fold result JSON into archive/."""
    payload = {
        "fold_index": idx,
        "fold_count": 5,
        "status": "completed",
        "composite": composite,
        "metrics": {"val_auc": composite, "val_bacc": composite,
                    "test_auc": composite, "test_bacc": composite},
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }
    (archive / f"fold_{idx}_result.json").write_text(json.dumps(payload))


def test_kill_with_n_completed_folds_returns_mean_composite(tmp_path: Path) -> None:
    """D-01 baseline: 3 of 5 folds → composite = mean of 3, status='partial'.

    This should be GREEN — confirms aggregate_folds already handles partial correctly.
    baseline — must stay GREEN.
    """
    archive = tmp_path / "archive"
    archive.mkdir()

    composites = [0.80, 0.82, 0.84]
    for i, c in enumerate(composites):
        _write_fold(archive, i, composite=c)

    result = aggregate_folds(archive, expected_fold_count=5)

    assert result["status"] == "partial", f"Expected 'partial', got {result['status']!r}"
    expected_composite = sum(composites) / len(composites)
    assert result["composite"] == pytest.approx(expected_composite, rel=1e-6), (
        f"Composite {result['composite']} != mean of {composites} = {expected_composite}"
    )
    assert result["composite"] != 0.0, "composite must not be 0.0 for partial result"


def test_partial_status_excluded_from_best_node(tmp_path: Path) -> None:
    """D-01: a partial node must NOT become best_node even if its composite is high.

    Graph with completed parent (composite=0.7) + partial child (composite=0.8):
    best_node must remain the completed parent.
    RED until D-01 quarantine guard in graph ships in Plan 06.
    """
    from automil.graph import ExperimentGraph

    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))

    # Add completed parent with composite=0.7
    parent_id = graph.add_executed(
        parent_id=None,
        description="baseline",
        techniques=["baseline"],
        metrics={"composite": 0.7},
        status="keep",
    )

    # Add a partial child with higher composite=0.8 via direct nodes dict mutation
    child_node = {
        "id": "node_0002",
        "parent_id": parent_id,
        "type": "executed",
        "status": "partial",   # quarantined status
        "description": "partial run",
        "techniques": [],
        "composite": 0.8,
        "global_delta": 0.0,
        "parent_delta": 0.1,
        "metrics": {"composite": 0.8, "partial_folds": 3, "expected_folds": 5},
        "vram_gb": 0.0,
        "elapsed_min": 0.0,
        "gpu": -1,
        "commit": None,
        "config_hash": None,
    }
    graph.nodes["node_0002"] = child_node
    graph.recompute_best()
    graph.save()

    # D-01 quarantine: best_node must still be the completed parent, not the partial child
    best = graph.best_node()
    assert best is not None, "best_node must not be None with a completed parent present"
    assert best.get("id") != "node_0002", (
        "D-01 not implemented: partial node (node_0002, composite=0.8) became best_node. "
        "Partial nodes must be quarantined from best_node selection."
    )
    assert best.get("id") == parent_id, (
        f"Expected best_node={parent_id!r} (completed, composite=0.7), "
        f"got best_node={best.get('id')!r}."
    )


def test_partial_status_excluded_from_keep_discard(tmp_path: Path) -> None:
    """D-01: a node with status='partial' must NOT be assigned keep or discard.

    RED until D-01 quarantine guard ships in Plan 06.
    """
    from automil.graph import ExperimentGraph

    graph_path = tmp_path / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))

    # Add a completed parent
    parent_id = graph.add_executed(
        parent_id=None,
        description="baseline",
        techniques=["baseline"],
        metrics={"composite": 0.7},
        status="keep",
    )

    # Add a partial child directly via nodes dict
    child_node = {
        "id": "node_0002",
        "parent_id": parent_id,
        "type": "executed",
        "status": "partial",
        "description": "partial run",
        "techniques": [],
        "composite": 0.75,
        "global_delta": 0.0,
        "parent_delta": 0.05,
        "metrics": {"composite": 0.75, "partial_folds": 2, "expected_folds": 5},
        "vram_gb": 0.0,
        "elapsed_min": 0.0,
        "gpu": -1,
        "commit": None,
        "config_hash": None,
    }
    graph.nodes["node_0002"] = child_node

    # Trigger keep/discard logic (recompute_best walks executed nodes)
    graph.recompute_best()
    graph.save()

    # Reload from disk to check persisted state
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node("node_0002")
    assert node is not None, "node_0002 must still exist after save/reload"

    # D-01: partial nodes must NOT have keep or discard status
    assert node.get("status") not in ("keep", "discard"), (
        f"D-01 not implemented: partial node got status={node.get('status')!r}. "
        "Partial nodes must retain 'partial' status, not be promoted to keep/discard."
    )
    assert node.get("status") == "partial", (
        f"Expected node_0002 to remain 'partial', got {node.get('status')!r}."
    )
