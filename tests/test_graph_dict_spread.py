"""DEC-04 / D-200: graph.py dict-spread regression tests.

Verifies the post-refactor storage shape:
  - node["metrics"] is the dict-spread of the consumer's metrics dict.
  - Framework-owned scalars (primary_value, parent_delta, global_delta, vram_gb,
    elapsed_min, gpu) stay at top level.
  - The four named autobench keys (val_auc, val_bacc, test_auc, test_bacc)
    NO LONGER live at top level.
  - Pareto dominance is primary_value-only (OQ-9 Option B).

Pitfall 7 anti-acceptance defenders:
  - test_consumer_extension_keys_round_trip: arbitrary consumer metric names
    survive ingestion.
  - test_node_metrics_no_silent_zero_default: framework does not bake
    autobench-key assumptions on minimal payloads.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from automil.graph import ExperimentGraph


@pytest.fixture
def empty_graph(tmp_path: Path) -> ExperimentGraph:
    """Return a fresh ExperimentGraph rooted in tmp_path."""
    return ExperimentGraph(tmp_path / "graph.json")


def test_add_executed_round_trips_arbitrary_metric_keys(empty_graph: ExperimentGraph):
    """D-200: dict-spread preserves all consumer keys (Pitfall 7b defender)."""
    nid = empty_graph.add_executed(
        parent_id=None,
        description="round-trip",
        techniques=[],
        status="keep",
        metrics={"primary_value": 0.7, "top1": 0.7, "top5": 0.93, "custom_score": 1.5},
    )
    n = empty_graph.get_node(nid)
    assert n["metrics"]["top1"] == 0.7
    assert n["metrics"]["top5"] == 0.93
    assert n["metrics"]["custom_score"] == 1.5
    assert n["primary_value"] == 0.7  # framework-owned scalar at top level


def test_sklearn_iris_two_key_metrics_stored(empty_graph: ExperimentGraph):
    """D-203 / DEC-02 enabling: sklearn-iris {accuracy, f1} round-trips."""
    nid = empty_graph.add_executed(
        parent_id=None,
        description="iris baseline",
        techniques=[],
        status="keep",
        metrics={"primary_value": 0.97, "accuracy": 0.97, "f1": 0.965},
    )
    n = empty_graph.get_node(nid)
    assert n["metrics"]["accuracy"] == 0.97
    assert n["metrics"]["f1"] == 0.965
    # Pitfall 7b: NO val_auc auto-default to 0.0
    assert "val_auc" not in n["metrics"]


def test_autobench_four_key_metrics_stored(empty_graph: ExperimentGraph):
    """Backwards-compat: autobench's 4-key shape still works."""
    nid = empty_graph.add_executed(
        parent_id=None,
        description="autobench-shaped",
        techniques=[],
        status="keep",
        metrics={
            "primary_value": 0.502,
            "val_auc": 0.81, "val_bacc": 0.78,
            "test_auc": 0.83, "test_bacc": 0.80,
        },
    )
    n = empty_graph.get_node(nid)
    assert n["metrics"]["val_auc"] == 0.81
    assert n["metrics"]["test_bacc"] == 0.80
    # The four named keys are NO LONGER at the top level (D-200).
    assert "val_auc" not in n
    assert "test_auc" not in n


def test_promote_uses_dict_spread(empty_graph: ExperimentGraph):
    """promote() must apply the same dict-spread as add_executed."""
    pid = empty_graph.add_proposed(
        parent_id=None,
        description="proposed",
        techniques=[],
        rationale="test",
    )
    empty_graph.mark_running(pid)
    empty_graph.promote(pid, {
        "primary_value": 0.6,
        "status": "keep",
        "accuracy": 0.6,
        "custom_metric": 42.0,
    })
    n = empty_graph.get_node(pid)
    assert n["primary_value"] == 0.6
    assert n["metrics"]["accuracy"] == 0.6
    assert n["metrics"]["custom_metric"] == 42.0


def test_pareto_dominance_is_primary_value_only(empty_graph: ExperimentGraph):
    """OQ-9 Option B: child with higher primary_value is keep, regardless of val_auc."""
    parent_id = empty_graph.add_executed(
        parent_id=None, description="parent", techniques=[], status="keep",
        metrics={"primary_value": 0.5, "val_auc": 0.9, "test_auc": 0.85},
    )
    child_id = empty_graph.add_executed(
        parent_id=parent_id, description="child", techniques=[], status="keep",
        metrics={"primary_value": 0.6, "val_auc": 0.7, "test_auc": 0.7},
    )
    empty_graph._reevaluate_descendants(parent_id)
    n = empty_graph.get_node(child_id)
    # Primary_value-only dominance: 0.6 > 0.5 means keep, even though auc dropped.
    assert n["status"] == "keep"


def test_node_metrics_no_silent_zero_default(empty_graph: ExperimentGraph):
    """Pitfall 7b: framework does NOT auto-add autobench keys to a minimal payload."""
    nid = empty_graph.add_executed(
        parent_id=None, description="minimal", techniques=[], status="keep",
        metrics={"primary_value": 0.3},
    )
    n = empty_graph.get_node(nid)
    assert "val_auc" not in n["metrics"]
    assert "test_auc" not in n["metrics"]
    # primary_value IS in node["metrics"] because dict(metrics) spread it.
    assert n["metrics"].get("primary_value") == 0.3
