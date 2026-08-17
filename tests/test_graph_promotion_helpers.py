"""Tests for ExperimentGraph.nominations_in_window and promotion_rate helpers (GTE-06 / D-144).

9 tests covering: empty graph, window filtering by ISO timestamp, promotion_rate
computation (all-pass, half-pass, windowed), and graceful handling of nodes
without a history key (legacy backward compat D-147).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Helper to build a minimal graph fixture
# ---------------------------------------------------------------------------

def _make_graph(tmp_path) -> ExperimentGraph:
    g = ExperimentGraph(path=str(tmp_path / "graph.json"))
    return g


def _add_node(g: ExperimentGraph, node_id: str, status: str, history: list | None = None) -> None:
    """Inject a node directly into the graph dict for test isolation."""
    node = {
        "id": node_id,
        "parent_id": None,
        "type": "executed",
        "status": status,
        "description": "test fixture",
        "primary_value": 0.80,
        "created_at": "2026-05-05T00:00:00+00:00",
    }
    if history is not None:
        node["history"] = history
    g.nodes[node_id] = node


def _nominated_event(days_ago: float = 0.0) -> dict:
    """Build a 'nominated' history event at the given offset from now."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "event": "nominated",
        "timestamp": ts.isoformat(),
        "agent_initiated": False,
    }


# ---------------------------------------------------------------------------
# nominations_in_window tests
# ---------------------------------------------------------------------------

def test_nominations_in_window_empty(tmp_path):
    """Graph with no nominated nodes returns an empty list."""
    g = _make_graph(tmp_path)
    _add_node(g, "node_0001", "keep")  # no history
    assert g.nominations_in_window(days=30) == []


def test_nominations_in_window_inside(tmp_path):
    """Node with a 'nominated' event 1 day ago appears in a 30-day window."""
    g = _make_graph(tmp_path)
    _add_node(g, "node_0001", "candidate", history=[_nominated_event(days_ago=1)])
    result = g.nominations_in_window(days=30)
    assert len(result) == 1
    # Must return the same dict reference as graph.nodes["node_0001"]
    assert result[0] is g.nodes["node_0001"]


def test_nominations_in_window_outside(tmp_path):
    """Node nominated 60 days ago is NOT in a 30-day window but IS in 90-day window."""
    g = _make_graph(tmp_path)
    _add_node(g, "node_0001", "candidate", history=[_nominated_event(days_ago=60)])
    assert g.nominations_in_window(days=30) == []
    assert len(g.nominations_in_window(days=90)) == 1


# ---------------------------------------------------------------------------
# promotion_rate tests
# ---------------------------------------------------------------------------

def test_promotion_rate_no_nominations(tmp_path):
    """Graph with no nominations returns 0.0 (D-144 zero-division guard)."""
    g = _make_graph(tmp_path)
    assert g.promotion_rate() == 0.0


def test_promotion_rate_all_pass(tmp_path):
    """3 nominations, all status=registered → promotion_rate == 1.0."""
    g = _make_graph(tmp_path)
    for i in range(1, 4):
        nid = f"node_{i:04d}"
        _add_node(g, nid, "registered", history=[_nominated_event(days_ago=1)])
    assert g.promotion_rate() == pytest.approx(1.0)


def test_promotion_rate_half(tmp_path):
    """4 nominations: 2 registered, 2 keep → promotion_rate == 0.5."""
    g = _make_graph(tmp_path)
    for i in range(1, 3):
        _add_node(g, f"node_{i:04d}", "registered", history=[_nominated_event(days_ago=1)])
    for i in range(3, 5):
        _add_node(g, f"node_{i:04d}", "keep", history=[_nominated_event(days_ago=1)])
    assert g.promotion_rate() == pytest.approx(0.5)


def test_promotion_rate_window(tmp_path):
    """Window filtering: old nomination (90d ago, registered) + new nomination (today, keep).

    promotion_rate(days=30) == 0.0  (only new one in window; it's keep, not registered)
    promotion_rate(days=180) == 0.5 (both in window; 1 of 2 registered)
    """
    g = _make_graph(tmp_path)
    # Old nomination — registered
    _add_node(g, "node_0001", "registered", history=[_nominated_event(days_ago=90)])
    # New nomination — keep (not yet promoted)
    _add_node(g, "node_0002", "keep", history=[_nominated_event(days_ago=1)])

    assert g.promotion_rate(days=30) == pytest.approx(0.0)
    assert g.promotion_rate(days=180) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Backward compatibility / edge-case tests
# ---------------------------------------------------------------------------

def test_nominations_in_window_handles_missing_history(tmp_path):
    """Nodes without a 'history' key (legacy) do not crash — they're simply skipped."""
    g = _make_graph(tmp_path)
    _add_node(g, "node_0001", "keep")  # no history key at all
    # Should return [] without raising KeyError
    result = g.nominations_in_window(days=30)
    assert result == []


def test_promotion_rate_handles_missing_history(tmp_path):
    """Mixed graph: some nodes have history, some don't — only nominated nodes are counted."""
    g = _make_graph(tmp_path)
    # Node with history — nominated and registered
    _add_node(g, "node_0001", "registered", history=[_nominated_event(days_ago=1)])
    # Legacy node with no history — should not crash or inflate denominator
    _add_node(g, "node_0002", "registered")
    # promotion_rate should be 1.0 (1 nominated, 1 promoted) — not 0.5
    assert g.promotion_rate() == pytest.approx(1.0)
