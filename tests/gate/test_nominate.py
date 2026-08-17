"""Tests for gate/nominate.py (GTE-05 / D-136, D-142).

8 behaviour tests covering: status flow, idempotency, history event shape,
agent_initiated flag, rejection of non-keep nodes, and save-discipline (no
auto-persist).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Shared fixture — a graph with one keep node
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_with_keep_node(tmp_path):
    g = ExperimentGraph(path=str(tmp_path / "graph.json"))
    # Inject a node directly via internal dict — simpler than the full submit path.
    g.nodes["node_0001"] = {
        "id": "node_0001",
        "parent_id": None,
        "type": "executed",
        "status": "keep",
        "description": "fixture",
        "primary_value": 0.85,
        "created_at": "2026-05-05T00:00:00+00:00",
    }
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nominate_keep_to_candidate(graph_with_keep_node):
    """Nominate mutates status from keep -> candidate."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    nominate("node_0001", g)
    assert g.nodes["node_0001"]["status"] == "candidate"


def test_nominate_appends_history_event(graph_with_keep_node):
    """After nominate, last history entry is a 'nominated' event with ISO timestamp."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    nominate("node_0001", g)
    history = g.nodes["node_0001"]["history"]
    assert len(history) >= 1
    last = history[-1]
    assert last["event"] == "nominated"
    # Timestamp must parse as ISO-8601
    ts = datetime.fromisoformat(last["timestamp"])
    assert ts.tzinfo is not None  # must be timezone-aware
    assert last["agent_initiated"] is False


def test_nominate_agent_initiated_flag(graph_with_keep_node):
    """agent_initiated=True is stamped in the history event."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    nominate("node_0001", g, agent_initiated=True)
    last = g.nodes["node_0001"]["history"][-1]
    assert last["agent_initiated"] is True


def test_nominate_idempotent(graph_with_keep_node):
    """Calling nominate twice leaves status=candidate and adds only ONE history event."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    nominate("node_0001", g)
    nominate("node_0001", g)  # second call — should be a no-op
    assert g.nodes["node_0001"]["status"] == "candidate"
    nominated_events = [
        e for e in g.nodes["node_0001"]["history"]
        if e.get("event") == "nominated"
    ]
    assert len(nominated_events) == 1, (
        "Idempotent second call must not append a second 'nominated' event"
    )


def test_nominate_rejects_non_keep_discard(graph_with_keep_node):
    """nominate raises ValueError for a discard node; message contains 'discard' and 'keep'."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    g.nodes["node_0001"]["status"] = "discard"
    with pytest.raises(ValueError) as exc:
        nominate("node_0001", g)
    msg = str(exc.value)
    assert "discard" in msg
    assert "keep" in msg


def test_nominate_rejects_running(graph_with_keep_node):
    """nominate raises ValueError for a running node; message contains 'running'."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    g.nodes["node_0001"]["status"] = "running"
    with pytest.raises(ValueError) as exc:
        nominate("node_0001", g)
    assert "running" in str(exc.value)


def test_nominate_unknown_node_raises(graph_with_keep_node):
    """nominate raises ValueError for an unknown node_id; message contains the id and 'not found'."""
    from automil.gate.nominate import nominate
    g = graph_with_keep_node
    with pytest.raises(ValueError) as exc:
        nominate("node_9999", g)
    msg = str(exc.value)
    assert "node_9999" in msg
    assert "not found" in msg


def test_nominate_does_not_save(tmp_path):
    """nominate mutates in-memory but does NOT call graph.save() — file remains stale."""
    from automil.gate.nominate import nominate
    graph_path = tmp_path / "g.json"
    g = ExperimentGraph(path=str(graph_path))
    g.nodes["node_0001"] = {
        "id": "node_0001",
        "parent_id": None,
        "type": "executed",
        "status": "keep",
        "description": "fixture",
        "primary_value": 0.85,
        "created_at": "2026-05-05T00:00:00+00:00",
    }
    g.save()  # persist the INITIAL state (status=keep)

    # Now nominate — must NOT auto-save
    nominate("node_0001", g)
    assert g.nodes["node_0001"]["status"] == "candidate"  # in-memory updated

    # Re-read the file — should still have status=keep (no auto-save)
    on_disk = json.loads(graph_path.read_text())
    assert on_disk["nodes"]["node_0001"]["status"] == "keep", (
        "nominate() must NOT call graph.save(); caller controls persistence"
    )
