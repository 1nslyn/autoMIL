"""Tests for /api/promotion-rate endpoint in viz/server.py (GTE-06 / D-144).

5 tests exercising:
  1. No graph.json → status 200, zeros, "no data" diagnostic
  2. 4 nominations (2 registered, 2 keep) → rate=0.5, "healthy"
  3. 20 nominations, 0 registered → rate=0.0, "too strict"
  4. 4 nominations, all 4 registered → rate=1.0, "too loose"
  5. Response always includes window_days=30

Calls the aiohttp handler directly with ``loop_context`` (no pytest-asyncio or
network socket required). Route registration is tested separately.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiohttp.test_utils import loop_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominated_event() -> dict:
    """Build a 'nominated' history event timestamped now (within any window)."""
    return {
        "event": "nominated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_initiated": False,
    }


def _build_graph(statuses: list[str]) -> dict:
    """Build a minimal graph.json dict with N nodes each having a nominated event."""
    nodes = {}
    for i, status in enumerate(statuses):
        nid = f"node_{i:04d}"
        nodes[nid] = {
            "id": nid,
            "parent_id": None,
            "type": "executed",
            "status": status,
            "description": "test fixture",
            "primary_value": 0.8,
            "history": [_nominated_event()],
        }
    return {"nodes": nodes, "meta": {}, "technique_stats": {}}


def _run(coro):
    """Execute a coroutine synchronously using loop_context."""
    with loop_context() as loop:
        return loop.run_until_complete(coro)


async def _promotion_payload() -> dict:
    """Call the request-independent JSON handler without binding a socket."""
    from automil.viz.server import promotion_rate_handler

    response = await promotion_rate_handler(None)
    assert response.status == 200
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_promotion_rate_endpoint_no_graph(tmp_path, monkeypatch):
    """GET /api/promotion-rate when GRAPH_FILE doesn't exist → 200 + safe defaults."""
    import automil.viz.server as srv

    # Point GRAPH_FILE at a path that doesn't exist
    monkeypatch.setattr(srv, "GRAPH_FILE", tmp_path / "automil" / "graph.json")

    async def _test():
        data = await _promotion_payload()
        assert data["promotion_rate"] == 0.0
        assert data["nominated"] == 0
        assert data["promoted"] == 0
        assert "no data" in data["health_diagnostic"].lower()
        assert data["window_days"] == 30

    _run(_test())


def test_promotion_rate_endpoint_with_data(tmp_path, monkeypatch):
    """4 nominations (2 registered, 2 keep) → rate=0.5, nominated=4, promoted=2, healthy."""
    import automil.viz.server as srv

    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_build_graph(["registered", "registered", "keep", "keep"])))
    monkeypatch.setattr(srv, "GRAPH_FILE", graph_file)

    async def _test():
        data = await _promotion_payload()
        assert data["promotion_rate"] == pytest.approx(0.5)
        assert data["nominated"] == 4
        assert data["promoted"] == 2
        assert "healthy" in data["health_diagnostic"].lower()
        assert data["window_days"] == 30

    _run(_test())


def test_promotion_rate_endpoint_health_low(tmp_path, monkeypatch):
    """20 nominations, 0 registered → rate=0.0 → 'too strict' diagnostic."""
    import automil.viz.server as srv

    # 20 nominations, none registered — rate will be 0.0 < 5% threshold
    statuses = ["keep"] * 20
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_build_graph(statuses)))
    monkeypatch.setattr(srv, "GRAPH_FILE", graph_file)

    async def _test():
        data = await _promotion_payload()
        assert data["promotion_rate"] == pytest.approx(0.0)
        assert "too strict" in data["health_diagnostic"].lower()

    _run(_test())


def test_promotion_rate_endpoint_health_high(tmp_path, monkeypatch):
    """4 nominations, all 4 registered → rate=1.0 → 'too loose' diagnostic."""
    import automil.viz.server as srv

    statuses = ["registered"] * 4
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_build_graph(statuses)))
    monkeypatch.setattr(srv, "GRAPH_FILE", graph_file)

    async def _test():
        data = await _promotion_payload()
        assert data["promotion_rate"] == pytest.approx(1.0)
        assert "too loose" in data["health_diagnostic"].lower()

    _run(_test())


def test_promotion_rate_endpoint_window_days_field(tmp_path, monkeypatch):
    """Response always includes window_days: 30 regardless of graph state."""
    import automil.viz.server as srv

    monkeypatch.setattr(srv, "GRAPH_FILE", tmp_path / "no_graph.json")

    async def _test():
        data = await _promotion_payload()
        assert data["window_days"] == 30

    _run(_test())


def test_promotion_rate_route_is_registered():
    """The direct handler under test remains exposed at the public API path."""
    from automil.viz.server import create_app

    paths = {
        resource.canonical
        for resource in create_app().router.resources()
    }
    assert "/api/promotion-rate" in paths
