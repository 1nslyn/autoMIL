"""Tests for promotion_rate line in `automil status` output (GTE-06 / D-144).

4 tests:
  6. Graph with 4 nominations (2 registered) → output has "Promotion rate.*30d.*50.0%.*2/4"
  7. Same graph → output contains "healthy"
  8. No graph.json → status does not crash; omits line OR shows "no data"
  9. Graph with zero nominations → line shows "no data" or "0%.*0/0"
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominated_event() -> dict:
    return {
        "event": "nominated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_initiated": False,
    }


def _build_graph(statuses: list[str]) -> dict:
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
    return {
        "schema_version": 1,
        "meta": {
            "best_primary_value": 0.8,
            "best_node_id": "node_0000",
            "total_executed": len(statuses),
            "total_proposed": 0,
            "next_id": len(statuses),
            "baseline_primary_value": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": nodes,
        "technique_stats": {},
    }


def _setup_project(tmp_path: Path, graph_data: dict | None = None) -> None:
    """Create a minimal automil project directory for the status command."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    # Create orchestrator dirs so status doesn't fail on glob
    orch = adir / "orchestrator"
    (orch / "queue").mkdir(parents=True)
    (orch / "completed").mkdir(parents=True)
    if graph_data is not None:
        (adir / "graph.json").write_text(json.dumps(graph_data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_status_includes_promotion_rate_line(tmp_path, monkeypatch):
    """4 nominations (2 registered, 2 keep) → output contains 'Promotion rate.*30d.*50.0%.*2/4'."""
    _setup_project(tmp_path, _build_graph(["registered", "registered", "keep", "keep"]))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0, result.output
    assert re.search(r"Promotion rate.*30d.*50\.0%.*2/4", result.output), (
        f"Expected promotion rate line not found in:\n{result.output}"
    )


def test_status_includes_health_diagnostic(tmp_path, monkeypatch):
    """Same 50% setup → output contains 'healthy' health diagnostic."""
    _setup_project(tmp_path, _build_graph(["registered", "registered", "keep", "keep"]))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0, result.output
    assert "healthy" in result.output.lower(), (
        f"Expected 'healthy' diagnostic not found in:\n{result.output}"
    )


def test_status_no_graph_omits_or_zeros_promotion_line(tmp_path, monkeypatch):
    """No graph.json → status does not crash; either omits the line or shows 'no data'."""
    _setup_project(tmp_path, graph_data=None)  # no graph.json
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    # Must not crash
    assert result.exit_code == 0, result.output
    # If it does print something about promotion rate, it must say "no data" (not a traceback)
    if "Promotion rate" in result.output:
        assert "no data" in result.output.lower(), (
            f"Expected 'no data' when no graph.json:\n{result.output}"
        )
    # Exception text must NOT appear
    assert "Traceback" not in result.output
    assert "Error" not in result.output


def test_status_zero_nominations_handles_gracefully(tmp_path, monkeypatch):
    """Graph with zero nominations → line shows 'no data' or '0%.*0/0', no crash."""
    # Build graph with 2 nodes but no history (zero nominations)
    nodes = {}
    for i in range(2):
        nid = f"node_{i:04d}"
        nodes[nid] = {
            "id": nid, "parent_id": None,
            "type": "executed", "status": "keep",
            "description": "test fixture", "primary_value": 0.8,
            # No "history" key — legacy format
        }
    graph_data = {
        "schema_version": 1,
        "meta": {
            "best_primary_value": 0.8, "best_node_id": "node_0000",
            "total_executed": 2, "total_proposed": 0, "next_id": 2,
            "baseline_primary_value": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    _setup_project(tmp_path, graph_data)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    # If present, must show "no data"
    if "Promotion rate" in result.output:
        assert "no data" in result.output.lower(), (
            f"Expected 'no data' for zero nominations:\n{result.output}"
        )
