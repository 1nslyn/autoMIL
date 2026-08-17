"""Tests for held-out node filtering in `automil rank` command (D-139 / GTE-01).

Verifies:
  - rank filters held-out nodes by default
  - --include-held-out shows held-out nodes
  - --include-held-out logs a WARNING citing D-139
  - Without the flag, no WARNING about held-out is emitted
  - Nodes without a metadata key are not filtered
"""
from __future__ import annotations

import json
import logging

import pytest
from click.testing import CliRunner

from automil.cli import main


def _write_graph(adir, nodes):
    """Write a minimal graph.json to adir (includes scoring keys required by recalculate_scores)."""
    graph = {
        "schema_version": 1,
        "meta": {
            "total_executed": len(nodes),
            "best_node_id": list(nodes.keys())[0],
            "best_primary_value": 0.9,
            "baseline_primary_value": 0.0,
            "total_proposed": 0,
            "next_id": len(nodes) + 1,
            "scoring": {
                "exploration_weight": 0.005,
                "novelty_weight": 0.003,
            },
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph))


@pytest.fixture
def rank_project(tmp_path, monkeypatch):
    """Project with 3 proposed pending nodes; node_0002 is held-out."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("")
    nodes = {
        "node_0001": {
            "id": "node_0001", "type": "proposed", "status": "pending",
            "parent_id": "root",
            "primary_value": 0.9, "potential": 0.5,
            "description": "search-cell win",
        },
        "node_0002": {
            "id": "node_0002", "type": "proposed", "status": "pending",
            "parent_id": "root",
            "primary_value": 0.85, "potential": 0.4,
            "description": "held-out eval",
            "metadata": {"held_out": True},
        },
        "node_0003": {
            "id": "node_0003", "type": "proposed", "status": "pending",
            "parent_id": "root",
            "primary_value": 0.88, "potential": 0.45,
            "description": "another search-cell",
        },
    }
    _write_graph(adir, nodes)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---- Test 1: default rank hides held-out ------------------------------------

def test_rank_filters_held_out_by_default(rank_project):
    """automil rank (no flags) must NOT show held-out node_0002."""
    runner = CliRunner()
    result = runner.invoke(main, ["rank", "--n", "10"])
    assert result.exit_code == 0, result.output
    assert "node_0001" in result.output
    assert "node_0003" in result.output
    assert "node_0002" not in result.output


# ---- Test 2: --include-held-out shows held-out node --------------------------

def test_rank_include_held_out_shows_held_out(rank_project):
    """automil rank --include-held-out must show all three nodes including node_0002."""
    runner = CliRunner()
    result = runner.invoke(main, ["rank", "--n", "10", "--max-per-branch", "10", "--include-held-out"])
    assert result.exit_code == 0, result.output
    assert "node_0001" in result.output
    assert "node_0002" in result.output
    assert "node_0003" in result.output


# ---- Test 3: --include-held-out logs WARNING with D-139 ----------------------

def test_rank_include_held_out_logs_warning(rank_project, caplog):
    """--include-held-out must emit a WARNING log message containing 'held-out' AND 'D-139'."""
    runner = CliRunner()
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(main, ["rank", "--n", "10", "--max-per-branch", "10", "--include-held-out"])
    assert result.exit_code == 0, result.output
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("held-out" in m or "held_out" in m for m in warning_messages), (
        f"No WARNING about held-out in log records. Records: {warning_messages}"
    )
    assert any("D-139" in m for m in warning_messages), (
        f"No D-139 citation in WARNING log. Records: {warning_messages}"
    )


# ---- Test 4: default rank emits no held-out warning -------------------------

def test_rank_default_no_warning(rank_project, caplog):
    """Without --include-held-out, no WARNING about held-out should be emitted."""
    runner = CliRunner()
    with caplog.at_level(logging.WARNING):
        result = runner.invoke(main, ["rank", "--n", "10"])
    assert result.exit_code == 0, result.output
    held_out_warnings = [
        r.message for r in caplog.records
        if r.levelno >= logging.WARNING and ("held-out" in r.message or "D-139" in r.message)
    ]
    assert held_out_warnings == [], f"Unexpected held-out warnings: {held_out_warnings}"


# ---- Test 5: nodes without metadata key are not filtered --------------------

def test_rank_filter_handles_missing_metadata(tmp_path, monkeypatch):
    """Nodes without a 'metadata' key must NOT be filtered out (treat as not-held-out)."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("")
    # node_0010 has no metadata key at all
    nodes = {
        "node_0010": {
            "id": "node_0010", "type": "proposed", "status": "pending",
            "parent_id": "root",
            "primary_value": 0.9, "potential": 0.5,
            "description": "no-metadata node",
        },
    }
    _write_graph(adir, nodes)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["rank", "--n", "10"])
    assert result.exit_code == 0, result.output
    assert "node_0010" in result.output
