"""Tests for automil nominate top-level CLI command (plan 05-09 / GTE-05 / D-142, D-145).

6 behaviour tests covering:
  T-1 keep_to_candidate      : nominate exits 0, output contains "Nominated", status=candidate
  T-2 agent_flag             : --agent stamps agent_initiated=True in history
  T-3 unknown_node           : non-zero exit + "not found"
  T-4 non_keep_exits_nonzero : discard node yields non-zero exit + helpful message
  T-5 idempotent             : second nominate is no-op (only one history event)
  T-6 persists_via_graph_save: re-reading graph.json after nominate shows candidate status
"""
from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Shared project fixture — minimal automil project with one keep node
# ---------------------------------------------------------------------------

@pytest.fixture
def project(tmp_path):
    """Minimal automil project dir with graph.json containing node_0001 (keep)."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("# fixture\n")
    (adir / "graph.json").write_text(
        json.dumps({
            "meta": {"total_executed": 1},
            "nodes": {
                "node_0001": {
                    "id": "node_0001",
                    "parent_id": None,
                    "type": "executed",
                    "status": "keep",
                    "primary_value": 0.85,
                    "description": "fixture keep node",
                    "history": [],
                }
            },
        })
    )
    return tmp_path


def _run_nominate(project_dir, args: list[str]):
    """Invoke `automil nominate ...` with cwd set to project_dir."""
    from click.testing import CliRunner
    from automil.cli import main

    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        result = CliRunner().invoke(main, ["nominate"] + args, catch_exceptions=False)
    finally:
        os.chdir(old_cwd)
    return result


# ---------------------------------------------------------------------------
# T-1: keep -> candidate happy path
# ---------------------------------------------------------------------------

def test_nominate_cli_keep_to_candidate(project):
    """Exit 0; output 'Nominated node_0001'; graph shows status=candidate."""
    result = _run_nominate(project, ["node_0001"])

    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}. Output: {result.output}"
    )
    assert "Nominated" in result.output and "node_0001" in result.output, (
        f"Expected 'Nominated node_0001' in output; got: {result.output}"
    )
    # Verify graph mutated
    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    assert graph_data["nodes"]["node_0001"]["status"] == "candidate", (
        f"Node status should be 'candidate'; got: "
        f"{graph_data['nodes']['node_0001']['status']!r}"
    )
    # History event has agent_initiated=False (operator path)
    history = graph_data["nodes"]["node_0001"].get("history", [])
    nominated_events = [e for e in history if e.get("event") == "nominated"]
    assert len(nominated_events) == 1, f"Expected 1 nominated event; got {nominated_events}"
    assert nominated_events[0]["agent_initiated"] is False, (
        f"agent_initiated should be False for operator nominate; got: "
        f"{nominated_events[0]['agent_initiated']!r}"
    )


# ---------------------------------------------------------------------------
# T-2: --agent flag stamps agent_initiated=True
# ---------------------------------------------------------------------------

def test_nominate_cli_agent_flag(project):
    """--agent flag causes history event with agent_initiated=True."""
    result = _run_nominate(project, ["node_0001", "--agent"])

    assert result.exit_code == 0, (
        f"Expected exit 0 with --agent; got {result.exit_code}. Output: {result.output}"
    )
    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    history = graph_data["nodes"]["node_0001"].get("history", [])
    nominated_events = [e for e in history if e.get("event") == "nominated"]
    assert len(nominated_events) == 1, (
        f"Expected 1 nominated event with --agent; got {nominated_events}"
    )
    assert nominated_events[0]["agent_initiated"] is True, (
        f"agent_initiated should be True with --agent flag; got: "
        f"{nominated_events[0]['agent_initiated']!r}"
    )


# ---------------------------------------------------------------------------
# T-3: Unknown node exits non-zero + "not found"
# ---------------------------------------------------------------------------

def test_nominate_cli_unknown_node_exits_nonzero(project):
    """Unknown node_id exits non-zero; output contains 'not found'."""
    result = _run_nominate(project, ["node_9999"])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for unknown node; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    assert "not found" in result.output.lower(), (
        f"Expected 'not found' in output; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-4: Non-keep (discard) node exits non-zero with status hint
# ---------------------------------------------------------------------------

def test_nominate_cli_non_keep_exits_nonzero(project):
    """Node with status=discard exits non-zero; output mentions 'discard' and 'keep'."""
    # Mutate fixture to discard status
    graph_path = project / "automil" / "graph.json"
    graph_data = json.loads(graph_path.read_text())
    graph_data["nodes"]["node_0001"]["status"] = "discard"
    graph_path.write_text(json.dumps(graph_data))

    result = _run_nominate(project, ["node_0001"])

    assert result.exit_code != 0, (
        f"Expected non-zero exit for discard node; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    out_lower = result.output.lower()
    assert "discard" in out_lower or "keep" in out_lower, (
        f"Expected status hint ('discard' or 'keep') in error output; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-5: Idempotent — second nominate is a no-op (only one history event)
# ---------------------------------------------------------------------------

def test_nominate_cli_idempotent(project):
    """Run nominate twice; both exit 0; only one history event recorded."""
    result1 = _run_nominate(project, ["node_0001"])
    assert result1.exit_code == 0, f"First nominate failed: {result1.output}"

    result2 = _run_nominate(project, ["node_0001"])
    assert result2.exit_code == 0, (
        f"Second nominate (idempotent) failed: {result2.output}"
    )

    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    history = graph_data["nodes"]["node_0001"].get("history", [])
    nominated_events = [e for e in history if e.get("event") == "nominated"]
    assert len(nominated_events) == 1, (
        f"Idempotent: expected exactly 1 nominated event; got {len(nominated_events)}: "
        f"{nominated_events}"
    )


# ---------------------------------------------------------------------------
# T-6: Persists via graph.save() — re-reading graph.json shows candidate
# ---------------------------------------------------------------------------

def test_nominate_cli_persists_via_graph_save(project):
    """After CLI nominate, a FRESH ExperimentGraph read shows status=candidate."""
    result = _run_nominate(project, ["node_0001"])
    assert result.exit_code == 0, f"nominate failed: {result.output}"

    # Re-read via a new ExperimentGraph instance
    from automil.graph import ExperimentGraph

    graph_path = str(project / "automil" / "graph.json")
    fresh_graph = ExperimentGraph(path=graph_path)
    status = fresh_graph.nodes["node_0001"]["status"]
    assert status == "candidate", (
        f"Fresh graph read should show status='candidate'; got {status!r}. "
        f"This confirms graph.save() was called by the CLI command."
    )
