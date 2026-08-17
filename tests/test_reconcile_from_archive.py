"""REC-02 / D-11: reconcile --from-archive refreshes existing nodes from archive result.json.

D-11: `automil reconcile --from-archive [<node>|all]` treats archive result.json
      as authoritative and refreshes existing nodes. Default reconcile stays
      missing-node recovery only (no surprise clobbers).

Pitfall 3 guard: running nodes must never be overwritten.

RED until Plan 06 wires --from-archive flag to reconcile.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def test_from_archive_refreshes_existing_node(cli_runner, tmp_path: Path, monkeypatch) -> None:
    """D-11: reconcile --from-archive updates existing node primary_value from archive result.json.

    After --from-archive ships: graph primary_value must change from 0.5 to 0.8.
    RED until Plan 06 wires --from-archive flag to reconcile.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])

    # Setup: create a graph with an executed node (primary_value=0.5) and archive result (primary_value=0.8)
    from automil.graph import ExperimentGraph
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="baseline",
        techniques=["baseline"],
        metrics={"primary_value": 0.5},
        status="keep",
    )
    graph.save()

    archive_dir = tmp_path / "automil" / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True)
    (archive_dir / "result.json").write_text(
        json.dumps({"primary_value": 0.8, "status": "completed", "metrics": {}})
    )

    # Invoke reconcile --from-archive
    result = cli_runner.invoke(main, ["reconcile", "--from-archive", node_id])

    # RED: --from-archive flag not yet added → must succeed AND refresh primary_value
    assert result.exit_code == 0, (
        f"D-11 not implemented: reconcile --from-archive failed (exit_code={result.exit_code}). "
        f"Output: {result.output!r}"
    )

    # Reload graph and check primary_value was refreshed
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None
    assert abs(node.get("primary_value", 0.0) - 0.8) < 0.01, (
        f"D-11 not implemented: node primary_value={node.get('primary_value')}, "
        "expected 0.8 after --from-archive refresh."
    )


def test_from_archive_skips_running_nodes(cli_runner, tmp_path: Path, monkeypatch) -> None:
    """D-11 Pitfall 3: running nodes must not be overwritten by --from-archive.

    After --from-archive ships: running node primary_value must remain unchanged.
    RED until Plan 06 wires --from-archive with the running-node guard.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])

    from automil.graph import ExperimentGraph
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="running node",
        techniques=["baseline"],
        metrics={"primary_value": 0.5},
        status="keep",
    )
    # Mark the node as running
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    archive_dir = tmp_path / "automil" / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True)
    (archive_dir / "result.json").write_text(
        json.dumps({"primary_value": 0.9, "status": "completed", "metrics": {}})
    )

    result = cli_runner.invoke(main, ["reconcile", "--from-archive", node_id])

    # D-11: must succeed and skip the running node
    assert result.exit_code == 0, (
        f"D-11 not implemented: reconcile --from-archive failed: {result.output!r}"
    )

    # Running node primary_value must be unchanged
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None
    assert abs(node.get("primary_value", 0.0) - 0.5) < 0.01, (
        f"D-11 Pitfall 3 guard not implemented: running node primary_value was overwritten. "
        f"Expected 0.5, got {node.get('primary_value')}."
    )
    assert "skip" in result.output.lower() or "running" in result.output.lower(), (
        "D-11: expected 'skip' or 'running' in output for skipped running node."
    )


def test_from_archive_maps_result_status_to_graph_vocabulary(cli_runner, tmp_path: Path, monkeypatch) -> None:
    """CR-03 regression: --from-archive must NOT write result.json status values
    (completed/budget_killed) directly into graph nodes. Graph status vocabulary
    is keep/discard/crash/partial/cancelled. Writing 'completed' corrupts
    _reevaluate_descendants and recompute_best which only match 'keep'/'discard'.

    Assert: result status='completed' with primary_value > parent → graph status='keep'.
    Assert: result status='budget_killed' with primary_value < parent → graph status='discard'.
    Assert: result status='crash' → graph status='crash' (passthrough).
    Assert: result status='partial' → graph status='partial' (quarantined, D-01).
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])

    from automil.graph import ExperimentGraph
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))

    # Parent node with primary_value=0.7 (used as comparison baseline).
    parent_id = graph.add_executed(
        parent_id=None,
        description="parent",
        techniques=["baseline"],
        metrics={"primary_value": 0.7},
        status="keep",
    )
    graph.nodes[parent_id]["primary_value"] = 0.7

    # Node A: result status=completed, primary_value=0.85 > parent 0.7 → expect keep.
    node_a = graph.add_proposed(parent_id=parent_id, description="completed-high", techniques=[])
    graph.nodes[node_a]["parent_id"] = parent_id

    # Node B: result status=budget_killed, primary_value=0.5 < parent 0.7 → expect discard.
    node_b = graph.add_proposed(parent_id=parent_id, description="budget-killed-low", techniques=[])
    graph.nodes[node_b]["parent_id"] = parent_id

    # Node C: result status=crash → expect crash (passthrough).
    node_c = graph.add_proposed(parent_id=parent_id, description="crash-node", techniques=[])
    graph.nodes[node_c]["parent_id"] = parent_id

    # Node D: result status=partial → expect partial (quarantined D-01).
    node_d = graph.add_proposed(parent_id=parent_id, description="partial-node", techniques=[])
    graph.nodes[node_d]["parent_id"] = parent_id

    graph.save()

    archive_root = tmp_path / "automil" / "orchestrator" / "archive"
    for nid, payload in [
        (node_a, {"primary_value": 0.85, "status": "completed",      "metrics": {}}),
        (node_b, {"primary_value": 0.50, "status": "budget_killed",  "metrics": {}}),
        (node_c, {"primary_value": 0.00, "status": "crash",          "metrics": {}}),
        (node_d, {"primary_value": 0.60, "status": "partial",        "metrics": {}}),
    ]:
        d = archive_root / nid
        d.mkdir(parents=True, exist_ok=True)
        (d / "result.json").write_text(json.dumps(payload))

    result = cli_runner.invoke(main, ["reconcile", "--from-archive", "all"])
    assert result.exit_code == 0, f"reconcile --from-archive failed: {result.output!r}"

    graph2 = ExperimentGraph(path=str(graph_path))

    status_a = graph2.get_node(node_a).get("status")
    assert status_a == "keep", (
        f"CR-03: result status='completed', primary_value=0.85>parent=0.70 → "
        f"expected graph status='keep', got {status_a!r}"
    )
    status_b = graph2.get_node(node_b).get("status")
    assert status_b == "discard", (
        f"CR-03: result status='budget_killed', primary_value=0.50<parent=0.70 → "
        f"expected graph status='discard', got {status_b!r}"
    )
    status_c = graph2.get_node(node_c).get("status")
    assert status_c == "crash", (
        f"CR-03: result status='crash' → expected graph status='crash', got {status_c!r}"
    )
    status_d = graph2.get_node(node_d).get("status")
    assert status_d == "partial", (
        f"CR-03: result status='partial' → expected graph status='partial' (D-01 quarantine), "
        f"got {status_d!r}"
    )
    # Verify raw result status is preserved in metadata for traceability.
    meta_a = graph2.get_node(node_a).get("metadata", {})
    assert meta_a.get("result_status") == "completed", (
        f"CR-03: raw result_status should be preserved in metadata for traceability, "
        f"got metadata={meta_a!r}"
    )


def test_default_reconcile_unchanged(cli_runner, tmp_path: Path, monkeypatch) -> None:
    """baseline — must stay GREEN. Default reconcile (no --from-archive) must not change
    existing executed nodes — only recovers missing nodes.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])

    # Default reconcile on an empty graph should succeed
    result = cli_runner.invoke(main, ["reconcile"])

    assert result.exit_code == 0, (
        f"Baseline broken: default reconcile failed: {result.output}"
    )
