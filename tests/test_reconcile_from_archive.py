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
    """D-11: reconcile --from-archive updates existing node composite from archive result.json.

    After --from-archive ships: graph composite must change from 0.5 to 0.8.
    RED until Plan 06 wires --from-archive flag to reconcile.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])

    # Setup: create a graph with an executed node (composite=0.5) and archive result (composite=0.8)
    from automil.graph import ExperimentGraph
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=str(graph_path))
    node_id = graph.add_executed(
        parent_id=None,
        description="baseline",
        techniques=["baseline"],
        metrics={"composite": 0.5},
        status="keep",
    )
    graph.save()

    archive_dir = tmp_path / "automil" / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True)
    (archive_dir / "result.json").write_text(
        json.dumps({"composite": 0.8, "status": "completed", "metrics": {}})
    )

    # Invoke reconcile --from-archive
    result = cli_runner.invoke(main, ["reconcile", "--from-archive", node_id])

    # RED: --from-archive flag not yet added → must succeed AND refresh composite
    assert result.exit_code == 0, (
        f"D-11 not implemented: reconcile --from-archive failed (exit_code={result.exit_code}). "
        f"Output: {result.output!r}"
    )

    # Reload graph and check composite was refreshed
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None
    assert abs(node.get("composite", 0.0) - 0.8) < 0.01, (
        f"D-11 not implemented: node composite={node.get('composite')}, "
        "expected 0.8 after --from-archive refresh."
    )


def test_from_archive_skips_running_nodes(cli_runner, tmp_path: Path, monkeypatch) -> None:
    """D-11 Pitfall 3: running nodes must not be overwritten by --from-archive.

    After --from-archive ships: running node composite must remain unchanged.
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
        metrics={"composite": 0.5},
        status="keep",
    )
    # Mark the node as running
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    archive_dir = tmp_path / "automil" / "orchestrator" / "archive" / node_id
    archive_dir.mkdir(parents=True)
    (archive_dir / "result.json").write_text(
        json.dumps({"composite": 0.9, "status": "completed", "metrics": {}})
    )

    result = cli_runner.invoke(main, ["reconcile", "--from-archive", node_id])

    # D-11: must succeed and skip the running node
    assert result.exit_code == 0, (
        f"D-11 not implemented: reconcile --from-archive failed: {result.output!r}"
    )

    # Running node composite must be unchanged
    graph2 = ExperimentGraph(path=str(graph_path))
    node = graph2.get_node(node_id)
    assert node is not None
    assert abs(node.get("composite", 0.0) - 0.5) < 0.01, (
        f"D-11 Pitfall 3 guard not implemented: running node composite was overwritten. "
        f"Expected 0.5, got {node.get('composite')}."
    )
    assert "skip" in result.output.lower() or "running" in result.output.lower(), (
        "D-11: expected 'skip' or 'running' in output for skipped running node."
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
