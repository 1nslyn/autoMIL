"""RED stubs for automil dequeue command (OPS-02).

Wave-0 Nyquist compliance — all stubs xfail until 13-03 implements dequeue.py.

Fixtures mirror test_cli_cancel_resubmit.py (self-contained: no cross-file imports).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_backends():
    """Save + restore BACKENDS registry around every test (PATTERNS.md §11)."""
    from automil.backends import BACKENDS  # noqa: PLC0415
    saved = dict(BACKENDS)
    yield
    BACKENDS.clear()
    BACKENDS.update(saved)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_adir(tmp_path: Path) -> Path:
    """Create a minimal automil/ directory structure under tmp_path.

    Creates orchestrator/queue/ (for dequeue tests) and
    orchestrator/running/local/ (D-169) so specs land in the right place.
    """
    (tmp_path / ".git").mkdir(exist_ok=True)
    adir = tmp_path / "automil"
    orch_dir = adir / "orchestrator"
    for sub in ("queue", "running", "archive"):
        (orch_dir / sub).mkdir(parents=True, exist_ok=True)
    # Per-backend running subdirectory (D-169).
    (orch_dir / "running" / "local").mkdir(parents=True, exist_ok=True)
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    return adir


def _write_graph(adir: Path, nodes: dict[str, Any]) -> None:
    """Write a graph.json with the given nodes dict."""
    graph = {
        "schema_version": 1,
        "meta": {
            "best_composite": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 10,
            "baseline_composite": 0.0,
            "scoring": {
                "exploration_weight": 0.005,
                "novelty_weight": 0.003,
            },
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))


# ---------------------------------------------------------------------------
# OPS-02 RED stubs (Wave 0 — Nyquist compliance)
# All xfail until plan 13-03 implements src/automil/cli/dequeue.py.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="OPS-02 not yet implemented", strict=True)
def test_dequeue_removes_queue_spec(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """dequeue removes orchestrator/queue/<node>.json and marks graph cancelled.

    OPS-02: queue spec path is FLAT (not backend-namespaced — unlike running specs).
    Uses graph.cancel() under locked_update to ensure serialization with daemon.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0020"

    # Write queue spec at the flat path (daemon:395 — orchestrator/queue/<node>.json).
    queue_spec_path = adir / "orchestrator" / "queue" / f"{node_id}.json"
    queue_spec_path.write_text(json.dumps({"id": node_id, "spec_version": 1}, indent=2))

    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "pending",
            "description": "dequeue removes queue spec test",
            "techniques": [],
            "metadata": {},
        }
    })

    result = cli_runner.invoke(main, ["dequeue", node_id], catch_exceptions=False)

    # Queue spec must be removed.
    assert not queue_spec_path.exists(), (
        f"queue spec still exists after dequeue: {queue_spec_path}"
    )

    # Graph node must be cancelled.
    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][node_id]["status"] == "cancelled", (
        f"expected status='cancelled', got {graph['nodes'][node_id].get('status')!r}"
    )


@pytest.mark.xfail(reason="OPS-02 not yet implemented", strict=True)
def test_dequeue_refuses_running(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """dequeue hard-fails for a running node with a cross-reference to 'automil cancel'.

    OPS-02 D-05: state guard must reject running nodes before any file operations.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0021"

    # Node is running — no queue spec needed (daemon already picked it up).
    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "running",
            "description": "running node dequeue test",
            "techniques": [],
            "metadata": {"backend": "local"},
        }
    })

    result = cli_runner.invoke(main, ["dequeue", node_id])

    assert result.exit_code != 0, "expected non-zero exit for running node"
    lower_out = result.output.lower()
    assert "cancel" in lower_out or "running" in lower_out, (
        f"expected 'cancel' or 'running' in output: {result.output!r}"
    )


@pytest.mark.xfail(reason="OPS-02 not yet implemented", strict=True)
def test_dequeue_pending_no_spec(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """dequeue on a pending node with no queue spec still marks it cancelled (idempotent).

    OPS-02 D-05: idempotent path — clears orphan nodes that have no queue file on disk.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    node_id = "node_0022"

    # Graph has pending node but NO queue spec on disk (orphan scenario).
    _write_graph(adir, {
        node_id: {
            "id": node_id,
            "parent_id": None,
            "type": "proposed",
            "status": "pending",
            "description": "pending no queue spec test",
            "techniques": [],
            "metadata": {},
        }
    })

    result = cli_runner.invoke(main, ["dequeue", node_id], catch_exceptions=False)

    assert result.exit_code == 0, f"expected exit 0 for idempotent dequeue: {result.output}"

    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][node_id]["status"] == "cancelled", (
        f"expected status='cancelled', got {graph['nodes'][node_id].get('status')!r}"
    )


@pytest.mark.xfail(reason="OPS-02 not yet implemented", strict=True)
def test_dequeue_unknown_node(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """dequeue hard-fails for unknown node id with a 'not found' message.

    OPS-02: _get_node_or_die must be called before graph.cancel() to avoid KeyError.
    The command must exist (not 'No such command') and must produce a node-not-found
    diagnostic — not a generic Click usage error.
    """
    from automil.cli import main  # noqa: PLC0415

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Empty graph — node does not exist.
    _write_graph(adir, {})

    result = cli_runner.invoke(main, ["dequeue", "node_9999"])

    assert result.exit_code != 0, "expected non-zero exit for unknown node"
    # Must not fail because the command doesn't exist — that would be the wrong reason.
    assert "No such command" not in result.output, (
        "dequeue command not registered yet — this is the reason the test fails"
    )
    assert "not found" in result.output.lower() or "unknown" in result.output.lower(), (
        f"expected 'not found' in dequeue output: {result.output!r}"
    )
