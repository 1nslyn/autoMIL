"""Daemon-level integration tests for _tick_cells, _running_in_cell, and cap-detection
in _handle_completion (CAP-02, CAP-04 / D-114, D-123, D-124).

Each test constructs a minimal ExperimentOrchestrator pointed at a tmp_path automil
overlay, injects in-memory state (self.running, self.graph, self.backend) as needed,
then invokes the method under test and asserts state transitions.

Tests:
    1. test_tick_cells_active_to_refusing_new — ACTIVE → REFUSING_NEW (no cancel)
    2. test_tick_cells_terminating_fires_cancel_with_cap_reason — REFUSING_NEW → TERMINATING;
       verifies cancel_reason written BEFORE backend.cancel() is called (Pitfall 4 ordering)
    3. test_tick_cells_finalized_when_running_empty — TERMINATING → FINALIZED
    4. test_tick_cells_idempotent_on_finalized — FINALIZED stays FINALIZED, no cancel
    5. test_running_in_cell_filters_by_metadata_cell_id — filter by cell_id in metadata
    6. test_handle_completion_with_cap_cancel_reason_calls_reconcile — partial folds → executed
    7. test_handle_completion_with_cap_cancel_zero_folds_marks_crash — zero folds → crash
    8. test_automil_fold_count_injected_into_subprocess_env — AUTOMIL_FOLD_COUNT in env
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orch(tmp_path: Path, config_yaml: str = "orchestrator: {}\n") -> Any:
    """Build a minimal ExperimentOrchestrator with an isolated automil/ overlay."""
    from automil.orchestrator import ExperimentOrchestrator

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text(config_yaml)
    (tmp_path / ".git").mkdir(exist_ok=True)
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    # D-169 (Phase 6): __init__ no longer pre-creates running/local/ to preserve
    # the D-168 startup guardrail. Tests that write running specs directly must
    # create orch.running_dir (= running/local/) themselves.
    orch.running_dir.mkdir(parents=True, exist_ok=True)
    return orch


def _write_cell_json(cells_dir: Path, cell_id: str, status: str,
                     started_at: float, budget_seconds: int = 21600,
                     safety_buffer_seconds: int = 1800) -> None:
    """Write a minimal cells/<cell_id>.json to disk."""
    cells_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id,
        "dataset": "test",
        "encoder": "enc",
        "parent_id": "node_0001",
        "started_at": started_at,
        "budget_seconds": budget_seconds,
        "safety_buffer_seconds": safety_buffer_seconds,
        "status": status,
    }
    (cells_dir / f"{cell_id}.json").write_text(json.dumps(payload, indent=2))


@dataclass
class _FakeRunningExp:
    """Minimal RunningExperiment stand-in for tests that don't need a real process."""
    id: str
    spec: dict
    gpu: int = 0
    process: Any = None
    log_file: Any = None
    log_path: Path = None
    started_at: float = 0.0
    timeout_at: float = 1e18
    estimated_vram_gb: float = 0.5


# ---------------------------------------------------------------------------
# Test 1: ACTIVE → REFUSING_NEW
# ---------------------------------------------------------------------------

def test_tick_cells_active_to_refusing_new(tmp_path: Path) -> None:
    """ACTIVE cell with < safety_buffer remaining transitions to refusing-new.

    No running experiments → backend.cancel must NOT be called.
    """
    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"

    cell_id = "aabbccdd11223344"
    # Place cell in ACTIVE with ~1700s remaining (< 1800s buffer → should flip)
    started_at = time.time() - (21600 - 1700)
    _write_cell_json(cells_dir, cell_id, "active", started_at)

    mock_backend = MagicMock()
    orch.backend = mock_backend
    orch.running = {}

    orch._tick_cells()

    # Re-read cell
    cell_json = json.loads((cells_dir / f"{cell_id}.json").read_text())
    assert cell_json["status"] == "refusing-new", (
        f"Expected refusing-new, got {cell_json['status']!r}"
    )
    mock_backend.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: REFUSING_NEW → TERMINATING (cancel order-of-operation)
# ---------------------------------------------------------------------------

def test_tick_cells_terminating_fires_cancel_with_cap_reason(tmp_path: Path) -> None:
    """REFUSING_NEW cell with expired budget → TERMINATING.

    Verifies:
    - running/<node>.json has metadata.cancel_reason='cap' BEFORE backend.cancel() fires
    - backend.cancel() is called exactly once with the handle
    - cells/<cell_id>.json transitions to 'terminating'
    """
    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"
    running_dir = orch.running_dir

    cell_id = "11223344aabbccdd"
    node_id = "node_0099"

    # Cell well past its budget (22000s > 21600s budget)
    started_at = time.time() - 22000
    _write_cell_json(cells_dir, cell_id, "refusing-new", started_at)

    # Write running/<node>.json with matching cell_id
    running_spec = {
        "id": node_id,
        "description": "test run",
        "metadata": {"cell_id": cell_id},
    }
    (running_dir / f"{node_id}.json").write_text(json.dumps(running_spec, indent=2))

    # Inject a fake RunningExperiment
    orch.running = {
        node_id: _FakeRunningExp(
            id=node_id,
            spec={"metadata": {"cell_id": cell_id}},
        )
    }

    # Track cancel_reason state AT the moment cancel() is called
    cancel_reason_at_call_time: list[str | None] = []

    def _cancel_side_effect(handle, signal=None):
        running_spec_path = running_dir / f"{handle.node_id}.json"
        try:
            data = json.loads(running_spec_path.read_text())
            cancel_reason_at_call_time.append(
                data.get("metadata", {}).get("cancel_reason")
            )
        except (json.JSONDecodeError, OSError):
            cancel_reason_at_call_time.append(None)

    mock_backend = MagicMock()
    mock_backend.cancel.side_effect = _cancel_side_effect
    orch.backend = mock_backend

    orch._tick_cells()

    # 1. Cell transitioned to terminating
    cell_json = json.loads((cells_dir / f"{cell_id}.json").read_text())
    assert cell_json["status"] == "terminating", (
        f"Expected terminating, got {cell_json['status']!r}"
    )

    # 2. backend.cancel called once
    assert mock_backend.cancel.call_count == 1

    # 3. cancel_reason was ALREADY 'cap' when cancel() fired (Pitfall 4)
    assert cancel_reason_at_call_time == ["cap"], (
        f"cancel_reason at cancel time: {cancel_reason_at_call_time!r}"
    )

    # 4. Disk annotation also present after the tick
    updated_spec = json.loads((running_dir / f"{node_id}.json").read_text())
    assert updated_spec["metadata"]["cancel_reason"] == "cap"


# ---------------------------------------------------------------------------
# Test 3: TERMINATING → FINALIZED when no running experiments
# ---------------------------------------------------------------------------

def test_tick_cells_finalized_when_running_empty(tmp_path: Path) -> None:
    """TERMINATING cell with no in-cell running experiments → FINALIZED."""
    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"

    cell_id = "55667788aabbccdd"
    started_at = time.time() - 25000
    _write_cell_json(cells_dir, cell_id, "terminating", started_at)

    mock_backend = MagicMock()
    orch.backend = mock_backend
    orch.running = {}

    orch._tick_cells()

    cell_json = json.loads((cells_dir / f"{cell_id}.json").read_text())
    assert cell_json["status"] == "finalized"
    mock_backend.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: FINALIZED stays FINALIZED (idempotency)
# ---------------------------------------------------------------------------

def test_tick_cells_idempotent_on_finalized(tmp_path: Path) -> None:
    """FINALIZED cell remains FINALIZED after two _tick_cells calls; no cancel."""
    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"

    cell_id = "ff00ff00ff00ff00"
    started_at = time.time() - 30000
    _write_cell_json(cells_dir, cell_id, "finalized", started_at)

    mock_backend = MagicMock()
    orch.backend = mock_backend
    orch.running = {}

    orch._tick_cells()
    orch._tick_cells()

    cell_json = json.loads((cells_dir / f"{cell_id}.json").read_text())
    assert cell_json["status"] == "finalized"
    mock_backend.cancel.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: _running_in_cell filters by metadata.cell_id
# ---------------------------------------------------------------------------

def test_running_in_cell_filters_by_metadata_cell_id(tmp_path: Path) -> None:
    """_running_in_cell returns only handles whose spec.metadata.cell_id matches."""
    orch = _make_orch(tmp_path)

    orch.running = {
        "node_001": _FakeRunningExp(id="node_001", spec={"metadata": {"cell_id": "abc"}}),
        "node_002": _FakeRunningExp(id="node_002", spec={"metadata": {"cell_id": "abc"}}),
        "node_003": _FakeRunningExp(id="node_003", spec={"metadata": {"cell_id": "xyz"}}),
        "node_004": _FakeRunningExp(id="node_004", spec={"metadata": {}}),  # no cell_id
    }

    handles = orch._running_in_cell("abc")
    node_ids = {h.node_id for h in handles}
    assert node_ids == {"node_001", "node_002"}, f"Got: {node_ids}"

    handles_xyz = orch._running_in_cell("xyz")
    assert {h.node_id for h in handles_xyz} == {"node_003"}

    handles_none = orch._running_in_cell("nonexistent")
    assert handles_none == []


# ---------------------------------------------------------------------------
# Test 6: _handle_completion cap-cancel with partial folds → executed
# ---------------------------------------------------------------------------

def test_handle_completion_with_cap_cancel_reason_calls_reconcile(tmp_path: Path) -> None:
    """cap-killed node with 2 completed folds reconciles to executed + budget_killed=True."""
    from automil.graph import ExperimentGraph

    orch = _make_orch(tmp_path)

    node_id = "node_cap01"
    archive_dir = orch.archive_dir
    running_dir = orch.running_dir
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True, exist_ok=True)

    # Write running/<node>.json with cancel_reason='cap'
    spec_data = {
        "id": node_id,
        "description": "cap test",
        "metadata": {"cell_id": "testcell", "cancel_reason": "cap"},
        "env": {"AUTOMIL_FOLD_COUNT": "5"},
    }
    (running_dir / f"{node_id}.json").write_text(json.dumps(spec_data, indent=2))
    # Also write archive/<node>/spec.json (for fallback path)
    (node_archive / "spec.json").write_text(json.dumps(spec_data, indent=2))

    # Write 2 fold files (partial 2-of-5)
    for i in range(2):
        fold_data = {
            "fold_index": i,
            "fold_count": 5,
            "status": "completed",
            "metrics": {"val_auc": 0.85, "val_bacc": 0.80, "test_auc": 0.84, "test_bacc": 0.79},
            "composite": 0.82,
            "elapsed_seconds": 400,
            "peak_vram_mb": 4500,
        }
        (node_archive / f"fold_{i}_result.json").write_text(json.dumps(fold_data, indent=2))

    # Set up a real graph with the node in running state
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=graph_path)
    # Add a proposed node then mark it running to simulate daemon state
    parent_nid = graph.add_proposed(
        parent_id=None, description="root", techniques=["baseline"],
    )
    proposed_nid = graph.add_proposed(
        parent_id=parent_nid, description="cap test node", techniques=["test_tech"],
    )
    # Override node_id to match our test node_id (add_proposed generates IDs)
    # We need to directly inject the node under node_id
    graph._data["nodes"][node_id] = {
        "id": node_id,
        "parent_id": None,
        "type": "running",
        "status": "running",
        "description": "cap test",
        "techniques": [],
        "composite": 0.0,
        "test_auc": 0.0,
        "test_bacc": 0.0,
        "val_auc": 0.0,
        "val_bacc": 0.0,
    }
    graph.save()
    orch.graph = graph

    # Build a fake RunningExperiment and a fake process (already exited)
    import subprocess
    mock_process = MagicMock()
    mock_process.pid = 99999
    mock_log = MagicMock()

    from automil.backends._orchestrator_daemon import RunningExperiment
    orch.running[node_id] = RunningExperiment(
        id=node_id,
        spec=spec_data,
        gpu=0,
        process=mock_process,
        log_file=mock_log,
        log_path=node_archive / "run.log",
        started_at=time.time() - 1000,
        timeout_at=time.time() + 3600,
        estimated_vram_gb=0.5,
    )
    orch.gpu_allocations[0] = [node_id]

    # Stub runner.collect_result and runner.cleanup_worktree
    orch.runner = MagicMock()
    orch.runner.collect_result.return_value = None
    orch.runner.worktree_path.return_value = tmp_path / "worktrees" / node_id
    orch.runner.cleanup_worktree.return_value = None

    # Call _handle_completion with returncode 0 (SIGTERM handler exited cleanly)
    orch._handle_completion(node_id, returncode=0)

    # 1. archive/<node>/result.json exists with status=partial, partial_folds=2
    result_path = node_archive / "result.json"
    assert result_path.exists(), "result.json should exist after reconcile"
    result = json.loads(result_path.read_text())
    assert result["status"] in ("partial", "completed"), f"Got status: {result['status']}"
    assert result["partial_folds"] == 2
    assert result["metadata"]["budget_killed"] is True

    # 2. Graph node promoted to executed with budget_killed=True
    reload_graph = ExperimentGraph(path=graph_path)
    gnode = reload_graph.get_node(node_id)
    assert gnode is not None, "Graph node should exist after reconcile"
    assert gnode["type"] == "executed", f"Got type: {gnode.get('type')}"
    assert gnode.get("metadata", {}).get("budget_killed") is True, (
        f"budget_killed missing: {gnode.get('metadata')}"
    )
    assert gnode["composite"] > 0, f"Composite should be positive, got {gnode['composite']}"


# ---------------------------------------------------------------------------
# Test 7: _handle_completion cap-cancel zero folds → crash
# ---------------------------------------------------------------------------

def test_handle_completion_with_cap_cancel_zero_folds_marks_crash(tmp_path: Path) -> None:
    """cap-killed node with 0 completed folds reconciles to crash + budget_killed=True."""
    from automil.graph import ExperimentGraph

    orch = _make_orch(tmp_path)

    node_id = "node_cap02"
    archive_dir = orch.archive_dir
    running_dir = orch.running_dir
    node_archive = archive_dir / node_id
    node_archive.mkdir(parents=True, exist_ok=True)

    # Write spec with cancel_reason='cap' (no fold files)
    spec_data = {
        "id": node_id,
        "description": "zero folds test",
        "metadata": {"cell_id": "testcell2", "cancel_reason": "cap"},
        "env": {"AUTOMIL_FOLD_COUNT": "5"},
    }
    (running_dir / f"{node_id}.json").write_text(json.dumps(spec_data, indent=2))
    (node_archive / "spec.json").write_text(json.dumps(spec_data, indent=2))
    (node_archive / "run.log").write_text("Process killed\n")

    # Set up graph node
    graph_path = tmp_path / "automil" / "graph.json"
    graph = ExperimentGraph(path=graph_path)
    graph._data["nodes"][node_id] = {
        "id": node_id,
        "parent_id": None,
        "type": "running",
        "status": "running",
        "description": "zero folds test",
        "techniques": [],
        "composite": 0.0,
        "test_auc": 0.0,
        "test_bacc": 0.0,
        "val_auc": 0.0,
        "val_bacc": 0.0,
    }
    graph.save()
    orch.graph = graph

    import subprocess
    mock_process = MagicMock()
    mock_log = MagicMock()

    from automil.backends._orchestrator_daemon import RunningExperiment
    orch.running[node_id] = RunningExperiment(
        id=node_id,
        spec=spec_data,
        gpu=0,
        process=mock_process,
        log_file=mock_log,
        log_path=node_archive / "run.log",
        started_at=time.time() - 100,
        timeout_at=time.time() + 3600,
        estimated_vram_gb=0.5,
    )
    orch.gpu_allocations[0] = [node_id]
    orch.runner = MagicMock()
    orch.runner.collect_result.return_value = None
    orch.runner.worktree_path.return_value = tmp_path / "worktrees" / node_id
    orch.runner.cleanup_worktree.return_value = None

    orch._handle_completion(node_id, returncode=-15)

    # 1. result.json with status=crash (D-06: canonical value, was 'crashed' pre-v1.1)
    result_path = node_archive / "result.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text())
    assert result["status"] == "crash", f"Got status: {result['status']}"
    assert result["metadata"]["budget_killed"] is True

    # 2. Graph node status == crash
    reload_graph = ExperimentGraph(path=graph_path)
    gnode = reload_graph.get_node(node_id)
    assert gnode is not None
    assert gnode["status"] == "crash", f"Got status: {gnode.get('status')}"
    assert gnode.get("metadata", {}).get("budget_killed") is True


# ---------------------------------------------------------------------------
# Test 8: AUTOMIL_FOLD_COUNT injected into subprocess env
# ---------------------------------------------------------------------------

def test_automil_fold_count_injected_into_subprocess_env(tmp_path: Path) -> None:
    """_build_subprocess_env includes AUTOMIL_FOLD_COUNT from training.fold_count config."""
    orch = _make_orch(
        tmp_path,
        config_yaml="orchestrator: {}\ntraining:\n  fold_count: 7\n",
    )

    env = orch._build_subprocess_env(
        gpu_id=0,
        node_id="node_0001",
        archive=tmp_path / "archive" / "node_0001",
        spec={"description": "fold count test", "env": {}},
    )

    assert "AUTOMIL_FOLD_COUNT" in env, "AUTOMIL_FOLD_COUNT must be injected into subprocess env"
    assert env["AUTOMIL_FOLD_COUNT"] == "7", (
        f"Expected '7' from config, got {env['AUTOMIL_FOLD_COUNT']!r}"
    )


# ---------------------------------------------------------------------------
# Test 9/10: activity-gated budget accrual through _tick_cells (P2.2)
# ---------------------------------------------------------------------------

def _write_agent_active_cell(cells_dir: Path, cell_id: str, now: float,
                             consumed: float, last_tick_offset: float) -> None:
    cells_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_id": cell_id, "dataset": "test", "encoder": "enc", "parent_id": "root",
        "started_at": now - 100, "budget_seconds": 21600, "safety_buffer_seconds": 1800,
        "status": "active", "mode": "agent_active", "idle_grace_seconds": 300,
        "consumed_active_seconds": consumed, "last_tick_at": now - last_tick_offset,
    }
    (cells_dir / f"{cell_id}.json").write_text(json.dumps(payload, indent=2))


def test_tick_cells_agent_active_accrues_while_agent_acting(tmp_path: Path, monkeypatch) -> None:
    """agent_active cell: a fresh .last_action_at → consumed_active_seconds advances."""
    import automil.backends._orchestrator_daemon as dm
    FIXED_NOW = 1_000_000.0
    monkeypatch.setattr(dm.time, "time", lambda: FIXED_NOW)

    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"
    _write_agent_active_cell(cells_dir, "active0000000001", FIXED_NOW,
                             consumed=10.0, last_tick_offset=5.0)
    # Agent acted 1s ago (< idle_grace 300) → active.
    (tmp_path / "automil" / ".last_action_at").write_text(str(FIXED_NOW - 1))

    orch.backend = MagicMock()
    orch.running = {}
    orch._tick_cells()

    updated = json.loads((cells_dir / "active0000000001.json").read_text())
    assert updated["consumed_active_seconds"] == 15.0  # 10 + 5s elapsed
    assert updated["last_tick_at"] == FIXED_NOW
    assert updated["status"] == "active"


def test_tick_cells_agent_active_frozen_while_waiting(tmp_path: Path, monkeypatch) -> None:
    """agent_active cell: no recent activity → budget clock frozen (agent waiting)."""
    import automil.backends._orchestrator_daemon as dm
    FIXED_NOW = 1_000_000.0
    monkeypatch.setattr(dm.time, "time", lambda: FIXED_NOW)

    orch = _make_orch(tmp_path)
    cells_dir = tmp_path / "automil" / "cells"
    _write_agent_active_cell(cells_dir, "idle000000000001", FIXED_NOW,
                             consumed=10.0, last_tick_offset=5.0)
    # Stale marker: agent's last action was 1h ago (>> idle_grace) → waiting.
    (tmp_path / "automil" / ".last_action_at").write_text(str(FIXED_NOW - 3600))

    orch.backend = MagicMock()
    orch.running = {}
    orch._tick_cells()

    updated = json.loads((cells_dir / "idle000000000001.json").read_text())
    assert updated["consumed_active_seconds"] == 10.0  # unchanged — clock paused
    assert updated["last_tick_at"] == FIXED_NOW         # tick still advanced


# ---------------------------------------------------------------------------
# Test 11: CR-01 regression — daemon supplies its own graph (no test injection)
# ---------------------------------------------------------------------------

def test_handle_completion_daemon_supplies_own_graph(tmp_path: Path) -> None:
    """CR-01 regression: _handle_completion must write all four terminal artifacts
    WITHOUT any external orch.graph injection.

    Before the CR-01 fix, self.graph was never assigned in __init__ so every
    real daemon completion raised AttributeError. Tests masked this by setting
    orch.graph = graph externally. This test proves the daemon wires its own
    graph: it creates an ExperimentOrchestrator, deliberately does NOT set
    orch.graph, calls _handle_completion, and asserts that graph.json +
    completed/<node>.json + archive result.json are all written.
    """
    from automil.graph import ExperimentGraph

    orch = _make_orch(tmp_path)
    # CR-01 regression guard: confirm __init__ assigned self.graph without test help.
    assert hasattr(orch, "graph"), (
        "CR-01 regression: ExperimentOrchestrator.__init__ must assign self.graph. "
        "Do not gate this on hasattr — the daemon must supply its own graph."
    )
    assert isinstance(orch.graph, ExperimentGraph), (
        f"self.graph must be an ExperimentGraph instance, got {type(orch.graph)}"
    )
    # DO NOT set orch.graph here — that is what we are testing.

    node_id = "node_cr01"
    node_archive = orch.archive_dir / node_id
    node_archive.mkdir(parents=True, exist_ok=True)

    # Write a minimal graph.json with a pending node so terminal_writer can update it.
    graph_path = tmp_path / "automil" / "graph.json"
    init_graph = ExperimentGraph(path=graph_path, technique_map=None)
    init_graph.nodes[node_id] = {
        "id": node_id,
        "type": "running",
        "status": "running",
        "description": "cr01 test",
        "techniques": [],
        "composite": 0.0,
    }
    init_graph.save()

    # Stub runner so no real worktrees are touched.
    result_payload = {
        "status": "completed",
        "composite": 0.82,
        "metrics": {"val_auc": 0.82},
        "elapsed_seconds": 100,
    }
    mock_process = MagicMock()
    mock_process.pid = 99998
    mock_log = MagicMock()
    from automil.backends._orchestrator_daemon import RunningExperiment
    orch.running[node_id] = RunningExperiment(
        id=node_id,
        spec={"id": node_id, "description": "cr01 test"},
        gpu=0,
        process=mock_process,
        log_file=mock_log,
        log_path=node_archive / "run.log",
        started_at=time.time() - 100,
        timeout_at=time.time() + 3600,
        estimated_vram_gb=0.5,
    )
    orch.gpu_allocations[0] = [node_id]
    orch.runner = MagicMock()
    orch.runner.collect_result.return_value = result_payload
    orch.runner.worktree_path.return_value = tmp_path / "worktrees" / node_id
    orch.runner.cleanup_worktree.return_value = None

    # Invoke _handle_completion WITHOUT injecting orch.graph.
    orch._handle_completion(node_id, returncode=0)

    # 1. completed/<node>.json must exist.
    completed_json = orch.completed_dir / f"{node_id}.json"
    assert completed_json.exists(), (
        "CR-01: completed/<node>.json not written — terminal_writer did not run. "
        "self.graph was likely missing (AttributeError swallowed by tick except)."
    )

    # 2. archive result.json must be valid JSON with correct composite.
    result_path = node_archive / "result.json"
    assert result_path.exists(), "CR-01: archive result.json not written by terminal_writer"
    result_written = json.loads(result_path.read_text())
    assert result_written.get("composite") == 0.82

    # 3. graph.json must reflect the node as executed (status keep or discard).
    reloaded = ExperimentGraph(path=graph_path, technique_map=None)
    gnode = reloaded.get_node(node_id)
    assert gnode is not None, "CR-01: graph node not found after _handle_completion"
    assert gnode.get("type") == "executed", (
        f"CR-01: graph node type should be 'executed', got {gnode.get('type')!r}. "
        "terminal_writer did not update the graph."
    )
    assert gnode.get("status") in ("keep", "discard"), (
        f"CR-01: graph status should be keep/discard, got {gnode.get('status')!r}."
    )


# ---------------------------------------------------------------------------
# Test 12: WR-02 regression — _was_cap_killed_completion uses backend-aware path
# ---------------------------------------------------------------------------

def test_was_cap_killed_completion_detects_slurm_annotation(tmp_path: Path) -> None:
    """WR-02 regression: _was_cap_killed_completion must find cancel_reason='cap'
    annotations written to running/slurm/<node>.json (non-local backends).

    Before the WR-02 fix the method read self.running_dir / f"{node_id}.json"
    (= running/local/<node>.json), so SLURM/Ray annotations were never found
    and _handle_cap_killed_completion was never called for those backends.

    This test writes the annotation to running/slurm/<node>.json (as _tick_cells
    does when metadata.backend='slurm') and asserts that
    _was_cap_killed_completion returns True.
    """
    orch = _make_orch(tmp_path)

    node_id = "node_slurm01"
    # Write the node spec with metadata.backend='slurm' into running/slurm/
    slurm_running_dir = orch.running_root / "slurm"
    slurm_running_dir.mkdir(parents=True, exist_ok=True)
    spec_data = {
        "id": node_id,
        "description": "slurm cap test",
        "metadata": {
            "backend": "slurm",
            "cell_id": "testcell_slurm",
            "cancel_reason": "cap",
        },
    }
    (slurm_running_dir / f"{node_id}.json").write_text(json.dumps(spec_data, indent=2))

    # Pre-condition: running/local/<node>.json must NOT exist so we know
    # the method is reading from the backend-aware path, not the local alias.
    local_path = orch.running_dir / f"{node_id}.json"
    assert not local_path.exists(), (
        "Test setup error: running/local/<node>.json must not exist for this assertion"
    )

    result = orch._was_cap_killed_completion(node_id)
    assert result is True, (
        "WR-02 regression: _was_cap_killed_completion must return True for a "
        "SLURM node with cancel_reason='cap' in running/slurm/<node>.json. "
        "If False, the method is still reading from running/local/ (self.running_dir)."
    )


def test_was_cap_killed_completion_local_still_works(tmp_path: Path) -> None:
    """WR-02 regression guard: local-backend cap annotation still detected after fix.

    Ensures the backend-aware path resolves correctly for 'local' nodes (the
    common case) and doesn't regress existing behaviour.
    """
    orch = _make_orch(tmp_path)

    node_id = "node_local_cap"
    spec_data = {
        "id": node_id,
        "description": "local cap test",
        "metadata": {
            "backend": "local",
            "cell_id": "testcell_local",
            "cancel_reason": "cap",
        },
    }
    # Write to running/local/ — the standard local-backend location
    (orch.running_dir / f"{node_id}.json").write_text(json.dumps(spec_data, indent=2))

    result = orch._was_cap_killed_completion(node_id)
    assert result is True, (
        "WR-02 regression: local-backend cap annotation must still be detected "
        "after the backend-aware path fix."
    )
