"""The launch path must honour the cell cap (CAP-1).

``_launch()`` and ``_get_pending()`` never consulted cell status, so a spec that
was already sitting in ``orchestrator/queue/`` when its cell flipped to
REFUSING_NEW still launched: only ``automil submit`` was gated. The cap was
therefore enforced at the front door and nowhere else, and any batch of queued
work outlived the budget that was supposed to bound it.

Refusal semantics chosen here mirror ``automil dequeue`` (the existing "withdraw
a queued spec" path): remove ``queue/<node>.json`` and ``graph.cancel()`` the
node. Leaving the spec queued is NOT viable — the cap state machine is monotone,
so a closed cell never re-opens, and the spec would block its node id forever
(submit refuses a duplicate queue entry, and children refuse a still-running
parent).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from automil.cells.state import Cell, CellStatus, write_cell
from automil.graph import ExperimentGraph

CELL_ID = "capcell000000001"
NODE_ID = "node_0007"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_orch(tmp_path: Path) -> Any:
    from automil.orchestrator import ExperimentOrchestrator

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir(exist_ok=True)
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    orch.running_dir.mkdir(parents=True, exist_ok=True)
    orch.queue_dir.mkdir(parents=True, exist_ok=True)
    return orch


def _write_cell(automil_dir: Path, *, status: CellStatus = CellStatus.ACTIVE,
                eval_budget: int | None = None, consumed_evals: int = 0,
                mode: str = "wall_clock", started_at: float | None = None,
                billed_node_ids: list[str] | None = None) -> Cell:
    cell = Cell(
        cell_id=CELL_ID,
        dataset="ds",
        encoder="enc",
        mil_model="clam sb",
        started_at=time.time() - 60 if started_at is None else started_at,
        budget_seconds=21600,
        safety_buffer_seconds=1800,
        status=status,
        mode=mode,
        eval_budget=eval_budget,
        consumed_evals=consumed_evals,
        billed_node_ids=list(billed_node_ids or []),
    )
    write_cell(cell, automil_dir / "cells")
    return cell


def _write_queued_spec(orch: Any, *, cell_id: str | None = CELL_ID,
                       node_id: str = NODE_ID) -> dict:
    metadata: dict = {"backend": "local"}
    if cell_id is not None:
        metadata["cell_id"] = cell_id
    spec = {
        "id": node_id,
        "description": "queued before the cell closed",
        "base_commit": "deadbeef",
        "overlay_dir": f"archive/{node_id}",
        "overlay_manifest": {},
        "deletions": [],
        "priority": 1,
        "estimated_vram_gb": 0.5,
        "metadata": metadata,
    }
    path = orch.queue_dir / f"{node_id}.json"
    path.write_text(json.dumps(spec, indent=2))
    return {**spec, "_file": path}


def _seed_graph(orch: Any, node_id: str = NODE_ID) -> ExperimentGraph:
    graph = ExperimentGraph(path=orch.automil_dir / "graph.json")
    graph.nodes[node_id] = {
        "id": node_id,
        "parent_id": None,
        "type": "proposed",
        "status": "running",
        "description": "queued before the cell closed",
        "techniques": [],
        "cell_id": CELL_ID,
    }
    graph.meta["total_proposed"] = 1
    graph.save()
    orch.graph = graph
    return graph


def _stub_runner(orch: Any, tmp_path: Path) -> None:
    wt = tmp_path / "wt" / NODE_ID
    wt.mkdir(parents=True, exist_ok=True)
    orch.runner = MagicMock()
    orch.runner.create_worktree.return_value = wt
    orch.runner.apply_overlay.return_value = None
    orch.runner.worktree_path.return_value = wt
    orch.runner.cleanup_worktree.return_value = None


class _FakePopen:
    """Records every launch attempt; stands in for a live training process."""

    calls: list = []
    pid = 4242

    def __init__(self, cmd, **kwargs):
        type(self).calls.append(cmd)

    def poll(self):
        return None


@pytest.fixture(autouse=True)
def _reset_popen_calls():
    _FakePopen.calls = []
    yield
    _FakePopen.calls = []


# ---------------------------------------------------------------------------
# The launch gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [CellStatus.REFUSING_NEW, CellStatus.TERMINATING, CellStatus.FINALIZED],
)
def test_launch_refuses_a_spec_whose_cell_has_closed(tmp_path, status):
    """CAP-1 reproducer: a queued spec must not launch into a closed cell."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=status)
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    assert _FakePopen.calls == [], (
        f"CAP-1: launched into a {status.value} cell — the cap is enforced at "
        f"submit only, so any already-queued spec outlives the budget."
    )
    assert NODE_ID not in orch.running
    assert not (orch.queue_dir / f"{NODE_ID}.json").exists(), (
        "a refused spec must be dequeued: a closed cell never re-opens, so "
        "leaving it queued strands the node id forever"
    )


def test_launch_proceeds_for_an_active_cell(tmp_path):
    """Regression guard: the gate must not block ordinary launches."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.ACTIVE)
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    assert len(_FakePopen.calls) == 1, "an ACTIVE cell must still launch"
    assert NODE_ID in orch.running


def test_launch_refuses_when_the_eval_budget_is_spent_but_status_is_stale(tmp_path):
    """Mid-tick overshoot: N queued specs must not all launch on 1 remaining eval.

    ``consumed_evals`` advances at launch, but ``status`` only advances on the
    next ``_tick_cells``. A status-only gate would admit every spec already in
    the loop's pending list.
    """
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.ACTIVE, eval_budget=3,
                consumed_evals=3)
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    assert _FakePopen.calls == [], (
        "the launch gate must read consumed_evals, not only the tick-lagged status"
    )


def test_launch_holds_agent_active_spec_during_outage_then_recovers(tmp_path):
    """Telemetry degradation is recoverable: keep the queue entry and retry."""
    from automil.cells.activity import (
        ActivityObservation,
        ingest_prometheus_metrics,
        record_hook_event,
    )

    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, mode="agent_active")
    record_hook_event(
        orch.automil_dir,
        CELL_ID,
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
    )
    ingest_prometheus_metrics(
        orch.automil_dir,
        'claude_code_active_time_total{session_id="session-1",type="cli"} 5\n',
    )
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)
    unavailable = ActivityObservation(
        available=False,
        sessions=(),
        observed_at=time.time(),
        error="endpoint unavailable",
    )

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0, activity_observation=unavailable)

    assert _FakePopen.calls == []
    assert (orch.queue_dir / f"{NODE_ID}.json").exists()
    assert ExperimentGraph(path=orch.automil_dir / "graph.json").get_node(
        NODE_ID
    )["status"] == "running"

    recovered = ActivityObservation(
        available=True,
        sessions=("session-1",),
        observed_at=time.time(),
        error=None,
    )
    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0, activity_observation=recovered)

    assert len(_FakePopen.calls) == 1
    assert NODE_ID in orch.running


def test_refusal_cancels_the_graph_node_and_archives_the_spec(tmp_path):
    """Refusal follows `automil dequeue` semantics: cancelled, not crashed."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.REFUSING_NEW)
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    reloaded = ExperimentGraph(path=orch.automil_dir / "graph.json")
    node = reloaded.get_node(NODE_ID)
    assert node["status"] == "cancelled", (
        f"expected cancelled (nothing ran), got {node['status']!r}"
    )
    assert node.get("cancel_reason") == "cap"
    assert reloaded.meta["total_proposed"] == 0, "the proposal counter must not drift"

    archived = json.loads((orch.archive_dir / NODE_ID / "spec.json").read_text())
    assert archived["metadata"]["cancel_reason"] == "cap"

    # A completed/ record would make `reconcile` promote this to an executed
    # node with composite 0.0 — it never ran, so there must not be one.
    assert not (orch.completed_dir / f"{NODE_ID}.json").exists()


def test_refusal_does_not_bill_an_evaluation(tmp_path):
    """A refused spec never dispatched, so it must not consume eval budget."""
    from automil.cells.state import read_cell

    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.REFUSING_NEW, eval_budget=10,
                consumed_evals=4)
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    cell = read_cell(orch.automil_dir / "cells" / f"{CELL_ID}.json")
    assert cell.consumed_evals == 4


def test_tick_refuses_closed_cell_specs_even_with_no_gpu_free(tmp_path):
    """The refusal must not be gated on GPU availability.

    ``tick()`` only reaches ``_launch`` once a GPU is allocated. If the gate
    lived solely there, a closed-cell spec would sit in the queue for as long as
    the cluster was busy, and the daemon would re-evaluate it every poll forever.
    """
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.REFUSING_NEW)
    _seed_graph(orch)
    _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)
    orch._find_best_gpu = MagicMock(return_value=None)  # cluster fully busy

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch.tick()

    assert not (orch.queue_dir / f"{NODE_ID}.json").exists()
    assert NODE_ID not in orch.running
    orch.runner.create_worktree.assert_not_called()
    orch._find_best_gpu.assert_not_called()


# ---------------------------------------------------------------------------
# A9 billed retries through closed cells (canary recovery, 2026-08-15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [CellStatus.REFUSING_NEW, CellStatus.TERMINATING, CellStatus.FINALIZED],
)
def test_billed_node_relaunches_through_any_eval_closed_state(tmp_path, status):
    """A billed node is paid-for work being completed, not new work.

    Eval-axis closure (REFUSING_NEW draining to TERMINATING/FINALIZED with
    running=0) must complete a billed retry, never cancel it: refusal stamps
    ``cap_refused`` onto a BILLED archived spec, leaving the freeze census
    permanently short of ``consumed_evals``. Observed live 2026-08-15: all 20
    billed promotion retries of the canary recovery were refused by
    terminating/finalized cells whose 7-day wall_clock budgets had days of
    headroom left.
    """
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=status, eval_budget=10,
                consumed_evals=10, billed_node_ids=[NODE_ID])
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    assert len(_FakePopen.calls) == 1, (
        f"a billed retry must relaunch through a {status.value} cell"
    )
    assert NODE_ID in orch.running
    archived = orch.archive_dir / NODE_ID / "spec.json"
    if archived.exists():
        assert not (json.loads(archived.read_text()).get("metadata") or {}).get(
            "cap_refused"
        ), "a billed spec must never carry a cap_refused stamp"


def test_billed_node_is_still_refused_by_an_expired_wall_clock(tmp_path):
    """The one wall that may refuse paid work: a genuinely expired wall_clock
    budget — the only axis denominated in the unit the retry would spend."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.FINALIZED, eval_budget=10,
                consumed_evals=10, billed_node_ids=[NODE_ID],
                started_at=time.time() - 36000)  # budget_seconds=21600: expired
    _seed_graph(orch)
    _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch.tick()

    # tick()'s state save legitimately queries nvidia-smi on GPU hosts; the
    # contract under test is that no TRAINING process is spawned.
    launches = [c for c in _FakePopen.calls if "nvidia-smi" not in str(c[0])]
    assert launches == [], (
        "an expired wall_clock budget is the hard wall: even billed work stays refused"
    )
    assert not (orch.queue_dir / f"{NODE_ID}.json").exists()


def test_billed_retry_does_not_double_bill(tmp_path):
    """Exactly-once: relaunching a billed node must not advance consumed_evals."""
    from automil.cells.state import read_cell

    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.FINALIZED, eval_budget=10,
                consumed_evals=10, billed_node_ids=[NODE_ID])
    _seed_graph(orch)
    spec = _write_queued_spec(orch)
    _stub_runner(orch, tmp_path)

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(spec, gpu_id=0)

    cell = read_cell(orch.automil_dir / "cells" / f"{CELL_ID}.json")
    assert cell.consumed_evals == 10
    assert cell.billed_node_ids.count(NODE_ID) == 1


# ---------------------------------------------------------------------------
# Legacy specs with no cell identity
# ---------------------------------------------------------------------------


def test_spec_without_cell_id_still_launches_but_is_reported(tmp_path, caplog):
    """Legacy / non-CLI specs are un-billable, so they are loud rather than silent.

    Refusing them would make the daemon destroy work submitted through
    ``Backend.submit`` (a first-class path — the gate uses it). Launching them
    silently would let unbilled evaluations dilute the equal-effort claim without
    trace. So: launch, and say so — the absent ``metadata.cell_id`` on the
    archived spec is the durable record.
    """
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, status=CellStatus.REFUSING_NEW)  # unrelated cell
    _seed_graph(orch)
    spec = _write_queued_spec(orch, cell_id=None)
    _stub_runner(orch, tmp_path)

    with caplog.at_level(logging.WARNING, logger="automil.backends._orchestrator_daemon"):
        with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
                   side_effect=_FakePopen):
            orch._launch(spec, gpu_id=0)

    assert len(_FakePopen.calls) == 1, "a cell-less spec must not be silently dropped"
    assert any("cell_id" in r.getMessage() for r in caplog.records), (
        f"expected a warning naming the missing cell_id; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_spec_referencing_an_unknown_cell_is_held_not_cancelled(tmp_path, caplog):
    """A declared but unresolved cell fails closed WITHOUT destroying the spec.

    It never launches unmetered, but it is held rather than refused: a missing
    or unreadable cell file may be transient, and refusal would irreversibly
    unlink the queue spec and cancel the node.
    """
    orch = _make_orch(tmp_path)
    (orch.automil_dir / "cells").mkdir(parents=True, exist_ok=True)
    _seed_graph(orch)
    spec = _write_queued_spec(orch, cell_id="doesnotexist0001")
    _stub_runner(orch, tmp_path)

    with caplog.at_level(logging.WARNING, logger="automil.backends._orchestrator_daemon"):
        with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
                   side_effect=_FakePopen):
            orch._launch(spec, gpu_id=0)

    assert len(_FakePopen.calls) == 0
    assert (orch.queue_dir / f"{NODE_ID}.json").exists(), (
        "a held spec must stay queued for a later tick"
    )
    assert any("doesnotexist0001"[:8] in r.getMessage() for r in caplog.records), (
        f"expected a warning naming the unknown cell; got "
        f"{[r.getMessage() for r in caplog.records]}"
    )
