"""The orchestrator bills evaluations to the cell that dispatched them (H-2).

``consumed_evals`` is the primary equal-effort axis, so the daemon — not the
agent, and not the training script — is its only writer. It is incremented at
LAUNCH: crashed, partial and budget-killed nodes all count, because equal effort
must mean equal ATTEMPTS. ``completed_evals`` is a reported secondary: it counts
only the terminal statuses that produced usable results, so the paper can quote
both attempts and usable results per cell.
"""
from __future__ import annotations

import json
from dataclasses import replace
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from automil.cells.state import Cell, CellStatus, read_cell, write_cell
from automil.graph import ExperimentGraph

CELL_ID = "evalcell00000001"


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


def _write_cell(automil_dir: Path, *, eval_budget: int | None = 10,
                consumed_evals: int = 0, completed_evals: int = 0) -> Cell:
    cell = Cell(
        cell_id=CELL_ID, dataset="ds", encoder="enc", mil_model="clam sb",
        started_at=time.time() - 60, budget_seconds=21600,
        safety_buffer_seconds=1800, status=CellStatus.ACTIVE, mode="wall_clock",
        eval_budget=eval_budget, consumed_evals=consumed_evals,
        completed_evals=completed_evals,
    )
    write_cell(cell, automil_dir / "cells")
    return cell


def _read_cell(orch: Any) -> Cell:
    return read_cell(orch.automil_dir / "cells" / f"{CELL_ID}.json")


def _spec(node_id: str, *, cell_id: str | None = CELL_ID) -> dict:
    metadata: dict = {"backend": "local"}
    if cell_id is not None:
        metadata["cell_id"] = cell_id
    return {
        "id": node_id,
        "description": f"eval {node_id}",
        "base_commit": "deadbeef",
        "overlay_dir": f"archive/{node_id}",
        "overlay_manifest": {},
        "deletions": [],
        "estimated_vram_gb": 0.5,
        "metadata": metadata,
    }


def _stub_runner(orch: Any, tmp_path: Path, node_id: str) -> Path:
    wt = tmp_path / "wt" / node_id
    wt.mkdir(parents=True, exist_ok=True)
    orch.runner = MagicMock()
    orch.runner.create_worktree.return_value = wt
    orch.runner.apply_overlay.return_value = None
    orch.runner.worktree_path.return_value = wt
    orch.runner.cleanup_worktree.return_value = None
    return wt


class _FakePopen:
    pid = 5150

    def __init__(self, cmd, **kwargs):
        pass

    def poll(self):
        return None


def _seed_graph(orch: Any, node_id: str) -> ExperimentGraph:
    graph = ExperimentGraph(path=orch.automil_dir / "graph.json")
    graph.nodes[node_id] = {
        "id": node_id, "parent_id": None, "type": "running", "status": "running",
        "description": f"eval {node_id}", "techniques": [], "composite": 0.0,
        "cell_id": CELL_ID,
    }
    graph.save()
    orch.graph = graph
    return graph


# ---------------------------------------------------------------------------
# consumed_evals — billed at dispatch
# ---------------------------------------------------------------------------


def test_launch_increments_consumed_evals(tmp_path):
    """H-2 reproducer: without this the eval budget can never be reached."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, consumed_evals=2)
    _seed_graph(orch, "node_0001")
    _stub_runner(orch, tmp_path, "node_0001")

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(_spec("node_0001"), gpu_id=0)

    assert _read_cell(orch).consumed_evals == 3


def test_consecutive_launches_accumulate(tmp_path):
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir)
    for node_id in ("node_0001", "node_0002", "node_0003"):
        _seed_graph(orch, node_id)
        _stub_runner(orch, tmp_path, node_id)
        with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
                   side_effect=_FakePopen):
            orch._launch(_spec(node_id), gpu_id=0)

    assert _read_cell(orch).consumed_evals == 3


def test_launch_without_a_cell_id_bills_nothing(tmp_path):
    """A cell-less spec has no budget to charge — it must not crash the launch."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, consumed_evals=1)
    _seed_graph(orch, "node_0001")
    _stub_runner(orch, tmp_path, "node_0001")

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(_spec("node_0001", cell_id=None), gpu_id=0)

    assert "node_0001" in orch.running
    assert _read_cell(orch).consumed_evals == 1


def test_a_pre_spawn_failure_is_billed(tmp_path):
    """A9: archived attempt <=> billed attempt, worktree failure included.

    The campaign freeze requires archived non-cap-refused specs == consumed
    evals exactly; billing only spawned processes let one pre-spawn failure
    (admissibility, base_commit, worktree, Popen) archive an unbilled spec and
    deadlock the cell permanently. Billing happens at archive time now, so the
    two counts agree by construction.
    """
    import subprocess as _sp

    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, consumed_evals=2)
    _seed_graph(orch, "node_0001")
    _stub_runner(orch, tmp_path, "node_0001")
    orch.runner.create_worktree.side_effect = _sp.CalledProcessError(1, "git worktree")

    orch._launch(_spec("node_0001"), gpu_id=0)

    assert _read_cell(orch).consumed_evals == 3
    spec_json = json.loads((orch.archive_dir / "node_0001" / "spec.json").read_text())
    assert not (spec_json.get("metadata") or {}).get("cap_refused"), (
        "the archived spec is a countable attempt, not a cap refusal"
    )


def test_a_cap_refused_spec_stays_unbilled_and_census_excluded(tmp_path):
    """The one legitimate archived-but-unbilled shape: cap refusal (pre-archive gate)."""
    from automil.cells.state import CellStatus as _CS

    orch = _make_orch(tmp_path)
    cell = _write_cell(orch.automil_dir, consumed_evals=2)
    cell = replace(cell, status=_CS.REFUSING_NEW)
    write_cell(cell, orch.automil_dir / "cells")
    _seed_graph(orch, "node_0001")
    _stub_runner(orch, tmp_path, "node_0001")

    orch._launch(_spec("node_0001"), gpu_id=0)

    assert _read_cell(orch).consumed_evals == 2
    spec_json = json.loads((orch.archive_dir / "node_0001" / "spec.json").read_text())
    assert (spec_json.get("metadata") or {}).get("cap_refused") is True


# ---------------------------------------------------------------------------
# completed_evals — reported secondary
# ---------------------------------------------------------------------------


def _complete(orch: Any, tmp_path: Path, node_id: str, result: dict) -> None:
    """Drive a full _handle_completion for a node with a stubbed result."""
    from automil.backends._orchestrator_daemon import RunningExperiment

    node_archive = orch.archive_dir / node_id
    node_archive.mkdir(parents=True, exist_ok=True)
    (node_archive / "run.log").write_text("training log\n")
    _stub_runner(orch, tmp_path, node_id)
    orch.runner.collect_result.return_value = result
    orch.running[node_id] = RunningExperiment(
        id=node_id, spec=_spec(node_id), gpu=0, process=MagicMock(),
        log_file=MagicMock(), log_path=node_archive / "run.log",
        started_at=time.time() - 10, timeout_at=time.time() + 3600,
        estimated_vram_gb=0.5,
    )
    orch.gpu_allocations[0] = [node_id]
    orch._handle_completion(node_id, returncode=0)


@pytest.mark.parametrize(
    ("status", "expected_completed"),
    [
        ("completed", 1),   # usable
        ("partial", 1),     # usable (some folds landed)
        ("crash", 0),       # attempted, not usable
        ("oom", 0),
        ("timeout", 0),
    ],
)
def test_completed_evals_counts_only_usable_results(tmp_path, status, expected_completed):
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, consumed_evals=1)
    _seed_graph(orch, "node_0001")

    _complete(orch, tmp_path, "node_0001", {
        "status": status, "composite": 0.5, "metrics": {"val_auc": 0.5},
        "elapsed_seconds": 10,
    })

    cell = _read_cell(orch)
    assert cell.completed_evals == expected_completed
    assert cell.consumed_evals == 1, (
        "completion must never re-bill the attempt — consumed_evals is a launch counter"
    )


def test_completed_evals_never_gates_the_cap(tmp_path):
    """A cell full of crashes is still exhausted: attempts are what is budgeted."""
    from automil.cells.cap import evals_exhausted

    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, eval_budget=2, consumed_evals=2, completed_evals=0)

    assert evals_exhausted(_read_cell(orch)) is True


def test_budget_killed_partial_counts_as_usable(tmp_path):
    """A cap-killed run that produced folds still yields a usable result."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, consumed_evals=1)
    node_id = "node_cap01"
    _seed_graph(orch, node_id)

    node_archive = orch.archive_dir / node_id
    sealed = node_archive / "certify"
    sealed.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        (sealed / f"fold_{i}_result.json").write_text(json.dumps({
            "fold_index": i, "fold_count": 5, "status": "completed",
            "metrics": {"val_auc": 0.8}, "composite": 0.8,
            "elapsed_seconds": 100, "peak_vram_mb": 1000,
        }))
    spec_data = {**_spec(node_id), "env": {"AUTOMIL_FOLD_COUNT": "5"}}
    spec_data["metadata"] = {**spec_data["metadata"], "cancel_reason": "cap"}
    (orch.running_dir / f"{node_id}.json").write_text(json.dumps(spec_data, indent=2))
    (node_archive / "spec.json").write_text(json.dumps(spec_data, indent=2))

    from automil.backends._orchestrator_daemon import RunningExperiment
    _stub_runner(orch, tmp_path, node_id)
    orch.runner.collect_result.return_value = None
    orch.running[node_id] = RunningExperiment(
        id=node_id, spec=spec_data, gpu=0, process=MagicMock(), log_file=MagicMock(),
        log_path=node_archive / "run.log", started_at=time.time() - 10,
        timeout_at=time.time() + 3600, estimated_vram_gb=0.5,
    )
    orch.gpu_allocations[0] = [node_id]

    orch._handle_completion(node_id, returncode=0)

    cell = _read_cell(orch)
    assert cell.completed_evals == 1, "a budget-killed run with folds is usable"
    assert cell.consumed_evals == 1


# ---------------------------------------------------------------------------
# End-to-end: the counter closes the cell
# ---------------------------------------------------------------------------


def test_exhausting_the_eval_budget_flips_the_cell_to_refusing_new(tmp_path):
    """Launch-count → cap → state machine, through the real daemon tick."""
    orch = _make_orch(tmp_path)
    _write_cell(orch.automil_dir, eval_budget=1, consumed_evals=0)
    _seed_graph(orch, "node_0001")
    _stub_runner(orch, tmp_path, "node_0001")

    with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
               side_effect=_FakePopen):
        orch._launch(_spec("node_0001"), gpu_id=0)

    assert _read_cell(orch).status is CellStatus.ACTIVE, "status lags until the next tick"

    orch._tick_cells()

    assert _read_cell(orch).status is CellStatus.REFUSING_NEW, (
        "an exhausted eval budget must move the cell along the same state machine "
        "the time cap uses"
    )
