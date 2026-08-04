"""An exhausted EVAL budget shrinks K_effective exactly like an exhausted clock (H-2).

``gate/evaluate.py`` skips held-out cells that are refusing new work, and
``gate/promote.py`` subtracts the skipped cells from K. Adding the eval-count
axis therefore reaches into the generalization gate: a cell can now be skipped
because it ran out of EVALUATIONS rather than seconds, which lowers K_effective
and can push a candidate into the ``inconclusive`` branch.

That coupling is deliberate — an eval-exhausted cell has no budget left to pay
for a held-out evaluation, and the gate must not quietly borrow one — so it is
pinned here rather than left to be discovered as a surprise.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from automil.backends.base import JobState
from automil.cells.state import Cell, CellStatus
from tests.gate.test_evaluate import (
    RecordingBackend,
    _make_graph,
    _make_manifest,
)

CELL_A = "cell_aaaa1111"


def _cell(cell_id: str, *, status: CellStatus = CellStatus.ACTIVE,
          eval_budget: int | None = None, consumed_evals: int = 0) -> Cell:
    return Cell(
        cell_id=cell_id, dataset="ds", encoder="enc", mil_model="clam sb",
        started_at=time.time() - 60,          # nowhere near the time wall
        budget_seconds=21600, safety_buffer_seconds=1800,
        status=status, mode="wall_clock",
        eval_budget=eval_budget, consumed_evals=consumed_evals,
    )


def test_eval_exhausted_cell_is_skipped_by_the_gate(monkeypatch):
    """The gate must not spend an evaluation a cell cannot pay for."""
    from automil.gate.evaluate import evaluate_candidate

    exhausted = _cell(CELL_A, status=CellStatus.ACTIVE, eval_budget=5, consumed_evals=5)
    monkeypatch.setattr(
        "automil.gate.evaluate.get_cell",
        lambda cid: exhausted if cid == CELL_A else None,
    )

    backend = RecordingBackend(initial_state=JobState.COMPLETED)
    results, skipped = evaluate_candidate(
        "node_0002", _make_manifest(), backend, _make_graph(),
        poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    assert skipped == [CELL_A], (
        "a cell whose eval budget is spent must be skipped even while its status "
        "still reads ACTIVE (status only advances on the next daemon tick)"
    )
    assert backend.submit_count == 2, "the other two cells must still be evaluated"
    assert len(results) == 2


def test_cell_with_eval_budget_remaining_is_still_evaluated():
    """Regression guard: the new axis must not skip healthy cells."""
    from automil.gate.evaluate import evaluate_candidate

    backend = RecordingBackend(initial_state=JobState.COMPLETED)
    results, skipped = evaluate_candidate(
        "node_0002", _make_manifest(), backend, _make_graph(),
        poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    assert skipped == []
    assert backend.submit_count == 3


def test_eval_exhaustion_can_push_a_candidate_to_inconclusive(tmp_path: Path, monkeypatch):
    """K_effective = K - |skipped|; below K_floor the gate reports inconclusive.

    This is the load-bearing consequence: an eval budget that is too tight
    relative to K does not silently weaken the gate, it stops it from
    concluding — the candidate stays 'candidate' rather than being promoted on
    thinner evidence.
    """
    from automil.gate.manifest import write_manifest
    from automil.gate.promote import promote

    graph = _make_graph()
    graph.path = tmp_path / "graph.json"
    graph.nodes["node_0002"]["status"] = "candidate"
    graph.save()

    manifest = _make_manifest()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    write_manifest(manifest, manifests_dir)

    # Two of the three held-out cells have spent their eval budget.
    exhausted = {
        "cell_aaaa1111": _cell("cell_aaaa1111", eval_budget=5, consumed_evals=5),
        "cell_bbbb2222": _cell("cell_bbbb2222", eval_budget=5, consumed_evals=5),
    }
    monkeypatch.setattr(
        "automil.gate.evaluate.get_cell", lambda cid: exhausted.get(cid),
    )

    promoted = promote(
        "node_0002",
        backend=RecordingBackend(initial_state=JobState.COMPLETED),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=tmp_path / "archive",
        K_floor=2,
    )

    assert promoted is False
    node = graph.nodes["node_0002"]
    assert node["status"] == "candidate", (
        f"expected the candidate to stay unpromoted; got {node['status']!r}"
    )
    gate_events = [h for h in node.get("history", []) if h.get("event") == "gate_result"]
    assert gate_events and gate_events[-1]["result"] == "inconclusive"
    assert gate_events[-1]["K_effective"] == manifest.K - 2
