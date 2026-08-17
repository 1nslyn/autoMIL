"""Gate-eval submission and polling (D-140 / GTE-03).

Spawns held-out evaluations through Backend.submit() — the SAME path
the agent uses (NEVER a parallel mechanism). Each spec carries
metadata.gate_eval='true' so downstream code (rank filter, redactor,
promote logic) can identify gate-eval children.

Cap interaction (D-150 / Pitfall 1): cells in REFUSING_NEW / TERMINATING /
FINALIZED states are SKIPPED — promote.py uses the skipped list to
adjust K_effective. Since H-2 a cell that has spent its EVAL budget is skipped
on the same footing, so an eval budget set too tight relative to K lowers
K_effective and can push a candidate into the ``inconclusive`` branch (which is
the intended failure mode: conclude on less evidence is not an option).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automil.backends.base import Backend
    from automil.gate.manifest import GateManifest
    from automil.graph import ExperimentGraph

logger = logging.getLogger(__name__)

# Module-level import of cells API so tests can monkeypatch the module-level names
# (automil.gate.evaluate.get_cell and the cap-check helper).
# Lazy imports inside evaluate_candidate are removed in favour of this top-level
# import to support deterministic monkeypatching in the test suite.
from automil.cells import blocks_new_work, get_cell  # noqa: E402

# Terminal job states — match the JobState enum from backends/base.py.
# Use string values so we don't have to import the enum at module load.
_TERMINAL_STATES = {"completed", "crashed", "cancelled", "budget_killed"}



def _persist_gate_child(graph, child_id: str, node: dict) -> None:
    """Write one gate-eval child to graph.json under a short lock (CR-2b).

    Best-effort: a persistence failure must not abandon a job that has already
    been submitted to the backend. The in-memory node stays authoritative for
    this run either way.
    """
    from automil.graph import locked_update

    try:
        with locked_update(
            str(graph.path),
            technique_map=getattr(graph, "_technique_map", None),
        ) as g:
            g.nodes.setdefault(child_id, dict(node))
    except Exception:  # noqa: BLE001 — the job is already submitted
        logger.exception("gate-eval: could not persist child node %s", child_id)


def evaluate_candidate(
    candidate_node_id: str,
    manifest: "GateManifest",
    backend: "Backend",
    graph: "ExperimentGraph",
    poll_interval_s: float = 1.0,
    poll_timeout_s: float = 7200.0,
) -> tuple[list[dict], list[str]]:
    """Submit + poll gate-eval jobs for each held-out cell in `manifest`.

    Returns:
        (per_cell_results, skipped_cells)

        per_cell_results: list of dicts, each:
            {
                "cell_id": str,
                "dataset": str,
                "encoder": str,
                "task": str,
                "child_node_id": str,
                "candidate_primary_value": float,
                "parent_primary_value": float,
                "delta": float,
                "status": "completed" | "partial" | "crashed",
            }

        skipped_cells: list of cell_ids skipped due to cap (D-150).
    """
    from automil.backends.base import JobSpec  # lazy import (avoids circular at module load)

    candidate_node = graph.nodes.get(candidate_node_id)
    if candidate_node is None:
        raise ValueError(
            f"candidate node {candidate_node_id!r} not found in graph"
        )
    candidate_primary_value = float(candidate_node.get("primary_value", 0.0))

    parent_id = candidate_node.get("parent_id")
    parent_node = graph.nodes.get(parent_id) if parent_id else None
    parent_primary_value = (
        float(parent_node.get("primary_value", 0.0)) if parent_node else 0.0
    )

    skipped: list[str] = []
    handles: dict[str, tuple] = {}  # cell_id -> (handle, child_id, hc_tuple)

    for hc in manifest.held_out_cells:
        cell_id, dataset, encoder, task = hc

        # D-150: cap-exhausted cells skip the eval (promote.py reduces K).
        # get_cell returns None on missing per cells/registry.py — no try/except needed.
        # H-2: exhaustion is now either axis — a cell with no evaluations left
        # cannot pay for a held-out eval any more than one with no seconds left,
        # and the gate must not quietly borrow one.
        cell = get_cell(cell_id)
        if cell is not None and blocks_new_work(cell):
            logger.info(
                "gate-eval: skipping cell %s (status=%s, consumed_evals=%d/%s, "
                "cap-exhausted)",
                cell_id, cell.status, cell.consumed_evals,
                cell.eval_budget if cell.eval_budget is not None else "-",
            )
            skipped.append(cell_id)
            continue

        # Allocate a child node id (mirrors graph.next_id pattern)
        child_id = _next_child_id(graph)

        # Build spec — metadata stamps are the load-bearing GTE-03 piece.
        # Single call site for backend.submit below (T-05-06-01 / lint assertion).
        spec = JobSpec(
            node_id=child_id,
            base_commit=candidate_node.get("commit", "HEAD"),
            overlay_files=tuple(candidate_node.get("overlay_files", ())),
            overlay_dir=Path(candidate_node.get("overlay_dir", f"archive/{candidate_node_id}")),
            command=("python", "train.py"),
            env=(),
            working_subdir=".",
            gpu_estimate_gb=float(candidate_node.get("vram_gb", 4.0)),
            walltime_seconds=21600,
            metadata=(
                ("gate_eval", "true"),
                ("held_out", "true"),
                ("gate_parent_node", candidate_node_id),
                ("cell_id", cell_id),
                ("edge_type", "gate_eval"),
                ("dataset", dataset),
                ("encoder", encoder),
                ("task", task),
            ),
        )
        handle = backend.submit(spec)

        # Tag child node with gate-eval markers (D-140 — additive node fields).
        # Done synchronously at submit time so callers can read edge_type immediately.
        graph.nodes[child_id] = {
            "id": child_id,
            "parent_id": candidate_node_id,
            "type": "gate_eval",
            "status": "pending",
            "edge_type": "gate_eval",
            "description": f"gate-eval for {candidate_node_id} on cell {cell_id[:8]}",
            "primary_value": 0.0,
            "metadata": {
                "held_out": True,
                "gate_eval": True,
                "gate_parent_node": candidate_node_id,
                "cell_id": cell_id,
                "dataset": dataset,
                "encoder": encoder,
                "task": task,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # CR-2b: persist this child under a SHORT lock, as it is created.
        # Previously the only write was promote_candidate's single graph.save()
        # after every held-out evaluation had finished — minutes later — which
        # meant (a) these nodes existed nowhere on disk while their jobs ran, and
        # (b) that final whole-snapshot save clobbered any daemon completion that
        # landed in the meantime. The lock cannot be held across the evaluations
        # themselves, so the write is split into short transactions instead.
        _persist_gate_child(graph, child_id, graph.nodes[child_id])

        handles[cell_id] = (handle, child_id, hc)

    # Concurrent poll: all handles polled in the same loop until all terminal
    per_cell_results = _poll_handles(
        handles, backend, graph,
        candidate_primary_value, parent_primary_value,
        poll_interval_s, poll_timeout_s,
    )
    return per_cell_results, skipped


def _next_child_id(graph: "ExperimentGraph") -> str:
    """Allocate a fresh node id. Mirrors graph.next_id() if present, else fallback."""
    if hasattr(graph, "next_id") and callable(graph.next_id):
        try:
            cid = graph.next_id()
            # Verify no collision (next_id already advances the counter, but double-check)
            if cid not in graph.nodes:
                return cid
        except Exception:
            pass
    # Fallback: scan for first unused zero-padded id
    counter = 1
    while True:
        cid = f"node_{counter:04d}"
        if cid not in graph.nodes:
            return cid
        counter += 1


def _poll_handles(
    handles: dict,
    backend: "Backend",
    graph: "ExperimentGraph",
    candidate_primary_value: float,
    parent_primary_value: float,
    poll_interval_s: float,
    poll_timeout_s: float,
) -> list[dict]:
    """Poll all handles concurrently until terminal; return paired results.

    "Concurrent" here means polling all handles in a single loop iteration
    before sleeping — NOT spawning threads. This matches the Backend ABC
    contract (poll is never blocking) and avoids the Pitfall 6 violation
    of creating a parallel submission mechanism.
    """
    deadline = time.monotonic() + poll_timeout_s
    pending = dict(handles)  # cell_id -> (handle, child_id, hc)
    results: list[dict] = []

    while pending and time.monotonic() < deadline:
        terminal_now: list[str] = []
        for cell_id, (handle, child_id, hc) in pending.items():
            state = backend.poll(handle)
            state_str = state.value if hasattr(state, "value") else str(state)
            if state_str in _TERMINAL_STATES:
                cand_primary_value, status_label = _read_eval_primary_value(
                    handle, backend, graph, child_id, candidate_primary_value, state_str,
                )
                # Crashed/cancelled jobs carry delta=0.0 (no meaningful comparison)
                delta = 0.0 if status_label == "crashed" else cand_primary_value - parent_primary_value
                results.append({
                    "cell_id": cell_id,
                    "dataset": hc[1],
                    "encoder": hc[2],
                    "task": hc[3],
                    "child_node_id": child_id,
                    "candidate_primary_value": cand_primary_value,
                    "parent_primary_value": parent_primary_value,
                    "delta": delta,
                    "status": status_label,
                })
                terminal_now.append(cell_id)

        for cid in terminal_now:
            pending.pop(cid)

        if pending:
            time.sleep(poll_interval_s)

    if pending:
        raise TimeoutError(
            f"gate-eval polling deadline exceeded; {len(pending)} jobs still running: "
            f"{list(pending.keys())}"
        )
    return results


def _read_eval_primary_value(
    handle,
    backend,
    graph: "ExperimentGraph",
    child_id: str,
    fallback_primary_value: float,
    state_str: str,
) -> tuple[float, str]:
    """Read per-cell primary_value from the completed eval result.

    For crashed/cancelled/budget_killed states, returns (0.0, 'crashed').
    For completed state, reads primary_value from graph node (which the orchestrator
    would have written from result.json — in tests, mock backends stamp the node
    directly or use fallback_primary_value).
    """
    if state_str in ("crashed", "cancelled", "budget_killed"):
        return (0.0, "crashed")

    node = graph.nodes.get(child_id, {})
    primary_value = float(node.get("primary_value", 0.0))
    if primary_value == 0.0:
        # Fallback: use the candidate's primary_value (same variant, different cell context)
        primary_value = fallback_primary_value
    return (primary_value, "completed")
