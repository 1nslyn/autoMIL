"""Single terminal-state writer for all four artifacts (REC-02 / D-09, D-10).

Fixed write order: graph node (via locked_update) → completed/<node>.json
→ archive result.json → results.tsv. Both _handle_completion and
_handle_cap_killed_completion delegate here. Never called from train.py.

D-01: partial results are quarantined — their graph node status stays
      "partial" (not "keep"/"discard") and they do not update best_node.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from automil.graph import ExperimentGraph

logger = logging.getLogger(__name__)

# D-06 (REC-03): canonicalize non-enum status drift values.
_STATUS_CANON: dict[str, str] = {
    "crashed": "crash",
    "oom": "crash",
    "timeout": "crash",
}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic tempfile+replace write — same filesystem guaranteed by dir=."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _canonicalize(result: dict, termination_reason: str | None = None) -> dict:
    """Return a copy of result with status canonicalized and termination_reason injected."""
    result = dict(result)
    raw_status = result.get("status", "crash")
    result["status"] = _STATUS_CANON.get(raw_status, raw_status)
    if termination_reason and "termination_reason" not in result:
        result["termination_reason"] = termination_reason
    return result


def write_terminal_state(
    *,
    node_id: str,
    result: dict,
    graph: "ExperimentGraph",
    completed_dir: Path,
    archive_dir: Path,
    results_tsv_writer: Callable,
    spec: dict,
    elapsed_s: float,
    gpu_id: int | str,
) -> None:
    """Write all four terminal artifacts in fixed order (D-09).

    Step 1: Canonicalize status (D-06).
    Step 2: Schema validation (fail-safe — log and continue).
    Step 3: Graph node update via locked_update (D-10, Pitfall 1/2 guard).
    Step 4: completed/<node>.json (atomic).
    Step 5: archive result.json (atomic).
    Step 6: results.tsv (delegated, sole-writer invariant D-10).

    Args:
        node_id:            Graph node ID.
        result:             Result dict (as-collected or synthesized).
        graph:              Live ExperimentGraph instance (used for its path and technique_map).
        completed_dir:      Path to orchestrator/completed/ directory.
        archive_dir:        Path to orchestrator/archive/<node_id>/ directory.
        results_tsv_writer: Callable with signature (node_id, result, description) -> None.
                            This is the daemon's _append_results_tsv, bound as a method.
        spec:               The experiment spec dict (for graph_metadata, description).
        elapsed_s:          Wall-clock elapsed seconds.
        gpu_id:             GPU ID used by this experiment.
    """
    # Step 1 — Canonicalize status
    result = _canonicalize(result)

    # Step 2 — Schema validation (fail-safe: log and continue)
    try:
        from automil.schemas import validate_result, ValidationError
        validate_result(result)
    except Exception as exc:
        # Support both ValidationError with .message and generic exceptions
        msg = getattr(exc, "message", str(exc))
        logger.warning(
            "result.json schema validation failed for %s: %s", node_id, msg
        )
        result = {
            "status": "crash",
            "composite": 0.0,
            "metrics": {},
            "error": f"result.json failed schema validation: {msg}",
        }

    # Val-firewall: quarantine test. ``held_out`` (test metrics) and the full
    # ``summary`` (which embeds test) are split into a sealed certify.json sidecar
    # and stripped from every agent-facing artifact (graph node, completed/,
    # results.tsv, archive/result.json). Read once by ``automil certify``.
    sealed = {k: result[k] for k in ("held_out", "summary") if k in result}
    result = {k: v for k, v in result.items() if k not in ("held_out", "summary")}

    raw_status = result.get("status", "crash")

    # Step 3 — Graph node update via locked_update (D-10, D-01)
    from automil.graph import locked_update, _accept, _accept_margin
    try:
        # _technique_map is the internal attribute on ExperimentGraph
        _tm = getattr(graph, "_technique_map", None)
        with locked_update(str(graph.path), technique_map=_tm) as g:
            gnode = g.get_node(node_id)
            if gnode is None:
                logger.warning(
                    "terminal_writer: node %s not found in graph — skipping graph update",
                    node_id,
                )
            else:
                parent_id = gnode.get("parent_id")
                parent = g.get_node(parent_id) if parent_id else None
                p_comp = parent.get("composite", 0.0) if parent else 0.0
                composite = result.get("composite", 0.0)

                # D-01: partial nodes stay quarantined — never get keep/discard
                # crash nodes stay crash (composite=0.0 should not become discard)
                if raw_status == "partial":
                    graph_status = "partial"  # D-01: quarantined
                elif raw_status == "crash":
                    graph_status = "crash"    # failure — not a keep/discard candidate
                else:
                    # completed, budget_killed, cancelled — Ladder-gated dominance
                    graph_status = (
                        "keep"
                        if _accept(composite, p_comp, _accept_margin(g.meta))
                        else "discard"
                    )

                gnode["type"] = "executed"
                gnode["status"] = graph_status
                gnode["composite"] = composite
                if result.get("metrics"):
                    gnode["metrics"] = dict(result["metrics"])
                # Propagate metadata from result (e.g. budget_killed flag)
                if result.get("metadata"):
                    gnode.setdefault("metadata", {}).update(result["metadata"])

                # Only re-evaluate descendants for non-partial, non-crash completions
                if raw_status not in ("partial", "crash"):
                    g._reevaluate_descendants(node_id)

                # D-01: only update best_node for non-partial, non-crash results
                if raw_status not in ("partial", "crash"):
                    if composite > g.meta.get("best_composite", 0.0):
                        g.meta["best_composite"] = composite
                        g.meta["best_node_id"] = node_id
                # g.save() called automatically on context exit
    except Exception:
        logger.exception(
            "terminal_writer: failed to update graph node %s — continuing with other artifacts",
            node_id,
        )

    # Step 4 — completed/<node>.json (atomic write)
    completion = {
        "id": node_id,
        "status": result.get("status", "crash"),
        "composite": result.get("composite", 0.0),
        "metrics": result.get("metrics", {}),
        "elapsed_seconds": result.get("elapsed_seconds", elapsed_s),
        "peak_vram_mb": result.get("peak_vram_mb", 0),
        "gpu": gpu_id,
        "completed_at": datetime.now().isoformat(),
        "graph_metadata": result.get("graph_metadata") or spec.get("graph_metadata") or {},
    }
    if result.get("termination_reason"):
        completion["termination_reason"] = result["termination_reason"]
    try:
        _atomic_write_json(completed_dir / f"{node_id}.json", completion)
    except Exception:
        logger.exception(
            "terminal_writer: failed to write completed/%s.json", node_id
        )

    # Step 5 — archive result.json (atomic write, sole writer per D-10)
    try:
        _atomic_write_json(archive_dir / "result.json", result)
    except Exception:
        logger.exception(
            "terminal_writer: failed to write archive result.json for %s", node_id
        )

    # Step 5b — sealed certify.json sidecar (test metrics + summary; val-firewall).
    # Not surfaced during search; consumed once by ``automil certify``.
    if sealed:
        try:
            _atomic_write_json(archive_dir / "certify.json", sealed)
        except Exception:
            logger.exception(
                "terminal_writer: failed to write certify.json for %s", node_id
            )

    # Step 6 — results.tsv (delegated, D-08: partial rows ARE written)
    try:
        results_tsv_writer(
            node_id,
            result,
            description=spec.get("description", ""),
        )
    except Exception:
        logger.exception(
            "terminal_writer: failed to append results.tsv row for %s", node_id
        )
