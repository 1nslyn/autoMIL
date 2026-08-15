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
import shutil
import tempfile
from collections.abc import Mapping
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


def _seal_stray(src: Path, dest: Path) -> None:
    """Relocate a stray test artifact from the node-archive root into certify/.

    The invariant this must restore: nothing test-bearing is left at the
    agent-visible node-archive root. ``Path.replace`` cannot rename onto a
    non-empty directory (ENOTEMPTY), and under born-sealing ``certify/`` usually
    already holds the born-sealed copy (e.g. ``certify/results`` always exists for
    the CLAM detail tree). So: if certify/ has no copy yet, move the stray in;
    otherwise the root stray is a bypass duplicate and is removed (certify/ is
    authoritative under born-sealing). Best-effort: logs, never raises.
    """
    if not dest.exists():
        try:
            src.replace(dest)  # atomic move into certify/ (file or directory)
            return
        except OSError as exc:
            # Leave the stray rather than destroy the only copy; the caller's
            # ERROR log already flags the (should-never-happen) bypass.
            logger.warning("terminal_writer: could not relocate stray %s: %s", src, exc)
            return
    # certify/ already holds the born-sealed copy → the root stray is a duplicate.
    try:
        if src.is_dir():
            shutil.rmtree(src, ignore_errors=True)
        elif src.exists():
            src.unlink()
    except OSError as exc:
        logger.warning("terminal_writer: could not remove stray duplicate %s: %s", src, exc)


def _seal_node_archive(archive_dir: Path, sealed: dict) -> None:
    """Write the sealed test sidecar (certify.json) into ``archive/<node>/certify/``.

    Val-firewall (Scope B): test-bearing artifacts are **born-sealed** — the
    orchestrator points AUTOMIL_RESULTS_DIR at ``archive/<node>/certify/`` so the
    per-fold writer, the ``results/`` detail tree, and the raw result.json write
    there directly, and ``collect_result`` copies its raw result.json there too.
    This writer therefore only has to drop ``certify.json`` (the held_out + summary
    sidecar) alongside them. The ``certify/`` subdir is documented off-limits to
    the agent and read only by ``automil certify``. Best-effort: a seal failure is
    logged, never raised, so it cannot break a completion.

    The stray sweep below is a regression backstop: under born-sealing the
    node-archive root holds no test artifact, so a non-empty sweep means a writer
    bypassed AUTOMIL_RESULTS_DIR during the run — an integrity problem that
    completion-time cleanup cannot retroactively undo, hence logged at ERROR. It
    still restores the invariant that matters: nothing test-bearing is left at the
    agent-visible root (see _seal_stray). Best-effort: never raises.
    """
    sealed_dir = archive_dir / "certify"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    if sealed:
        _atomic_write_json(sealed_dir / "certify.json", sealed)

    strays: list[Path] = list(archive_dir.glob("fold_*_result.json"))
    if (archive_dir / "results").is_dir():
        strays.append(archive_dir / "results")
    if strays:
        logger.error(
            "terminal_writer: born-sealing bypassed for %s — test artifact(s) at "
            "node-archive root: %s (a writer ignored AUTOMIL_RESULTS_DIR); sealing",
            archive_dir.name, [s.name for s in strays],
        )
        for s in strays:
            _seal_stray(s, sealed_dir / s.name)


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
    gpu_id: int | str | None,
    accelerator: str = "cuda",
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
        gpu_id:             Physical GPU ID, or ``None`` for CPU execution.
        accelerator:        Execution substrate (``cuda``, ``rocm``, or ``cpu``).
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

    # Step 2a — A6 (claims-alignment): a held-out-named key INSIDE ``metrics``
    # is a val-firewall violation, not a stylistic nit — it would enter every
    # agent-facing surface and the recomputed composite (test driving
    # selection). Fail closed exactly like a schema failure: the node crashes
    # with a pointer naming the offending keys, and no certify.json is minted.
    from automil.firewall import held_out_metric_keys as _held_out_metric_keys

    _leaking = _held_out_metric_keys(result.get("metrics"))
    if _leaking:
        logger.error(
            "val-firewall violation for %s: held-out-named metrics key(s) %s — "
            "`metrics` is the validation-only selection block; test belongs in "
            "the sealed `held_out` block. Failing the node closed.",
            node_id, ", ".join(_leaking),
        )
        result = {
            "status": "crash",
            "composite": 0.0,
            "metrics": {},
            "error": (
                "val-firewall violation: held-out-named key(s) in `metrics`: "
                f"{', '.join(_leaking)}. `metrics` is validation-only; test "
                "metrics belong in the sealed `held_out` block."
            ),
        }

    # Val-firewall: quarantine test. ``held_out`` (test metrics) and the full
    # ``summary`` (which embeds test) are split into a sealed certify.json sidecar
    # and stripped from every agent-facing artifact (graph node, completed/,
    # results.tsv, archive/result.json). Read once by ``automil certify``.
    sealed = {k: result[k] for k in ("held_out", "summary") if k in result}
    result = {k: v for k, v in result.items() if k not in ("held_out", "summary")}

    # Step 2c — B1 (claims-alignment): the SE that gates the Ladder keep-margin
    # must not be trusted verbatim off agent-editable training output. Recompute
    # it from the result's own val-only per-fold evidence when present; the
    # recomputed value replaces the reported one here so every downstream
    # artifact (graph node, completed/, archive result.json) round-trips it.
    from automil.scoring import recompute_composite_se as _recompute_se

    _se_recomputed = _recompute_se(result)
    if _se_recomputed is not None:
        _se_reported = result.get("composite_se")
        if isinstance(_se_reported, (int, float)) and not isinstance(_se_reported, bool) \
                and abs(float(_se_reported) - _se_recomputed) > 1e-6:
            logger.warning(
                "terminal_writer: reported composite_se %.6f for %s disagrees with "
                "the value recomputed from validation_folds (%.6f); using the "
                "recomputed value (CR-4 noise floor must not be self-reported).",
                float(_se_reported), node_id, _se_recomputed,
            )
        result = {**result, "composite_se": _se_recomputed}

    raw_status = result.get("status", "crash")

    # Step 2b — CR-1b (audit 2026-07-23): derive the selection signal from the
    # declared VALIDATION metrics instead of trusting the reported scalar.
    # result.json is written by agent-editable training code, so a composite
    # computed from the sealed test block would otherwise drive selection
    # undetected — the exact leak the val-firewall exists to prevent. The
    # recomputed value is authoritative; a disagreement is logged at ERROR and
    # recorded on the node so the audit trail survives.
    # Resolved inside the lock (the formula lives in graph meta.scoring).
    composite_recomputed: float | None = None
    composite_disagreement: dict | None = None

    # Step 3 — Graph node update via locked_update (D-10, D-01)
    from automil.graph import (locked_update, keep_or_discard, merged_metadata,
                           node_composite_se as _node_se_reader)
    from automil.scoring import fold_composite_entries as _fold_entries

    # One projection serves the graph node AND the completion artifact below —
    # the round-trip contract requires them identical, so compute once.
    _folds = _fold_entries(result)
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

                # CR-1b: recompute from the val metrics; the val-derived value wins.
                from automil.scoring import composite_disagrees, recompute_composite
                _formula = (g.meta.get("scoring") or {}).get("formula")
                try:
                    composite_recomputed = recompute_composite(
                        result.get("metrics") or {}, _formula
                    )
                except ValueError as exc:
                    logger.error("terminal_writer: %s — trusting reported composite", exc)
                    composite_recomputed = None
                if composite_recomputed is not None and composite_disagrees(
                    composite, composite_recomputed
                ):
                    logger.error(
                        "terminal_writer: VAL-FIREWALL — reported composite %.6f for %s "
                        "disagrees with the value recomputed from its val metrics "
                        "(%.6f, formula=%r). Using the val-derived value; the reported "
                        "scalar may have been computed from test.",
                        composite, node_id, composite_recomputed, _formula,
                    )
                    composite_disagreement = {
                        "reported": composite,
                        "recomputed": composite_recomputed,
                        "formula": _formula,
                    }
                if composite_recomputed is not None:
                    composite = composite_recomputed

                # Paired margin: attach the child evidence BEFORE the accept so
                # this node pairs with its parent now and serves as a pairable
                # incumbent for its own children later. Assign-or-CLEAR: a
                # re-ingest without usable folds must not leave a previous
                # run's vector beside a new composite.
                if _folds is not None:
                    gnode["fold_composites"] = _folds
                else:
                    gnode.pop("fold_composites", None)
                gnode["composite"] = composite

                # D-01: partial nodes stay quarantined — never get keep/discard
                # crash nodes stay crash (composite=0.0 should not become discard)
                if raw_status == "partial":
                    graph_status = "partial"  # D-01: quarantined
                elif raw_status == "crash":
                    graph_status = "crash"    # failure — not a keep/discard candidate
                else:
                    # completed, budget_killed, cancelled — Ladder-gated dominance
                    graph_status = keep_or_discard(g.meta, parent, gnode)

                # M-7: the daemon's terminal path never maintained the counters.
                # `meta.total_executed` is the UCB exploration denominator
                # (graph.py: sqrt(log(total) / (1 + child_count))), so a campaign
                # driven entirely by the daemon explored against a frozen count
                # while every CLI path kept it moving. Idempotent on `type`: a
                # re-processed completion must not double-count.
                if gnode.get("type") != "executed":
                    g.meta["total_executed"] = g.meta.get("total_executed", 0) + 1
                    g.meta["total_proposed"] = max(
                        0, g.meta.get("total_proposed", 0) - 1
                    )
                gnode["type"] = "executed"
                gnode["status"] = graph_status
                gnode["composite"] = composite
                # CR-4: the measured cross-fold SE travels with the composite, so
                # this node can serve as an incumbent whose noise sets the bar for
                # its own children. None when <2 folds were estimable.
                gnode["composite_se"] = _node_se_reader({"composite_se": result.get("composite_se")})
                # CELL-1: backfill budget-cell membership for nodes that did not
                # come through `automil submit` (Backend.submit paths stamp the
                # spec but never touch the graph). Submit-time identity wins.
                _spec_cell_id = (spec.get("metadata") or {}).get("cell_id")
                if _spec_cell_id and not gnode.get("cell_id"):
                    gnode["cell_id"] = _spec_cell_id
                if result.get("metrics"):
                    gnode["metrics"] = dict(result["metrics"])
                # Propagate metadata from result (e.g. budget_killed flag).
                # L-8a: copy-on-write via merged_metadata — a plain setdefault+
                # update/assign would mutate node["metadata"] in place, and that
                # dict object can be aliased with another node's (gate/evaluate.py
                # creates gate-eval children via a shallow dict(node) copy).
                if isinstance(result.get("metadata"), Mapping):
                    gnode["metadata"] = merged_metadata(gnode, result["metadata"])
                # CR-1b: durable audit trail when the reported scalar could not be
                # explained by the node's own validation metrics.
                if composite_disagreement is not None:
                    gnode["metadata"] = merged_metadata(
                        gnode, {"composite_disagreement": composite_disagreement}
                    )

                # Only re-evaluate descendants for non-partial, non-crash completions
                if raw_status not in ("partial", "crash"):
                    g._reevaluate_descendants(node_id)

                # D-01 + H-6 (audit 2026-07-23): only touch best for non-partial,
                # non-crash completions, and recompute it from keep nodes only. An
                # inline ``composite > best`` could set best to a node that is (or
                # just became, via _reevaluate_descendants above) discarded — e.g.
                # under a Ladder δ>0 a within-margin child is discarded yet would
                # win the inline strict-``>`` update.
                if raw_status not in ("partial", "crash"):
                    g.recompute_best()
                # g.save() called automatically on context exit
    except Exception:
        logger.exception(
            "terminal_writer: failed to update graph node %s — continuing with other artifacts",
            node_id,
        )

    # CR-1b: keep the downstream artifacts (completed/, archive result.json,
    # results.tsv) consistent with the graph's authoritative val-derived
    # composite. Rebuilt immutably — never mutating the caller's dict.
    if composite_recomputed is not None:
        result = {**result, "composite": composite_recomputed}
        if composite_disagreement is not None:
            existing_metadata = (
                result.get("metadata")
                if isinstance(result.get("metadata"), Mapping)
                else {}
            )
            result = {
                **result,
                "metadata": {
                    **existing_metadata,
                    "composite_disagreement": composite_disagreement,
                },
            }

    # Step 4 — completed/<node>.json (atomic write)
    completion = {
        "id": node_id,
        "status": result.get("status", "crash"),
        "composite": result.get("composite", 0.0),
        # CR-4: reconcile() rebuilds nodes from this artifact, so the measured
        # noise has to survive the round trip or a recovered node would silently
        # revert to the bare predeclared margin.
        "composite_se": _node_se_reader({"composite_se": result.get("composite_se")}),
        # Paired margin: same round-trip contract for the fold projection —
        # without it a recovered node falls back to the marginal basis. Same
        # single projection the graph node received above.
        "fold_composites": _folds,
        "metrics": result.get("metrics", {}),
        "elapsed_seconds": result.get("elapsed_seconds", elapsed_s),
        "peak_vram_mb": result.get("peak_vram_mb", 0),
        "accelerator": accelerator,
        "gpu": gpu_id,
        "completed_at": datetime.now().isoformat(),
        "budget_killed": bool(
            result.get("metadata", {}).get("budget_killed", False)
            if isinstance(result.get("metadata"), Mapping) else False
        ),
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

    # Step 5b — val-firewall: seal ALL test-bearing artifacts (certify.json +
    # the per-fold fold_*_result.json / results/ detritus) under the off-limits
    # archive/<node>/certify/ subdir. The agent-visible node archive is left with
    # only result.json (val) + run.log. Read once by ``automil certify``.
    try:
        _seal_node_archive(archive_dir, sealed)
    except Exception:
        logger.exception(
            "terminal_writer: failed to seal test artifacts for %s", node_id
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
