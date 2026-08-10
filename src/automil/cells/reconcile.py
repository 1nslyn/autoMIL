"""Per-fold result aggregation + budget-kill reconciliation (CAP-03, CAP-04 / D-119, D-123)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not one.

    A fold whose composite was unestimable serializes as ``null``
    (``runtime_helpers.json_safe``). ``float(None)`` raises TypeError, and this
    aggregator runs inside the SIGTERM handler — an uncaught raise there loses
    the entire partial flush. Coercing to 0.0 would be no better: this reader's
    contract is to distinguish missing data from zero-valued data (Pitfall 4).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None

# D-06 (REC-03): canonical status enum. "crashed" was emitted pre-v1.1 — normalize on write.
_STATUS_CANON: dict[str, str] = {
    "crashed": "crash",
}


def aggregate_folds(node_archive: Path, expected_fold_count: int) -> dict:
    """Walk archive/<node>/fold_*_result.json; return a result.json payload (D-119).

    Pure reader. Malformed fold files are skipped with logger.warning, NOT silently
    used as zeros (Pitfall 4 defence: aggregator must distinguish missing data from
    zero-valued data).

    Args:
        node_archive: Directory containing fold_<i>_result.json files.
                      For SIGTERM handler: Path.cwd() (D-121).
                      For post-cancel reconcile: archive/<node_id>/ (D-123).
        expected_fold_count: K from training.fold_count config (D-120).

    Returns:
        {
            "status":          "completed" if n==expected else "partial" else "crashed",
            "composite":       float (mean of per-fold composites; 0.0 if zero folds),
            "metrics":         {key: mean across folds},
            "partial_folds":   int,
            "expected_folds":  int,
            "elapsed_seconds": int (sum),
            "peak_vram_mb":    int (max),
        }

    Status rules (D-119):
        - All K folds present → status: "completed", partial_folds == expected_fold_count
        - 1 ≤ folds < K → status: "partial", partial_folds: <n>
        - 0 folds → status: "crashed", composite: 0.0
    """
    if not node_archive.exists():
        return _crashed_payload(expected_fold_count)
    fold_files = sorted(node_archive.glob("fold_*_result.json"))
    if not fold_files:
        return _crashed_payload(expected_fold_count)

    composites: list[float] = []
    metrics_by_key: dict[str, list[float]] = {}
    held_out_by_key: dict[str, list[float]] = {}
    elapsed_total = 0
    peak_vram = 0

    for ff in fold_files:
        try:
            data = json.loads(ff.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed fold file %s: %s", ff, exc)
            continue
        # An unestimable composite is `null`, and a fold that carries one is not
        # evidence — it must not count toward partial_folds or dilute the mean.
        # Its resource usage below still does: the fold really did run.
        composite = _finite(data.get("composite"))
        if composite is None:
            logger.warning(
                "Skipping fold with no estimable composite (%r) in %s",
                data.get("composite"), ff,
            )
        else:
            composites.append(composite)
        for k, v in data.get("metrics", {}).items():
            value = _finite(v)
            if value is None:
                logger.warning("Skipping non-numeric metric %s=%r in %s", k, v, ff)
                continue
            metrics_by_key.setdefault(k, []).append(value)
        # held_out (test) aggregated in parallel but kept sealed — terminal_writer
        # routes it to certify.json, never into agent-facing artifacts (val-firewall).
        for k, v in data.get("held_out", {}).items():
            value = _finite(v)
            if value is None:
                logger.warning("Skipping non-numeric held_out %s=%r in %s", k, v, ff)
                continue
            held_out_by_key.setdefault(k, []).append(value)
        elapsed_total += int(data.get("elapsed_seconds", 0) or 0)
        peak_vram = max(peak_vram, int(data.get("peak_vram_mb", 0) or 0))

    n = len(composites)
    if n == 0:
        return _crashed_payload(expected_fold_count)

    # B1 (claims-alignment): the fold composites are in hand — compute the SE
    # here so budget-killed / partial nodes carry a measured noise floor for
    # the Ladder margin instead of silently dropping to the bare δ.
    from automil.scoring import cross_fold_se

    return {
        "status": "completed" if n == expected_fold_count else "partial",
        "composite": sum(composites) / n,
        "composite_se": cross_fold_se(composites),
        "metrics": {k: sum(v) / len(v) for k, v in metrics_by_key.items()},
        "held_out": {k: sum(v) / len(v) for k, v in held_out_by_key.items()},
        "partial_folds": n,
        "expected_folds": expected_fold_count,
        "elapsed_seconds": elapsed_total,
        "peak_vram_mb": peak_vram,
    }


def _crashed_payload(expected_fold_count: int) -> dict:
    return {
        "status": "crash",  # D-06: canonical value (was "crashed")
        "composite": 0.0,
        "metrics": {},
        "held_out": {},
        "partial_folds": 0,
        "expected_folds": expected_fold_count,
        "elapsed_seconds": 0,
        "peak_vram_mb": 0,
    }


def reconcile_budget_kill(
    node_id: str,
    archive_dir: Path,
    graph: Any,
    expected_fold_count: int,
) -> dict:
    """Post-cap-cancel reconciliation (CAP-04 / D-123).

    Aggregates whatever fold files are present in archive/<node_id>/,
    writes archive/<node_id>/result.json with metadata.budget_killed=True,
    and returns the payload dict so the caller can drive graph updates.

    STUB — Plan 04-08 wires this into the daemon's _handle_completion path
    and adds the graph mutation calls (graph.add_executed / graph.mark_failed
    + _reevaluate_descendants). For Wave 3 this stub:
      1. Aggregates fold files via aggregate_folds()
      2. Tags metadata.budget_killed=True per D-124
      3. Writes archive/<node_id>/result.json
      4. Returns the payload dict

    The graph-mutation portion (D-123 steps 2b and 3b) lands in Plan 04-08
    alongside the daemon _tick_cells integration where the graph reference
    is in scope.

    D-124 discriminator:
        ≥1 fold → payload["status"] in ("partial", "completed") — caller sets
                  graph node status: executed, metadata.budget_killed=True
        0 folds → payload["status"] == "crashed" — caller sets graph node
                  status: crashed, metadata.budget_killed=True

    Args:
        node_id:              Graph node id of the budget-killed experiment.
        archive_dir:          Parent directory; fold files born-sealed at
                              archive_dir/node_id/certify/ (Scope B val-firewall).
        graph:                ExperimentGraph instance (unused in stub — Plan 04-08).
        expected_fold_count:  K from AUTOMIL_FOLD_COUNT env / config (D-120).

    Returns:
        result.json payload dict with metadata.budget_killed=True.
    """
    # Val-firewall (Scope B): fold_*_result.json are born-sealed under
    # archive/<node>/certify/ (AUTOMIL_RESULTS_DIR), so budget-kill reconciliation
    # aggregates from there — never the agent-visible node-archive root.
    node_archive = archive_dir / node_id / "certify"
    payload = aggregate_folds(node_archive, expected_fold_count)
    payload.setdefault("metadata", {})["budget_killed"] = True
    # D-10 (REC-02): archive result.json is written solely by terminal_writer._atomic_write_json.
    # The mkdir call stays — the sealed dir must exist before aggregation/terminal_writer.
    node_archive.mkdir(parents=True, exist_ok=True)
    logger.info(
        "reconcile_budget_kill %s: status=%s partial_folds=%d/%d composite=%.4f",
        node_id,
        payload["status"],
        payload["partial_folds"],
        payload["expected_folds"],
        payload["composite"],
    )
    return payload
