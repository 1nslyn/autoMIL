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

    A fold whose primary_value was unestimable serializes as ``null``
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
            "primary_value":       float (mean of per-fold primary values; 0.0 if zero folds),
            "metrics":         {key: mean across folds},
            "partial_folds":   int,
            "expected_folds":  int,
            "elapsed_seconds": int (sum),
            "peak_vram_mb":    int (max),
        }

    Status rules (D-119):
        - All K folds present → status: "completed", partial_folds == expected_fold_count
        - 1 ≤ folds < K → status: "partial", partial_folds: <n>
        - 0 folds → status: "crashed", primary_value: 0.0
    """
    if not node_archive.exists():
        return _crashed_payload(expected_fold_count)
    fold_files = sorted(node_archive.glob("fold_*_result.json"))
    if not fold_files:
        return _crashed_payload(expected_fold_count)

    primary_values: list[float] = []
    metrics_by_key: dict[str, list[float]] = {}
    held_out_by_key: dict[str, list[float]] = {}
    key_signatures: set[tuple[frozenset, frozenset]] = set()
    fold_entries: list[dict] = []
    elapsed_total = 0
    peak_vram = 0

    for ff in fold_files:
        try:
            data = json.loads(ff.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed fold file %s: %s", ff, exc)
            continue
        # An unestimable primary_value is `null`, and a fold that carries one is not
        # evidence — it must not count toward partial_folds or dilute the mean.
        # Its resource usage below still does: the fold really did run.
        # Resource usage is counted for every fold that ran, estimable or not.
        elapsed_total += int(data.get("elapsed_seconds", 0) or 0)
        peak_vram = max(peak_vram, int(data.get("peak_vram_mb", 0) or 0))

        # ONE rule: a fold contributes everything or nothing. An unestimable
        # value anywhere in it -- the primary_value, a `metrics` entry, a `held_out`
        # entry -- drops the whole fold.
        #
        # Skipping only the offending VALUE was the subtler half of this bug.
        # `primary_value` would then be averaged over N folds while a surviving
        # sibling metric was averaged over N+1, and the two are supposed to
        # describe the same evidence:
        #
        #   - on the val side CR-1b recomputes the primary_value from `metrics`, so
        #     the mismatch fires terminal_writer's VAL-FIREWALL ERROR ("the
        #     reported scalar may have been computed from test") on a benign
        #     coverage gap -- degrading the one alarm that is supposed to mean
        #     test leaked into selection -- and then the mixed-denominator
        #     recompute WINS and becomes the node's authoritative primary_value,
        #     its error direction set by whether the dropped folds happened to
        #     be good or bad;
        #   - on the test side nothing recomputes `held_out`, so a `test_auc`
        #     averaged over 2 folds sat beside a `test_bacc` averaged over 3
        #     under `status: completed`, and that block is what terminal_writer
        #     seals into certify.json -- the number that goes in the table.
        #
        offenders = [
            f"{block}.{k}"
            for block in ("metrics", "held_out")
            for k, v in (data.get(block) or {}).items()
            if _finite(v) is None
        ]
        primary_value = _finite(data.get("primary_value"))
        if primary_value is None:
            offenders.insert(0, f"primary_value={data.get('primary_value')!r}")
        # fold_index is part of the same all-or-nothing contract: a fold that
        # cannot be placed in the fold vector must not count toward the
        # aggregate either, or primary_value/primary_se run over N folds while
        # validation_folds carries N−1 — and the ingest-side SE recompute
        # (which prefers validation_folds) would silently replace the N-fold
        # SE and fire the val-firewall tamper WARNING on a benign naming gap.
        fold_index = data.get("fold_index")
        if not isinstance(fold_index, int) or isinstance(fold_index, bool):
            try:  # fold_<i>_result.json — the writer's own naming contract
                fold_index = int(ff.name.split("_")[1])
            except (IndexError, ValueError):
                fold_index = None
        if fold_index is None:
            offenders.append(f"fold_index={data.get('fold_index')!r}")
        if offenders:
            logger.warning(
                "Skipping fold %s: unestimable %s (a fold contributes all of its "
                "values or none, so every reported mean shares one denominator)",
                ff.name, ", ".join(offenders),
            )
            continue

        # Two files claiming the same fold_index would both count toward
        # `n == expected_fold_count` (a phantom "completed") while the fold
        # map collapses them to one entry — evidence inconsistency this
        # reader cannot adjudicate, same class as a mixed key-set schema:
        # fail the recovery closed.
        if any(entry["fold_index"] == fold_index for entry in fold_entries):
            logger.warning(
                "Fold archive %s carries duplicate fold_index %d (%s); the "
                "evidence is inconsistent — failing the recovery closed.",
                node_archive, fold_index, ff.name,
            )
            return _crashed_payload(
                expected_fold_count,
                elapsed_seconds=elapsed_total, peak_vram_mb=peak_vram,
            )

        primary_values.append(primary_value)
        # Cross-fold key-set signature: every counted fold must describe the
        # SAME evidence schema. Every writer emits a fixed key set per task
        # family, so two folds disagreeing on keys means the code surface
        # changed mid-run — evidence this reader cannot adjudicate.
        key_signatures.add((
            frozenset((data.get("metrics") or {}).keys()),
            frozenset((data.get("held_out") or {}).keys()),
        ))
        for k, v in (data.get("metrics") or {}).items():
            metrics_by_key.setdefault(k, []).append(_finite(v))
        # held_out (test) aggregated in parallel but kept sealed — terminal_writer
        # routes it to certify.json, never into agent-facing artifacts (val-firewall).
        for k, v in (data.get("held_out") or {}).items():
            held_out_by_key.setdefault(k, []).append(_finite(v))
        # The recovery path must emit the same per-fold evidence contract as a
        # normal completion: validation_folds is what carries fold primary values
        # (paired keep-margin) and the val-prediction hash (no-op detection)
        # onto the node and its round-trip artifacts. Val-only projection —
        # held_out stays in its sealed aggregate above. fold_index was
        # resolved (or the whole fold skipped) in the offender check, so the
        # entry list and the aggregates share one denominator by construction.
        from automil.firewall import held_out_metric_keys

        fold_metrics = dict(data.get("metrics") or {})
        for leak in held_out_metric_keys(fold_metrics):
            # A held-out-named key inside a fold's metrics trips the
            # ingest firewall for the whole node anyway; the entry
            # projection is DECLARED val-only, so it enforces the
            # vocabulary itself rather than relying on that.
            fold_metrics.pop(leak, None)
        fold_entries.append({
            "fold_index": fold_index,
            "metrics": fold_metrics,
            "primary_value": primary_value,
            "val_predictions_sha256": data.get("val_predictions_sha256"),
        })

    n = len(primary_values)
    if n == 0:
        return _crashed_payload(
            expected_fold_count,
            elapsed_seconds=elapsed_total, peak_vram_mb=peak_vram,
        )

    # ONE evidence schema across all counted folds, or nothing. A mixed
    # archive (e.g. 2-key and 3-key held_out from a mid-run code change)
    # would average different keys over different denominators and seal the
    # result under `status: completed`/`partial` — the exact
    # mixed-denominator failure the per-fold all-or-nothing rule exists to
    # prevent, one level up. The aggregator is a pure reader with no access
    # to the declared schema, so it cannot pick a side: fail the recovery
    # closed instead. (No arm emits sparse per-fold diagnostics in
    # `metrics`/`held_out` — every writer emits a fixed set per family — so
    # this tolerance-free rule costs no honest evidence.)
    if len(key_signatures) > 1:
        logger.warning(
            "Fold archive %s carries %d different metric key-set schemas "
            "across its folds; the evidence is inconsistent (mid-run code "
            "surface change?) — failing the recovery closed.",
            node_archive, len(key_signatures),
        )
        return _crashed_payload(
            expected_fold_count,
            elapsed_seconds=elapsed_total, peak_vram_mb=peak_vram,
        )

    # B1 (claims-alignment): the fold primary values are in hand — compute the SE
    # here so budget-killed / partial nodes carry a measured noise floor for
    # the Ladder margin instead of silently dropping to the bare δ.
    from automil.scoring import cross_fold_se

    return {
        # Completeness is the INDEX SET, not the count: {0,1,3} with three
        # expected folds is three files but not the declared evidence.
        "status": (
            "completed"
            if {e["fold_index"] for e in fold_entries}
            == set(range(expected_fold_count))
            else "partial"
        ),
        "primary_value": sum(primary_values) / n,
        "primary_se": cross_fold_se(primary_values),
        # Every value here came from a fold that contributed ALL of its values
        # AND all counted folds share one key-set schema, so every mean and
        # `primary_value` share one denominator by construction.
        "metrics": {k: sum(v) / len(v) for k, v in metrics_by_key.items()},
        "held_out": {k: sum(v) / len(v) for k, v in held_out_by_key.items()},
        "validation_folds": sorted(fold_entries, key=lambda e: e["fold_index"]),
        "partial_folds": n,
        "expected_folds": expected_fold_count,
        "elapsed_seconds": elapsed_total,
        "peak_vram_mb": peak_vram,
    }


def _crashed_payload(
    expected_fold_count: int, *,
    elapsed_seconds: int = 0, peak_vram_mb: int = 0,
) -> dict:
    return {
        "status": "crash",  # D-06: canonical value (was "crashed")
        "primary_value": 0.0,
        "metrics": {},
        "held_out": {},
        "partial_folds": 0,
        "expected_folds": expected_fold_count,
        # Fail-closed on EVIDENCE, not on telemetry: folds that ran really
        # did burn this time and VRAM, and zeroing it corrupts the archive
        # and TSV accounting for exactly the runs an operator investigates.
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
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
        "reconcile_budget_kill %s: status=%s partial_folds=%d/%d primary_value=%.4f",
        node_id,
        payload["status"],
        payload["partial_folds"],
        payload["expected_folds"],
        payload["primary_value"],
    )
    return payload
