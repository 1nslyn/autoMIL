"""Single-experiment runner: all folds for one (task, encoder, model) combo."""

from __future__ import annotations

import json
import math
import os

import torch

from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.clam.dataset import create_dataset, load_fold_splits
from autobench.pipeline.evaluate import (
    compute_confidence_intervals,
    pooled_val_block,
    val_prediction_hashes,
)
from autobench.pipeline.hparams import apply_overrides_to_exp_cfg
from autobench.pipeline.results_cache import resolve_results_dir
from autobench.pipeline.clam.train import train_fold
from autobench.pipeline.policy_dispatch import PolicyRuntime


def _write_fold_result_json(
    fold_index: int, result: dict, *, ordinal: bool = False,
) -> None:
    """Write archive/<node>/fold_<i>_result.json per autoMIL CAP-03 / D-118.

    No-op when AUTOMIL_RESULTS_DIR is unset (e.g., running outside the
    autoMIL orchestrator).

    ``ordinal`` is the DECLARED task flag (``exp_cfg.task.ordinal``), never
    sniffed from whether ``qwk`` happens to be present in the metrics dicts —
    the same rule as ``summary_to_result_json`` on the aggregate side. When
    set, ``val_qwk`` joins the recorded fold evidence and ``test_qwk`` the
    sealed ``held_out`` block, both clamped at 0 (kappa is defined on
    [-1, 1]; every campaign consumer requires recorded values in [0, 1] —
    the raw value stays recoverable from the sealed summary and
    predictions.csv). A declared-but-missing qwk is recorded as ``null`` and
    invalidates the fold like any other lost component.

    Pitfall 5: compute_extended_metrics() returns flat floats at the per-fold
    level. However, this helper defensively unwraps both flat-float and
    {"mean": ..., "std": ...} CI-dict shapes in case the caller supplies a
    post-CI metrics dict.

    An unestimable metric is written as ``null``, never NaN: a NaN token makes
    the file invalid JSON and the aggregator that reads it back
    (``automil.cells.reconcile.aggregate_folds``) skips a null-composite fold
    rather than averaging a hole into the partial result.
    """
    from pathlib import Path

    from automil.runtime_helpers import _atomic_write_json

    results_dir = os.environ.get("AUTOMIL_RESULTS_DIR")
    if not results_dir:
        return  # not running under autoMIL orchestrator

    # AUTOMIL_FOLD_COUNT is set by the orchestrator from
    # `automil/config.yaml: training.fold_count`. The literal fallback here
    # matters only when this helper is invoked outside the orchestrator
    # (e.g. local debugging without env vars); it tracks the benchmark's
    # 5-fold lab standard. Production paths always have the env var set.
    fold_count = int(os.environ.get("AUTOMIL_FOLD_COUNT", "5"))

    def _unwrap(metric) -> float | None:
        # auc_roc may be float OR {"mean": float, ...} (CI-dict shape).
        # None means "not estimable" and is kept distinct from a real 0.0.
        if isinstance(metric, dict):
            metric = metric.get("mean")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            return None
        value = float(metric)
        return value if math.isfinite(value) else None

    test_m = result.get("test_metrics", {}) or {}
    val_m = result.get("val_metrics", {}) or {}
    if "c_index" in test_m:
        # Survival: composite is the VALIDATION concordance index (selection signal).
        # Test lives in a sealed ``held_out`` block — never surfaced to the agent
        # during search; read once by ``automil certify`` (val-firewall).
        metrics = {"val_c_index": _unwrap(val_m.get("c_index"))}
        held_out = {"test_c_index": _unwrap(test_m.get("c_index"))}
        primary = "val_c_index"
    else:
        def _clamped_qwk(metric) -> float | None:
            value = _unwrap(metric)
            return None if value is None else max(0.0, value)

        metrics = {
            "val_auc":  _unwrap(val_m.get("auc_roc")),
            "val_bacc": _unwrap(val_m.get("balanced_accuracy")),
        }
        held_out = {
            "test_auc":  _unwrap(test_m.get("auc_roc")),
            "test_bacc": _unwrap(test_m.get("balanced_accuracy")),
        }
        if ordinal:
            metrics["val_qwk"] = _clamped_qwk(val_m.get("qwk"))
            held_out["test_qwk"] = _clamped_qwk(test_m.get("qwk"))
        primary = "val_auc"
    # Selection is the primary validation metric alone (scoring.formula:
    # val_auc / val_c_index); companions stay recorded but no longer vote —
    # see run_experiment._composite_components, the aggregate-side authority
    # this per-fold value must mirror. A fold that lost ANY recorded
    # component (companion included) carries a null composite: fold validity
    # spans the full evidence set, matching _per_fold_composites and the
    # campaign's ingest validator.
    composite = (
        None if any(value is None for value in metrics.values())
        else metrics[primary]
    )

    payload = {
        "fold_index":      fold_index,
        "fold_count":      fold_count,
        "status":          "completed",
        "metrics":         metrics,
        "held_out":        held_out,
        "composite":       composite,
        "elapsed_seconds": int(result.get("elapsed_seconds", 0) or 0),
        "peak_vram_mb":    int(result.get("peak_vram_mb", 0) or 0),
        # A4': no-op detector, ENTRY level — never inside `metrics` (the
        # exact-key-locked CR-1b input: every value votes under the `mean`
        # reducer, and any extra key fails the campaign schema lock). Both paths
        # carry it: the full-run summary -> validation_folds projection here,
        # and the cap-kill aggregator (automil.cells.reconcile.aggregate_folds),
        # which rebuilds entries from the sealed fold files. Null means the
        # fold predates hashing, not "not implemented".
        "val_predictions_sha256": result.get("val_predictions_sha256"),
    }
    # Atomic: these files are written in exactly the window a cap-kill SIGTERM
    # can land, and a torn one is silently dropped by the aggregator.
    _atomic_write_json(Path(results_dir) / f"fold_{fold_index}_result.json", payload)


def run_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: torch.device,
    wandb_project: str | None = None,
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single CLAM experiment and return aggregated results."""
    # A1 (claims-alignment): consume the opaque --hparams channel. CLAM trains
    # off the shared ModelConfig + TrainConfig, so this is the one arm where the
    # opaque keys land on exp_cfg itself — before results-dir resolution (CR-5b
    # cache identity) and before exp_cfg.save (honest provenance).
    apply_overrides_to_exp_cfg(exp_cfg, arm="clam")
    # CR-5b: seed is now a path segment and the rest of the config is
    # fingerprinted into a sidecar, so a re-run at a different seed or
    # hyperparameter can no longer resume these folds' metrics.json.
    results_dir = resolve_results_dir(exp_cfg, benchmark_dir, results_dir)

    # H-3: no `arm_cfg` here, and that is the honest record rather than an
    # omission -- CLAM is the ONE arm that genuinely trains off the shared
    # ModelConfig + TrainConfig (they were designed around it). config.json
    # therefore carries `arm: null` and an empty superseded-fields list, which is
    # exactly what a reader needs to trust its `train` block.
    exp_cfg.save(os.path.join(results_dir, "config.json"))
    policy_runtime = PolicyRuntime.from_experiment(exp_cfg)

    fold_results: list[dict] = []
    if exp_cfg.is_survival:
        # Survival uses an adapter-side trainer over the CLAM model.
        from autobench.pipeline.clam.survival_train import train_survival_fold

        for fold in exp_cfg.selected_folds:
            fold_policy_runtime = policy_runtime.for_fold()
            result = train_survival_fold(
                exp_cfg, benchmark_dir, fold, results_dir, device,
                policy_runtime=fold_policy_runtime,
            )
            fold_results.append(result)
            _write_fold_result_json(fold, result, ordinal=exp_cfg.task.ordinal)
    else:
        dataset = create_dataset(
            exp_cfg, benchmark_dir, task_csv_name=exp_cfg.task.name,
        )
        # Splits directory: splits/{strategy}/{task}/
        splits_subdir = os.path.join(exp_cfg.strategy, exp_cfg.task.name)
        for fold in exp_cfg.selected_folds:
            fold_policy_runtime = policy_runtime.for_fold()
            train_split, val_split, test_split = load_fold_splits(
                dataset, benchmark_dir, splits_subdir, fold,
                task_csv_name=exp_cfg.task.name,
            )
            result = train_fold(
                exp_cfg, train_split, val_split, test_split,
                fold, results_dir, device, wandb_project=wandb_project,
                policy_runtime=fold_policy_runtime,
            )
            fold_results.append(result)
            _write_fold_result_json(fold, result, ordinal=exp_cfg.task.ordinal)

    test_fold_metrics = [fr["test_metrics"] for fr in fold_results]
    val_fold_metrics = [fr["val_metrics"] for fr in fold_results]
    elapsed_seconds_total = sum(fr.get("elapsed_seconds", 0) or 0 for fr in fold_results)

    exp_summary = {
        "dataset": exp_cfg.dataset,
        "experiment_id": exp_cfg.experiment_id,
        "task": exp_cfg.task.name,
        "encoder": exp_cfg.encoder_key,
        "embed_dim": exp_cfg.embed_dim,
        "model_type": exp_cfg.model.model_type,
        "survival_loss": exp_cfg.survival_loss,
        "framework": exp_cfg.framework.value,
        "strategy": exp_cfg.strategy,
        "n_folds": exp_cfg.n_folds,
        "fold_indices": list(exp_cfg.selected_folds),
        "elapsed_seconds_total": elapsed_seconds_total,
        "seed": exp_cfg.train.seed,
        "test": compute_confidence_intervals(test_fold_metrics),
        "val": compute_confidence_intervals(val_fold_metrics),
        # CR-3: pooled cross-fold val concordance (survival only; {} otherwise).
        "val_pooled": pooled_val_block(fold_results),
        "per_fold_test": test_fold_metrics,
        "per_fold_val": val_fold_metrics,
        # A4': positional with per_fold_val; one hash home — see
        # val_prediction_hashes.
        "per_fold_val_predictions_sha256": val_prediction_hashes(fold_results),
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=2)

    return exp_summary
