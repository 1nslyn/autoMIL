"""TITAN experiment runner: all folds for one (task, strategy) combo.

TITAN pins ``encoder_key="titan"`` (design spec §7 -- TITAN *is* the
encoder, there is no tile-encoder axis to sweep), so the grid yields
exactly one experiment per (task, fold) with the single ``titan`` model.
Structurally identical to ``nnmil/runner.py``: per-fold ``metrics.json``,
aggregated ``summary.json`` via the shared ``compute_confidence_intervals``,
and the same autoMIL ``fold_<i>_result.json`` archive hook.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.evaluate import (
    compute_confidence_intervals,
    pooled_val_block,
    val_prediction_hashes,
)
from autobench.pipeline.results_cache import resolve_results_dir
from autobench.pipeline.titan.config import TitanHeadConfig, apply_train_overrides
from autobench.pipeline.titan.dataset import build_split_dataset, build_survival_split_dataset
from autobench.pipeline.titan.survival_train import train_titan_survival_fold
from autobench.pipeline.titan.train import train_titan_fold
from autobench.pipeline.policy_dispatch import PolicyRuntime

import pandas as pd


def run_titan_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single TITAN experiment and return aggregated results.

    Returns a summary dict in the SAME schema as the CLAM/nnMIL runners,
    enabling seamless aggregation. Per-fold archive files
    (``fold_<i>_result.json`` under ``AUTOMIL_RESULTS_DIR``) are written
    via the shared helper from ``clam.runner`` -- without these, autoMIL's
    cap-killed reconcile path would always report ``partial_folds=0`` for
    a cap-killed TITAN node even if several folds had completed.
    """
    manifest_path = os.path.join(
        benchmark_dir, "titan", exp_cfg.task.name, "manifest.json",
    )
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"TITAN manifest not found: {manifest_path}. "
            "Run data preparation first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)
    features_dir = manifest["features_dir"]

    # Size the linear probe from the ACTUAL detected slide-embedding dim
    # (design spec §7: never hard-code 768). Replace the grid placeholder
    # embed_dim so train_titan_fold builds nn.Linear(true_dim, n_classes)
    # and config.json/summary record the real dimension.
    exp_cfg = replace(exp_cfg, embed_dim=int(manifest["embed_dim"]))
    policy_runtime = PolicyRuntime.from_experiment(exp_cfg)

    # H-3 / CR-5b: apply the opaque channel's train-side slice (max_epochs,
    # early_stopping) HERE — before results-dir resolution and exp_cfg.save —
    # so cache identity and the archived config.json record the effective
    # values. The trainers read exp_cfg.train as already-effective.
    apply_train_overrides(exp_cfg)

    # CR-5 (audit 2026-07-23): honor an explicit isolated results_dir
    # (AUTOMIL_RESULTS_DIR under the orchestrator) so per-fold metrics.json is
    # never resumed across experiments/seeds/variants. Falls back to the shared
    # benchmark_dir path for standalone (non-orchestrated) runs.
    # CR-5b: the head config is built inside the trainer, so stamp its *defaults*
    # here — the overrides layered on top of them come from exp_cfg, which is
    # already in the fingerprint. Together that covers TITAN's whole surface.
    _head_cfg = TitanHeadConfig()
    results_dir = resolve_results_dir(
        exp_cfg, benchmark_dir, results_dir, arm_cfg=_head_cfg,
    )
    # H-3: TITAN is genuinely MIXED -- the probe's lr/weight_decay/patience come
    # from TitanHeadConfig (1e-3 / 1e-4), while max_epochs and seed are read off
    # the shared TrainConfig. The emitted `train_fields_superseded_by_arm` names
    # exactly which side won for each field, so the split is readable from the
    # artifact instead of only from this comment.
    exp_cfg.save(os.path.join(results_dir, "config.json"), arm_cfg=_head_cfg)

    task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{exp_cfg.task.name}.csv")
    task_df = pd.read_csv(task_csv)

    fold_results: list[dict] = []
    for fold in exp_cfg.selected_folds:
        fold_policy_runtime = policy_runtime.for_fold()
        split_csv = os.path.join(
            benchmark_dir, "splits", exp_cfg.strategy, exp_cfg.task.name,
            f"splits_{fold}.csv",
        )
        if exp_cfg.is_survival:
            train_ds = build_survival_split_dataset(split_csv, "train", task_df, features_dir)
            val_ds = build_survival_split_dataset(split_csv, "val", task_df, features_dir)
            test_ds = build_survival_split_dataset(split_csv, "test", task_df, features_dir)
            result = train_titan_survival_fold(
                exp_cfg, train_ds, val_ds, test_ds, fold, results_dir, device=device,
                policy_runtime=fold_policy_runtime,
            )
        else:
            train_ds = build_split_dataset(
                split_csv, "train", task_df, exp_cfg.task.label_dict, features_dir,
            )
            val_ds = build_split_dataset(
                split_csv, "val", task_df, exp_cfg.task.label_dict, features_dir,
            )
            test_ds = build_split_dataset(
                split_csv, "test", task_df, exp_cfg.task.label_dict, features_dir,
            )
            result = train_titan_fold(
                exp_cfg, train_ds, val_ds, test_ds, fold, results_dir, device=device,
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
