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

from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.evaluate import compute_confidence_intervals
from autobench.pipeline.titan.dataset import build_split_dataset
from autobench.pipeline.titan.train import train_titan_fold

import pandas as pd


def run_titan_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Run all folds for a single TITAN experiment and return aggregated results.

    Returns a summary dict in the SAME schema as the CLAM/nnMIL runners,
    enabling seamless aggregation. Per-fold archive files
    (``fold_<i>_result.json`` under ``AUTOMIL_RESULTS_DIR``) are written
    via the shared helper from ``clam.runner`` -- without these, autoMIL's
    cap-killed reconcile path would always report ``partial_folds=0`` for
    a cap-killed TITAN node even if several folds had completed.
    """
    results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    exp_cfg.save(os.path.join(results_dir, "config.json"))

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

    task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{exp_cfg.task.name}.csv")
    task_df = pd.read_csv(task_csv)

    fold_results: list[dict] = []
    for fold in range(exp_cfg.n_folds):
        split_csv = os.path.join(
            benchmark_dir, "splits", exp_cfg.strategy, exp_cfg.task.name,
            f"splits_{fold}.csv",
        )
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
        )
        fold_results.append(result)
        _write_fold_result_json(fold, result)

    test_fold_metrics = [fr["test_metrics"] for fr in fold_results]
    val_fold_metrics = [fr["val_metrics"] for fr in fold_results]

    exp_summary = {
        "experiment_id": exp_cfg.experiment_id,
        "task": exp_cfg.task.name,
        "encoder": exp_cfg.encoder_key,
        "embed_dim": exp_cfg.embed_dim,
        "model_type": exp_cfg.model.model_type,
        "framework": exp_cfg.framework.value,
        "strategy": exp_cfg.strategy,
        "n_folds": exp_cfg.n_folds,
        "seed": exp_cfg.train.seed,
        "test": compute_confidence_intervals(test_fold_metrics),
        "val": compute_confidence_intervals(val_fold_metrics),
        "per_fold_test": test_fold_metrics,
        "per_fold_val": val_fold_metrics,
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=2)

    return exp_summary
