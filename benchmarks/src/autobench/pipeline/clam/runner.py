"""Single-experiment runner: all folds for one (task, encoder, model) combo."""

from __future__ import annotations

import json
import os

import torch

from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.clam.dataset import create_dataset, load_fold_splits
from autobench.pipeline.evaluate import compute_confidence_intervals
from autobench.pipeline.clam.train import train_fold


def _write_fold_result_json(fold_index: int, result: dict) -> None:
    """Write archive/<node>/fold_<i>_result.json per autoMIL CAP-03 / D-118.

    No-op when AUTOMIL_RESULTS_DIR is unset (e.g., running outside the
    autoMIL orchestrator).

    Pitfall 5: compute_extended_metrics() returns flat floats at the per-fold
    level. However, this helper defensively unwraps both flat-float and
    {"mean": ..., "std": ...} CI-dict shapes in case the caller supplies a
    post-CI metrics dict.
    """
    import json as _json
    from pathlib import Path

    results_dir = os.environ.get("AUTOMIL_RESULTS_DIR")
    if not results_dir:
        return  # not running under autoMIL orchestrator

    # AUTOMIL_FOLD_COUNT is set by the orchestrator from
    # `automil/config.yaml: training.fold_count` (default 10 post-Level-D).
    # The literal "10" fallback here matters only when this helper is
    # invoked outside the orchestrator (e.g. local debugging without env
    # vars); production paths always have it set.
    fold_count = int(os.environ.get("AUTOMIL_FOLD_COUNT", "10"))

    def _unwrap(metric):
        # auc_roc may be float OR {"mean": float, ...} (CI-dict shape)
        if isinstance(metric, dict):
            return float(metric.get("mean", 0.0))
        try:
            return float(metric or 0.0)
        except (TypeError, ValueError):
            return 0.0

    test_m = result.get("test_metrics", {}) or {}
    val_m = result.get("val_metrics", {}) or {}
    if "c_index" in test_m:
        # Survival: composite is the test concordance index.
        metrics = {
            "val_c_index":  _unwrap(val_m.get("c_index")),
            "test_c_index": _unwrap(test_m.get("c_index")),
        }
        composite = metrics["test_c_index"]
    else:
        metrics = {
            "val_auc":   _unwrap(val_m.get("auc_roc")),
            "val_bacc":  _unwrap(val_m.get("balanced_accuracy")),
            "test_auc":  _unwrap(test_m.get("auc_roc")),
            "test_bacc": _unwrap(test_m.get("balanced_accuracy")),
        }
        composite = (metrics["test_auc"] + metrics["test_bacc"]) / 2.0

    payload = {
        "fold_index":      fold_index,
        "fold_count":      fold_count,
        "status":          "completed",
        "metrics":         metrics,
        "composite":       composite,
        "elapsed_seconds": int(result.get("elapsed_seconds", 0) or 0),
        "peak_vram_mb":    int(result.get("peak_vram_mb", 0) or 0),
    }
    fold_path = Path(results_dir) / f"fold_{fold_index}_result.json"
    fold_path.write_text(_json.dumps(payload, indent=2))


def run_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: torch.device,
    wandb_project: str | None = None,
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single CLAM experiment and return aggregated results."""
    if results_dir is None:
        results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    exp_cfg.save(os.path.join(results_dir, "config.json"))

    task_csv_name = exp_cfg.task.name

    dataset = create_dataset(exp_cfg, benchmark_dir, task_csv_name=task_csv_name)

    # Splits directory: splits/{strategy}/{task}/
    splits_subdir = os.path.join(exp_cfg.strategy, exp_cfg.task.name)

    fold_results: list[dict] = []
    for fold in range(exp_cfg.n_folds):
        train_split, val_split, test_split = load_fold_splits(
            dataset, benchmark_dir, splits_subdir, fold,
        )
        result = train_fold(
            exp_cfg, train_split, val_split, test_split,
            fold, results_dir, device, wandb_project=wandb_project,
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
