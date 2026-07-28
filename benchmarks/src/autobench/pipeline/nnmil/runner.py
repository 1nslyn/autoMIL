"""nnMIL experiment runner: all folds for one (task, encoder, model, strategy) combo."""

from __future__ import annotations

import json
import os

from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.evaluate import compute_confidence_intervals, pooled_val_block
from autobench.pipeline.nnmil.prepare import nnmil_plan_dir
from autobench.pipeline.nnmil.train import train_nnmil_fold


def run_nnmil_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single nnMIL experiment and return aggregated results.

    Returns a summary dict in the SAME schema as the CLAM runner
    (``run_experiment``), enabling seamless aggregation.

    Per-fold archive files (fold_<i>_result.json under
    AUTOMIL_RESULTS_DIR) are written via the shared helper from
    ``clam.runner``. The autoMIL cap-killed reconcile path
    (``automil/cells/reconcile.py::reconcile_budget_kill``) walks
    ``archive/<node>/fold_*_result.json``; without these files, nnMIL
    cap-killed nodes would always be reported as ``partial_folds=0``
    and marked crashed even when several folds had completed cleanly.
    """
    # CR-5 (audit 2026-07-23): honor an explicit isolated results_dir
    # (AUTOMIL_RESULTS_DIR under the orchestrator) so per-fold metrics.json is
    # never resumed across experiments/seeds/variants. Falls back to the shared
    # benchmark_dir path for standalone (non-orchestrated) runs.
    if results_dir is None:
        results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    exp_cfg.save(os.path.join(results_dir, "config.json"))

    # Locate the plan file for this (task, encoder, strategy[, survival_loss])
    plan_path = os.path.join(
        nnmil_plan_dir(
            benchmark_dir, exp_cfg.strategy, exp_cfg.task.name,
            exp_cfg.encoder_key, survival_loss=exp_cfg.survival_loss,
        ),
        "dataset_plan.json",
    )
    if not os.path.exists(plan_path):
        raise FileNotFoundError(
            f"nnMIL plan not found: {plan_path}. "
            f"Run data preparation first."
        )

    fold_results: list[dict] = []
    for fold in range(exp_cfg.n_folds):
        result = train_nnmil_fold(
            exp_cfg, plan_path, fold, results_dir, device=device,
        )
        fold_results.append(result)
        _write_fold_result_json(fold, result)

    test_fold_metrics = [fr["test_metrics"] for fr in fold_results]
    val_fold_metrics = [fr["val_metrics"] for fr in fold_results]
    elapsed_seconds_total = sum(fr.get("elapsed_seconds", 0) or 0 for fr in fold_results)

    exp_summary = {
        "experiment_id": exp_cfg.experiment_id,
        "task": exp_cfg.task.name,
        "encoder": exp_cfg.encoder_key,
        "embed_dim": exp_cfg.embed_dim,
        "model_type": exp_cfg.model.model_type,
        "survival_loss": exp_cfg.survival_loss,
        "framework": exp_cfg.framework.value,
        "strategy": exp_cfg.strategy,
        "n_folds": exp_cfg.n_folds,
        "elapsed_seconds_total": elapsed_seconds_total,
        "seed": exp_cfg.train.seed,
        "test": compute_confidence_intervals(test_fold_metrics),
        "val": compute_confidence_intervals(val_fold_metrics),
        # CR-3: pooled cross-fold val concordance (survival only; {} otherwise).
        "val_pooled": pooled_val_block(fold_results),
        "per_fold_test": test_fold_metrics,
        "per_fold_val": val_fold_metrics,
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=2)

    return exp_summary
