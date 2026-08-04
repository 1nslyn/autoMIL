"""ABMIL experiment runner: all folds for one (task, encoder, model, strategy) combo.

Clone of ``dtfd/runner.py`` -- same summary schema, same per-fold archive
contract, same resume-by-metrics.json behavior -- but drives the standard
one-tier ABMIL trainer. ABMIL reuses nnMIL's H5-bag prep, so the patch-feature
directory is resolved from the nnMIL ``dataset_plan.json`` (its
``feature_dir``), guaranteeing ABMIL and nnMIL/DTFD read byte-identical bags
from a single source of truth.
"""

from __future__ import annotations

import json
import os

import torch

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.abmil.dataset import load_abmil_split, load_abmil_survival_split
from autobench.pipeline.abmil.survival_train import train_abmil_survival_fold
from autobench.pipeline.abmil.train import train_abmil_fold
from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.hparams import all_overrides, apply_overrides
from autobench.pipeline.results_cache import resolve_results_dir
from autobench.pipeline.evaluate import compute_confidence_intervals, pooled_val_block
from autobench.pipeline.policy_dispatch import PolicyRuntime


def _resolve_h5_dir(benchmark_dir: str, exp_cfg: ExperimentConfig) -> str:
    """Read the H5 feature dir from the nnMIL plan (shared prep artifact)."""
    leaf = f"{exp_cfg.task.name}_{exp_cfg.encoder_key}"
    if exp_cfg.survival_loss is not None:
        leaf = f"{leaf}_{exp_cfg.survival_loss}"
    plan_path = os.path.join(
        benchmark_dir, "nnmil", exp_cfg.strategy, leaf, "dataset_plan.json",
    )
    if not os.path.exists(plan_path):
        raise FileNotFoundError(
            f"nnMIL/ABMIL plan not found: {plan_path}. Run data preparation "
            "first (ABMIL reuses nnMIL's H5-bag prep)."
        )
    with open(plan_path) as f:
        plan = json.load(f)
    h5_dir = plan.get("feature_dir")
    if not h5_dir:
        raise ValueError(f"plan {plan_path} has no 'feature_dir' entry")
    return h5_dir


def run_abmil_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
    cfg: ABMILConfig | None = None,
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single ABMIL experiment; return aggregated summary.

    Signature mirrors ``run_dtfd_experiment``/``run_nnmil_experiment`` so
    orchestrator / ``_gpu_worker`` dispatch is uniform. Emits the SAME
    ``summary.json`` schema as CLAM/nnMIL/DTFD and per-fold
    ``fold_<i>_result.json`` archive files (for autoMIL cap-kill reconcile),
    plus per-fold ``metrics.json``.
    """
    if cfg is None:
        cfg = ABMILConfig()
    # H-3 (audit 2026-07-23): ABMIL previously read ONLY `seed` off exp_cfg, so
    # --lr / --max_epochs / --patience and any agentic variant's CLAM_ARGS were
    # silently discarded. ABMILConfig stays the source of truth for defaults;
    # explicitly-set overrides are layered on top (None values are dropped, so an
    # unset flag can never pull this arm onto the shared schedule).
    cfg = apply_overrides(cfg, all_overrides(exp_cfg), arm="abmil")
    torch_device = torch.device(device)
    policy_runtime = PolicyRuntime.from_experiment(exp_cfg)

    # CR-5 (audit 2026-07-23): honor an explicit isolated results_dir
    # (AUTOMIL_RESULTS_DIR under the orchestrator) so per-fold metrics.json is
    # never resumed across experiments/seeds/variants. Falls back to the shared
    # benchmark_dir path for standalone (non-orchestrated) runs.
    # CR-5b: `cfg` is passed too — ABMIL's M/L/dropout live outside exp_cfg, so a
    # change there must invalidate this cache as surely as a change to lr.
    results_dir = resolve_results_dir(exp_cfg, benchmark_dir, results_dir, arm_cfg=cfg)
    # H-3: ABMIL trains off ABMILConfig (lr 5e-4 / wd 1e-4 / 20 epochs), not the
    # shared TrainConfig that config.json also carries.
    exp_cfg.save(os.path.join(results_dir, "config.json"), arm_cfg=cfg)

    h5_dir = _resolve_h5_dir(benchmark_dir, exp_cfg)
    task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{exp_cfg.task.name}.csv")
    splits_dir = os.path.join(benchmark_dir, "splits", exp_cfg.strategy, exp_cfg.task.name)
    label_dict = exp_cfg.task.label_dict
    num_classes = exp_cfg.task.n_classes
    model_type = exp_cfg.model.model_type

    fold_results: list[dict] = []
    for fold in exp_cfg.selected_folds:
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        metrics_path = os.path.join(fold_dir, "metrics.json")

        # Resume: skip completed folds (matches nnmil/train, dtfd/runner).
        if os.path.exists(metrics_path):
            print(f"    [ABMIL fold {fold}] already completed, loading from disk")
            with open(metrics_path) as f:
                result = json.load(f)
            fold_results.append(result)
            _write_fold_result_json(fold, result)
            continue

        fold_policy_runtime = policy_runtime.for_fold()

        split_csv = os.path.join(splits_dir, f"splits_{fold}.csv")
        if exp_cfg.is_survival:
            train_samples = load_abmil_survival_split(task_csv, split_csv, h5_dir, "train")
            val_samples = load_abmil_survival_split(task_csv, split_csv, h5_dir, "val")
            test_samples = load_abmil_survival_split(task_csv, split_csv, h5_dir, "test")
            raw = train_abmil_survival_fold(
                model_type, train_samples, val_samples, test_samples,
                embed_dim=exp_cfg.embed_dim, survival_loss=exp_cfg.survival_loss or "cox",
                nll_bins=exp_cfg.task.nll_bins, cfg=cfg, device=torch_device,
                seed=exp_cfg.train.seed + fold, fold_dir=fold_dir,
                policy_runtime=fold_policy_runtime,
            )
        else:
            train_slides = load_abmil_split(task_csv, split_csv, h5_dir, label_dict, "train")
            val_slides = load_abmil_split(task_csv, split_csv, h5_dir, label_dict, "val")
            test_slides = load_abmil_split(task_csv, split_csv, h5_dir, label_dict, "test")
            raw = train_abmil_fold(
                model_type, train_slides, val_slides, test_slides,
                embed_dim=exp_cfg.embed_dim, num_classes=num_classes,
                cfg=cfg, device=torch_device, seed=exp_cfg.train.seed + fold,
                policy_runtime=fold_policy_runtime,
            )

        result = {
            "test_metrics": raw["test_metrics"],
            "val_metrics": raw["val_metrics"],
            # CR-3: carry the val risk records through for pooled concordance.
            **({"val_records": raw["val_records"]} if "val_records" in raw else {}),
            "fold": fold,
            "elapsed_seconds": int(raw.get("elapsed_seconds", 0) or 0),
        }
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)

        fold_results.append(result)
        _write_fold_result_json(fold, result)

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
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=2)

    return exp_summary
