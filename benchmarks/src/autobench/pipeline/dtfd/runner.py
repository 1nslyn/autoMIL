"""DTFD-MIL experiment runner: all folds for one (task, encoder, strategy) combo.

Clone of ``nnmil/runner.py`` — same summary schema, same per-fold archive
contract — but drives the real two-tier DTFD trainer. DTFD reuses nnMIL's
H5-bag prep, so the patch-feature directory is resolved from the nnMIL
``dataset_plan.json`` (its ``feature_dir``), guaranteeing DTFD and nnMIL read
byte-identical bags from a single source of truth.
"""

from __future__ import annotations

import json
import os
import time

import torch

from autobench.pipeline.clam.runner import _write_fold_result_json
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.dtfd.dataset import load_dtfd_split, load_dtfd_survival_split
from autobench.pipeline.dtfd.survival_train import train_dtfd_survival_fold
from autobench.pipeline.dtfd.train import train_dtfd_fold
from autobench.pipeline.hparams import apply_overrides, overrides_from_exp_cfg
from autobench.pipeline.evaluate import compute_confidence_intervals, pooled_val_block


def _resolve_h5_dir(benchmark_dir: str, exp_cfg: ExperimentConfig) -> str:
    """Read the H5 feature dir from the nnMIL plan (shared prep artifact).

    nnMIL's own prep (``nnmil/prepare.py``) appends ``_{survival_loss}`` to
    the plan leaf for survival tasks (each loss gets its own plan, since
    ``num_classes``/binning differ) — mirror that suffix here so survival
    experiments resolve to the correct plan directory instead of falling
    back to the classification one.
    """
    leaf = f"{exp_cfg.task.name}_{exp_cfg.encoder_key}"
    if exp_cfg.survival_loss is not None:
        leaf = f"{leaf}_{exp_cfg.survival_loss}"
    plan_path = os.path.join(
        benchmark_dir, "nnmil", exp_cfg.strategy, leaf, "dataset_plan.json",
    )
    if not os.path.exists(plan_path):
        raise FileNotFoundError(
            f"nnMIL/DTFD plan not found: {plan_path}. Run data preparation "
            "first (DTFD reuses nnMIL's H5-bag prep)."
        )
    with open(plan_path) as f:
        plan = json.load(f)
    h5_dir = plan.get("feature_dir")
    if not h5_dir:
        raise ValueError(f"plan {plan_path} has no 'feature_dir' entry")
    return h5_dir


def run_dtfd_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
    cfg: DTFDConfig | None = None,
    results_dir: str | None = None,
) -> dict:
    """Run all folds for a single DTFD-MIL experiment; return aggregated summary.

    Signature mirrors ``run_nnmil_experiment`` so orchestrator / ``_gpu_worker``
    dispatch is uniform. Emits the SAME ``summary.json`` schema as CLAM/nnMIL and
    per-fold ``fold_<i>_result.json`` archive files (for autoMIL cap-kill
    reconcile), plus per-fold ``metrics.json``.
    """
    if cfg is None:
        cfg = DTFDConfig()
    # H-3 (audit 2026-07-23): DTFD read ONLY `seed` off exp_cfg, so CLI flags and
    # agentic variant args were silently discarded. DTFDConfig stays the source of
    # truth for its paper-exact defaults (lr Main:29, wd Main:30); only
    # explicitly-set overrides are layered on (note: its field is `wd`, mapped by
    # FIELD_ALIASES so a canonical `weight_decay` override still lands).
    cfg = apply_overrides(cfg, overrides_from_exp_cfg(exp_cfg), arm="dtfd")
    torch_device = torch.device(device)

    # CR-5 (audit 2026-07-23): honor an explicit isolated results_dir
    # (AUTOMIL_RESULTS_DIR under the orchestrator) so per-fold metrics.json is
    # never resumed across experiments/seeds/variants. Falls back to the shared
    # benchmark_dir path for standalone (non-orchestrated) runs.
    if results_dir is None:
        results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    exp_cfg.save(os.path.join(results_dir, "config.json"))

    if exp_cfg.is_survival and exp_cfg.survival_loss == "cox":
        raise ValueError(
            "DTFD-MIL survival does not support cox: its two-tier pseudo-bag "
            "distillation supervises each pseudo-bag with the slide's own "
            "target, but cox's partial-likelihood loss needs a risk set of "
            "different patients' relative event-time ordering -- there is no "
            "such comparison within one slide's pseudo-bags. nllsurv (a "
            "discrete time-bin classification problem) is the only survival "
            "loss DTFD supports."
        )

    h5_dir = _resolve_h5_dir(benchmark_dir, exp_cfg)
    task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{exp_cfg.task.name}.csv")
    splits_dir = os.path.join(benchmark_dir, "splits", exp_cfg.strategy, exp_cfg.task.name)
    label_dict = exp_cfg.task.label_dict
    num_classes = exp_cfg.task.n_classes

    fold_results: list[dict] = []
    for fold in range(exp_cfg.n_folds):
        fold_dir = os.path.join(results_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        metrics_path = os.path.join(fold_dir, "metrics.json")

        # Resume: skip completed folds (matches nnmil/train).
        if os.path.exists(metrics_path):
            print(f"    [DTFD fold {fold}] already completed, loading from disk")
            with open(metrics_path) as f:
                result = json.load(f)
            fold_results.append(result)
            _write_fold_result_json(fold, result)
            continue

        split_csv = os.path.join(splits_dir, f"splits_{fold}.csv")
        if exp_cfg.is_survival:
            train_samples = load_dtfd_survival_split(task_csv, split_csv, h5_dir, "train")
            val_samples = load_dtfd_survival_split(task_csv, split_csv, h5_dir, "val")
            test_samples = load_dtfd_survival_split(task_csv, split_csv, h5_dir, "test")
            raw = train_dtfd_survival_fold(
                train_samples, val_samples, test_samples,
                embed_dim=exp_cfg.embed_dim, nll_bins=exp_cfg.task.nll_bins,
                cfg=cfg, device=torch_device, seed=exp_cfg.train.seed + fold,
            )
            elapsed_seconds = int(raw.get("elapsed_seconds", 0) or 0)
        else:
            train_slides = load_dtfd_split(task_csv, split_csv, h5_dir, label_dict, "train")
            val_slides = load_dtfd_split(task_csv, split_csv, h5_dir, label_dict, "val")
            test_slides = load_dtfd_split(task_csv, split_csv, h5_dir, label_dict, "test")
            start = time.time()
            raw = train_dtfd_fold(
                train_slides, val_slides, test_slides,
                embed_dim=exp_cfg.embed_dim, num_classes=num_classes,
                cfg=cfg, device=torch_device, seed=exp_cfg.train.seed + fold,
            )
            elapsed_seconds = int(time.time() - start)

        result = {
            "test_metrics": raw["test_metrics"],
            "val_metrics": raw["val_metrics"],
            # CR-3: carry the val risk records through for pooled concordance.
            **({"val_records": raw["val_records"]} if "val_records" in raw else {}),
            "fold": fold,
            "elapsed_seconds": elapsed_seconds,
        }
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)

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
