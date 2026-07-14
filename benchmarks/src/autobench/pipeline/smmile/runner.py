"""SMMILe experiment runner: all folds for one (task, encoder) combo.

Produces the same summary schema as CLAM and nnMIL runners.
Patch-level detection scores are saved per fold for tumor ROI extraction.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import torch

from autobench.pipeline.evaluate import compute_confidence_intervals
from autobench.pipeline.smmile.config import SMMILeConfig
from autobench.pipeline.smmile.dataset import create_smmile_split
from autobench.pipeline.smmile.train import train_smmile_fold
from autobench.config import DatasetConfig


def run_smmile_experiment(
    task_name: str,
    encoder_key: str,
    benchmark_dir: str,
    smmile_dir: str,
    encoder_dims: dict[str, int] | None = None,
    label_dict: dict[str, int] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    device: torch.device | None = None,
    cfg: SMMILeConfig | None = None,
    split_subdir: str | None = None,
) -> dict:
    """Run all folds of SMMILe for a single (task, encoder) combination.

    Returns summary dict with per-fold and aggregated metrics.
    Patch-level detection scores are saved under each fold directory
    for downstream tumor ROI visualization and pathologist review.
    """
    if cfg is None:
        cfg = SMMILeConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if encoder_dims is None:
        raise ValueError("encoder_dims must be provided (from DatasetConfig.encoder_dims)")
    fea_dim = encoder_dims[encoder_key]
    npy_dir = os.path.join(smmile_dir, "features_npy", encoder_key)
    sp_dir = os.path.join(smmile_dir, "superpixels", encoder_key)

    # Load and encode labels
    csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")
    slide_data = pd.read_csv(csv_path)
    if label_dict is None:
        label_dict = {"neg": 0, "pos": 1}  # fallback for binary tasks
    slide_data["label"] = slide_data["label"].map(label_dict)

    # Filter to slides with NIC files
    has_npy = slide_data["slide_id"].apply(
        lambda sid: os.path.exists(os.path.join(npy_dir, f"{sid}_0_{cfg.patch_size}.npy"))
    )
    n_missing = (~has_npy).sum()
    if n_missing > 0:
        missing = slide_data.loc[~has_npy, "slide_id"].tolist()
        print(f"[WARNING] Dropping {n_missing} slides missing NIC files: {missing[:5]}")
        slide_data = slide_data[has_npy].reset_index(drop=True)

    # Results path
    exp_code = f"smmile_single_{encoder_key}"
    results_dir = os.path.join(
        smmile_dir, "results", task_name, encoder_key, f"{exp_code}_s{seed}",
    )
    os.makedirs(results_dir, exist_ok=True)

    # Splits
    if split_subdir is None:
        split_subdir = task_name
    splits_dir = os.path.join(benchmark_dir, "splits", split_subdir)

    # Per-fold archive helper: cap-killed reconcile walks
    # archive/<node>/fold_*_result.json; nnMIL and CLAM both emit these
    # via _write_fold_result_json, so SMMILe must too for symmetric
    # autoMIL recovery on cap-killed runs.
    from autobench.pipeline.clam.runner import _write_fold_result_json

    fold_results: list[dict] = []
    for fold in range(n_folds):
        print(f"\n  [{task_name}/{encoder_key}] Fold {fold}/{n_folds}")
        split_csv = os.path.join(splits_dir, f"splits_{fold}.csv")

        train_ds, val_ds, test_ds = create_smmile_split(
            slide_data, split_csv, npy_dir, sp_dir, cfg.patch_size,
        )
        print(f"    train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        result = train_smmile_fold(
            train_ds, val_ds, test_ds,
            fold=fold, results_dir=results_dir,
            fea_dim=fea_dim, device=device, seed=seed, cfg=cfg,
        )
        fold_results.append(result)
        _write_fold_result_json(fold, result)

    # Aggregate across folds
    test_fold_metrics = [fr["test_metrics"] for fr in fold_results]
    val_fold_metrics = [fr["val_metrics"] for fr in fold_results]

    summary = {
        "experiment_id": f"smmile__{task_name}__{encoder_key}__s{seed}",
        "task": task_name,
        "encoder": encoder_key,
        "embed_dim": fea_dim,
        "model_type": "smmile_single",
        "framework": "smmile",
        "strategy": split_subdir,
        "n_folds": n_folds,
        "seed": seed,
        "test": compute_confidence_intervals(test_fold_metrics),
        "val": compute_confidence_intervals(val_fold_metrics),
        "per_fold_test": test_fold_metrics,
        "per_fold_val": val_fold_metrics,
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: {summary_path}")
    if os.environ.get("AUTOMIL_CERTIFY"):  # val-firewall: seal test from stdout during search
        for metric in ("auc_roc", "balanced_accuracy"):
            if metric in summary["test"]:
                m = summary["test"][metric]
                print(f"    test_{metric}: {m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]")

    return summary
