"""Training wrapper around CLAM's core_utils.train().

Calls CLAM's train() directly for all training logic.
Only adds: seed_everything, fold-level resume, extended metrics,
predictions CSV output, and optional wandb logging.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from autobench.pipeline.clam._imports import clam_train, initiate_model, summary
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.determinism import seed_everything
from autobench.pipeline.evaluate import compute_extended_metrics

# CLAM's core_utils and utils.utils use a module-level ``device`` variable.
# We patch it to match the device selected by the benchmark runner so that
# data movement inside CLAM is consistent with where we place the model.
import utils.core_utils as _clam_core_utils  # noqa: E402
import utils.utils as _clam_utils  # noqa: E402


def _set_clam_device(device: torch.device) -> None:
    """Align CLAM's module-level device with the benchmark's chosen device."""
    _clam_core_utils.device = device
    _clam_utils.device = device


# ---------------------------------------------------------------------------
# Reproducibility
# CLAM has an identical seed_torch() in main.py, but it's defined after
# module-level parser.parse_args() and captures a module-level device
# variable — importing it triggers argparse side effects. We use the shared
# autobench.pipeline.determinism.seed_everything (L-2), which replicates the
# same logic and is the canonical copy every other trainer imports too.
# ---------------------------------------------------------------------------
# Build the args namespace that CLAM's train() expects
# ---------------------------------------------------------------------------


def _make_clam_args(
    exp_cfg: ExperimentConfig,
    fold_dir: str,
    *,
    log_data: bool = False,
    policy_runtime: Any | None = None,
) -> SimpleNamespace:
    """Construct the full args namespace expected by clam_train()."""
    return SimpleNamespace(
        # Model
        model_type=exp_cfg.model.model_type,
        model_size=exp_cfg.model.model_size,
        n_classes=exp_cfg.task.n_classes,
        embed_dim=exp_cfg.embed_dim,
        drop_out=exp_cfg.model.dropout,
        subtyping=(exp_cfg.task.n_classes > 2),
        B=exp_cfg.model.B,
        # H-3b: read from ModelConfig instead of hardcoding. Defaults are the
        # previous literals, so this changes no dispatched experiment.
        inst_loss=exp_cfg.model.inst_loss,
        no_inst_cluster=exp_cfg.model.no_inst_cluster,
        bag_weight=exp_cfg.model.bag_weight,
        bag_loss=exp_cfg.model.bag_loss,
        # Training
        max_epochs=exp_cfg.train.max_epochs,
        opt=exp_cfg.train.optimizer,
        lr=exp_cfg.train.lr,
        reg=exp_cfg.train.weight_decay,
        early_stopping=exp_cfg.train.early_stopping,
        patience=exp_cfg.train.patience,
        stop_epoch=exp_cfg.train.stop_epoch,
        weighted_sample=exp_cfg.train.weighted_sample,
        # Infrastructure
        results_dir=fold_dir,
        log_data=log_data,
        testing=False,
        policy_runtime=policy_runtime,
    )


# ---------------------------------------------------------------------------
# Full fold training
# ---------------------------------------------------------------------------


def train_fold(
    exp_cfg: ExperimentConfig,
    train_split: Any,
    val_split: Any,
    test_split: Any,
    fold: int,
    results_dir: str,
    device: torch.device,
    wandb_project: str | None = None,
    policy_runtime: Any | None = None,
) -> dict[str, Any]:
    """Train one fold.

    Delegates entirely to CLAM's train() for model creation, training,
    validation, and evaluation. Adds fold-level resume, extended metrics,
    and predictions CSV on top.
    """
    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    predictions_path = os.path.join(fold_dir, "predictions.csv")
    metrics_path = os.path.join(fold_dir, "metrics.json")

    # Ensure CLAM's internal device matches our device
    _set_clam_device(device)

    # Resume: skip if already completed
    if os.path.exists(predictions_path) and os.path.exists(metrics_path):
        print(f"\n    [fold {fold}] Already completed, loading from disk")
        with open(metrics_path) as f:
            return json.load(f)

    seed_everything(exp_cfg.train.seed)

    # --- wandb setup (captures CLAM's tensorboard writes automatically) ---
    wb_run = None
    if wandb_project:
        import wandb

        wb_run = wandb.init(
            project=wandb_project,
            group=exp_cfg.experiment_id,
            name=f"{exp_cfg.task.name}/{exp_cfg.encoder_key}/{exp_cfg.model.model_type}/fold_{fold}",
            tags=[exp_cfg.task.name, exp_cfg.encoder_key, exp_cfg.model.model_type],
            config={
                "task": exp_cfg.task.name,
                "encoder": exp_cfg.encoder_key,
                "embed_dim": exp_cfg.embed_dim,
                "model_type": exp_cfg.model.model_type,
                "model_size": exp_cfg.model.model_size,
                "dropout": exp_cfg.model.dropout,
                "bag_weight": exp_cfg.model.bag_weight,
                "B": exp_cfg.model.B,
                "fold": fold,
                "max_epochs": exp_cfg.train.max_epochs,
                "lr": exp_cfg.train.lr,
                "weight_decay": exp_cfg.train.weight_decay,
                "seed": exp_cfg.train.seed,
                "n_folds": exp_cfg.n_folds,
            },
            sync_tensorboard=True,
            reinit="finish_previous",
        )

    # --- Call CLAM's train() directly ---
    # Enable tensorboard (log_data) when wandb is active so CLAM writes
    # per-epoch metrics that wandb captures via sync_tensorboard.
    args = _make_clam_args(
        exp_cfg,
        fold_dir,
        log_data=bool(wandb_project),
        policy_runtime=policy_runtime,
    )
    datasets = (train_split, val_split, test_split)

    _timer_start = time.perf_counter()

    test_results_dict, test_auc, val_auc, test_acc, val_acc = clam_train(
        datasets, fold, args,
    )

    # --- Extended metrics (not in CLAM) ---
    slide_ids = list(test_results_dict.keys())
    all_probs = np.array([test_results_dict[s]["prob"] for s in slide_ids]).squeeze()
    all_labels = np.array([test_results_dict[s]["label"] for s in slide_ids], dtype=int)
    all_preds = all_probs.argmax(axis=1)

    test_metrics = compute_extended_metrics(
        all_labels, all_probs, all_preds, exp_cfg.task.n_classes,
        ordinal=exp_cfg.task.ordinal,
    )

    # Re-run summary on val to get extended val metrics
    # (CLAM's train() only returns val_auc and val_acc)
    ckpt_path = os.path.join(fold_dir, f"s_{fold}_checkpoint.pt")
    if os.path.exists(ckpt_path):
        # Load the model CLAM saved to get val predictions
        from autobench.pipeline.clam._imports import get_split_loader

        val_loader = get_split_loader(val_split)
        # Reconstruct model from checkpoint for val summary
        val_results_dict, _, _, _ = summary(
            initiate_model(args, ckpt_path, device),
            val_loader,
            exp_cfg.task.n_classes,
        )
        val_slide_ids = list(val_results_dict.keys())
        val_probs = np.array([val_results_dict[s]["prob"] for s in val_slide_ids]).squeeze()
        val_labels = np.array([val_results_dict[s]["label"] for s in val_slide_ids], dtype=int)
        val_preds = val_probs.argmax(axis=1)
        val_metrics = compute_extended_metrics(
            val_labels, val_probs, val_preds, exp_cfg.task.n_classes,
            ordinal=exp_cfg.task.ordinal,
        )
    else:
        val_metrics = {"auc_roc": val_auc, "accuracy": val_acc}

    # --- Log final metrics to wandb ---
    if wb_run is not None:
        import wandb

        # Val-firewall: test metrics are sealed during search. Only surface
        # them on the wandb dashboard for an explicit certify pass.
        if os.environ.get("AUTOMIL_CERTIFY"):
            wandb.log({f"final_test/{k}": v for k, v in test_metrics.items()})
        wandb.log({f"final_val/{k}": v for k, v in val_metrics.items()})
        wandb.finish()

    # --- Save predictions CSV (not in CLAM) ---
    pred_data = {"slide_id": slide_ids, "y_true": all_labels}
    for i in range(all_probs.shape[1]):
        pred_data[f"y_prob_{i}"] = all_probs[:, i]
    pred_data["y_hat"] = all_preds
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(predictions_path, index=False)

    # --- Save fold metrics JSON (not in CLAM) ---
    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "fold": fold,
        # FOLD-TIMING CONTRACT: covers the whole fold, final evaluation and
        # prediction writing included, so the number is comparable across arms
        # and task types.
        "elapsed_seconds": time.perf_counter() - _timer_start,
    }
    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result
