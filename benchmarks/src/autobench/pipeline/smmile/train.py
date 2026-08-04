"""SMMILe single-fold training: two-stage (base + refinement).

Stage 1: Base model with superpixel sampling + weighted dropout
Stage 2: Instance refinement with MRF constraint (loads Stage 1 checkpoint)

After training, patch-level detection scores are saved for tumor ROI extraction.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from autobench.pipeline.determinism import seed_everything
from autobench.pipeline.smmile._imports import SMMILe_SINGLE
from autobench.pipeline.smmile.config import SMMILeConfig
from autobench.pipeline.smmile.dataset import make_smmile_loader
from autobench.pipeline.smmile.evaluate import evaluate_smmile_model


def _create_model(fea_dim: int, cfg: SMMILeConfig, device: torch.device) -> SMMILe_SINGLE:
    """Instantiate SMMILe_SINGLE and move to device."""
    model = SMMILe_SINGLE(
        gate=True,
        size_arg=cfg.model_size,
        dropout=cfg.dropout,
        n_classes=1,
        n_refs=cfg.n_refs,
        drop_rate=cfg.drop_rate,
        fea_dim=fea_dim,
    )
    # Use relocate() which handles all submodule placement
    model.relocate()
    return model


class _EarlyStopping:
    def __init__(self, patience: int = 20, stop_epoch: int = 50):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, epoch: int, val_loss: float, model, ckpt_path: str) -> bool:
        score = -val_loss
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), ckpt_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.should_stop = True
        return self.should_stop


def _train_one_epoch(
    model, loader, optimizer, loss_fn, device, cfg, inst_refinement=False,
) -> float:
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for data, label, cors, inst_label in loader:
        data = data.to(device)
        # SMMILe_SINGLE expects float label for BCE
        label_t = label.to(device).float()

        mask = cors[1]
        sp = cors[2]
        adj = cors[3]

        _, Y_prob, Y_hat, ref_score, results_dict = model(
            data, mask, sp, adj,
            label=label_t,
            group_numbers=cfg.G,
            superpixels=cfg.superpixel,
            sp_smooth=cfg.sp_smooth,
            drop_with_score=cfg.drop_with_score,
            drop_times=cfg.D,
            instance_eval=inst_refinement,
            inst_rate=cfg.inst_rate,
            mrf=cfg.mrf if inst_refinement else False,
            tau=cfg.tau,
            consistency=cfg.consistency,
        )

        loss = loss_fn(Y_prob[0], label_t.float())
        for prob in Y_prob[1:]:
            loss += loss_fn(prob, label_t.float()) / len(Y_prob[1:])

        if inst_refinement:
            inst_loss = results_dict.get("instance_loss", 0)
            if inst_loss > 0:
                loss += inst_loss
            mrf_loss = results_dict.get("mrf_loss", 0)
            if cfg.mrf and mrf_loss > 0:
                loss += mrf_loss

        consist_loss = results_dict.get("consist_loss", 0)
        if consist_loss > 0:
            loss += consist_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _validate(model, loader, loss_fn, device, cfg, inst_refinement=False) -> float:
    """Validate, return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, label, cors, inst_label in loader:
            data = data.to(device)
            label_t = label.to(device).float()
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]

            _, Y_prob, Y_hat, ref_score, results_dict = model(
                data, mask, sp, adj,
                label=label_t,
                superpixels=cfg.superpixel,
                sp_smooth=cfg.sp_smooth,
                group_numbers=cfg.G,
                instance_eval=inst_refinement,
            )

            loss = loss_fn(Y_prob[0], label_t.float())
            for prob in Y_prob[1:]:
                loss += loss_fn(prob, label_t.float()) / len(Y_prob[1:])
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_smmile_fold(
    train_dataset: Any,
    val_dataset: Any,
    test_dataset: Any,
    fold: int,
    results_dir: str,
    fea_dim: int,
    device: torch.device,
    seed: int = 42,
    cfg: SMMILeConfig | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train one fold (two-stage) and return metrics dict.

    Saves patch-level detection scores for tumor ROI extraction after
    Stage 2 refinement completes.

    Returns: {"test_metrics": {...}, "val_metrics": {...}, "fold": int}
    """
    if cfg is None:
        cfg = SMMILeConfig()

    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    metrics_path = os.path.join(fold_dir, "metrics.json")
    predictions_path = os.path.join(fold_dir, "predictions.csv")

    # Resume: skip if already completed
    if os.path.exists(metrics_path) and os.path.exists(predictions_path):
        print(f"    [fold {fold}] Already completed, loading from disk")
        with open(metrics_path) as f:
            return json.load(f)

    seed_everything(seed + fold)

    train_loader = make_smmile_loader(train_dataset, training=True, weighted=cfg.weighted_sample)
    val_loader = make_smmile_loader(val_dataset)
    test_loader = make_smmile_loader(test_dataset)

    loss_fn = F.binary_cross_entropy

    # ---- Stage 1: Base training ----
    print(f"    [fold {fold}] Stage 1: {cfg.stage1_epochs} epochs")
    model = _create_model(fea_dim, cfg, device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    early_stop = _EarlyStopping(patience=cfg.patience, stop_epoch=cfg.stop_epoch)
    s1_ckpt = os.path.join(fold_dir, f"s_{fold}_stage1_best.pt")

    for epoch in range(cfg.stage1_epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg)
        val_loss = _validate(model, val_loader, loss_fn, device, cfg)
        scheduler.step(val_loss)
        if early_stop.step(epoch, val_loss, model, s1_ckpt):
            print(f"      Stage 1 early stop at epoch {epoch}")
            break

    if os.path.exists(s1_ckpt):
        model.load_state_dict(torch.load(s1_ckpt, weights_only=True))

    # ---- Stage 2: Instance refinement ----
    if cfg.inst_refinement:
        print(f"    [fold {fold}] Stage 2: {cfg.stage2_epochs} epochs (refinement + MRF)")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        early_stop = _EarlyStopping(patience=cfg.patience, stop_epoch=10)
        s2_ckpt = os.path.join(fold_dir, f"s_{fold}_checkpoint_best.pt")

        for epoch in range(cfg.stage2_epochs):
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, loss_fn, device, cfg,
                inst_refinement=True,
            )
            val_loss = _validate(
                model, val_loader, loss_fn, device, cfg, inst_refinement=True,
            )
            scheduler.step(val_loss)
            if early_stop.step(epoch, val_loss, model, s2_ckpt):
                print(f"      Stage 2 early stop at epoch {epoch}")
                break

        if os.path.exists(s2_ckpt):
            model.load_state_dict(torch.load(s2_ckpt, weights_only=True))

    # ---- Evaluate (with patch scores for tumor ROI) ----
    test_result = evaluate_smmile_model(
        model, test_loader, device,
        superpixel=cfg.superpixel, sp_smooth=cfg.sp_smooth,
        G=cfg.G, inst_refinement=cfg.inst_refinement,
        extract_patch_scores=True,
    )
    val_result = evaluate_smmile_model(
        model, val_loader, device,
        superpixel=cfg.superpixel, sp_smooth=cfg.sp_smooth,
        G=cfg.G, inst_refinement=cfg.inst_refinement,
    )

    # Save predictions
    pred_df = pd.DataFrame({
        "slide_id": test_result["slide_ids"],
        "y_true": test_result["y_true"],
        "y_prob_1": test_result["y_prob"],
        "y_hat": test_result["y_hat"],
    })
    pred_df.to_csv(predictions_path, index=False)

    # Save per-patch detection scores for tumor ROI extraction
    patch_scores = test_result.get("patch_scores", {})
    if patch_scores:
        roi_dir = os.path.join(fold_dir, "patch_scores")
        os.makedirs(roi_dir, exist_ok=True)
        for sid, score_data in patch_scores.items():
            np.savez_compressed(
                os.path.join(roi_dir, f"{sid}.npz"),
                coords=score_data["coords"],
                det_scores=score_data["det_scores"],
            )
        print(f"    [fold {fold}] Saved patch scores for {len(patch_scores)} slides")

        # Auto-generate heatmap overlays for pathologist review
        # NOTE: wsi_dir must be passed via kwargs or set via env var
        from autobench.pipeline.smmile.visualize import generate_fold_heatmaps

        wsi_dir = kwargs.get("wsi_dir") or os.environ.get("AUTOBENCH_WSI_DIR", "")
        if wsi_dir:
            heatmap_dir = os.path.join(fold_dir, "heatmaps")
            print(f"    [fold {fold}] Generating heatmaps...")
            generate_fold_heatmaps(roi_dir, wsi_dir, heatmap_dir)
        else:
            print(f"    [fold {fold}] Skipping heatmaps (no wsi_dir provided)")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(fold_dir, f"s_{fold}_checkpoint.pt"))

    fold_result = {
        "test_metrics": test_result["metrics"],
        "val_metrics": val_result["metrics"],
        "fold": fold,
    }
    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    return fold_result
