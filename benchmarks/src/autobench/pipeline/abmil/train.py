"""Single-fold ABMIL training: standard CE + Adam loop over full H5 bags.

Unlike DTFD's two-tier pseudo-bag distillation, ABMIL is a STANDARD one-tier
MIL trainer: one forward per slide (full bag, no pseudo-bag split), one
CrossEntropy loss, one Adam optimizer. Early stopping is on val AUC (shared
metric) with best-state restore -- same discipline as ``dtfd/train.py``.
"""

from __future__ import annotations

import os

import copy
import random
import time

import numpy as np
import torch

from autobench.pipeline.abmil.config import ABMILConfig
from autobench.pipeline.abmil.dataset import ABMILSlide
from autobench.pipeline.abmil.model import build_abmil_model
from autobench.pipeline.determinism import seed_everything as _seed_everything
from autobench.pipeline.evaluate import compute_extended_metrics, write_predictions_csv
from autobench.pipeline.policy_dispatch import PolicyRuntime


def _train_one_epoch(
    model: torch.nn.Module,
    slides: list[ABMILSlide],
    ce_cri: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    py_rng: random.Random,
) -> float:
    """Run one training epoch; return mean loss over slides."""
    model.train()
    order = list(range(len(slides)))
    py_rng.shuffle(order)

    losses: list[float] = []
    for i in order:
        slide = slides[i]
        features = slide.features.to(device).unsqueeze(0)  # [1, N, in_dim]
        label_t = torch.LongTensor([slide.label]).to(device)

        out = model(features)
        loss = ce_cri(out["logits"], label_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    slides: list[ABMILSlide],
    num_classes: int,
    device: torch.device,
    ordinal: bool = False,
    predictions_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a split -> shared-schema metrics dict.

    ``predictions_path`` persists the per-slide predictions so any
    confusion-matrix-derived metric added later is a recomputation rather than a
    retrain. This arm previously saved nothing but metrics.json -- not even a
    checkpoint -- so a missing metric was unrecoverable.
    """
    model.eval()

    probs: list[np.ndarray] = []
    labels: list[int] = []
    for slide in slides:
        features = slide.features.to(device).unsqueeze(0)  # [1, N, in_dim]
        out = model(features)
        prob = torch.softmax(out["logits"], dim=1).squeeze(0).cpu().numpy()
        probs.append(prob)
        labels.append(slide.label)

    y_probs = np.vstack(probs)  # [num_slides, num_classes]
    y_true = np.asarray(labels, dtype=int)
    y_pred = y_probs.argmax(axis=1)
    if predictions_path:
        write_predictions_csv(
            predictions_path,
            [getattr(sl, "slide_id", None) for sl in slides],
            y_true, y_probs, y_pred,
        )
    return compute_extended_metrics(y_true, y_probs, y_pred, num_classes, ordinal=ordinal)


def _val_auc(metrics: dict[str, float]) -> float:
    """Validation AUC used for early stopping (NaN-safe -> -inf)."""
    auc = metrics.get("auc_roc", float("nan"))
    return float(auc) if not np.isnan(auc) else float("-inf")


def train_abmil_fold(
    model_type: str,
    train_slides: list[ABMILSlide],
    val_slides: list[ABMILSlide],
    test_slides: list[ABMILSlide],
    embed_dim: int,
    num_classes: int,
    cfg: ABMILConfig,
    device: torch.device,
    seed: int,
    policy_runtime: PolicyRuntime | None = None,
    ordinal: bool = False,
    fold_dir: str | None = None,
) -> dict:
    """Train one ABMIL fold and return shared-schema test/val metrics.

    ``model_type`` selects ``"abmil"`` (non-gated) or ``"abmil_gated"``.
    Restores pristine torch grad state on return so it cannot perturb a
    following experiment's metrics under a shared process (mirrors
    ``dtfd/train.py::train_dtfd_fold``).
    """
    grad_was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        _seed_everything(seed)
        py_rng = random.Random(seed)

        if not train_slides:
            raise ValueError("ABMIL train split is empty; cannot train a fold.")

        model = build_abmil_model(
            model_type, in_dim=embed_dim, num_classes=num_classes,
            M=cfg.M, L=cfg.L, dropout=cfg.dropout,
        ).to(device)
        ce_cri = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        policy_runtime = policy_runtime or PolicyRuntime()
        optimizer = policy_runtime.wrap_optimizer(optimizer)

        best_auc = float("-inf")
        best_snap: dict | None = None
        epochs_no_improve = 0

        start = time.time()
        for _epoch in range(cfg.max_epochs):
            _train_one_epoch(model, train_slides, ce_cri, optimizer, device, py_rng)

            if val_slides:
                cur_metrics = _evaluate(model, val_slides, num_classes, device, ordinal=ordinal)
                cur = _val_auc(cur_metrics)
                if cur > best_auc:
                    best_auc = cur
                    best_snap = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                default_stop = cfg.early_stopping and epochs_no_improve >= cfg.patience
                if policy_runtime.should_stop(
                    default_stop, epoch=_epoch, metrics={"val_auc": cur},
                ):
                    break
        elapsed_seconds = time.time() - start

        if best_snap is not None:
            model.load_state_dict(best_snap)

        test_metrics = (
            _evaluate(model, test_slides, num_classes, device, ordinal=ordinal,
                      predictions_path=(os.path.join(fold_dir, "predictions.csv") if fold_dir else None)) if test_slides else {}
        )
        val_metrics = (
            _evaluate(model, val_slides, num_classes, device, ordinal=ordinal,
                      predictions_path=(os.path.join(fold_dir, "predictions_val.csv") if fold_dir else None)) if val_slides else {}
        )

        return {
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "elapsed_seconds": elapsed_seconds,
        }
    finally:
        torch.set_grad_enabled(grad_was_enabled)
