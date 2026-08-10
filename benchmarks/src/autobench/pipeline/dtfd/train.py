"""Single-fold DTFD-MIL training: faithful two-tier loop over the reference.

Ports ``train_attention_preFeature_DTFD`` (``lib/DTFD-MIL/Main_DTFD_MIL.py:272-376``):
per slide a random pseudo-bag split (re-drawn every epoch), a tier-1 forward
(DimReduction + gated Attention + Classifier_1fc) per pseudo-bag producing the
pseudo-bag prediction and its AFS distilled feature, then a tier-2
Attention_with_Classifier over the stacked pseudo-bag features. Two optimizers
over disjoint module groups; ``loss0.backward(retain_graph=True)`` then
``loss1.backward()`` with per-module grad clipping between (Main:351-365).

Fidelity notes:
- ``distill='AFS'`` is locked, so the pseudo-bag feature is the attention-feature
  sum (``af_inst_feat``, Main:337/343-344). The CAM-based top-k ranking
  (Main:326-335) is only consumed by MaxS/MaxMinS and is intentionally omitted.
- Early stopping is on val AUC (shared metric) with best-state restore — a clean,
  fair replacement for the reference's best-after-80%-epochs bookkeeping. The
  optimization math per step is unchanged.
"""

from __future__ import annotations

import os

import copy
import random
import time

import numpy as np
import torch

from autobench.pipeline.determinism import seed_everything as _seed_everything
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.dtfd.dataset import DTFDSlide, min_bag_size
from autobench.pipeline.dtfd.eval import evaluate_dtfd, val_auc
from autobench.pipeline.dtfd.model import DTFDBundle, build_dtfd_bundle
from autobench.pipeline.policy_dispatch import PolicyRuntime


def _pseudo_bag_forward(
    bundle: DTFDBundle,
    features: torch.Tensor,
    label_t: torch.Tensor,
    cfg: DTFDConfig,
    device: torch.device,
    py_rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One slide's tier-1 forward → (sub_preds, sub_labels, pseudo_feats).

    ``sub_preds`` is ``numGroup x num_cls`` (tier-1 pseudo-bag logits),
    ``sub_labels`` is ``numGroup`` (slide label repeated), ``pseudo_feats`` is
    ``numGroup x mDim`` (AFS distilled features fed to tier-2).
    """
    n = features.shape[0]
    feat_index = list(range(n))
    py_rng.shuffle(feat_index)
    chunks = [c.tolist() for c in np.array_split(np.array(feat_index), cfg.numGroup)]
    chunks = [c for c in chunks if len(c) > 0]

    sub_preds: list[torch.Tensor] = []
    sub_labels: list[torch.Tensor] = []
    pseudo_feats: list[torch.Tensor] = []

    for idx in chunks:
        idx_t = torch.LongTensor(idx).to(device)
        sub_feat = torch.index_select(features, dim=0, index=idx_t)   # n x embed_dim
        mid_feat = bundle.dim_reduction(sub_feat)                     # n x mDim
        attn = bundle.attention(mid_feat).squeeze(0)                  # n (softmax'd)
        att_feats = torch.einsum("ns,n->ns", mid_feat, attn)         # n x mDim
        att_feat = torch.sum(att_feats, dim=0).unsqueeze(0)          # 1 x mDim
        pred = bundle.classifier(att_feat)                           # 1 x num_cls

        sub_preds.append(pred)
        sub_labels.append(label_t)
        pseudo_feats.append(att_feat)  # AFS: attention-feature sum (Main:337)

    return (
        torch.cat(sub_preds, dim=0),
        torch.cat(sub_labels, dim=0),
        torch.cat(pseudo_feats, dim=0),
    )


def _train_one_epoch(
    bundle: DTFDBundle,
    slides: list[DTFDSlide],
    cfg: DTFDConfig,
    ce_cri: torch.nn.Module,
    opt0: torch.optim.Optimizer,
    opt1: torch.optim.Optimizer,
    device: torch.device,
    py_rng: random.Random,
) -> float:
    """Run one training epoch; return mean tier-2 loss over slides.

    Tier-2 loss is the honest signal that the two-tier objective is learning
    (the smoke test asserts it decreases).
    """
    bundle.train()
    order = list(range(len(slides)))
    py_rng.shuffle(order)

    tier2_losses: list[float] = []
    for i in order:
        slide = slides[i]
        features = slide.features.to(device)
        label_t = torch.LongTensor([slide.label]).to(device)

        sub_preds, sub_labels, pseudo_feats = _pseudo_bag_forward(
            bundle, features, label_t, cfg, device, py_rng
        )

        # --- tier 1 ---
        loss0 = ce_cri(sub_preds, sub_labels).mean()
        opt0.zero_grad()
        loss0.backward()
        torch.nn.utils.clip_grad_norm_(bundle.dim_reduction.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(bundle.attention.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(bundle.classifier.parameters(), cfg.grad_clip)
        opt0.step()

        # --- tier 2 ---
        # Detach the distilled pseudo-bag features from the tier-1 graph. The
        # reference (Main:359-365) leaves them attached and relies on
        # retain_graph, but optimizer1 = Adam(att_cls only) NEVER applies the
        # tier-1 gradients that loss1 would produce (they are overwritten by the
        # next optimizer0.zero_grad()), so tier-1 is updated solely by loss0 and
        # tier-2 solely by loss1. Detaching is therefore equivalent in applied
        # updates and avoids the torch>=2.x in-place-version error triggered by
        # optimizer0.step() mutating tier-1 weights before loss1.backward().
        slide_pred = bundle.att_cls(pseudo_feats.detach())           # 1 x num_cls
        loss1 = ce_cri(slide_pred, label_t).mean()
        opt1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(bundle.att_cls.parameters(), cfg.grad_clip)
        opt1.step()

        tier2_losses.append(float(loss1.item()))

    return float(np.mean(tier2_losses)) if tier2_losses else float("nan")


def _snapshot(bundle: DTFDBundle) -> dict:
    return {
        "dim_reduction": copy.deepcopy(bundle.dim_reduction.state_dict()),
        "attention": copy.deepcopy(bundle.attention.state_dict()),
        "classifier": copy.deepcopy(bundle.classifier.state_dict()),
        "att_cls": copy.deepcopy(bundle.att_cls.state_dict()),
    }


def _restore(bundle: DTFDBundle, snap: dict) -> None:
    bundle.dim_reduction.load_state_dict(snap["dim_reduction"])
    bundle.attention.load_state_dict(snap["attention"])
    bundle.classifier.load_state_dict(snap["classifier"])
    bundle.att_cls.load_state_dict(snap["att_cls"])


def train_dtfd_fold(
    train_slides: list[DTFDSlide],
    val_slides: list[DTFDSlide],
    test_slides: list[DTFDSlide],
    embed_dim: int,
    num_classes: int,
    cfg: DTFDConfig,
    device: torch.device,
    seed: int,
    return_history: bool = False,
    policy_runtime: PolicyRuntime | None = None,
    ordinal: bool = False,
    fold_dir: str | None = None,
) -> dict:
    """Train one DTFD fold and return shared-schema test/val metrics.

    Restores pristine torch grad state on return so it cannot perturb a
    following experiment's metrics under a shared process (design spec §6/§11).

    Args:
        return_history: if True, also include ``epoch_tier2_loss`` (per-epoch
            mean tier-2 loss) for the smoke test's loss-decrease assertion.
    """
    grad_was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    # FOLD-TIMING CONTRACT: every fold trainer reports elapsed_seconds over its
    # whole body, so the number is comparable across arms and task types. This
    # one used to be timed by the runner instead — same span, but a second
    # mechanism that drifted from its survival sibling's.
    start = time.time()
    try:
        _seed_everything(seed)
        py_rng = random.Random(seed)

        if not train_slides:
            raise ValueError("DTFD train split is empty; cannot train a fold.")
        cfg.validate(min_bag_size(train_slides))

        bundle = build_dtfd_bundle(embed_dim, num_classes, cfg).to(device)
        ce_cri = torch.nn.CrossEntropyLoss(reduction="none").to(device)

        raw0 = torch.optim.Adam(bundle.tier1_parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        raw1 = torch.optim.Adam(bundle.att_cls.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        policy_runtime = policy_runtime or PolicyRuntime()
        opt0 = policy_runtime.wrap_optimizer(raw0, role="tier1")
        opt1 = policy_runtime.wrap_optimizer(raw1, role="tier2")
        # A policy may legitimately return a duck-typed wrapper, which no torch
        # LR scheduler will accept; scheduler_target resolves what to attach to.
        sched0 = torch.optim.lr_scheduler.MultiStepLR(
            policy_runtime.scheduler_target(opt0, raw0, role="tier1"),
            [cfg.lr_decay_step], gamma=cfg.lr_decay_ratio,
        )
        sched1 = torch.optim.lr_scheduler.MultiStepLR(
            policy_runtime.scheduler_target(opt1, raw1, role="tier2"),
            [cfg.lr_decay_step], gamma=cfg.lr_decay_ratio,
        )
        sched0 = policy_runtime.wrap_scheduler(sched0, role="tier1")
        sched1 = policy_runtime.wrap_scheduler(sched1, role="tier2")

        best_auc = float("-inf")
        best_snap: dict | None = None
        epochs_no_improve = 0
        history: list[float] = []

        for epoch in range(cfg.max_epochs):
            mean_tier2 = _train_one_epoch(
                bundle, train_slides, cfg, ce_cri, opt0, opt1, device, py_rng
            )
            history.append(mean_tier2)
            sched0.step()
            sched1.step()

            if val_slides:
                cur = val_auc(bundle, val_slides, cfg, num_classes, device, seed)
                if cur > best_auc:
                    best_auc = cur
                    best_snap = _snapshot(bundle)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                default_stop = cfg.early_stopping and epochs_no_improve >= cfg.patience
                if policy_runtime.should_stop(
                    default_stop, epoch=epoch, metrics={"val_auc": cur},
                ):
                    break

        if best_snap is not None:
            _restore(bundle, best_snap)

        test_metrics = (
            evaluate_dtfd(bundle, test_slides, cfg, num_classes, device, seed, ordinal=ordinal,
                          predictions_path=(os.path.join(fold_dir, "predictions.csv") if fold_dir else None))
            if test_slides else {}
        )
        val_metrics = (
            evaluate_dtfd(bundle, val_slides, cfg, num_classes, device, seed, ordinal=ordinal,
                          predictions_path=(os.path.join(fold_dir, "predictions_val.csv") if fold_dir else None))
            if val_slides else {}
        )

        result: dict = {
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "elapsed_seconds": time.time() - start,
        }
        if return_history:
            result["epoch_tier2_loss"] = history
        return result
    finally:
        torch.set_grad_enabled(grad_was_enabled)
