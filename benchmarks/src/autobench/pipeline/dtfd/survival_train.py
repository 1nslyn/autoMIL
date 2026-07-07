"""nllsurv-only survival training for DTFD-MIL's two-tier pseudo-bag distillation.

Preserves DTFD's defining trick: each random pseudo-bag is supervised with
the SAME target as the slide it was split from (mirrors
``dtfd/train.py::_pseudo_bag_forward`` repeating the slide's classification
label into ``sub_labels``). For nllsurv, that target is the slide's
discretized (bin_idx, censor) pair -- a valid per-slide analog since nllsurv
is a discrete time-bin classification problem in disguise. ``NLLSurvLoss``
replaces ``CrossEntropyLoss`` at both tiers; everything else about the
two-tier optimization (disjoint tier-1/tier-2 optimizers, grad clipping,
detached tier-2 input) is unchanged from the classification trainer.

Cox is NOT supported here: its partial-likelihood loss needs a risk set of
*different patients*' relative event-time ordering, and there is no such
comparison available within one slide's own pseudo-bags. See
``dtfd/runner.py::run_dtfd_experiment`` for the explicit guard.
"""

from __future__ import annotations

import random
import sys
import time

import numpy as np
import torch

from autobench import LIB_ROOT
from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.dtfd.dataset import DTFDSurvivalSlide, min_bag_size
from autobench.pipeline.dtfd.eval import _split_pseudo_bags
from autobench.pipeline.dtfd.model import DTFDBundle, build_dtfd_bundle
from autobench.pipeline.dtfd.train import _restore, _snapshot

# The framework-agnostic survival core lives under the vendored nnMIL tree;
# import it adapter -> lib (the normal autobench direction).
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))
from nnMIL.training.losses.survival_loss import survival_c_index  # noqa: E402
from nnMIL.training.losses.survival_loss_nll import NLLSurvLoss  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _event_time_bin_edges(times, statuses, n_bins: int) -> np.ndarray:
    """NLL bin edges from event (uncensored) train times — same PORPOISE/MCAT
    convention as the CLAM/ABMIL/TITAN survival trainers."""
    times = np.asarray(times, dtype=float)
    statuses = np.asarray(statuses)
    event_times = times[statuses == 1]
    src = event_times if len(np.unique(event_times)) >= n_bins else times
    edges = np.quantile(src, np.linspace(0, 1, n_bins + 1))
    eps = 1e-6
    for i in range(1, len(edges)):  # force strictly increasing
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def _time_to_bin(time_val: float, edges: np.ndarray) -> int:
    b = int(np.digitize([time_val], edges[1:-1])[0])
    return int(np.clip(b, 0, len(edges) - 2))


def _nllsurv_risk(logits: torch.Tensor) -> torch.Tensor:
    """Scalar risk from nllsurv hazard logits ``(B, n_bins)`` — lower predicted
    survival = higher risk, oriented like the cox score (higher = earlier event)."""
    survival = torch.cumprod(1 - torch.sigmoid(logits), dim=1)
    return -survival.sum(dim=1)


def _pseudo_bag_forward_survival(
    bundle: DTFDBundle,
    features: torch.Tensor,
    bin_t: torch.Tensor,
    censor_t: torch.Tensor,
    cfg: DTFDConfig,
    device: torch.device,
    py_rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One slide's tier-1 forward -> (sub_preds, sub_bins, sub_censor, pseudo_feats).

    Identical pseudo-bag split to ``dtfd.train._pseudo_bag_forward``; the only
    difference is repeating the slide's (bin_idx, censor) survival pair
    instead of its classification label.
    """
    n = features.shape[0]
    feat_index = list(range(n))
    py_rng.shuffle(feat_index)
    chunks = [c.tolist() for c in np.array_split(np.array(feat_index), cfg.numGroup)]
    chunks = [c for c in chunks if len(c) > 0]

    sub_preds: list[torch.Tensor] = []
    sub_bins: list[torch.Tensor] = []
    sub_censor: list[torch.Tensor] = []
    pseudo_feats: list[torch.Tensor] = []

    for idx in chunks:
        idx_t = torch.LongTensor(idx).to(device)
        sub_feat = torch.index_select(features, dim=0, index=idx_t)   # n x embed_dim
        mid_feat = bundle.dim_reduction(sub_feat)                     # n x mDim
        attn = bundle.attention(mid_feat).squeeze(0)                  # n (softmax'd)
        att_feats = torch.einsum("ns,n->ns", mid_feat, attn)         # n x mDim
        att_feat = torch.sum(att_feats, dim=0).unsqueeze(0)          # 1 x mDim
        pred = bundle.classifier(att_feat)                           # 1 x n_bins

        sub_preds.append(pred)
        sub_bins.append(bin_t)
        sub_censor.append(censor_t)
        pseudo_feats.append(att_feat)

    return (
        torch.cat(sub_preds, dim=0),
        torch.cat(sub_bins, dim=0),
        torch.cat(sub_censor, dim=0),
        torch.cat(pseudo_feats, dim=0),
    )


def _train_one_epoch_survival(
    bundle: DTFDBundle,
    slides: list[DTFDSurvivalSlide],
    edges: np.ndarray,
    cfg: DTFDConfig,
    loss_fn: NLLSurvLoss,
    opt0: torch.optim.Optimizer,
    opt1: torch.optim.Optimizer,
    device: torch.device,
    py_rng: random.Random,
) -> None:
    bundle.train()
    order = list(range(len(slides)))
    py_rng.shuffle(order)

    for i in order:
        slide = slides[i]
        features = slide.features.to(device)
        bin_t = torch.LongTensor([_time_to_bin(slide.time, edges)]).to(device)
        censor_t = torch.LongTensor([1 - slide.status]).to(device)  # 1=censored, 0=event

        sub_preds, sub_bins, sub_censor, pseudo_feats = _pseudo_bag_forward_survival(
            bundle, features, bin_t, censor_t, cfg, device, py_rng
        )

        # --- tier 1 ---
        loss0 = loss_fn(sub_preds, sub_bins, sub_censor)
        opt0.zero_grad()
        loss0.backward()
        torch.nn.utils.clip_grad_norm_(bundle.dim_reduction.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(bundle.attention.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(bundle.classifier.parameters(), cfg.grad_clip)
        opt0.step()

        # --- tier 2 --- (same detach rationale as classification: opt1 only
        # ever applies gradients to att_cls, so detaching pseudo_feats from
        # the tier-1 graph is equivalent in applied updates; see train.py:125-132)
        slide_pred = bundle.att_cls(pseudo_feats.detach())           # 1 x n_bins
        loss1 = loss_fn(slide_pred, bin_t, censor_t)
        opt1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(bundle.att_cls.parameters(), cfg.grad_clip)
        opt1.step()


@torch.no_grad()
def _slide_survival_logits(
    bundle: DTFDBundle,
    features: torch.Tensor,
    cfg: DTFDConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Tier-2 raw hazard logits ``(1, n_bins)`` for one slide (AFS distillation).

    Mirrors ``dtfd/eval.py::_slide_prob`` but returns raw logits (no final
    softmax) since ``_nllsurv_risk`` needs sigmoid hazards, not class probs.
    """
    feats = features.to(device)
    mid_feat = bundle.dim_reduction(feats)                       # N x mDim
    attn = bundle.attention(mid_feat, isNorm=False).squeeze(0)   # N

    chunks = _split_pseudo_bags(feats.shape[0], cfg.numGroup, rng)
    pseudo_feats: list[torch.Tensor] = []
    for idx in chunks:
        idx_t = torch.LongTensor(idx).to(device)
        sub_mid = mid_feat.index_select(dim=0, index=idx_t)      # n x mDim
        sub_attn = torch.softmax(attn.index_select(dim=0, index=idx_t), dim=0)  # n
        att_feats = torch.einsum("ns,n->ns", sub_mid, sub_attn)  # n x mDim
        pseudo_feats.append(torch.sum(att_feats, dim=0).unsqueeze(0))  # 1 x mDim (AFS)

    slide_pseudo = torch.cat(pseudo_feats, dim=0)                # numGroup x mDim
    return bundle.att_cls(slide_pseudo)                          # 1 x n_bins


def _val_loss(
    bundle: DTFDBundle,
    slides: list[DTFDSurvivalSlide],
    edges: np.ndarray,
    cfg: DTFDConfig,
    loss_fn: NLLSurvLoss,
    device: torch.device,
    seed: int,
) -> float:
    bundle.eval()
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for slide in slides:
        logits = _slide_survival_logits(bundle, slide.features, cfg, device, rng)
        bin_t = torch.LongTensor([_time_to_bin(slide.time, edges)]).to(device)
        censor_t = torch.LongTensor([1 - slide.status]).to(device)
        losses.append(float(loss_fn(logits, bin_t, censor_t).item()))
    return float(np.mean(losses)) if losses else float("nan")


def _c_index(
    bundle: DTFDBundle,
    slides: list[DTFDSurvivalSlide],
    cfg: DTFDConfig,
    device: torch.device,
    seed: int,
) -> float:
    bundle.eval()
    rng = np.random.default_rng(seed)
    risks, statuses, times, pids = [], [], [], []
    for slide in slides:
        logits = _slide_survival_logits(bundle, slide.features, cfg, device, rng)
        risks.append(float(_nllsurv_risk(logits).item()))
        statuses.append(slide.status)
        times.append(slide.time)
        pids.append(slide.patient_id)
    ci = survival_c_index(
        torch.tensor(risks, dtype=torch.float32),
        torch.tensor(statuses, dtype=torch.float32),
        torch.tensor(times, dtype=torch.float32),
        pids,
    )
    return float(ci) if ci is not None else float("nan")


def train_dtfd_survival_fold(
    train_samples: list[DTFDSurvivalSlide],
    val_samples: list[DTFDSurvivalSlide],
    test_samples: list[DTFDSurvivalSlide],
    embed_dim: int,
    nll_bins: int,
    cfg: DTFDConfig,
    device: torch.device,
    seed: int,
) -> dict:
    """Train one DTFD-MIL survival (nllsurv) fold; return shared-schema c-index metrics.

    Restores pristine torch grad state on return (mirrors
    ``train_dtfd_fold``'s discipline for a shared-process orchestrator).
    """
    grad_was_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        _seed_everything(seed)
        py_rng = random.Random(seed)

        if not train_samples:
            raise ValueError("DTFD survival train split is empty; cannot train a fold.")
        cfg.validate(min_bag_size(train_samples))

        bundle = build_dtfd_bundle(embed_dim, nll_bins, cfg).to(device)
        loss_fn = NLLSurvLoss()
        edges = _event_time_bin_edges(
            [s.time for s in train_samples], [s.status for s in train_samples], nll_bins,
        )

        opt0 = torch.optim.Adam(bundle.tier1_parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        opt1 = torch.optim.Adam(bundle.att_cls.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        sched0 = torch.optim.lr_scheduler.MultiStepLR(
            opt0, [cfg.lr_decay_step], gamma=cfg.lr_decay_ratio
        )
        sched1 = torch.optim.lr_scheduler.MultiStepLR(
            opt1, [cfg.lr_decay_step], gamma=cfg.lr_decay_ratio
        )

        # Select on val LOSS, not val c-index: with only a handful of events
        # per val fold the c-index is near-random, so maximizing it would
        # overfit to noise (same rationale as CLAM/ABMIL/TITAN's survival
        # trainers). DTFDBundle bundles 4 separate nn.Modules with no unified
        # state_dict, so EarlyStoppingSurvival's single-module checkpointing
        # doesn't apply directly -- reuse DTFD's own manual snapshot/restore
        # helpers instead, keyed on val loss.
        best_loss = float("inf")
        best_snap: dict | None = None
        epochs_no_improve = 0

        start = time.time()
        for epoch in range(cfg.max_epochs):
            _train_one_epoch_survival(
                bundle, train_samples, edges, cfg, loss_fn, opt0, opt1, device, py_rng
            )
            sched0.step()
            sched1.step()

            if val_samples:
                v_loss = _val_loss(bundle, val_samples, edges, cfg, loss_fn, device, seed)
                v_cidx = _c_index(bundle, val_samples, cfg, device, seed)
                print(
                    f"    [DTFD-surv] epoch {epoch + 1}: "
                    f"val_loss={v_loss:.4f} val_c_index={v_cidx:.4f}"
                )
                if v_loss < best_loss:
                    best_loss = v_loss
                    best_snap = _snapshot(bundle)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if cfg.early_stopping and epochs_no_improve >= cfg.patience:
                    break
        elapsed_seconds = time.time() - start

        if best_snap is not None:
            _restore(bundle, best_snap)

        return {
            "test_metrics": {
                "c_index": _c_index(bundle, test_samples, cfg, device, seed)
                if test_samples else float("nan"),
            },
            "val_metrics": {
                "c_index": _c_index(bundle, val_samples, cfg, device, seed)
                if val_samples else float("nan"),
            },
            "elapsed_seconds": elapsed_seconds,
        }
    finally:
        torch.set_grad_enabled(grad_was_enabled)
