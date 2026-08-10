"""DTFD-MIL evaluation — tier-2 slide predictions scored by the SHARED metric.

FAIRNESS-CRITICAL: the reference scores with its own optimal-threshold
``eval_metric`` (``lib/DTFD-MIL/utils.py:22-40``); we DISCARD that and feed the
tier-2 softmax probabilities + labels into ``compute_extended_metrics`` — the
identical code CLAM / nnMIL / ABMIL use — so ``test_auc`` / ``test_bacc`` are
computed by one code path across the whole roster (design spec §6, risk #1).

Structurally ports ``test_attention_DTFD_preFeat_MultipleMean``
(``Main_DTFD_MIL.py:151-269``) with ``num_MeanInference = 1`` (reference default).
"""

from __future__ import annotations

import numpy as np
import torch

from autobench.pipeline.dtfd.config import DTFDConfig
from autobench.pipeline.dtfd.dataset import DTFDSlide
from autobench.pipeline.dtfd.model import DTFDBundle
from autobench.pipeline.evaluate import compute_extended_metrics


def _split_pseudo_bags(n_patches: int, num_group: int, rng: np.random.Generator) -> list[list[int]]:
    """Random pseudo-bag partition of instance indices (Main:194-197 / 311-314)."""
    feat_index = list(range(n_patches))
    rng.shuffle(feat_index)
    chunks = np.array_split(np.array(feat_index), num_group)
    return [c.tolist() for c in chunks if len(c) > 0]


@torch.no_grad()
def _slide_prob(
    bundle: DTFDBundle,
    features: torch.Tensor,
    cfg: DTFDConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tier-2 softmax probability vector for one slide (AFS distillation).

    Ports the eval inner loop (Main:186-249) for ``distill='AFS'``: attention is
    computed once over the full bag (unnormalized), then per pseudo-bag the
    attention weights are re-softmaxed within the chunk and summed into one
    pseudo-bag feature; the tier-2 classifier consumes the stacked pseudo-bag
    features.
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
    slide_logits = bundle.att_cls(slide_pseudo)                 # 1 x num_cls
    return torch.softmax(slide_logits, dim=1).squeeze(0).cpu().numpy()


def evaluate_dtfd(
    bundle: DTFDBundle,
    slides: list[DTFDSlide],
    cfg: DTFDConfig,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Evaluate a split → shared-schema metrics dict.

    Returns exactly the keys ``compute_extended_metrics`` emits
    (auc_roc, accuracy, balanced_accuracy, f1, plus sensitivity/specificity
    on binary or macro_recall/macro_specificity_ovr on multi-class).
    """
    bundle.eval()
    rng = np.random.default_rng(seed)

    probs: list[np.ndarray] = []
    labels: list[int] = []
    for slide in slides:
        probs.append(_slide_prob(bundle, slide.features, cfg, device, rng))
        labels.append(slide.label)

    y_probs = np.vstack(probs)                 # [num_slides, num_classes]
    y_true = np.asarray(labels, dtype=int)
    y_pred = y_probs.argmax(axis=1)
    return compute_extended_metrics(y_true, y_probs, y_pred, num_classes)


def val_auc(
    bundle: DTFDBundle,
    slides: list[DTFDSlide],
    cfg: DTFDConfig,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> float:
    """Validation AUC used for early stopping (NaN-safe → -inf)."""
    metrics = evaluate_dtfd(bundle, slides, cfg, num_classes, device, seed)
    auc = metrics.get("auc_roc", float("nan"))
    return float(auc) if not np.isnan(auc) else float("-inf")
