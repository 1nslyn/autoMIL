"""Standard (non-gated) and gated ABMIL — faithful ports of Ilse et al., 2018.

Reference: M. Ilse, J. M. Tomczak, M. Welling, "Attention-based Deep Multiple
Instance Learning," ICML 2018. Ported from the authors' reference
implementation, ``Attention`` and ``GatedAttention`` in
``benchmarks/lib/AttentionDeepMIL/model.py`` (imported, not reimplemented, via
``autobench.pipeline.abmil._imports``).

Two adaptations from the reference, both confined to the ends of the network;
the attention core in the middle is verbatim for each variant:

1. The reference's ``feature_extractor_part1/2`` is a ``Conv2d`` stack that
   embeds 28x28 MNIST tiles. We operate on precomputed patch-encoder features
   instead, so that Conv2d extractor is replaced by a ``Linear(in_dim -> M)``
   + ``ReLU`` projection -- functionally the same role as the reference's
   ``feature_extractor_part2``, which is the only part of the original
   extractor relevant once instance embeddings already exist.
2. The reference's binary ``Sigmoid`` head is replaced by a
   ``Linear(M -> num_classes)`` head emitting K-class logits (no sigmoid), so
   these models train with cross-entropy and report through the same metrics
   path as the rest of the benchmark roster instead of a bespoke bernoulli
   objective.

``ABMILGated``'s gating (``attention_V``, ``attention_U``, ``attention_w``,
softmax over instances, weighted sum) matches the reference exactly, including
the paper's hidden dimensions M=500, L=128; it is a straight move of
``lib/nnMIL/network_architecture/models/ab_mil_gated.py::AB_MIL_Gated`` out of
nnMIL and into this framework-owned package. ``ABMIL`` mirrors the same
adaptation applied to the reference's non-gated ``Attention`` instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    """Standard (non-gated) attention MIL over precomputed instance features.

    Matches the trainer I/O contract: ``forward`` takes batched input
    ``[B, N, in_dim]`` and returns a dict with key ``'logits'`` (and,
    optionally, ``'WSI_feature'`` / ``'WSI_attn'``).
    """

    def __init__(self, in_dim=1024, M=500, L=128, num_classes=2, dropout=0.0, K=1):
        super().__init__()
        if K != 1:
            raise ValueError(
                f"ABMIL only supports a single attention branch (K=1); got K={K}. "
                "The reference Attention hardcodes ATTENTION_BRANCHES=1."
            )
        self.in_dim = in_dim
        self.M = M
        self.L = L
        self.K = K
        self.num_classes = num_classes

        # Adaptation 1: linear projection replaces the reference's Conv2d
        # feature_extractor_part1/2 (this plays the same role as
        # feature_extractor_part2 once instance embeddings already exist).
        projection = [nn.Linear(in_dim, self.M), nn.ReLU()]
        if dropout:
            projection.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*projection)

        # Non-gated attention, verbatim from the reference Attention.
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.K),
        )

        # Adaptation 2: K-class logit head replaces the reference's binary
        # Linear(M, 1) + Sigmoid classifier.
        self.classifier = nn.Linear(self.M * self.K, self.num_classes)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """Forward pass.

        Args:
            x: instance features, ``[B, N, in_dim]``.
            return_WSI_attn: if True, include per-instance attention weights
                under key ``'WSI_attn'``, shape ``[B, N, K]``.
            return_WSI_feature: if True, include the pooled slide feature
                under key ``'WSI_feature'``, shape ``[B, K*M]``.

        Returns:
            dict with key ``'logits'``, shape ``[B, num_classes]``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected input of shape [B, N, in_dim], got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {D}")

        forward_return = {}

        H = self.feature_extractor(x)  # [B, N, M]

        A = self.attention(H)  # [B, N, K]
        A = A.transpose(-1, -2)  # [B, K, N]
        A = F.softmax(A, dim=-1)  # softmax over instances (N)

        Z = torch.bmm(A, H)  # [B, K, M] weighted sum of instance embeddings
        Z = Z.reshape(B, self.K * self.M)  # [B, K*M]

        logits = self.classifier(Z)  # [B, num_classes]

        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = Z
        if return_WSI_attn:
            forward_return["WSI_attn"] = A.transpose(-1, -2)  # [B, N, K]

        return forward_return


class ABMILGated(nn.Module):
    """Gated-attention MIL over precomputed instance features.

    Matches the trainer I/O contract: ``forward`` takes batched input
    ``[B, N, in_dim]`` and returns a dict with key ``'logits'`` (and,
    optionally, ``'WSI_feature'`` / ``'WSI_attn'``).
    """

    def __init__(self, in_dim=1024, M=500, L=128, num_classes=2, dropout=0.0, K=1):
        super().__init__()
        if K != 1:
            raise ValueError(
                f"ABMILGated only supports a single attention branch (K=1); got K={K}. "
                "The reference GatedAttention hardcodes ATTENTION_BRANCHES=1."
            )
        self.in_dim = in_dim
        self.M = M
        self.L = L
        self.K = K
        self.num_classes = num_classes

        # Adaptation 1: linear projection replaces the reference's Conv2d
        # feature_extractor_part1/2 (this plays the same role as
        # feature_extractor_part2 once instance embeddings already exist).
        projection = [nn.Linear(in_dim, self.M), nn.ReLU()]
        if dropout:
            projection.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*projection)

        # Gated attention, verbatim from the reference GatedAttention.
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(self.L, self.K)

        # Adaptation 2: K-class logit head replaces the reference's binary
        # Linear(M, 1) + Sigmoid classifier.
        self.classifier = nn.Linear(self.M * self.K, self.num_classes)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """Forward pass.

        Args:
            x: instance features, ``[B, N, in_dim]``.
            return_WSI_attn: if True, include per-instance attention weights
                under key ``'WSI_attn'``, shape ``[B, N, K]``.
            return_WSI_feature: if True, include the pooled slide feature
                under key ``'WSI_feature'``, shape ``[B, K*M]``.

        Returns:
            dict with key ``'logits'``, shape ``[B, num_classes]``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected input of shape [B, N, in_dim], got shape {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {D}")

        forward_return = {}

        H = self.feature_extractor(x)  # [B, N, M]

        A_V = self.attention_V(H)  # [B, N, L]
        A_U = self.attention_U(H)  # [B, N, L]
        A = self.attention_w(A_V * A_U)  # [B, N, K] (element-wise gate, as in the reference)

        A = A.transpose(-1, -2)  # [B, K, N]
        A = F.softmax(A, dim=-1)  # softmax over instances (N)

        Z = torch.bmm(A, H)  # [B, K, M] weighted sum of instance embeddings
        Z = Z.reshape(B, self.K * self.M)  # [B, K*M]

        logits = self.classifier(Z)  # [B, num_classes]

        forward_return["logits"] = logits
        if return_WSI_feature:
            forward_return["WSI_feature"] = Z
        if return_WSI_attn:
            forward_return["WSI_attn"] = A.transpose(-1, -2)  # [B, N, K]

        return forward_return


def build_abmil_model(
    model_type: str,
    in_dim: int,
    num_classes: int,
    M: int = 500,
    L: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    """Instantiate an ABMIL variant by key.

    ``"abmil"`` -> non-gated (``ABMIL``); ``"abmil_gated"`` -> gated
    (``ABMILGated``). Paper-exact hidden dims M=500, L=128 by default for
    both, matching the locked decision for the gated port.
    """
    if model_type == "abmil":
        return ABMIL(in_dim=in_dim, M=M, L=L, num_classes=num_classes, dropout=dropout, K=1)
    elif model_type == "abmil_gated":
        return ABMILGated(in_dim=in_dim, M=M, L=L, num_classes=num_classes, dropout=dropout, K=1)
    raise ValueError(f"Unknown ABMIL model_type: {model_type!r}. Expected 'abmil' or 'abmil_gated'.")
