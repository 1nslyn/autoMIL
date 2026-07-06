"""Assemble the four DTFD-MIL reference modules into an immutable bundle.

Mirrors the instantiation in ``lib/DTFD-MIL/Main_DTFD_MIL.py:64-67`` exactly;
the modules themselves are imported from the vendored reference (never
reimplemented) via ``dtfd._imports``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from autobench.pipeline.dtfd._imports import (
    Attention_Gated,
    Attention_with_Classifier,
    Classifier_1fc,
    DimReduction,
)
from autobench.pipeline.dtfd.config import DTFDConfig


@dataclass(frozen=True)
class DTFDBundle:
    """The four DTFD-MIL sub-networks, held together for one experiment.

    - ``dim_reduction``: ``embed_dim -> mDim`` projection (+ optional res blocks)
    - ``attention``:     gated attention over instances (tier-1)
    - ``classifier``:    tier-1 pseudo-bag classifier (also the CAM source)
    - ``att_cls``:       tier-2 ``Attention_with_Classifier`` (slide prediction)
    """

    dim_reduction: nn.Module
    attention: nn.Module
    classifier: nn.Module
    att_cls: nn.Module

    def to(self, device) -> "DTFDBundle":
        """Move every module to ``device`` (returns self; modules move in place)."""
        self.dim_reduction.to(device)
        self.attention.to(device)
        self.classifier.to(device)
        self.att_cls.to(device)
        return self

    def train(self) -> None:
        for m in (self.dim_reduction, self.attention, self.classifier, self.att_cls):
            m.train()

    def eval(self) -> None:
        for m in (self.dim_reduction, self.attention, self.classifier, self.att_cls):
            m.eval()

    def tier1_parameters(self) -> list:
        """Parameters optimized by the tier-1 optimizer (Main:99-105)."""
        params: list = []
        params += list(self.classifier.parameters())
        params += list(self.attention.parameters())
        params += list(self.dim_reduction.parameters())
        return params


def build_dtfd_bundle(
    embed_dim: int,
    num_classes: int,
    cfg: DTFDConfig,
) -> DTFDBundle:
    """Instantiate the DTFD-MIL modules for a given feature dim and class count.

    ``embed_dim`` is the patch-encoder feature dimension (``in_chn`` in the
    reference); ``num_classes`` sizes both tier heads.
    """
    classifier = Classifier_1fc(cfg.mDim, num_classes, cfg.droprate)
    attention = Attention_Gated(cfg.mDim)
    dim_reduction = DimReduction(embed_dim, cfg.mDim, numLayer_Res=cfg.numLayer_Res)
    att_cls = Attention_with_Classifier(
        L=cfg.mDim, num_cls=num_classes, droprate=cfg.droprate_2
    )
    return DTFDBundle(
        dim_reduction=dim_reduction,
        attention=attention,
        classifier=classifier,
        att_cls=att_cls,
    )
