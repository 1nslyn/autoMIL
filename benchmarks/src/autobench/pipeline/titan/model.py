"""TITAN linear-probe model.

TITAN's slide encoder already produces one frozen embedding per slide --
there is no bag, no patches, no aggregator to learn. Per the design spec
(§7) the locked decision is a **linear probe**: a single ``Linear`` layer
mapping the frozen embedding directly to class logits.
"""

from __future__ import annotations

import torch
from torch import nn


class TitanLinearProbe(nn.Module):
    """Linear probe on a frozen TITAN slide embedding.

    Returns raw logits of shape ``[batch, num_classes]`` (unnormalized),
    matching the CE-loss / ``compute_extended_metrics`` softmax-probs
    contract used by every other arm.
    """

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(embedding)
