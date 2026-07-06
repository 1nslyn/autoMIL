"""TITAN linear-probe head configuration.

Per the design spec (§7), the locked decision is a frozen-embedding
**linear probe**: ``Linear(embed_dim, num_classes)``. This dataclass holds
the small set of training knobs the probe needs; it does not duplicate
anything already carried by ``ExperimentConfig.train`` (max_epochs, seed,
etc.) -- those are read directly off the shared ``TrainConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TitanHeadConfig:
    """Linear-probe hyperparameters for the TITAN arm.

    ``head`` mirrors ``autobench.config.TitanDef.head`` (currently only
    ``"linear"`` is implemented; the design spec allows a trivial MLP
    option later without changing this contract).
    """

    head: str = "linear"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
