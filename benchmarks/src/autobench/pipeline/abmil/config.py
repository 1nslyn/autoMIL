"""ABMIL hyperparameter configuration.

Defaults are the paper-exact hidden dims (Ilse et al., 2018) plus the shared
benchmark training schedule (matches ``TrainConfig`` defaults in
``autobench.pipeline.config``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ABMILConfig:
    """Immutable ABMIL hyperparameters."""

    # Architecture (paper-exact: Ilse et al., 2018)
    M: int = 500       # instance embedding dim
    L: int = 128       # attention hidden dim
    dropout: float = 0.0

    # Optimization
    lr: float = 2e-4
    weight_decay: float = 1e-5

    # Training schedule
    max_epochs: int = 200
    early_stopping: bool = True
    patience: int = 20
