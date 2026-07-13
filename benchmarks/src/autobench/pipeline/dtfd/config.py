"""DTFD-MIL hyperparameter configuration.

Defaults are the reference values from ``lib/DTFD-MIL/Main_DTFD_MIL.py``
(argparse block ``:21-50`` and the optimizer/scheduler setup ``:104-108``).
See docs/design/2026-07-05-mil-model-integration-design.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DTFDConfig:
    """Immutable DTFD-MIL hyperparameters (reference defaults).

    ``distill`` is LOCKED to ``"AFS"`` (attention-feature-sum): it sidesteps
    per-class instance ranking, so it is robust to variable bag sizes and safe
    for multi-class tasks (the reference's MaxS/MaxMinS rank on
    ``patch_pred_softmax[:, -1]``, a binary assumption). See §6.
    """

    # Pseudo-bag / distillation
    numGroup: int = 4          # number of pseudo-bags per slide (Main:39)
    total_instance: int = 4    # total distilled instances (Main:40); AFS ignores
    distill: str = "AFS"       # LOCKED (Main:50) — attention-feature-sum

    # Architecture
    mDim: int = 512            # DimReduction output / attention width (Main:43)
    numLayer_Res: int = 0      # residual blocks in DimReduction (Main:47)
    droprate: float = 0.0      # tier-1 classifier dropout (Main:27)
    droprate_2: float = 0.0    # tier-2 classifier dropout (Main:28)

    # Optimization
    lr: float = 1e-4           # Main:29
    wd: float = 1e-4           # weight_decay, Main:30
    grad_clip: float = 5.0     # grad_clipping, Main:44
    lr_decay_ratio: float = 0.2  # MultiStepLR gamma, Main:31

    # Training schedule
    max_epochs: int = 200      # EPOCH, Main:21
    lr_decay_step: int = 100   # epoch_step "[100]", Main:22
    early_stopping: bool = True
    patience: int = 20         # epochs w/o val-AUC improvement before stop

    def validate(self, n_patches: int) -> None:
        """Guard invariants against the smallest bag in the dataset.

        ``numGroup`` chunks are produced by ``np.array_split`` over the patch
        indices; with ``numGroup > n_patches`` some chunks are empty and the
        tier-1 forward degenerates. Fail fast with a clear message.
        """
        if self.numGroup < 1:
            raise ValueError(f"numGroup must be >= 1, got {self.numGroup}")
        if self.numGroup > n_patches:
            raise ValueError(
                f"numGroup ({self.numGroup}) exceeds the smallest bag size "
                f"({n_patches} patches). Reduce numGroup or drop tiny slides."
            )
        if self.distill != "AFS":
            raise ValueError(
                f"distill is locked to 'AFS' for the benchmark, got {self.distill!r}"
            )
