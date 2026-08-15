"""TITAN linear-probe head configuration.

Per the design spec (§7), the locked decision is a frozen-embedding
**linear probe**: ``Linear(embed_dim, num_classes)``. This dataclass holds
the small set of training knobs the probe needs; it does not duplicate
anything already carried by ``ExperimentConfig.train`` (max_epochs, seed,
etc.) -- those are read directly off the shared ``TrainConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


#: Declared TITAN knobs that live on the shared TrainConfig, not the head.
_TRAIN_SIDE = ("max_epochs", "early_stopping")


def resolve_head_config(
    exp_cfg: Any, head_cfg: TitanHeadConfig | None = None,
) -> TitanHeadConfig:
    """Apply this experiment's overrides to TITAN's mixed-provenance config.

    H-3: ``TitanHeadConfig`` stays the source of truth for lr/weight_decay/
    patience; only explicitly-set overrides are layered on. ``max_epochs`` and
    ``early_stopping`` are excluded from the head — this arm reads those
    straight off ``exp_cfg.train`` (its documented mixed provenance), so
    routing them into the head would double-apply and trip the fail-loud
    guard.

    The OPAQUE channel's values for those two are applied onto
    ``exp_cfg.train`` here (mutating it in place, the same contract as CLAM's
    ``apply_overrides_to_exp_cfg``). Before this seam existed an
    ``--hparams`` ``max_epochs``/``early_stopping`` on TITAN was silently
    dropped — head filtering excluded them and nothing else consumed the
    opaque channel — the exact H-3 defect on this arm. Both trainers resolve
    through this one function so the wiring is testable without training.
    """
    from autobench.pipeline.hparams import all_overrides, apply_overrides

    if head_cfg is None:
        head_cfg = TitanHeadConfig()
    opaque = getattr(exp_cfg, "hparam_overrides", None) or {}
    train_slice = {
        k: v for k, v in opaque.items() if k in _TRAIN_SIDE and v is not None
    }
    if train_slice:
        exp_cfg.train = apply_overrides(exp_cfg.train, train_slice, arm="titan")
    head_overrides = {
        k: v for k, v in all_overrides(exp_cfg).items() if k not in _TRAIN_SIDE
    }
    return apply_overrides(head_cfg, head_overrides, arm="titan")
