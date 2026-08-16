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


def apply_train_overrides(exp_cfg: Any) -> None:
    """Apply the opaque channel's train-side slice onto ``exp_cfg.train``.

    H-3: ``max_epochs`` and ``early_stopping`` are TITAN's documented mixed
    provenance — both trainers read them straight off the shared
    ``TrainConfig``. Mutates ``exp_cfg`` in place, the same contract as
    CLAM's ``apply_overrides_to_exp_cfg``: it must run at the RUNNER level,
    before results-dir resolution and ``exp_cfg.save``, so CR-5b cache
    identity and the archived ``config.json`` record the EFFECTIVE values.
    (This slice used to be applied inside the trainers via
    ``resolve_head_config`` — after the runner had already saved
    ``config.json``, so the archived provenance lied about ``max_epochs``.)

    Before this seam existed an ``--hparams`` ``max_epochs``/
    ``early_stopping`` on TITAN was silently dropped — head filtering
    excluded them and nothing else consumed the opaque channel — the exact
    H-3 defect on this arm.
    """
    from autobench.pipeline.hparams import apply_overrides

    opaque = getattr(exp_cfg, "hparam_overrides", None) or {}
    train_slice = {
        k: v for k, v in opaque.items() if k in _TRAIN_SIDE and v is not None
    }
    if train_slice:
        exp_cfg.train = apply_overrides(exp_cfg.train, train_slice, arm="titan")


def resolve_head_config(
    exp_cfg: Any, head_cfg: TitanHeadConfig | None = None,
) -> TitanHeadConfig:
    """Layer this experiment's HEAD-side overrides onto the head config.

    H-3: ``TitanHeadConfig`` stays the source of truth for lr/weight_decay/
    patience; only explicitly-set overrides are layered on. ``max_epochs``
    and ``early_stopping`` are excluded from the head — routing them into it
    would double-apply and trip the fail-loud guard. Head filtering ONLY: no
    ``exp_cfg`` mutation. The train-side slice is applied by
    ``apply_train_overrides`` at the runner level (before results-dir
    resolution and ``exp_cfg.save``), so by the time a trainer calls this,
    ``exp_cfg.train`` is already effective.
    """
    from autobench.pipeline.hparams import all_overrides, apply_overrides

    if head_cfg is None:
        head_cfg = TitanHeadConfig()
    head_overrides = {
        k: v for k, v in all_overrides(exp_cfg).items() if k not in _TRAIN_SIDE
    }
    return apply_overrides(head_cfg, head_overrides, arm="titan")
