"""The menu a TPE search draws a dev-cell attempt from: the menu-AutoML arm.

One sample is one attempt: the composite recipe's switches
(``benchmarks/experiments/dev/_recipes/composite.py`` class constants) plus the
trainer's own hyperparameters through ``--hparams``, the two channels a campaign
agent has inside the sealed seams. The choices are the settings the agents' 900
attempts used, so the menu covers what the agents found and a comparison between
the two is a comparison of search methods, not of search spaces.

Every switch has an off choice and every trainer hyperparameter but the learning
rate has a NATIVE choice (the key is left out and the trainer keeps its default):
the native run is a point of the menu, and a search that finds nothing can say so.
The learning rate is continuous on a log scale from a quarter to four times the
arm's own default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

NATIVE = "native"
LR_SPAN = 4.0
BALANCED = "BALANCED"
#: Recipe switches every arm draws from; None is off (the seam falls through to the trainer).
COMMON_CONSTANTS: Mapping[str, tuple[Any, ...]] = {
    "EMA_DECAY": (None, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995),
    "LS_EPS": (None, 0.05, 0.1, 0.2),
    "WARMUP_STEPS": (None, 128, 256, 512),
    "COSINE_EPOCHS": (None, 20, 40),
    "MIN_EPOCHS": (None, 12, 20),
}
#: Drawn by key: a categorical choice is a scalar, and UNFREEZE is (epoch, epoch, factor).
UNFREEZE: Mapping[str, tuple[int, int, float] | None] = {
    "off": None, "4-12-0.1": (4, 12, 0.1), "8-16-0.1": (8, 16, 0.1), "8-16-0.3": (8, 16, 0.3),
}
TILE_CAPS = (None, 512, 1024, 2048)
#: Switches one arm draws from: bag subsampling where that seam is live, nnMIL's body lr scale.
ARM_CONSTANTS: Mapping[str, Mapping[str, tuple[Any, ...]]] = {
    "abmil": {"MAX_TILES": TILE_CAPS},
    "clam_mb": {"MAX_TILES": TILE_CAPS},
    "dtfd_mil": {"MAX_TILES": TILE_CAPS},
    "simple_mil": {"BODY_LR_SCALE": (None, 0.1, 0.2, 0.5)},
}
#: The trainer's learning rate: its field name and default per arm (abmil/config.py,
#: pipeline/config.py for CLAM, dtfd/config.py, the plan nnmil/prepare.py writes).
NATIVE_LR: Mapping[str, tuple[str, float]] = {
    "abmil": ("lr", 5e-4),
    "clam_mb": ("lr", 1e-4),
    "dtfd_mil": ("lr", 1e-4),
    "simple_mil": ("learning_rate", 3e-4),
}
#: The other trainer hyperparameters, under the arm's own field names; NATIVE keeps the default.
ARM_HPARAMS: Mapping[str, Mapping[str, tuple[Any, ...]]] = {
    "abmil": {
        "weight_decay": (NATIVE, 1e-5, 1e-3, 1e-2),
        "dropout": (NATIVE, 0.1, 0.25, 0.5),
        "max_epochs": (NATIVE, 40, 80),
    },
    "clam_mb": {
        "weight_decay": (NATIVE, 1e-4, 1e-3, 1e-2),
        "dropout": (NATIVE, 0.0, 0.1, 0.5),
        "weighted_sample": (NATIVE, False),
    },
    "dtfd_mil": {
        "wd": (NATIVE, 1e-5, 1e-3, 1e-2),
        "droprate": (NATIVE, 0.25, 0.5, 0.7),
        "droprate_2": (NATIVE, 0.25, 0.5, 0.7),
        "lr_decay_step": (NATIVE, 9, 14, 30),
    },
    "simple_mil": {
        "weight_decay": (NATIVE, 1e-4, 1e-3, 1e-1),
        "dropout": (NATIVE, 0.0, 0.1, 0.5),
        "batch_size": (NATIVE, 4, 8, 12),
        "warmup_epochs": (NATIVE, 2, 5, 20),
    },
}


class Draws(Protocol):
    """What ``sample`` asks of a trial (``optuna.trial.Trial`` or ``FixedTrial``)."""

    def suggest_categorical(self, name: str, choices: Any) -> Any: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = ...) -> float: ...


@dataclass(frozen=True)
class Sample:
    """One drawn attempt."""

    params: Mapping[str, Any]      # every draw, as the study records it
    constants: Mapping[str, Any]   # the recipe switches that are on
    hparams: Mapping[str, Any]     # the trainer hyperparameters set, the learning rate always among them


def arms() -> tuple[str, ...]:
    return tuple(ARM_CONSTANTS)


def _check(arm: str, positive_share: object) -> None:
    if arm not in ARM_CONSTANTS:
        raise ValueError(f"arm {arm!r} has no menu; the menus are for {sorted(ARM_CONSTANTS)}")
    if positive_share is not None and not (
        isinstance(positive_share, float) and 0.0 < positive_share < 1.0
    ):
        raise ValueError(f"positive_share must be a float strictly between 0 and 1, got {positive_share!r}")


def _switches(trial: Draws, arm: str, positive_share: float | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(params, constants)`` of the recipe switches."""
    params: dict[str, Any] = {}
    for name, choices in {**COMMON_CONSTANTS, **ARM_CONSTANTS[arm]}.items():
        params[name] = trial.suggest_categorical(name, choices)
    params["UNFREEZE"] = trial.suggest_categorical("UNFREEZE", tuple(UNFREEZE))
    if positive_share is not None:
        params[BALANCED] = trial.suggest_categorical(BALANCED, (False, True))
    constants = {name: value for name, value in params.items()
                 if name not in ("UNFREEZE", BALANCED) and value is not None}
    if UNFREEZE[params["UNFREEZE"]] is not None:
        constants["UNFREEZE"] = UNFREEZE[params["UNFREEZE"]]
    if params.get(BALANCED):
        constants["POSITIVE_SHARE"] = positive_share
    return params, constants


def _trainer_knobs(trial: Draws, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(params, hparams)`` of the trainer's own hyperparameters."""
    field, default = NATIVE_LR[arm]
    params: dict[str, Any] = {field: trial.suggest_float(field, default / LR_SPAN, default * LR_SPAN, log=True)}
    for name, choices in ARM_HPARAMS[arm].items():
        params[name] = trial.suggest_categorical(name, choices)
    return params, {name: value for name, value in params.items() if value != NATIVE}


def sample(trial: Draws, arm: str, *, positive_share: float | None = None) -> Sample:
    """Draw one attempt of ``arm`` from the menu.

    ``positive_share`` (the share of label 1 among a two-class cell's training bags)
    adds the class-balancing switch; without it the menu has none.
    """
    _check(arm, positive_share)
    switch_params, constants = _switches(trial, arm, positive_share)
    knob_params, hparams = _trainer_knobs(trial, arm)
    return Sample({**switch_params, **knob_params}, constants, hparams)


def _literal(value: object) -> str:
    return f"{value:.3g}" if isinstance(value, float) else repr(value)


def describe(sample: Sample) -> str:
    """The attempt in one line: every switch that is on and every trainer hyperparameter set."""
    items = [*sample.constants.items(), *sample.hparams.items()]
    return ", ".join(f"{name}={_literal(value)}" for name, value in items)
