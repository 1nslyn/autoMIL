"""Shared probes for the composite-recipe tests: build a recipe from the template, drive its seams."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import io
import itertools
import sys
from pathlib import Path

import torch

from autobench.pipeline import policy_smoke
from autobench.pipeline.policy_step import TrainStep

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "dev_recipe.py"
TEMPLATE = REPO_ROOT / "benchmarks" / "experiments" / "dev" / "_recipes" / "composite.py"
SEAL_CONFIG_DIR = REPO_ROOT / "benchmarks" / "experiments" / "tcga_luad" / "automil"

NODE_ID, BASE_COMMIT, CREATED_AT = "node_0002", "abc1234", "2026-10-15T00:00:00+00:00"

#: One dose per switch, sane for the tiny fixtures (bags of 16 to 24 instances, a few epochs).
SWITCHES: dict[str, dict[str, object]] = {
    "ema": {"EMA_DECAY": 0.9},
    "label_smoothing": {"LS_EPS": 0.2},
    "positive_share": {"POSITIVE_SHARE": 0.3},
    "class_weights": {"CLASS_WEIGHTS": (1.0, 2.0)},
    "max_tiles": {"MAX_TILES": 12},
    "keep_frac": {"KEEP_FRAC": 0.5},
    "warmup": {"WARMUP_STEPS": 4, "WARMUP_START": 0.2},
    "cosine": {"COSINE_EPOCHS": 4},
    "step_anneal": {"STEP_ANNEAL": (1, 0.2)},
    "min_epochs": {"MIN_EPOCHS": 2},
    "unfreeze": {"UNFREEZE": (1, 3, 0.2)},
    "body_lr_scale": {"BODY_LR_SCALE": 0.2},
}
ALL_ON: dict[str, object] = {
    name: value for switch in SWITCHES.values() for name, value in switch.items()
}


def switches_for(recipe_arm: str) -> dict[str, dict[str, object]]:
    """The switches the recipe takes on ``recipe_arm``: nnMIL (simple_mil) refuses the tile
    caps, every other arm refuses the body scale (composite.py ``_validate``)."""
    refused = ("max_tiles", "keep_frac") if recipe_arm == "simple_mil" else ("body_lr_scale",)
    return {name: switch for name, switch in SWITCHES.items() if name not in refused}


def all_on(recipe_arm: str) -> dict[str, object]:
    """Every constant ``recipe_arm`` takes, switched on together."""
    return {
        name: value for switch in switches_for(recipe_arm).values() for name, value in switch.items()
    }

_names = itertools.count()


@functools.lru_cache(maxsize=None)
def dev_recipe():
    """The operator script, loaded by path (a standalone script, not a package module)."""
    spec = importlib.util.spec_from_file_location("dev_recipe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unique_name(prefix: str = "recipe") -> str:
    """A variant name no earlier test registered (the registry is process-global)."""
    return f"{prefix}_{next(_names)}"


def instantiate(name: str | None = None, template: Path = TEMPLATE, **constants) -> str:
    return dev_recipe().instantiate_recipe(
        template, name=unique_name() if name is None else name, constants=constants,
        node_id=NODE_ID, base_commit=BASE_COMMIT, created_at=CREATED_AT,
    )


def write_recipe(tmp_path: Path, **constants) -> Path:
    name = unique_name()
    path = tmp_path / f"{name}.py"
    path.write_text(instantiate(name, **constants))
    return path


def recipe_class(tmp_path: Path, **constants) -> type:
    """The policy class of a recipe instantiated with ``constants``."""
    return policy_smoke.load_policy_class(write_recipe(tmp_path, **constants))


def captured(run, workdir, runtime):
    """One fold's comparable result and the ``[epoch k] ...`` trajectory it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = run(workdir, runtime)
    lines = [line for line in buffer.getvalue().splitlines() if line.startswith("[epoch ")]
    return result, lines


class _Recording:
    """Optimizer mixin: keep, per step, the learning rate each group stepped with and the weights left."""

    def _start_recording(self) -> None:
        self.lrs: list[list[float]] = []
        self.weights: list[list[torch.Tensor]] = []

    def step(self, *args, **kwargs):
        self.lrs.append([group["lr"] for group in self.param_groups])
        result = super().step(*args, **kwargs)
        self.weights.append(
            [p.detach().clone() for group in self.param_groups for p in group["params"]]
        )
        return result


class RecordingSGD(_Recording, torch.optim.SGD):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self._start_recording()


class RecordingAdamW(_Recording, torch.optim.AdamW):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        self._start_recording()


WIDTH = 6


def body_head_model(seed: int = 0) -> torch.nn.Module:
    """The feature projection (the widest layer), a ReLU and a head, in that order."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2),
    )


def train_epochs(policy, model, optimizer, *, epochs: int, steps_per_epoch: int, validate=True) -> None:
    """``train_step`` on one bag per step, then the validation hooks, as an arm drives them.

    ``optimizer`` is what ``policy.wrap_optimizer`` returned for ``model``'s parameters.
    """
    criterion = torch.nn.CrossEntropyLoss()
    bag = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([1])

    def backward_on(features, objective):
        logits = model(features).mean(dim=0, keepdim=True)
        loss = objective.loss(criterion, logits)
        loss.backward()
        return logits, loss

    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            policy.train_step(TrainStep(
                bag, labels, n_classes=2, epoch=epoch, policy="probe", backward_on=backward_on,
                step=optimizer.step, zero_grad=optimizer.zero_grad, generator=lambda: None,
            ))
        if validate:
            policy.before_validation(epoch=epoch)
            policy.should_stop(default=False, epoch=epoch, metrics={})
