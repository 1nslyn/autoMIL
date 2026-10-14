"""Smoke-run a PolicyVariant module through the seams the benchmark trainers use.

Run by ``automil submit`` (``registry.policy_smoke`` in the cell config) before
an attempt is charged. The policy is instantiated through the same
``PolicyRuntime`` the trainers use and driven over a tiny model the way the
cell's trainer drives it. A classification trainer trains every batch through
the runtime's ``train_step`` (the policy's, or the native zero_grad, pass,
step), on the batch its arm steps on (``--arm``): one bag (ABMIL; CLAM, which
keeps eight instances), a batch of slide vectors (TITAN), a padded batch
through a ``GradScaler`` as on CUDA (nnMIL), or both DTFD tiers on one bag.
The survival adapters step without ``train_step``, under the call orders
they use: every non-DTFD adapter zeroes before the forward pass, both DTFD
tiers between the forward and the backward pass, nnMIL's through a
``GradScaler``. Then the stopping seam exactly as the cell's arm drives it
(the arm's metrics dict, its first validated epoch, and on DTFD after both
tiers and their schedulers exist on the same runtime) for the cell's task
family (``--task-family classification`` or ``survival``), and, for DTFD or an
unspecified arm, the DTFD seam: both tier optimizers wrapped, then both
``MultiStepLR`` schedulers built on the targets the trainer resolves and
wrapped, before either tier trains.

Every check runs on the device the trainers train on (CUDA when the host has
it), so a policy that mixes a CPU tensor into a GPU bag or parameter fails
here as it would fail in the run.

A policy that overrides ``transform_bag``, ``train_step`` or
``before_validation`` has those seams driven too, through the runtime the
trainers use, seeded as a fold's is: the bag transform on the bag the arm
hands it (CLAM: eight instances must survive; TITAN: one vector per slide,
shape kept), the training step on the arm's batch. Each must give the same
result on two runs and draw nothing from the global RNGs. The survival
trainers call none of these seams, so overriding one there is refused: the
cell would run without it.

Exit codes: 0 the policy passed; 1 it failed a seam (the diagnostics name the
batch, the call order or the role); 2 usage (missing file, unknown task family
or arm, no PolicyVariant subclass in the module).

Usage: python -m autobench.pipeline.policy_smoke [--task-family FAMILY] [--arm ARM] <path/to/policy.py>
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

#: How the survival adapters step (no ``train_step`` there): every non-DTFD
#: adapter zeroes before the forward pass, DTFD's tiers between the forward
#: and the backward pass.
CALL_ORDERS: tuple[str, ...] = (
    "zero_grad -> forward -> backward -> step",
    "forward -> zero_grad -> backward -> step",
)
STEPS_PER_ORDER = 3
ROLES = ("tier1", "tier2")
TASK_FAMILIES = ("classification", "survival")
ARMS = ("abmil", "clam", "dtfd", "nnmil", "titan")
#: The train-only seams of PolicyVariant that no survival trainer calls.
SEAMS = ("transform_bag", "train_step", "before_validation")
BAG_INSTANCES = 16
BAG_WIDTH = 8
N_CLASSES = 2
#: CLAM's instance-level loss samples ``ModelConfig.B`` instances per class branch.
CLAM_MIN_INSTANCES = 8
#: The bag each arm's classification trainer hands the bag transform: what it
#: is, how many instances it has, and what the arm's model needs back.
BAG_CASES = {
    "bag": (f"a bag of {BAG_INSTANCES} instances", BAG_INSTANCES, {}),
    "clam": (f"a CLAM bag of {BAG_INSTANCES} instances, {CLAM_MIN_INSTANCES} kept", BAG_INSTANCES,
             {"min_instances": CLAM_MIN_INSTANCES}),
    "titan": ("a TITAN slide vector, shape kept", 1, {"keep_shape": True}),
}
ARM_BAG_CASE = {"abmil": "bag", "dtfd": "bag", "nnmil": "bag", "clam": "clam", "titan": "titan"}
#: The batch each arm's classification trainer trains on through ``train_step``:
#: what it is, its shape, and what the arm's model needs back. DTFD steps one
#: bag through both tiers (``_run_dtfd_tiers``).
STEP_CASES = {
    "bag": (f"one bag of {BAG_INSTANCES} instances", (BAG_INSTANCES, BAG_WIDTH), {}),
    "clam": (f"one CLAM bag of {BAG_INSTANCES} instances, {CLAM_MIN_INSTANCES} kept",
             (BAG_INSTANCES, BAG_WIDTH), {"min_instances": CLAM_MIN_INSTANCES}),
    "titan": ("a batch of 8 TITAN slide vectors, shape kept", (8, BAG_WIDTH), {"keep_shape": True}),
    "nnmil": (f"a padded nnMIL batch of 4 bags of {BAG_INSTANCES} through a GradScaler, shape kept",
              (4, BAG_INSTANCES, BAG_WIDTH), {"keep_shape": True}),
}
ARM_STEP_CASE = {"abmil": "bag", "clam": "clam", "titan": "titan", "nnmil": "nnmil"}


def _step_cases(arm: str | None) -> tuple[str, ...]:
    if arm is None:
        return tuple(STEP_CASES)
    return (ARM_STEP_CASE[arm],) if arm in ARM_STEP_CASE else ()


def _survival_order(arm: str | None) -> str:
    return CALL_ORDERS[1] if arm == "dtfd" else CALL_ORDERS[0]


class SmokeFailure(Exception):
    """A seam the trainers use rejected the policy."""


def load_policy_class(path: Path) -> type:
    """Import ``path`` and return the one PolicyVariant subclass it defines."""
    from automil.registry.variants import PolicyVariant

    name = f"_policy_smoke_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    classes = [
        value for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, PolicyVariant)
        and value is not PolicyVariant and value.__module__ == module.__name__
    ]
    if len(classes) != 1:
        raise ValueError(
            f"{path} must define exactly one PolicyVariant subclass, found {len(classes)}"
        )
    return classes[0]


def _device():
    """Where every trainer puts its model and its bags: CUDA when the host has it."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _two_layers(hidden: int = 4):
    """Two layers so the second layer's weight is saved for backward: an
    in-place change of it between forward and backward trips autograd's
    version check, which is how the rehearsal crash surfaced. Seeded, and
    built without moving the global RNGs, which the step checks watch."""
    import torch
    from torch import nn

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(BAG_WIDTH, hidden), nn.ReLU(), nn.Linear(hidden, N_CLASSES),
        )
    return model.to(_device())


def _model():
    return _two_layers()


def _batch(shape: tuple[int, ...], n_labels: int):
    import torch

    generator = torch.Generator().manual_seed(0)
    features = torch.randn(*shape, generator=generator)
    labels = torch.randint(0, N_CLASSES, (n_labels,), generator=generator)
    return features.to(_device()), labels.to(_device())


def _model_and_batch():
    """The survival adapters' step: a batch of 8 rows, one label each."""
    return (_model(), *_batch((8, BAG_WIDTH), 8))


def _logits(model, features, case: str):
    """The arm's forward: a bag pools to one row, a padded batch to one row per bag."""
    if case in ("bag", "clam"):
        return model(features).mean(dim=0, keepdim=True)
    if case == "nnmil":
        return model(features).mean(dim=1)
    return model(features)


def _run_order(order: str, policy_cls: type, *, scaled: bool = False) -> None:
    """A survival adapter's call order for a few steps; ``scaled`` drives the
    step through a ``GradScaler`` as nnMIL's survival trainers do on CUDA (the
    scaler unscales the gradients it finds through the wrapper's
    ``param_groups``, so a wrapper that hands it copies fails here as it fails
    there)."""
    import torch
    from torch import nn

    model, features, labels = _model_and_batch()
    runtime = _fold_runtime(policy_cls)
    raw = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer = runtime.wrap_optimizer(raw)
    scaler = torch.amp.GradScaler(_device().type, enabled=True) if scaled else None
    criterion = nn.CrossEntropyLoss()
    for _ in range(STEPS_PER_ORDER):
        loss = _train_step(order, model, optimizer, criterion, features, labels, scaler)
        if not torch.isfinite(loss):
            raise SmokeFailure(f"loss became non-finite under {order}")


def _train_step(order, model, optimizer, criterion, features, labels, scaler=None):
    """One optimizer step in the given call order."""
    # nnMIL reads the learning rate off the wrapper every epoch; every
    # scheduler the trainers attach mutates the same param_groups.
    optimizer.param_groups[0]["lr"]
    if order.startswith("zero_grad"):
        optimizer.zero_grad()
    loss = criterion(model(features), labels)
    if order.startswith("forward -> zero_grad"):
        optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    if order.endswith("zero_grad"):
        optimizer.zero_grad()
    return loss


def _policy_step(runtime, model, optimizer, features, labels, case: str, *, epoch: int, scaler=None):
    """One classification step as the arm's trainer takes it: through ``train_step``."""
    from torch import nn

    criterion = nn.CrossEntropyLoss()
    # nnMIL reads the learning rate off the wrapper every epoch; every
    # scheduler the trainers attach mutates the same param_groups.
    optimizer.param_groups[0]["lr"]

    def backward_on(batch, objective):
        logits = _logits(model, batch, case)
        loss = objective.loss(criterion, logits)
        (loss if scaler is None else scaler.scale(loss)).backward()
        return logits, loss

    def step():
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

    return runtime.train_step(
        features, labels, epoch=epoch, n_classes=N_CLASSES, backward_on=backward_on,
        step=step, zero_grad=optimizer.zero_grad,
        grad_scale=None if scaler is None else scaler.get_scale, **STEP_CASES[case][2],
    )


def _fold_runtime(policy_cls: type):
    from autobench.pipeline.policy_dispatch import PolicyRuntime

    return PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold(
        seed=0, fold=0,
    )


def _arm_stepper(runtime, case: str) -> Callable[[int], float | None]:
    """The arm's classification step on a fresh model; each call returns the
    step's loss (None when the policy skipped the batch)."""
    import torch

    model = _model()
    shape = STEP_CASES[case][1]
    features, labels = _batch(shape, 1 if case in ("bag", "clam") else shape[0])
    optimizer = runtime.wrap_optimizer(torch.optim.Adam(model.parameters(), lr=1e-2))
    scaler = torch.amp.GradScaler(_device().type) if case == "nnmil" else None

    def step(epoch: int) -> float | None:
        loss = _policy_step(
            runtime, model, optimizer, features, labels, case, epoch=epoch, scaler=scaler,
        )
        return None if loss is None else float(loss)

    return step


def _run_steps(policy_cls: type, case: str) -> list[float | None]:
    step = _arm_stepper(_fold_runtime(policy_cls), case)
    return [step(epoch) for epoch in range(STEPS_PER_ORDER)]


def _same_losses(first: list[float | None], second: list[float | None]) -> bool:
    import math

    return all(
        a == b or (a is not None and b is not None and math.isnan(a) and math.isnan(b))
        for a, b in zip(first, second)
    )


def _check_steps(policy_cls: type, run: Callable[[], list[float | None]], what: str) -> None:
    """The steps stay finite; an overriding policy's are reproducible and leave
    the global RNGs where they were."""
    import math

    from autobench.pipeline.policy_dispatch import defines_seam

    before = _global_rng_state()
    first = run()
    if defines_seam(policy_cls, "train_step"):
        second = run()
        if _global_rng_state() != before:
            raise SmokeFailure("train_step() drew from a global RNG; draw from step.generator")
        if not _same_losses(first, second):
            raise SmokeFailure(
                "train_step() is not reproducible: two runs seeded alike trained differently"
            )
    if any(loss is not None and not math.isfinite(loss) for loss in first):
        raise SmokeFailure(f"loss became non-finite on {what}")


def _run_train_step(policy_cls: type, case: str) -> None:
    _check_steps(policy_cls, lambda: _run_steps(policy_cls, case), STEP_CASES[case][0])


def _stop_metrics(
    arm: str | None, family: str, epoch: int, loss: float, *, undefined: bool = False,
) -> dict[str, float]:
    """Exactly the per-epoch validation metrics the arm's trainer passes to
    ``should_stop`` for the family (a policy reading a key its cell never
    supplies must fail here, one reading a key it does must pass).

    ``undefined`` is the epoch whose metric could not be computed (a
    single-class validation fold, non-finite predictions): ABMIL and DTFD
    hand the policy ``-inf`` for the AUC (DTFD ``+inf`` for the loss), every
    other loop and every survival adapter ``nan``. A policy that crashes on
    those values would crash the real loop after the attempt is charged.
    """
    if undefined:
        nan, inf = float("nan"), float("inf")
        if family == "survival":
            return {"val_loss": nan, "val_c_index": nan}
        if arm == "dtfd":
            return {"val_auc": -inf, "val_loss": inf}
        if arm == "abmil":
            return {"val_auc": -inf, "val_loss": nan}
        if arm == "clam":
            return {"val_loss": nan, "val_error": nan, "val_auc": nan}
        if arm == "nnmil":
            return {"val_loss": nan, "val_bacc": nan, "val_f1": nan, "val_auc": nan}
        return {"val_auc": nan, "val_loss": nan}
    rising = 0.5 + 0.05 * epoch
    if family == "survival":
        return {"val_loss": loss, "val_c_index": rising}
    if arm == "clam":
        return {"val_loss": loss, "val_error": 0.5 - 0.05 * epoch, "val_auc": rising}
    if arm == "nnmil":
        return {"val_loss": loss, "val_bacc": rising, "val_f1": rising, "val_auc": rising}
    return {"val_auc": rising, "val_loss": loss}


def _first_stop_epoch(arm: str | None, family: str) -> int:
    """Every loop the campaign runs asks from epoch 0, including nnMIL's
    survival trainer for the locked ``nllsurv`` loss (the porpoise trainer);
    only nnMIL's Cox trainer, which no cell runs, warms up for two epochs."""
    return 0


def _dtfd_tiers(runtime, device):
    """Both DTFD tiers as the trainer builds them: two parameter sets, both
    optimizers wrapped, then both ``MultiStepLR`` schedulers built on the
    targets the trainer resolves and wrapped, before either tier trains (one
    policy instance wraps both, so state a policy keeps per wrap on itself
    collides here as it does there)."""
    import torch
    from torch import nn

    # Two layers on each tier, so a weight the backward pass needs is saved
    # on both: an in-place change between forward and backward trips
    # autograd's version check whichever role it targets.
    # The tiers' parameters differ in shape, as DTFD's do, so state a policy
    # carries from one tier's wrap into the other's fails here as it does there.
    tier1, tier2 = _two_layers(4).to(device), _two_layers(3).to(device)
    raws = [torch.optim.Adam(module.parameters(), lr=1e-2) for module in (tier1, tier2)]
    optimizers = [runtime.wrap_optimizer(raw, role=role) for role, raw in zip(ROLES, raws)]
    schedulers = [
        torch.optim.lr_scheduler.MultiStepLR(
            runtime.scheduler_target(optimizer, raw, role=role), milestones=[1, 2], gamma=0.2,
        )
        for role, optimizer, raw in zip(ROLES, optimizers, raws)
    ]
    schedulers = [runtime.wrap_scheduler(sched, role=role) for role, sched in zip(ROLES, schedulers)]
    return list(zip((tier1, tier2), optimizers, schedulers))


def _step_dtfd_tiers(tiers, features, labels) -> None:
    """The DTFD survival adapter: each tier forward, zero_grad, backward, step."""
    from torch import nn

    criterion = nn.CrossEntropyLoss()
    for module, optimizer, scheduler in tiers:
        loss = criterion(module(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def _dtfd_policy_stepper(runtime) -> Callable[[int], float | None]:
    """The DTFD classification trainer: one bag, both tiers in one ``train_step``
    (both reset, both passes, both steps), then both schedulers."""
    from torch import nn

    tiers = _dtfd_tiers(runtime, _device())
    features, labels = _batch(STEP_CASES["bag"][1], 1)
    criterion = nn.CrossEntropyLoss()

    def zero_grad():
        for _, optimizer, _ in tiers:
            optimizer.zero_grad()

    first_size: list[int] = []  # the trainer draws one pseudo-bag split per slide

    def backward_on(bag, objective):
        if not first_size:
            first_size.append(bag.shape[0])
        elif bag.shape[0] != first_size[0]:
            raise ValueError(
                "a second DTFD pass must use a bag of the first pass's size "
                "(its pseudo-bag split is drawn once per slide)"
            )
        for module, _, _ in tiers:
            logits = module(bag).mean(dim=0, keepdim=True)
            loss = objective.loss(criterion, logits)
            loss.backward()
        return logits, loss

    def step():
        for _, optimizer, _ in tiers:
            optimizer.step()

    def run(epoch: int) -> float | None:
        first_size.clear()
        loss = runtime.train_step(
            features, labels, epoch=epoch, n_classes=N_CLASSES, backward_on=backward_on,
            step=step, zero_grad=zero_grad,
        )
        for _, _, scheduler in tiers:
            scheduler.step()
        return None if loss is None else float(loss)

    return run


def _run_dtfd_tiers(policy_cls: type, family: str) -> None:
    if family == "survival":
        _, features, labels = _model_and_batch()
        tiers = _dtfd_tiers(_fold_runtime(policy_cls), features.device)
        for _ in range(STEPS_PER_ORDER):
            _step_dtfd_tiers(tiers, features, labels)
        return

    def run() -> list[float | None]:
        step = _dtfd_policy_stepper(_fold_runtime(policy_cls))
        return [step(epoch) for epoch in range(STEPS_PER_ORDER)]

    _check_steps(policy_cls, run, "both DTFD tiers on one bag")


def _run_stopping(policy_cls: type, arm: str | None, family: str) -> None:
    """The stopping seam as the arm's trainer drives it: the arm's own step
    between two decisions (what a policy arms in ``should_stop`` and performs
    in the next ``zero_grad`` lands where the trainer puts that call), on DTFD
    after both tiers and their schedulers exist on the same runtime (a policy
    may hold its scheduler and read it here), from the arm's first validated
    epoch, with the arm's metrics."""
    import torch
    from torch import nn

    runtime = _fold_runtime(policy_cls)
    last = {"loss": 0.7}
    if family == "classification":
        stepper = (
            _dtfd_policy_stepper(runtime) if arm == "dtfd"
            else _arm_stepper(runtime, ARM_STEP_CASE.get(arm, "bag"))
        )

        def step(epoch: int) -> None:
            loss = stepper(epoch)
            if loss is not None:
                last["loss"] = loss
    elif arm == "dtfd":
        _, features, labels = _model_and_batch()
        tiers = _dtfd_tiers(runtime, features.device)

        def step(epoch: int) -> None:
            _step_dtfd_tiers(tiers, features, labels)
    else:
        model, features, labels = _model_and_batch()
        optimizer = runtime.wrap_optimizer(torch.optim.Adam(model.parameters(), lr=1e-2))
        criterion = nn.CrossEntropyLoss()

        def step(epoch: int) -> None:
            last["loss"] = float(_train_step(
                _survival_order(arm), model, optimizer, criterion, features, labels,
            ).detach())

    first = _first_stop_epoch(arm, family)
    for epoch in range(first, first + STEPS_PER_ORDER):
        step(epoch)
        # The middle epoch carries the arm's undefined-metric values.
        metrics = _stop_metrics(
            arm, family, epoch, last["loss"], undefined=(epoch == first + 1),
        )
        try:
            runtime.should_stop(False, epoch=epoch, metrics=metrics)
        except Exception as exc:
            raise RuntimeError(
                f"should_stop(epoch={epoch}, metrics={metrics}) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _global_rng_state() -> tuple:
    """The process-wide torch (CPU and, on a GPU host, CUDA), numpy and Python
    RNG states, comparable with ``==``."""
    import random

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    cuda_state = (torch.cuda.get_rng_state().numpy().tobytes(),) if torch.cuda.is_available() else ()
    return (
        torch.get_rng_state().numpy().tobytes(),
        numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:],
        random.getstate(),
        *cuda_state,
    )


def _transform_bags(policy_cls: type, rows: int, constraints: dict) -> list:
    """Three epochs of one bag through a fresh fold runtime, as a trainer drives it."""
    import torch

    bag = torch.randn(rows, BAG_WIDTH, generator=torch.Generator().manual_seed(0)).to(_device())
    runtime = _fold_runtime(policy_cls)
    return [
        runtime.transform_bag(bag, label=epoch % 2, epoch=epoch, **constraints)
        for epoch in range(STEPS_PER_ORDER)
    ]


def _run_transform_bag(policy_cls: type, rows: int, constraints: dict) -> None:
    """The runtime accepts what the policy returns; two runs agree; the global RNGs stay put."""
    import torch

    before = _global_rng_state()
    first = _transform_bags(policy_cls, rows, constraints)
    second = _transform_bags(policy_cls, rows, constraints)
    if _global_rng_state() != before:
        raise SmokeFailure(
            "transform_bag() drew from a global RNG; draw from the generator argument"
        )
    if not all(torch.equal(a, b) for a, b in zip(first, second)):
        raise SmokeFailure(
            "transform_bag() is not reproducible: two runs seeded alike returned different bags"
        )


def _run_before_validation(policy_cls: type, arm: str | None) -> None:
    """The hook after the optimizer is wrapped and a step has run, as a classification
    trainer calls it (the survival trainers never do)."""
    runtime = _fold_runtime(policy_cls)
    if arm == "dtfd":
        _dtfd_policy_stepper(runtime)(0)
    else:
        _arm_stepper(runtime, ARM_STEP_CASE.get(arm, "bag"))(0)
    runtime.before_validation(epoch=0)


def _refuse_survival(seam: str) -> None:
    raise SmokeFailure(
        f"{seam}() is not wired for survival trainers: this cell would run without it"
    )


def _seam_checks(
    policy_cls: type, arm: str | None, family: str,
) -> list[tuple[str, Callable[[], None]]]:
    """One check per data seam the policy overrides (the training step runs
    for every policy, with its own checks)."""
    from autobench.pipeline.policy_dispatch import defines_seam

    checks: list[tuple[str, Callable[[], None]]] = []
    for seam in SEAMS:
        if not defines_seam(policy_cls, seam):
            continue
        if family == "survival":
            checks.append((f"{seam} seam (survival)", lambda seam=seam: _refuse_survival(seam)))
        elif seam == "transform_bag":
            for case in ((ARM_BAG_CASE[arm],) if arm else tuple(BAG_CASES)):
                what, rows, constraints = BAG_CASES[case]
                checks.append((
                    f"transform_bag seam ({what})",
                    lambda rows=rows, constraints=constraints: _run_transform_bag(
                        policy_cls, rows, constraints,
                    ),
                ))
        elif seam == "before_validation":
            checks.append((
                "before_validation seam",
                lambda: _run_before_validation(policy_cls, arm),
            ))
    return checks


def _step_checks(policy_cls: type, arm: str | None, family: str) -> list[tuple[str, Callable[[], None]]]:
    """How the cell's trainer steps: ``train_step`` on the arm's batch for
    classification, the adapters' call orders for survival."""
    if family == "survival":
        orders = CALL_ORDERS if arm is None else (_survival_order(arm),)
        checks = [(order, lambda order=order: _run_order(order, policy_cls)) for order in orders]
        if arm in (None, "nnmil"):
            checks.append((f"{CALL_ORDERS[0]} through a GradScaler (nnMIL on CUDA)",
                           lambda: _run_order(CALL_ORDERS[0], policy_cls, scaled=True)))
        return checks
    return [
        (f"train_step on {STEP_CASES[case][0]}", lambda case=case: _run_train_step(policy_cls, case))
        for case in _step_cases(arm)
    ]


def _checks(policy_cls: type, arm: str | None, family: str) -> Sequence[tuple[str, Callable[[], None]]]:
    checks = [
        *_step_checks(policy_cls, arm, family),
        (f"stopping seam ({arm or 'generic'} {family} metrics)",
         lambda: _run_stopping(policy_cls, arm, family)),
    ]
    if arm in (None, "dtfd"):
        checks.append(("DTFD tiers tier1/tier2 with MultiStepLR schedulers",
                       lambda: _run_dtfd_tiers(policy_cls, family)))
    checks.extend(_seam_checks(policy_cls, arm, family))
    return tuple(checks)


def smoke(path: Path, family: str = "classification", arm: str | None = None) -> list[str]:
    """Run every check; return the list of failure descriptions (empty = pass)."""
    policy_cls = load_policy_class(path)
    failures = []
    for label, check in _checks(policy_cls, arm, family):
        try:
            check()
        except Exception as exc:  # the policy is untrusted code: any failure is a verdict
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append(f"[{label}] {detail}")
    return failures


def _parse(args: list[str]) -> tuple[Path, str, str | None] | None:
    """``[--task-family FAMILY] [--arm ARM] <path>``; ``None`` on a usage error."""
    options = {"--task-family": "classification", "--arm": None}
    positional: list[str] = []
    tokens = list(args)
    while tokens:
        token = tokens.pop(0)
        if token in options:
            if not tokens:
                return None
            options[token] = tokens.pop(0)
        else:
            positional.append(token)
    family, arm = options["--task-family"], options["--arm"]
    if len(positional) != 1 or family not in TASK_FAMILIES or (arm is not None and arm not in ARMS):
        return None
    return Path(positional[0]), family, arm


def main(argv: Sequence[str] | None = None) -> int:
    parsed = _parse(list(sys.argv[1:] if argv is None else argv))
    if parsed is None:
        print(__doc__, file=sys.stderr)
        return 2
    path, family, arm = parsed
    if not path.is_file():
        print(f"policy_smoke: no such file: {path}", file=sys.stderr)
        return 2
    try:
        failures = smoke(path, family, arm)
    except Exception as exc:  # import or shape errors: usage, not a seam verdict
        print(f"policy_smoke: cannot load {path}: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(f"policy_smoke: {path.name} failed {len(failures)} check(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"policy_smoke: {path.name} passed the {arm or 'generic'} {family} training "
          "steps and stopping seam"
          + (" and the DTFD tiers with their schedulers" if arm in (None, "dtfd") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
