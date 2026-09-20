"""Smoke-run a PolicyVariant module through the seams the benchmark trainers use.

Run by ``automil submit`` (``registry.policy_smoke`` in the cell config) before
an attempt is charged. The policy is instantiated through the same
``PolicyRuntime`` the trainers use and driven over a tiny model under each
call order that exists in the trainers: TITAN, nnMIL and the non-DTFD survival
adapters zero the gradients before the forward pass; ABMIL classification and
both DTFD tiers zero them between the forward and the backward pass; CLAM zeros
them after the step; nnMIL's order is also driven through a ``GradScaler`` as
on CUDA. Then the stopping seam exactly as the cell's arm drives it
(``--arm``: the arm's metrics dict, its first validated epoch, and on DTFD
after both tiers and their schedulers exist on the same runtime) for the
cell's task family (``--task-family classification`` or ``survival``), and,
for DTFD or an unspecified arm, the DTFD seam: both tier optimizers wrapped,
then both ``MultiStepLR`` schedulers built on the targets the trainer
resolves and wrapped, before either tier trains.

Exit codes: 0 the policy passed; 1 it failed a seam (the diagnostics name the
call order or the role); 2 usage (missing file, unknown task family or arm, no
PolicyVariant subclass in the module).

Usage: python -m autobench.pipeline.policy_smoke [--task-family FAMILY] [--arm ARM] <path/to/policy.py>
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

CALL_ORDERS: tuple[str, ...] = (
    "zero_grad -> forward -> backward -> step",
    "forward -> zero_grad -> backward -> step",
    "forward -> backward -> step -> zero_grad",
)
STEPS_PER_ORDER = 3
ROLES = ("tier1", "tier2")
TASK_FAMILIES = ("classification", "survival")
ARMS = ("abmil", "clam", "dtfd", "nnmil", "titan")
#: The call order each arm's CLASSIFICATION trainer uses between two stopping
#: decisions; every non-DTFD survival adapter zeroes before the forward pass.
ARM_ORDER = {
    "titan": CALL_ORDERS[0], "nnmil": CALL_ORDERS[0],
    "abmil": CALL_ORDERS[1], "dtfd": CALL_ORDERS[1],
    "clam": CALL_ORDERS[2],
}


def _stop_order(arm: str | None, family: str) -> str:
    if family == "survival" and arm != "dtfd":
        return CALL_ORDERS[0]
    return ARM_ORDER.get(arm, CALL_ORDERS[0])


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


def _model_and_batch():
    import torch
    from torch import nn

    torch.manual_seed(0)
    # Two layers so the second layer's weight is saved for backward: an
    # in-place change of it between forward and backward trips autograd's
    # version check, which is how the rehearsal crash surfaced.
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    features = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))
    return model, features, labels


def _run_order(order: str, policy_cls: type, *, scaled: bool = False) -> None:
    """One call order for a few steps; ``scaled`` drives the step through a
    ``GradScaler`` as nnMIL does on CUDA (the scaler unscales the gradients it
    finds through the wrapper's ``param_groups``, so a wrapper that hands it
    copies fails here as it fails there)."""
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    raw = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer = runtime.wrap_optimizer(raw)
    scaler = torch.amp.GradScaler("cpu", enabled=True) if scaled else None
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


def _dtfd_tiers(runtime, features, labels):
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
    tier1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    tier2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3))
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
    from torch import nn

    criterion = nn.CrossEntropyLoss()
    for module, optimizer, scheduler in tiers:
        loss = criterion(module(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def _run_dtfd_tiers(policy_cls: type) -> None:
    from autobench.pipeline.policy_dispatch import PolicyRuntime

    _, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    tiers = _dtfd_tiers(runtime, features, labels)
    for _ in range(STEPS_PER_ORDER):
        _step_dtfd_tiers(tiers, features, labels)


def _run_stopping(policy_cls: type, arm: str | None, family: str) -> None:
    """The stopping seam as the arm's trainer drives it: the arm's own call
    order between two decisions (what a policy arms in ``should_stop`` and
    performs in the next ``zero_grad`` lands where the trainer puts that
    call), on DTFD after both tiers and their schedulers exist on the same
    runtime (a policy may hold its scheduler and read it here), from the
    arm's first validated epoch, with the arm's metrics."""
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    if arm == "dtfd":
        tiers = _dtfd_tiers(runtime, features, labels)
        step = lambda: _step_dtfd_tiers(tiers, features, labels)  # noqa: E731
        loss_value = lambda: 0.7  # noqa: E731
    else:
        order = _stop_order(arm, family)
        optimizer = runtime.wrap_optimizer(torch.optim.Adam(model.parameters(), lr=1e-2))
        criterion = nn.CrossEntropyLoss()
        last = {"loss": 0.7}

        def step() -> None:
            last["loss"] = float(
                _train_step(order, model, optimizer, criterion, features, labels).detach()
            )

        loss_value = lambda: last["loss"]  # noqa: E731
    first = _first_stop_epoch(arm, family)
    for epoch in range(first, first + STEPS_PER_ORDER):
        step()
        # The middle epoch carries the arm's undefined-metric values.
        metrics = _stop_metrics(
            arm, family, epoch, loss_value(), undefined=(epoch == first + 1),
        )
        try:
            runtime.should_stop(False, epoch=epoch, metrics=metrics)
        except Exception as exc:
            raise RuntimeError(
                f"should_stop(epoch={epoch}, metrics={metrics}) raised "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def _checks(policy_cls: type, arm: str | None, family: str) -> Sequence[tuple[str, Callable[[], None]]]:
    checks = [
        *[(order, lambda order=order: _run_order(order, policy_cls)) for order in CALL_ORDERS],
        (f"{CALL_ORDERS[0]} through a GradScaler (nnMIL on CUDA)",
         lambda: _run_order(CALL_ORDERS[0], policy_cls, scaled=True)),
        (f"stopping seam ({arm or 'generic'} {family} metrics)",
         lambda: _run_stopping(policy_cls, arm, family)),
    ]
    if arm in (None, "dtfd"):
        checks.append(("DTFD tiers tier1/tier2 with MultiStepLR schedulers",
                       lambda: _run_dtfd_tiers(policy_cls)))
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
    print(f"policy_smoke: {path.name} passed {len(CALL_ORDERS)} call orders and the "
          f"{arm or 'generic'} {family} stopping seam"
          + (" and the DTFD tiers with their schedulers" if arm in (None, "dtfd") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
