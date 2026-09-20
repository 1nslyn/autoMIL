"""Smoke-run a PolicyVariant module through the seams the benchmark trainers use.

Run by ``automil submit`` (``registry.policy_smoke`` in the cell config) before
an attempt is charged. The policy is instantiated through the same
``PolicyRuntime`` the trainers use and driven over a tiny model under each
call order that exists in the trainers: TITAN, nnMIL and the non-DTFD survival
adapters zero the gradients before the forward pass; ABMIL classification and
both DTFD tiers zero them between the forward and the backward pass; CLAM zeros
them after the step. Then the stopping seam with the metrics dict the cell's
task family passes (``--task-family classification``: ``val_auc`` and
``val_loss``; ``survival``: ``val_c_index`` and ``val_loss``), and the DTFD
seam, the only trainer that passes a scheduler: one optimizer per tier role,
a ``MultiStepLR`` on the target the trainer resolves, wrapped per role.

Exit codes: 0 the policy passed; 1 it failed a seam (the diagnostics name the
call order or the role); 2 usage (missing file, unknown task family, no
PolicyVariant subclass in the module).

Usage: python -m autobench.pipeline.policy_smoke [--task-family FAMILY] <path/to/policy.py>
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


def _run_order(order: str, policy_cls: type) -> None:
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    raw = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer = runtime.wrap_optimizer(raw)
    criterion = nn.CrossEntropyLoss()
    for _ in range(STEPS_PER_ORDER):
        if order.startswith("zero_grad"):
            optimizer.zero_grad()
        loss = criterion(model(features), labels)
        if order.startswith("forward -> zero_grad"):
            optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if order.endswith("zero_grad"):
            optimizer.zero_grad()
        if not torch.isfinite(loss):
            raise SmokeFailure(f"loss became non-finite under {order}")


def _stop_metrics(family: str, epoch: int, loss: float) -> dict[str, float]:
    """The per-epoch validation metrics the family's trainers pass to ``should_stop``."""
    primary = "val_c_index" if family == "survival" else "val_auc"
    return {primary: 0.5 + 0.05 * epoch, "val_loss": loss}


def _run_stopping(policy_cls: type, family: str) -> None:
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    optimizer = runtime.wrap_optimizer(torch.optim.Adam(model.parameters(), lr=1e-2))
    criterion = nn.CrossEntropyLoss()
    for epoch in range(STEPS_PER_ORDER):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        runtime.should_stop(
            False, epoch=epoch, metrics=_stop_metrics(family, epoch, float(loss.detach())),
        )


def _run_dtfd_tiers(policy_cls: type) -> None:
    """Both DTFD tiers as the trainer drives them: an optimizer per role, a
    ``MultiStepLR`` on the target ``scheduler_target`` resolves, the
    scheduler wrapped per role, gradients zeroed between forward and backward."""
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    criterion = nn.CrossEntropyLoss()
    for role in ROLES:
        raw = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer = runtime.wrap_optimizer(raw, role=role)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            runtime.scheduler_target(optimizer, raw, role=role), milestones=[1, 2], gamma=0.2,
        )
        scheduler = runtime.wrap_scheduler(scheduler, role=role)
        for _ in range(STEPS_PER_ORDER):
            loss = criterion(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


def _checks(policy_cls: type, family: str) -> Sequence[tuple[str, Callable[[], None]]]:
    return (
        *[(order, lambda order=order: _run_order(order, policy_cls)) for order in CALL_ORDERS],
        (f"stopping seam ({family} metrics)", lambda: _run_stopping(policy_cls, family)),
        ("DTFD tiers tier1/tier2 with MultiStepLR schedulers", lambda: _run_dtfd_tiers(policy_cls)),
    )


def smoke(path: Path, family: str = "classification") -> list[str]:
    """Run every check; return the list of failure descriptions (empty = pass)."""
    policy_cls = load_policy_class(path)
    failures = []
    for label, check in _checks(policy_cls, family):
        try:
            check()
        except Exception as exc:  # the policy is untrusted code: any failure is a verdict
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append(f"[{label}] {detail}")
    return failures


def _parse(args: list[str]) -> tuple[Path, str] | None:
    """``[--task-family FAMILY] <path>``; ``None`` on a usage error."""
    family = "classification"
    positional: list[str] = []
    tokens = list(args)
    while tokens:
        token = tokens.pop(0)
        if token == "--task-family":
            if not tokens:
                return None
            family = tokens.pop(0)
        else:
            positional.append(token)
    if len(positional) != 1 or family not in TASK_FAMILIES:
        return None
    return Path(positional[0]), family


def main(argv: Sequence[str] | None = None) -> int:
    parsed = _parse(list(sys.argv[1:] if argv is None else argv))
    if parsed is None:
        print(__doc__, file=sys.stderr)
        return 2
    path, family = parsed
    if not path.is_file():
        print(f"policy_smoke: no such file: {path}", file=sys.stderr)
        return 2
    try:
        failures = smoke(path, family)
    except Exception as exc:  # import or shape errors: usage, not a seam verdict
        print(f"policy_smoke: cannot load {path}: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(f"policy_smoke: {path.name} failed {len(failures)} check(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"policy_smoke: {path.name} passed {len(CALL_ORDERS)} call orders, the "
          f"{family} stopping seam, and the DTFD tiers with their schedulers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
