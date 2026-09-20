"""Smoke-run a PolicyVariant module through the seams the benchmark trainers use.

Run by ``automil submit`` (``registry.policy_smoke`` in the cell config) before
an attempt is charged. The policy is instantiated through the same
``PolicyRuntime`` the trainers use and driven over a tiny model under each
call order that exists in the trainers: TITAN, nnMIL and the non-DTFD survival
adapters zero the gradients before the forward pass; ABMIL classification and
both DTFD tiers zero them between the forward and the backward pass; CLAM zeros
them after the step. Then the scheduler seam (a real ``StepLR`` on the target
the trainer would pick), the stopping seam with the classification metrics
dict, and the two DTFD roles.

Exit codes: 0 the policy passed; 1 it failed a seam (the diagnostics name the
call order); 2 usage (missing file, no PolicyVariant subclass in the module).

Usage: python -m autobench.pipeline.policy_smoke <path/to/policy.py>
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


def _run_scheduler_and_stop(policy_cls: type) -> None:
    import torch
    from torch import nn

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, features, labels = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    raw = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer = runtime.wrap_optimizer(raw)
    scheduler = torch.optim.lr_scheduler.StepLR(
        runtime.scheduler_target(optimizer, raw, role="main"), step_size=1,
    )
    scheduler = runtime.wrap_scheduler(scheduler)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(STEPS_PER_ORDER):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        runtime.should_stop(
            False, epoch=epoch,
            metrics={"val_auc": 0.5 + 0.05 * epoch, "val_loss": float(loss.detach())},
        )


def _run_roles(policy_cls: type) -> None:
    import torch

    from autobench.pipeline.policy_dispatch import PolicyRuntime

    model, _, _ = _model_and_batch()
    runtime = PolicyRuntime(name=policy_cls.__name__, policy_factory=policy_cls).for_fold()
    for role in ROLES:
        raw = torch.optim.Adam(model.parameters(), lr=1e-2)
        wrapped = runtime.wrap_optimizer(raw, role=role)
        wrapped.zero_grad()
        wrapped.step()


def _checks(policy_cls: type) -> Sequence[tuple[str, Callable[[], None]]]:
    return (
        *[(order, lambda order=order: _run_order(order, policy_cls)) for order in CALL_ORDERS],
        ("scheduler and stopping seams", lambda: _run_scheduler_and_stop(policy_cls)),
        ("optimizer roles tier1/tier2", lambda: _run_roles(policy_cls)),
    )


def smoke(path: Path) -> list[str]:
    """Run every check; return the list of failure descriptions (empty = pass)."""
    policy_cls = load_policy_class(path)
    failures = []
    for label, check in _checks(policy_cls):
        try:
            check()
        except Exception as exc:  # the policy is untrusted code: any failure is a verdict
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            failures.append(f"[{label}] {detail}")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(args[0])
    if not path.is_file():
        print(f"policy_smoke: no such file: {path}", file=sys.stderr)
        return 2
    try:
        failures = smoke(path)
    except Exception as exc:  # import or shape errors: usage, not a seam verdict
        print(f"policy_smoke: cannot load {path}: {exc}", file=sys.stderr)
        return 2
    if failures:
        print(f"policy_smoke: {path.name} failed {len(failures)} check(s):", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"policy_smoke: {path.name} passed {len(CALL_ORDERS)} call orders, "
          f"the scheduler and stopping seams, and the DTFD roles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
