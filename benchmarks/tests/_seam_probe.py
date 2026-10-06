"""Shared probes for the tests that drive the train-only policy seams through real arms."""

from __future__ import annotations

from collections.abc import Callable

import torch
from automil.registry.variants.policy import PolicyVariant

from autobench.pipeline.policy_dispatch import PolicyRuntime

#: The seed every probe fold is run under (``PolicyRuntime.for_fold(seed=, fold=)``).
FOLD_SEED = 42


def recording_policy(log: list, transform: Callable) -> type[PolicyVariant]:
    """A policy that logs every seam call and hands each training bag through ``transform``.

    ``log`` collects ``("bag", epoch, label, shape)`` per transformed bag,
    ``("validate", epoch)`` per ``before_validation`` and ``("stop", epoch)`` per
    ``should_stop``, in call order.
    """

    class Recording(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

        def transform_bag(self, features, *, label, epoch, generator):
            log.append(("bag", epoch, label, tuple(features.shape)))
            return transform(features)

        def before_validation(self, *, epoch):
            log.append(("validate", epoch))

        def should_stop(self, *, default, epoch, metrics):
            log.append(("stop", epoch))
            return default

    return Recording


def weight_shifting_policy() -> type[PolicyVariant]:
    """A policy whose hook shifts every weight it was handed an optimizer for.

    The epoch's validation scores whatever the hook leaves behind, so the first
    epoch's metrics differ from a native run's only if the hook ran before it.
    """

    class Shifting(PolicyVariant):
        def wrap_optimizer(self, opt):
            seen = getattr(self, "weights", [])
            self.weights = [*seen, *(p for group in opt.param_groups for p in group["params"])]
            return opt

        def before_validation(self, *, epoch):
            with torch.no_grad():
                for weight in self.weights:
                    weight.add_(1.0)

    return Shifting


def fold_runtime(policy: type[PolicyVariant] | None = None) -> PolicyRuntime:
    """The runtime a runner hands one fold: native when ``policy`` is None."""
    runtime = PolicyRuntime(name="recording" if policy else None, policy_factory=policy)
    return runtime.for_fold(seed=FOLD_SEED, fold=0)


def seam_calls(log: list) -> list[tuple[str, int]]:
    """The ``(kind, epoch)`` sequence of a ``recording_policy`` log."""
    return [(kind, epoch) for kind, epoch, *_ in log]


def expected_calls(epochs: int, bags_per_epoch: int) -> list[tuple[str, int]]:
    """Per epoch: every training bag, then the pre-validation hook, then the stopping decision.

    No validation or test bag is ever transformed, so the bags are exactly the training set.
    """
    return [
        call
        for epoch in range(epochs)
        for call in [("bag", epoch)] * bags_per_epoch + [("validate", epoch), ("stop", epoch)]
    ]


def first_half(bag):
    """A transform that really changes a bag: keep its first half."""
    return bag[: bag.shape[0] // 2]
