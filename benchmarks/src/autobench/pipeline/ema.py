"""Weight averaging that a train-only policy can switch on (protocol v5).

A policy that wants the averaged model evaluated and selected composes three
seams it already has:

- ``wrap_optimizer`` / ``wrap_optimizer_for``: fold every real optimizer step
  into the average (a step that ``GradScaler`` skips never reaches ``step()``,
  so it never moves the average);
- ``before_validation``: put the averaged weights in place;
- ``should_stop``: put the raw weights back before training resumes.

Every arm validates, and snapshots a selected epoch, between the last two, so
the averaged weights are what is scored and what the fold finally restores::

    class Ema(PolicyVariant):
        def __init__(self):
            self.average = WeightAverage(decay=0.99)

        def wrap_optimizer(self, opt):
            return self.average.track(opt)

        def wrap_optimizer_for(self, opt, *, role):
            return self.average.track(opt, role=role)

        def before_validation(self, *, epoch):
            self.average.swap_in()

        def should_stop(self, *, default, epoch, metrics):
            self.average.restore()
            return default

Parameters only: buffers (e.g. BatchNorm running statistics) are not
averaged.
"""
from __future__ import annotations

from typing import Any, Callable

import torch


class _TrackedOptimizer:
    """Delegate everything to ``inner``; report each completed ``step()``."""

    def __init__(self, inner: Any, on_step: Callable[[], None]) -> None:
        self._inner = inner
        self._on_step = on_step

    def step(self, *args: Any, **kwargs: Any) -> Any:
        result = self._inner.step(*args, **kwargs)
        self._on_step()
        return result

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class WeightAverage:
    """Exponential moving average of the parameters of one or more optimizers."""

    def __init__(self, decay: float) -> None:
        if isinstance(decay, bool) or not 0.0 < float(decay) < 1.0:
            raise ValueError(f"decay must lie strictly between 0 and 1, got {decay!r}")
        self.decay = float(decay)
        self._params: dict[str, list[torch.nn.Parameter]] = {}
        self._average: dict[str, list[torch.Tensor]] = {}
        self._raw: dict[str, list[torch.Tensor]] | None = None

    def track(self, optimizer: Any, *, role: str = "main") -> _TrackedOptimizer:
        """Start averaging ``optimizer``'s parameters; return the optimizer to train with."""
        if role in self._params:
            raise ValueError(f"optimizer role {role!r} is already tracked")
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self._params[role] = params
        self._average[role] = [p.detach().clone() for p in params]
        return _TrackedOptimizer(optimizer, lambda: self._update(role))

    @torch.no_grad()
    def _update(self, role: str) -> None:
        for average, param in zip(self._average[role], self._params[role]):
            average.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> None:
        """Replace the live weights with their average, keeping a copy to restore."""
        if self._raw is not None:
            raise RuntimeError("the averaged weights are already in place")
        self._raw = {
            role: [param.detach().clone() for param in params]
            for role, params in self._params.items()
        }
        for role, params in self._params.items():
            for param, average in zip(params, self._average[role]):
                param.copy_(average)

    @torch.no_grad()
    def restore(self) -> None:
        """Put the raw weights back; a no-op when nothing was swapped in."""
        if self._raw is None:
            return
        for role, params in self._params.items():
            for param, raw in zip(params, self._raw[role]):
                param.copy_(raw)
        self._raw = None
