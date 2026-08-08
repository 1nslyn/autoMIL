"""PolicyVariant ABC — parent-agnostic training-policy variants (REG-01 / D-21).

Policies wrap the optimizer / scheduler to implement single-point strategies —
Lookahead, gradient clipping, per-group learning rates, custom schedules inside
a wrapped ``step()`` — and refine stopping decisions from supplied validation
metrics. Two honesty notes (claims-alignment C-b): ``step`` is a
consumer-optional part of the contract that none of the shipped benchmark
trainers currently invoke (they call the wrapped optimizer's ``step()``
directly), and SAM-class two-pass optimizers are out of reach through this
seam regardless — no closure re-evaluates the loss at the perturbed point.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class PolicyVariant(ABC):
    """Parent-agnostic training-policy variant.

    The protected consumer training loop owns forward passes, the defining MIL
    loss, validation, and result writing.  A policy can only adapt the optimizer
    and scheduler objects handed to it, or refine an already-computed stopping
    decision.  This is the deliberately narrow, train-only source seam used by
    architecture-preserving searches.
    """

    @abstractmethod
    def wrap_optimizer(self, opt: Any) -> Any:
        """Wrap (or replace) the optimizer; return the wrapped instance."""

    def wrap_scheduler(self, sched: Any) -> Any:
        """Optional scheduler wrapping. Default returns the input unchanged."""
        return sched

    def wrap_optimizer_for(self, opt: Any, *, role: str) -> Any:
        """Role-aware optimizer seam; legacy policies keep working unchanged."""
        return self.wrap_optimizer(opt)

    def wrap_scheduler_for(self, sched: Any, *, role: str) -> Any:
        """Role-aware scheduler seam; legacy policies keep working unchanged."""
        return self.wrap_scheduler(sched)

    def should_stop(
        self,
        *,
        default: bool,
        epoch: int,
        metrics: dict[str, float],
    ) -> bool:
        """Refine the protected trainer's stopping decision.

        The default is an identity operation, so opening this seam cannot alter
        any baseline run.  Only validation/training metrics may be supplied by
        consumers; held-out metrics remain outside the policy surface.
        """
        return default

    def step(self, loss: Any, opt: Any) -> None:
        """Default step: delegate to opt.step() after backward.

        Consumer-optional: a trainer MAY route its step through this hook
        (e.g. for gradient accumulation or loss-aware stepping), but none of
        the shipped benchmark trainers do — they call the wrapped optimizer's
        ``step()`` directly, so policies must not rely on this being invoked.
        Note SAM-style two-step optimization is unreachable either way: this
        hook receives the loss tensor, not a closure, so the loss cannot be
        re-evaluated at the perturbed point.
        """
        opt.step()
