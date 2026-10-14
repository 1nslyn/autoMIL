"""One optimizer step on one training batch, as a policy's ``train_step`` sees it.

Each arm hands the runtime three pieces of its own training step: a pass
(forward, the arm's defining loss, backward), the optimizer step and the
gradient reset. A policy recombines those pieces, so it can take two passes
before one step (sharpness-aware minimization), skip a step (gradient
accumulation), or train toward a soft target or with a per-bag weight. It never
holds the graph or writes a loss: the loss is always the arm's own, computed by
the arm at the target the policy asked for.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

#: Passes a policy may take per step: the pass, plus one at perturbed weights.
MAX_PASSES = 2


def check_bag(
    out: Any,
    reference: torch.Tensor,
    *,
    policy: str | None,
    seam: str,
    keep_shape: bool,
    min_instances: int,
) -> torch.Tensor:
    """A policy-made bag keeps the arm's dtype, device, feature shape and minimum size."""
    if (
        not isinstance(out, torch.Tensor)
        or out.dtype != reference.dtype
        or out.device != reference.device
        or out.ndim != reference.ndim
        or out.shape[1:] != reference.shape[1:]
    ):
        raise TypeError(
            f"policy {policy!r} {seam}() bag must be a tensor with the arm's dtype, "
            f"device and feature shape {tuple(reference.shape[1:])}"
        )
    if keep_shape and out.shape != reference.shape:
        raise TypeError(
            f"policy {policy!r} {seam}() must keep this arm's bag shape "
            f"{tuple(reference.shape)}"
        )
    if out.shape[0] < min_instances:
        raise ValueError(
            f"policy {policy!r} {seam}() left {out.shape[0]} instances; "
            f"this arm needs at least {min_instances}"
        )
    return out


@dataclass(frozen=True)
class Objective:
    """What one pass trains toward: the batch's labels unless a policy set a target or weight.

    ``target`` is ``[B]`` class indices or ``[B, C]`` class probabilities;
    ``weight`` is ``[B]``. Both None reproduce the arm's native loss exactly.
    """

    labels: torch.Tensor
    target: torch.Tensor | None = None
    weight: torch.Tensor | None = None

    @property
    def classes(self) -> torch.Tensor:
        """One class per bag: the label, or the policy target's most likely class."""
        if self.target is None:
            return self.labels
        return self.target if self.target.ndim == 1 else self.target.argmax(dim=-1)

    def loss(self, loss_fn: Any, logits: torch.Tensor, *, repeat: int = 1) -> torch.Tensor:
        """The arm's loss on ``logits``; ``repeat`` rows per bag (DTFD's pseudo-bags)."""
        labels = self.labels.repeat(repeat) if repeat > 1 else self.labels
        if self.target is None and self.weight is None:
            return loss_fn(logits, labels)
        if not isinstance(loss_fn, torch.nn.CrossEntropyLoss) or loss_fn.weight is not None:
            raise TypeError(
                "a policy target or weight needs the arm's loss to be unweighted "
                f"cross-entropy; this arm trains with {type(loss_fn).__name__}"
            )
        target = labels if self.target is None else self.target.repeat(
            repeat, *([1] * (self.target.ndim - 1)),
        )
        per_bag = F.cross_entropy(
            logits, target, reduction="none", label_smoothing=loss_fn.label_smoothing,
        )
        if self.weight is not None:
            per_bag = per_bag * self.weight.repeat(repeat)
        if loss_fn.reduction == "mean":
            return per_bag.mean()
        if loss_fn.reduction == "sum":
            return per_bag.sum()
        return per_bag

    def weighted(self, loss: torch.Tensor) -> torch.Tensor:
        """A one-bag loss the arm combined itself (CLAM's bag + instance), times the bag's weight."""
        return loss if self.weight is None else loss * self.weight[0]


class TrainStep:
    """One training batch's step: ``backward_on`` passes, at most one ``step``.

    ``features`` is the batch as the arm trains on it (one ``[N, D]`` bag, a
    padded ``[B, N, D]`` nnMIL batch or TITAN's ``[B, D]`` slide vectors);
    ``labels`` is ``[B]``. Never modify ``features`` in place.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        n_classes: int,
        epoch: int,
        policy: str | None,
        backward_on: Callable[[torch.Tensor, Objective], tuple[torch.Tensor, torch.Tensor]],
        step: Callable[[], Any],
        zero_grad: Callable[[], Any],
        generator: Callable[[], torch.Generator],
        grad_scale: Callable[[], float] | None = None,
        keep_shape: bool = False,
        min_instances: int = 1,
    ) -> None:
        self.features = features
        self.labels = labels
        self.n_classes = int(n_classes)
        self.epoch = int(epoch)
        self.loss: torch.Tensor | None = None
        self.passes = 0
        self.steps = 0
        self._policy = policy
        self._backward_on = backward_on
        self._step = step
        self._zero_grad = zero_grad
        self._generator = generator
        self._grad_scale = grad_scale
        self._keep_shape = keep_shape
        self._min_instances = min_instances

    @property
    def generator(self) -> torch.Generator:
        """A CPU generator private to this seam and fold; draw randomness only from it."""
        return self._generator()

    @property
    def grad_scale(self) -> float:
        """The factor the arm's gradients carry (nnMIL's loss scaler); 1.0 elsewhere."""
        return 1.0 if self._grad_scale is None else float(self._grad_scale())

    def backward_on(
        self,
        features: torch.Tensor | None = None,
        *,
        target: Any = None,
        weight: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward ``features`` (default: this batch), take the arm's loss, backpropagate.

        Gradients add to those already held. Returns the detached logits and loss.
        """
        if self.passes == MAX_PASSES:
            raise RuntimeError(
                f"policy {self._policy!r} train_step() took more than {MAX_PASSES} passes"
            )
        if features is None:
            features = self.features
        else:
            check_bag(
                features, self.features, policy=self._policy, seam="train_step",
                keep_shape=self._keep_shape, min_instances=self._min_instances,
            )
        objective = Objective(self.labels, self._target(target), self._weight(weight))
        logits, loss = self._backward_on(features, objective)
        self.passes += 1
        if self.loss is None:
            self.loss = loss.detach()
        return logits.detach(), loss.detach()

    def step(self) -> None:
        """Apply the held gradients with the arm's optimizer step (once per batch)."""
        if self.steps:
            raise RuntimeError(f"policy {self._policy!r} train_step() stepped twice")
        if not self.passes:
            raise RuntimeError(
                f"policy {self._policy!r} train_step() stepped before any backward_on()"
            )
        self._step()
        self.steps += 1

    def zero_grad(self) -> None:
        """Reset the gradients of every optimizer the arm trains with."""
        self._zero_grad()

    def _target(self, target: Any) -> torch.Tensor | None:
        if target is None:
            return None
        batch = self.labels.shape[0]
        if isinstance(target, int) and not isinstance(target, bool) and batch == 1:
            target = torch.tensor([target])
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"policy {self._policy!r} target must be a tensor")
        target = target.to(self.labels.device)
        if target.dtype in (torch.int32, torch.int64):
            if target.shape != (batch,) or not bool(
                ((target >= 0) & (target < self.n_classes)).all()
            ):
                raise ValueError(
                    f"policy {self._policy!r} class target must be [{batch}] "
                    f"indices in [0, {self.n_classes})"
                )
            return target.long()
        if not target.is_floating_point():
            raise TypeError(f"policy {self._policy!r} target must be integer or float")
        if target.ndim == 1 and batch == 1:
            target = target.unsqueeze(0)
        target = target.float()
        if (
            target.shape != (batch, self.n_classes)
            or not bool(torch.isfinite(target).all())
            or bool((target < 0).any())
            or not torch.allclose(target.sum(dim=-1), torch.ones(batch, device=target.device), atol=1e-4)
        ):
            raise ValueError(
                f"policy {self._policy!r} probability target must be [{batch}, "
                f"{self.n_classes}] rows of non-negative values summing to 1"
            )
        return target

    def _weight(self, weight: Any) -> torch.Tensor | None:
        if weight is None:
            return None
        batch = self.labels.shape[0]
        if isinstance(weight, (int, float)) and not isinstance(weight, bool):
            weight = torch.full((batch,), float(weight))
        if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
            raise TypeError(f"policy {self._policy!r} weight must be a float or float tensor")
        weight = weight.to(self.labels.device).float()
        if weight.ndim == 0:
            weight = weight.expand(batch)
        if (
            weight.shape != (batch,)
            or not bool(torch.isfinite(weight).all())
            or bool((weight < 0).any())
        ):
            raise ValueError(
                f"policy {self._policy!r} weight must be [{batch}] finite non-negative values"
            )
        return weight
