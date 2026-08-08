"""A2 (claims-alignment): CLAM classification's stopping seam sees val metrics.

CLAM classification was the one trainer that passed ``metrics={}`` to
``should_stop`` — a metric-driven stopping policy was structurally impossible
on 15 of the 130 campaign cells while every other arm supplied real values.
``validate`` / ``validate_clam`` now return ``(stop, val_metrics)`` and the
train loop threads the dict through.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

# Installs benchmarks/lib/CLAM on sys.path (same mechanism the trainer uses).
import autobench.pipeline.clam._imports  # noqa: F401
import utils.core_utils as cu

EXPECTED_KEYS = {"val_loss", "val_error", "val_auc"}


class _TinyClam(nn.Module):
    """Minimal model honoring CLAM's forward contracts for both validators."""

    k_sample = 2

    def forward(self, data, label=None, instance_eval=False):
        logit = data.float().mean().reshape(1, 1)
        logits = torch.cat([-logit, logit], dim=1)
        y_prob = torch.softmax(logits, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        instance_dict = {
            "instance_loss": torch.tensor(0.1),
            "inst_preds": np.array([0, 1]),
            "inst_labels": np.array([0, 1]),
        }
        return logits, y_prob, y_hat, None, instance_dict


def _loader():
    # One clearly-negative and one clearly-positive bag -> both classes present,
    # so roc_auc_score is defined and equals 1.0 for this separable pair.
    return [
        (torch.full((4, 8), -1.0), torch.tensor([0])),
        (torch.full((4, 8), 1.0), torch.tensor([1])),
    ]


@pytest.fixture(autouse=True)
def _cpu_device(monkeypatch):
    monkeypatch.setattr(cu, "device", torch.device("cpu"))


def _assert_contract(result):
    stop, metrics = result
    assert stop is False
    assert set(metrics) == EXPECTED_KEYS
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} must be a real scalar"
    assert metrics["val_auc"] == 1.0


def test_validate_returns_stop_and_val_metrics():
    result = cu.validate(
        0, 0, _TinyClam(), _loader(), 2, loss_fn=nn.CrossEntropyLoss(),
    )
    _assert_contract(result)


def test_validate_clam_returns_stop_and_val_metrics():
    result = cu.validate_clam(
        0, 0, _TinyClam(), _loader(), 2, loss_fn=nn.CrossEntropyLoss(),
    )
    _assert_contract(result)


def test_policy_should_stop_receives_the_metrics():
    """The PolicyRuntime guard accepts exactly what the validators now emit."""
    from autobench.pipeline.policy_dispatch import PolicyRuntime

    class _Recorder:
        seen: dict | None = None

        def wrap_optimizer(self, opt):
            return opt

        def wrap_optimizer_for(self, opt, *, role):
            return opt

        def wrap_scheduler_for(self, sched, *, role):
            return sched

        def should_stop(self, *, default, epoch, metrics):
            _Recorder.seen = metrics
            return bool(default)

    runtime = PolicyRuntime(name="recorder", policy=_Recorder())
    _stop, metrics = cu.validate(
        0, 0, _TinyClam(), _loader(), 2, loss_fn=nn.CrossEntropyLoss(),
    )
    runtime.should_stop(False, epoch=0, metrics=metrics)
    assert _Recorder.seen is not None
    assert set(_Recorder.seen) == EXPECTED_KEYS
