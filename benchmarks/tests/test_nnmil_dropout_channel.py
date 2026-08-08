"""A8 (claims-alignment): nnMIL's declared-tunable dropout must actually land.

`model_factory` hardcoded `SimpleMIL(..., dropout=True)`, so the training
plan's `dropout` value — declared searchable in `search_space.py` and applied
to the plan by `apply_overrides_to_plan` — was silently discarded: `nn.Dropout(0.25)`
ran regardless. On the campaign's 30 nnmil cells an agent tuning `dropout`
burned attempts on baseline-identical runs (the H-3 defect class).
"""
from __future__ import annotations

from torch import nn

# Installs benchmarks/lib/nnMIL on sys.path (same mechanism the trainer uses).
import autobench.pipeline.nnmil._imports  # noqa: F401
from network_architecture.model_factory import create_mil_model
from network_architecture.models.simple_mil import SimpleMIL


def _make(**kwargs):
    return create_mil_model(
        "simple_mil", input_dim=32, hidden_dim=16, num_classes=2, **kwargs
    )


def test_factory_passes_the_plan_rate_through():
    model = _make(dropout=0.4)
    assert isinstance(model.drop, nn.Dropout)
    assert model.drop.p == 0.4


def test_factory_default_reproduces_the_original_quarter():
    """No override -> exactly the previous hardcoded behavior (0.25)."""
    model = _make()
    assert isinstance(model.drop, nn.Dropout)
    assert model.drop.p == 0.25


def test_zero_rate_disables_dropout():
    model = _make(dropout=0.0)
    assert isinstance(model.drop, nn.Identity)


def test_legacy_bool_contract_is_preserved():
    assert SimpleMIL(dropout=True).drop.p == 0.25
    assert isinstance(SimpleMIL(dropout=False).drop, nn.Identity)
