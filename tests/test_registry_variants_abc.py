"""Coverage for the three sibling ABCs + import surface (REG-01 / D-21 / D-24)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_model_variant_abstract_cannot_instantiate():
    from automil.registry.variants.model import ModelVariant
    with pytest.raises(TypeError, match=r"abstract"):
        ModelVariant()  # type: ignore[abstract]


def test_loss_variant_abstract_cannot_instantiate():
    from automil.registry.variants.loss import LossVariant
    with pytest.raises(TypeError, match=r"abstract"):
        LossVariant()  # type: ignore[abstract]


def test_policy_variant_abstract_cannot_instantiate():
    from automil.registry.variants.policy import PolicyVariant
    with pytest.raises(TypeError, match=r"abstract"):
        PolicyVariant()  # type: ignore[abstract]


def test_subclass_without_forward_cannot_instantiate():
    from automil.registry.variants.model import ModelVariant

    class Bad(ModelVariant):
        pass

    with pytest.raises(TypeError, match=r"abstract"):
        Bad()  # type: ignore[abstract]


def test_concrete_model_variant_returns_parent_shape_d24():
    """D-24: forward returns whatever parent returns; framework does NOT
    impose AggregatorOutput in Phase 1.
    """
    from automil.registry.variants.model import ModelVariant

    class StubModel(ModelVariant):
        def forward(self, features, coords=None):
            # CLAM-shape: (logits, Y_prob, Y_hat, instance_dict)
            return (None, None, None, {})

    m = StubModel()
    out = m.forward(features=None)
    assert isinstance(out, tuple) and len(out) == 4


def test_instance_attention_default_returns_none():
    from automil.registry.variants.model import ModelVariant

    class StubModel(ModelVariant):
        def forward(self, features, coords=None):
            return None

    m = StubModel()
    assert m.instance_attention(features=None) is None


def test_concrete_loss_variant_callable():
    from automil.registry.variants.loss import LossVariant

    class StubLoss(LossVariant):
        def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
            return 0.42

    loss = StubLoss()
    assert loss(logits=None, targets=None) == pytest.approx(0.42)


def test_policy_wrap_scheduler_default_identity():
    from automil.registry.variants.policy import PolicyVariant

    class StubPolicy(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    sched = MagicMock()
    p = StubPolicy()
    assert p.wrap_scheduler(sched) is sched


def test_policy_step_default_delegates_to_opt():
    from automil.registry.variants.policy import PolicyVariant

    class StubPolicy(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    opt = MagicMock()
    p = StubPolicy()
    p.step(loss=None, opt=opt)
    opt.step.assert_called_once()


def test_policy_transform_bag_default_returns_the_bag_unchanged():
    from automil.registry.variants.policy import PolicyVariant

    class StubPolicy(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    bag = MagicMock()
    out = StubPolicy().transform_bag(bag, label=1, epoch=0, generator=MagicMock())
    assert out is bag
    assert bag.method_calls == []


def test_policy_before_validation_default_is_a_no_op():
    from automil.registry.variants.policy import PolicyVariant

    class StubPolicy(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    assert StubPolicy().before_validation(epoch=0) is None


def test_no_top_level_torch_import_in_model_py():
    """D-24 + TYPE_CHECKING guard: framework must not require torch at import time
    (the synthetic-consumer round-trip in Plan 01-12 is torch-free)."""
    from pathlib import Path
    # Use __file__-relative path to be robust against monkeypatch.chdir() in other tests.
    path = Path(__file__).parent.parent / "src" / "automil" / "registry" / "variants" / "model.py"
    content = path.read_text()
    # Grep gate: filter comments + TYPE_CHECKING block guards. Lines that begin
    # with 'import torch' or 'from torch ' at module scope (not inside a TYPE_CHECKING
    # block) would be a defect.
    non_comment_lines = [
        ln for ln in content.splitlines()
        if not ln.strip().startswith("#") and ln.strip()
    ]
    in_type_checking = False
    offending = []
    for ln in non_comment_lines:
        if ln.strip().startswith("if TYPE_CHECKING"):
            in_type_checking = True
            continue
        # crude: TYPE_CHECKING block ends when indent returns to col 0 with non-blank
        if in_type_checking and ln and not ln.startswith(" ") and not ln.startswith("\t"):
            in_type_checking = False
        if in_type_checking:
            continue
        if ln.startswith("import torch") or ln.startswith("from torch "):
            offending.append(ln)
    assert not offending, f"Top-level torch import found: {offending}"


def test_package_re_exports_abcs_and_spec():
    from automil.registry import ModelVariant, LossVariant, PolicyVariant, VariantSpec
    assert ModelVariant is not None
    assert LossVariant is not None
    assert PolicyVariant is not None
    assert VariantSpec is not None


def test_variants_subpackage_re_exports_abcs():
    from automil.registry.variants import ModelVariant, LossVariant, PolicyVariant
    assert all(
        cls is not None for cls in (ModelVariant, LossVariant, PolicyVariant)
    )
