"""Coverage for VariantSpec frozen dataclass and Kind type alias (REG-01 / D-22 / D-23)."""
from __future__ import annotations

import dataclasses
from typing import get_args

import pytest


def _spec_fields() -> dict:
    """Minimal-valid VariantSpec construction kwargs."""
    return {
        "name": "clam_mb_v0176",
        "kind": "model",
        "parent": "clam_mb",
        "base_commit": "abc1234",
        "primary_value": 0.8074,
        "node_id": "node_0176",
        "created_at": "2026-05-02T10:00:00Z",
    }


def test_construction_happy_path():
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_fields())
    assert spec.name == "clam_mb_v0176"
    assert spec.kind == "model"
    assert spec.parent == "clam_mb"
    assert spec.primary_value == pytest.approx(0.8074)


def test_frozen_mutation_refused():
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_fields())
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.primary_value = 0.99  # type: ignore[misc]


def test_mutations_tuple_immutable():
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_fields(), mutations=("ce_smooth=0.008",))
    assert isinstance(spec.mutations, tuple)
    with pytest.raises(AttributeError):
        spec.mutations.append("foo")  # type: ignore[attr-defined]


def test_mutations_default_empty():
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_fields())
    assert spec.mutations == ()


def test_loss_kind_allows_none_parent():
    from automil.registry.spec import VariantSpec
    kwargs = _spec_fields()
    kwargs.update(kind="loss", parent=None)
    spec = VariantSpec(**kwargs)
    assert spec.kind == "loss"
    assert spec.parent is None


def test_policy_kind_allows_none_parent():
    from automil.registry.spec import VariantSpec
    kwargs = _spec_fields()
    kwargs.update(kind="policy", parent=None)
    spec = VariantSpec(**kwargs)
    assert spec.kind == "policy"
    assert spec.parent is None


def test_kind_type_alias_is_literal_with_three_values():
    from automil.registry.spec import Kind
    # Literal["model", "loss", "policy"] surfaces via typing.get_args.
    assert set(get_args(Kind)) == {"model", "loss", "policy"}


def test_phase_1_kind_exhaustiveness_d23():
    """D-23: Phase 1 ships kind=model|loss|policy ONLY.

    Phase 5 / Phase 8 will widen with `recipe` / `inference`; when that
    happens this test fails-loud and reminds the planner to revisit
    downstream contract assumptions.
    """
    from automil.registry.spec import Kind
    kinds = set(get_args(Kind))
    assert "recipe" not in kinds, "D-23 violated: `recipe` is Phase 5+"
    assert "inference" not in kinds, "D-23 violated: `inference` is Phase 5+"


def test_structural_equality():
    from automil.registry.spec import VariantSpec
    a = VariantSpec(**_spec_fields())
    b = VariantSpec(**_spec_fields())
    assert a == b


def test_hashable_for_dict_key():
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_fields())
    d = {spec: "value"}  # frozen dataclass is hashable
    assert d[spec] == "value"
