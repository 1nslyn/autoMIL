"""Coverage for @register decorator + resolvers + singleton isolation (REG-02 / D-27 / D-35)."""
from __future__ import annotations

import multiprocessing as mp
import sys

import pytest


def _spec_kwargs(**overrides):
    base = {
        "name": "clam_mb_v0176",
        "kind": "model",
        "parent": "clam_mb",
        "base_commit": "abc1234",
        "primary_value": 0.8074,
        "node_id": "node_0176",
        "created_at": "2026-05-02T10:00:00Z",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Clear registry singletons before each test to prevent cross-test pollution.

    PATTERNS.md §"Open codebase questions" #3: module-level dicts persist across
    test functions; this fixture is the canonical isolation pattern (cited in
    D-47 and Plan 01-01's `discretion` block).
    """
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


# --- register happy path ---

def test_register_model_variant_happy_path():
    from automil.registry import register, VariantSpec, ModelVariant
    from automil.registry._state import MODEL_VARIANTS

    @register(VariantSpec(**_spec_kwargs()))
    class ClamMbV0176(ModelVariant):
        def forward(self, features, coords=None):
            return None

    assert MODEL_VARIANTS[("clam_mb", "clam_mb_v0176")] is ClamMbV0176


def test_register_loss_variant_happy_path():
    from automil.registry import register, VariantSpec, LossVariant
    from automil.registry._state import LOSS_VARIANTS

    @register(VariantSpec(**_spec_kwargs(name="ce_smooth008", kind="loss", parent=None)))
    class CeSmooth008(LossVariant):
        def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
            return 0.0

    assert LOSS_VARIANTS["ce_smooth008"] is CeSmooth008


def test_register_policy_variant_happy_path():
    from automil.registry import register, VariantSpec, PolicyVariant
    from automil.registry._state import POLICY_VARIANTS

    @register(VariantSpec(**_spec_kwargs(name="sam_lookahead", kind="policy", parent=None)))
    class SamLookahead(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    assert POLICY_VARIANTS["sam_lookahead"] is SamLookahead


# --- error semantics ---

def test_duplicate_model_name_hard_fails():
    from automil.registry import register, VariantSpec, ModelVariant, RegistrationError

    @register(VariantSpec(**_spec_kwargs()))
    class First(ModelVariant):
        def forward(self, features, coords=None): return None

    with pytest.raises(RegistrationError, match=r"already registered|duplicate"):
        @register(VariantSpec(**_spec_kwargs()))
        class Second(ModelVariant):
            def forward(self, features, coords=None): return None


def test_register_message_suggests_resolution():
    from automil.registry import register, VariantSpec, ModelVariant, RegistrationError

    @register(VariantSpec(**_spec_kwargs()))
    class First(ModelVariant):
        def forward(self, features, coords=None): return None

    with pytest.raises(RegistrationError) as exc_info:
        @register(VariantSpec(**_spec_kwargs()))
        class Second(ModelVariant):
            def forward(self, features, coords=None): return None
    # Operator-friendly: the error names both classes AND suggests how to fix.
    msg = str(exc_info.value)
    assert "clam_mb_v0176" in msg
    assert "rename" in msg.lower() or "--name" in msg or "port-variant" in msg


def test_model_kind_without_parent_rejected():
    from automil.registry import register, VariantSpec, ModelVariant, RegistrationError
    with pytest.raises(RegistrationError, match=r"parent"):
        @register(VariantSpec(**_spec_kwargs(parent=None)))
        class Bad(ModelVariant):
            def forward(self, features, coords=None): return None


def test_loss_kind_with_parent_rejected():
    from automil.registry import register, VariantSpec, LossVariant, RegistrationError
    with pytest.raises(RegistrationError, match=r"parent"):
        @register(VariantSpec(**_spec_kwargs(name="x", kind="loss", parent="clam_mb")))
        class Bad(LossVariant):
            def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
                return 0.0


def test_policy_kind_with_parent_rejected():
    from automil.registry import register, VariantSpec, PolicyVariant, RegistrationError
    with pytest.raises(RegistrationError, match=r"parent"):
        @register(VariantSpec(**_spec_kwargs(name="x", kind="policy", parent="clam_mb")))
        class Bad(PolicyVariant):
            def wrap_optimizer(self, opt): return opt


def test_kind_class_mismatch_rejected():
    from automil.registry import register, VariantSpec, LossVariant, RegistrationError
    with pytest.raises(RegistrationError, match=r"ModelVariant|kind"):
        @register(VariantSpec(**_spec_kwargs()))  # kind="model" but class is LossVariant
        class WrongKind(LossVariant):
            def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
                return 0.0


def test_decorator_returns_class_unchanged():
    from automil.registry import register, VariantSpec, ModelVariant

    @register(VariantSpec(**_spec_kwargs()))
    class ClamMbV0176(ModelVariant):
        def forward(self, features, coords=None): return None

    assert ClamMbV0176.__name__ == "ClamMbV0176"
    # Calling resolve returns the same class object identity.
    from automil.registry import resolve_model
    assert resolve_model("clam_mb", "clam_mb_v0176") is ClamMbV0176


# --- resolvers ---

def test_resolve_model_missing_lists_available():
    from automil.registry import register, VariantSpec, ModelVariant, resolve_model

    @register(VariantSpec(**_spec_kwargs()))
    class A(ModelVariant):
        def forward(self, features, coords=None): return None

    with pytest.raises(KeyError) as exc_info:
        resolve_model("clam_mb", "does_not_exist")
    msg = str(exc_info.value)
    assert "clam_mb_v0176" in msg
    assert "available" in msg.lower()


def test_resolve_loss_missing_lists_available():
    from automil.registry import register, VariantSpec, LossVariant, resolve_loss

    @register(VariantSpec(**_spec_kwargs(name="ce_smooth008", kind="loss", parent=None)))
    class A(LossVariant):
        def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
            return 0.0

    with pytest.raises(KeyError) as exc_info:
        resolve_loss("does_not_exist")
    msg = str(exc_info.value)
    assert "ce_smooth008" in msg


def test_resolve_policy_missing_lists_available():
    from automil.registry import register, VariantSpec, PolicyVariant, resolve_policy

    @register(VariantSpec(**_spec_kwargs(name="sam_lookahead", kind="policy", parent=None)))
    class A(PolicyVariant):
        def wrap_optimizer(self, opt): return opt

    with pytest.raises(KeyError) as exc_info:
        resolve_policy("does_not_exist")
    msg = str(exc_info.value)
    assert "sam_lookahead" in msg


def test_clear_registry_empties_all_dicts():
    from automil.registry import register, VariantSpec, ModelVariant, LossVariant, PolicyVariant
    from automil.registry._state import (
        MODEL_VARIANTS, LOSS_VARIANTS, POLICY_VARIANTS, SPEC_STORE,
        _clear_registry,
    )

    @register(VariantSpec(**_spec_kwargs()))
    class M(ModelVariant):
        def forward(self, features, coords=None): return None

    @register(VariantSpec(**_spec_kwargs(name="ce_smooth008", kind="loss", parent=None)))
    class L(LossVariant):
        def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
            return 0.0

    @register(VariantSpec(**_spec_kwargs(name="sam_lookahead", kind="policy", parent=None)))
    class P(PolicyVariant):
        def wrap_optimizer(self, opt): return opt

    assert MODEL_VARIANTS and LOSS_VARIANTS and POLICY_VARIANTS and SPEC_STORE
    _clear_registry()
    assert MODEL_VARIANTS == {}
    assert LOSS_VARIANTS == {}
    assert POLICY_VARIANTS == {}
    assert SPEC_STORE == {}


def test_spec_store_populated_on_register():
    from automil.registry import register, VariantSpec, ModelVariant
    from automil.registry._state import SPEC_STORE

    spec = VariantSpec(**_spec_kwargs())

    @register(spec)
    class M(ModelVariant):
        def forward(self, features, coords=None): return None

    assert SPEC_STORE[("model", "clam_mb", "clam_mb_v0176")] == spec


# --- isolation between tests via autouse fixture ---

def test_isolation_first_registers_clam_mb_v0176(_isolated_registry=None):
    from automil.registry import register, VariantSpec, ModelVariant
    from automil.registry._state import MODEL_VARIANTS

    @register(VariantSpec(**_spec_kwargs()))
    class M(ModelVariant):
        def forward(self, features, coords=None): return None

    assert ("clam_mb", "clam_mb_v0176") in MODEL_VARIANTS


def test_isolation_second_registers_clam_mb_v0176_again():
    """If the autouse fixture works, this test passes — the previous test's
    registration was cleared.
    """
    from automil.registry import register, VariantSpec, ModelVariant
    from automil.registry._state import MODEL_VARIANTS

    @register(VariantSpec(**_spec_kwargs()))
    class M(ModelVariant):
        def forward(self, features, coords=None): return None

    assert ("clam_mb", "clam_mb_v0176") in MODEL_VARIANTS
    # And the previous test's class object is GONE — only this test's M remains.
    assert MODEL_VARIANTS[("clam_mb", "clam_mb_v0176")].__qualname__.endswith("M")


# --- fork safety smoke (D-27 specifics) ---

def _child_register_and_check():
    """Run inside a forked child. Returns True if child's registry has the variant."""
    from automil.registry import register, VariantSpec, ModelVariant
    from automil.registry._state import MODEL_VARIANTS

    @register(VariantSpec(**_spec_kwargs()))
    class M(ModelVariant):
        def forward(self, features, coords=None): return None

    return ("clam_mb", "clam_mb_v0176") in MODEL_VARIANTS


@pytest.mark.skipif(sys.platform == "win32", reason="fork() not available on Windows")
def test_fork_safe_child_repopulates_registry():
    """D-27 specifics: orchestrator forks worker processes; each worker's
    copy of the singleton dicts is independent. The child registers and
    sees its own entry.
    """
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=1) as pool:
        result = pool.apply(_child_register_and_check)
    assert result is True
