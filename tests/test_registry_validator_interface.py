"""Coverage for InterfaceValidator (REG-03 / D-30 interface)."""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Inline module bodies.  Each test writes one to tmp_path/x.py.
# ---------------------------------------------------------------------------

HAPPY_MODEL = '''
"""Test model variant."""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="testmodel", kind="model", parent="testparent",
    base_commit="abc1234", composite=0.5, node_id="node_0001",
    created_at="2026-05-02T10:00:00Z",
))
class TestModel(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

HAPPY_LOSS = '''
"""Test loss variant."""
from automil.registry import register, VariantSpec, LossVariant


@register(VariantSpec(
    name="testloss", kind="loss", parent=None,
    base_commit="abc1234", composite=0.5, node_id="node_0002",
    created_at="2026-05-02T10:00:00Z",
))
class TestLoss(LossVariant):
    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
        return 0.0
'''

HAPPY_POLICY = '''
"""Test policy variant."""
from automil.registry import register, VariantSpec, PolicyVariant


@register(VariantSpec(
    name="testpolicy", kind="policy", parent=None,
    base_commit="abc1234", composite=0.5, node_id="node_0003",
    created_at="2026-05-02T10:00:00Z",
))
class TestPolicy(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def step(self, loss, opt):
        opt.step()
'''

MISSING_FORWARD = '''
"""Bad model variant — no forward."""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="bad", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0004",
    created_at="2026-05-02T10:00:00Z",
))
class Bad(ModelVariant):
    pass
'''

WRONG_SIGNATURE_EXTRA_ARG = '''
"""Model variant with extra positional arg."""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="extraarg", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0010",
    created_at="2026-05-02T10:00:00Z",
))
class ExtraArg(ModelVariant):
    def forward(self, features, coords, weights):
        return None
'''

TIGHTER_DEFAULT_MODEL = '''
"""Model variant with coords having no default (tightening ABC's optional).
This is a PERMISSIVE case — the validator should NOT reject it."""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="tightdefault", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0011",
    created_at="2026-05-02T10:00:00Z",
))
class TightDefault(ModelVariant):
    def forward(self, features, coords):
        return None
'''

KIND_MISMATCH = '''
"""LossVariant subclass with kind=model — wrong ABC."""
from automil.registry import register, VariantSpec, LossVariant


@register(VariantSpec(
    name="x", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0005",
    created_at="2026-05-02T10:00:00Z",
))
class Wrong(LossVariant):
    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
        return 0.0
'''

MULTIPLE_REGISTRATIONS = '''
"""Two @register-decorated classes — D-26 'single .py file per variant'."""
from automil.registry import register, VariantSpec, ModelVariant, LossVariant


@register(VariantSpec(
    name="m", kind="model", parent="p",
    base_commit="abc1234", composite=0.5, node_id="node_0006",
    created_at="2026-05-02T10:00:00Z",
))
class M(ModelVariant):
    def forward(self, features, coords=None): return None


@register(VariantSpec(
    name="l", kind="loss", parent=None,
    base_commit="abc1234", composite=0.5, node_id="node_0007",
    created_at="2026-05-02T10:00:00Z",
))
class L(LossVariant):
    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
        return 0.0
'''

NO_REGISTER_DECORATOR = '''
"""Module with no @register call — not a variant module."""
from automil.registry import ModelVariant


class Anonymous(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

SYNTAX_ERROR = '''
"""Bad syntax."""
def foo(:
    pass
'''


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


def _write_module(tmp_path: Path, body: str, name: str = "x.py") -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_happy_path_model_variant(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    path = _write_module(tmp_path, HAPPY_MODEL)
    InterfaceValidator().check(path)  # No exception.


def test_happy_path_loss_variant(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    path = _write_module(tmp_path, HAPPY_LOSS)
    InterfaceValidator().check(path)


def test_happy_path_policy_variant(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    path = _write_module(tmp_path, HAPPY_POLICY)
    InterfaceValidator().check(path)


def test_interface_validation_never_imports_candidate_dependencies(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator

    body = HAPPY_POLICY.replace(
        'from automil.registry import register, VariantSpec, PolicyVariant',
        'from automil.registry import register, VariantSpec, PolicyVariant\n'
        'import package_that_must_never_be_imported_by_validation',
    )
    path = _write_module(tmp_path, body)
    InterfaceValidator().check(path)


def test_validation_is_idempotent_and_preserves_live_registry(tmp_path):
    """Static validation must neither read nor mutate live registrations."""
    from automil.registry import PolicyVariant, VariantSpec, register
    from automil.registry._state import (
        LOSS_VARIANTS,
        MODEL_VARIANTS,
        POLICY_VARIANTS,
        SPEC_STORE,
    )
    from automil.registry.validators.interface import InterfaceValidator

    @register(VariantSpec(
        name="testpolicy", kind="policy", parent=None,
        base_commit="existing", composite=0.1, node_id="node_existing",
        created_at="2026-05-02T10:00:00Z",
    ))
    class ExistingPolicy(PolicyVariant):
        def wrap_optimizer(self, opt):
            return opt

    stores = (MODEL_VARIANTS, LOSS_VARIANTS, POLICY_VARIANTS, SPEC_STORE)
    before = tuple(dict(store) for store in stores)
    path = _write_module(tmp_path, HAPPY_POLICY)

    InterfaceValidator().check(path)
    InterfaceValidator().check(path)

    assert tuple(dict(store) for store in stores) == before
    assert POLICY_VARIANTS["testpolicy"] is ExistingPolicy


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_missing_forward_method_rejected(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, MISSING_FORWARD)
    with pytest.raises(ValidationError) as exc_info:
        InterfaceValidator().check(path)
    err = exc_info.value
    assert err.validator_name == "interface"
    assert "forward" in err.reason
    assert err.path == path
    # Operator-friendly: fix suggestion names what to add.
    assert "forward" in err.fix_suggestion
    assert "ModelVariant" in err.fix_suggestion or "ABC" in err.fix_suggestion


def test_wrong_signature_extra_positional_arg_rejected(tmp_path):
    """A model variant whose forward(self, features, coords, weights) has an
    extra positional arg beyond the ABC — rejected with naming the offending
    parameter or arg count mismatch."""
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, WRONG_SIGNATURE_EXTRA_ARG)
    with pytest.raises(ValidationError) as exc_info:
        InterfaceValidator().check(path)
    err = exc_info.value
    assert err.validator_name == "interface"
    # Reason should mention forward or signature/parameter issue
    assert "forward" in err.reason.lower() or "signature" in err.reason.lower() or "param" in err.reason.lower()


def test_tighter_default_is_permissive(tmp_path):
    """A model variant with forward(self, features, coords) — no default for
    coords — should be allowed; validator is permissive on default tightening."""
    from automil.registry.validators.interface import InterfaceValidator
    path = _write_module(tmp_path, TIGHTER_DEFAULT_MODEL)
    InterfaceValidator().check(path)  # Must NOT raise


def test_kind_abc_mismatch_rejected(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, KIND_MISMATCH)
    with pytest.raises(ValidationError) as exc_info:
        InterfaceValidator().check(path)
    err = exc_info.value
    assert err.validator_name == "interface"
    # Names BOTH the kind and the actual ABC for clarity.
    assert "model" in err.reason.lower()
    assert "LossVariant" in err.reason or "loss" in err.reason.lower()


def test_multiple_register_classes_rejected(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, MULTIPLE_REGISTRATIONS)
    with pytest.raises(ValidationError, match=r"single|one|D-26|multiple"):
        InterfaceValidator().check(path)


def test_no_register_decorator_rejected(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, NO_REGISTER_DECORATOR)
    with pytest.raises(ValidationError) as exc_info:
        InterfaceValidator().check(path)
    assert "register" in exc_info.value.reason.lower()


def test_syntax_error_rejected(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, SYNTAX_ERROR)
    with pytest.raises(ValidationError) as exc_info:
        InterfaceValidator().check(path)
    # Line number from SyntaxError surfaces in the ValidationError.
    assert exc_info.value.line is not None
    assert exc_info.value.line >= 1


def test_file_does_not_exist(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    with pytest.raises(ValidationError, match=r"not found|missing|exist"):
        InterfaceValidator().check(tmp_path / "nonexistent.py")


# ---------------------------------------------------------------------------
# Error format / meta
# ---------------------------------------------------------------------------

def test_validation_error_str_includes_path_and_line(tmp_path):
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    path = _write_module(tmp_path, MISSING_FORWARD)
    try:
        InterfaceValidator().check(path)
    except ValidationError as e:
        s = str(e)
        assert str(path) in s
        assert "interface" in s
        assert "forward" in s.lower()


def test_runtime_safe_module_path_resolution(tmp_path):
    """The validator must accept Path objects in deep nested directories
    (e.g., <consumer>/<dataset>/automil/variants/<parent>/testmodel.py).
    """
    from automil.registry.validators.interface import InterfaceValidator

    nested = tmp_path / "automil" / "variants" / "testparent"
    nested.mkdir(parents=True)
    path = nested / "testmodel.py"
    path.write_text(HAPPY_MODEL)
    InterfaceValidator().check(path)


def test_validator_short_circuits_on_first_failure(tmp_path):
    """Interface raises on the first finding; does NOT accumulate errors."""
    from automil.registry.validators.interface import InterfaceValidator
    from automil.registry.errors import ValidationError

    # A module with BOTH missing-forward AND multiple registrations.
    bad = MISSING_FORWARD + "\n\n" + HAPPY_LOSS  # two @register decorators
    path = _write_module(tmp_path, bad)
    with pytest.raises(ValidationError):
        InterfaceValidator().check(path)
