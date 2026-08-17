"""Coverage for scanner + regenerate_init_py (REG-02 scan, D-29)."""
from __future__ import annotations

from pathlib import Path

import pytest


REGISTER_MODEL_TEMPLATE = '''"""{name} variant.

Parent: {parent}
Base commit: abc1234
Primary_value: 0.5
Node ID: node_{nodenum}
Mutations:
"""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="{name}", kind="model", parent="{parent}",
    base_commit="abc1234", primary_value=0.5, node_id="node_{nodenum}",
    created_at="2026-05-02T10:00:00Z",
))
class {classname}(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

REGISTER_LOSS_TEMPLATE = '''"""{name} loss."""
from automil.registry import register, VariantSpec, LossVariant


@register(VariantSpec(
    name="{name}", kind="loss", parent=None,
    base_commit="abc1234", primary_value=0.5, node_id="node_{nodenum}",
    created_at="2026-05-02T10:00:00Z",
))
class {classname}(LossVariant):
    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
        return 0.0
'''

REGISTER_POLICY_TEMPLATE = '''"""{name} policy."""
from automil.registry import register, VariantSpec, PolicyVariant


@register(VariantSpec(
    name="{name}", kind="policy", parent=None,
    base_commit="abc1234", primary_value=0.5, node_id="node_{nodenum}",
    created_at="2026-05-02T10:00:00Z",
))
class {classname}(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
'''

UNIMPORTABLE_MODULE = '''"""Bad import."""
import _automil_definitely_does_not_exist
'''


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


def _make_variants_root(tmp_path: Path) -> Path:
    root = tmp_path / "variants"
    root.mkdir()
    (root / "_losses").mkdir()
    (root / "_policies").mkdir()
    (root / "_candidates").mkdir()
    return root


def test_scan_finds_model_variant(tmp_path):
    from automil.registry.scanner import scan_variants
    from automil.registry._state import MODEL_VARIANTS

    root = _make_variants_root(tmp_path)
    (root / "clam_mb").mkdir()
    (root / "clam_mb" / "v0176.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="v0176", parent="clam_mb", nodenum="0176", classname="V0176",
        )
    )

    result = scan_variants(root)
    assert ("clam_mb", "v0176") in MODEL_VARIANTS
    assert any(p.name == "v0176.py" for p in result.imported)
    assert result.failed == ()


def test_scan_finds_loss_variant(tmp_path):
    from automil.registry.scanner import scan_variants
    from automil.registry._state import LOSS_VARIANTS

    root = _make_variants_root(tmp_path)
    (root / "_losses" / "ce_smooth.py").write_text(
        REGISTER_LOSS_TEMPLATE.format(
            name="ce_smooth", nodenum="0042", classname="CeSmooth",
        )
    )

    scan_variants(root)
    assert "ce_smooth" in LOSS_VARIANTS


def test_scan_finds_policy_variant(tmp_path):
    from automil.registry.scanner import scan_variants
    from automil.registry._state import POLICY_VARIANTS

    root = _make_variants_root(tmp_path)
    (root / "_policies" / "sam.py").write_text(
        REGISTER_POLICY_TEMPLATE.format(
            name="sam", nodenum="0050", classname="Sam",
        )
    )

    scan_variants(root)
    assert "sam" in POLICY_VARIANTS


def test_scan_finds_candidate_variant(tmp_path):
    """D-25 _candidates/ -- Phase 5 GTE will populate; scanner doesn't skip it."""
    from automil.registry.scanner import scan_variants
    from automil.registry._state import MODEL_VARIANTS

    root = _make_variants_root(tmp_path)
    (root / "_candidates" / "candidate_001.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="candidate_001", parent="some_parent", nodenum="0099", classname="Cand",
        )
    )

    scan_variants(root)
    # The variant registers -- _candidates is just a directory the scanner walks.
    assert ("some_parent", "candidate_001") in MODEL_VARIANTS


def test_scan_skips_init_py(tmp_path):
    from automil.registry.scanner import scan_variants

    root = _make_variants_root(tmp_path)
    (root / "clam_mb").mkdir()
    (root / "clam_mb" / "__init__.py").write_text("# generated\n")
    (root / "clam_mb" / "v0001.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="v0001", parent="clam_mb", nodenum="0001", classname="V0001",
        )
    )

    result = scan_variants(root)
    assert any(p.name == "__init__.py" for p in result.skipped)
    assert any(p.name == "v0001.py" for p in result.imported)


def test_scan_skips_private_helpers(tmp_path):
    from automil.registry.scanner import scan_variants

    root = _make_variants_root(tmp_path)
    (root / "clam_mb").mkdir()
    (root / "clam_mb" / "_helper.py").write_text("# helper\n")
    (root / "clam_mb" / "v0001.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="v0001", parent="clam_mb", nodenum="0001", classname="V0001",
        )
    )

    result = scan_variants(root)
    assert any(p.name == "_helper.py" for p in result.skipped)


def test_scan_failed_import_does_not_crash(tmp_path):
    from automil.registry.scanner import scan_variants
    from automil.registry._state import MODEL_VARIANTS

    root = _make_variants_root(tmp_path)
    (root / "clam_mb").mkdir()
    (root / "clam_mb" / "bad.py").write_text(UNIMPORTABLE_MODULE)
    (root / "clam_mb" / "good.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="good", parent="clam_mb", nodenum="0002", classname="Good",
        )
    )

    result = scan_variants(root)
    # Failed import recorded:
    assert any(p.name == "bad.py" for p, _ in result.failed)
    # Good module still imported:
    assert ("clam_mb", "good") in MODEL_VARIANTS


def test_scan_failed_import_duplicate_registration(tmp_path):
    from automil.registry.scanner import scan_variants

    root = _make_variants_root(tmp_path)
    (root / "clam_mb").mkdir()
    (root / "clam_mb" / "first.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="dup", parent="clam_mb", nodenum="0001", classname="First",
        )
    )
    (root / "clam_mb" / "second.py").write_text(
        REGISTER_MODEL_TEMPLATE.format(
            name="dup", parent="clam_mb", nodenum="0001", classname="Second",
        )
    )

    result = scan_variants(root)
    # Exactly one of them succeeded; the other is in failed.
    assert len(result.imported) == 1
    assert len(result.failed) == 1


def test_scan_missing_variants_root_returns_empty(tmp_path):
    from automil.registry.scanner import scan_variants
    result = scan_variants(tmp_path / "missing_dir")
    assert result.imported == ()
    assert result.failed == ()


def test_regenerate_init_py_alphabetic_order(tmp_path):
    from automil.registry.scanner import regenerate_init_py

    kind_dir = tmp_path / "clam_mb"
    kind_dir.mkdir()
    (kind_dir / "v_b.py").write_text("# b\n")
    (kind_dir / "v_a.py").write_text("# a\n")
    (kind_dir / "v_c.py").write_text("# c\n")

    rendered = regenerate_init_py(kind_dir)
    # The on-disk file matches.
    assert (kind_dir / "__init__.py").read_text() == rendered
    # Imports in alphabetic order.
    lines = [ln for ln in rendered.splitlines() if ln.startswith("from . import")]
    assert lines == [
        "from . import v_a  # noqa: F401",
        "from . import v_b  # noqa: F401",
        "from . import v_c  # noqa: F401",
    ]


def test_regenerate_init_py_idempotent_body(tmp_path):
    from automil.registry.scanner import regenerate_init_py

    kind_dir = tmp_path / "clam_mb"
    kind_dir.mkdir()
    (kind_dir / "v_a.py").write_text("# a\n")

    first = regenerate_init_py(kind_dir)
    second = regenerate_init_py(kind_dir)

    # Strip the timestamp comment (line starting with `# generated-at:`) and compare.
    def _strip_ts(s: str) -> str:
        return "\n".join(ln for ln in s.splitlines() if not ln.startswith("# generated-at:"))

    assert _strip_ts(first) == _strip_ts(second)


def test_regenerate_init_py_atomic_write(tmp_path):
    from automil.registry.scanner import regenerate_init_py

    kind_dir = tmp_path / "clam_mb"
    kind_dir.mkdir()
    (kind_dir / "v_a.py").write_text("# a\n")

    regenerate_init_py(kind_dir)
    # No .tmp leftover.
    leftover = list(kind_dir.glob("__init__.py*.tmp")) + list(kind_dir.glob("*.tmp"))
    assert leftover == []


def test_regenerate_init_py_empty_dir(tmp_path):
    from automil.registry.scanner import regenerate_init_py

    kind_dir = tmp_path / "_losses"
    kind_dir.mkdir()

    rendered = regenerate_init_py(kind_dir)
    # Header present, no `from . import` lines.
    assert "AUTO-GENERATED" in rendered
    assert "from . import" not in rendered


def test_regenerate_init_py_skips_init_and_private(tmp_path):
    from automil.registry.scanner import regenerate_init_py

    kind_dir = tmp_path / "clam_mb"
    kind_dir.mkdir()
    (kind_dir / "v_a.py").write_text("# a\n")
    (kind_dir / "__init__.py").write_text("# old\n")
    (kind_dir / "_helper.py").write_text("# helper\n")

    rendered = regenerate_init_py(kind_dir)
    assert "from . import v_a" in rendered
    assert "from . import __init__" not in rendered
    assert "from . import _helper" not in rendered
