"""Coverage for Manifest write/read/cross-check (REG-08 manifest format / D-44)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

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
        "mutations": ("ce_smooth=0.008", "sam_lookahead"),
    }
    base.update(overrides)
    return base


def _manifest_kwargs(**overrides):
    from automil.registry.spec import VariantSpec
    base = {
        "spec": VariantSpec(**_spec_kwargs()),
        "source_node": "node_0176",
        "source_overlay_files": ("benchmarks/lib/CLAM/models/model_clam.py",),
        "ported_at": "2026-05-02T10:00:00Z",
        "tool_version": "automil 0.1.0",
    }
    base.update(overrides)
    return base


def test_write_read_roundtrip(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    path = tmp_path / "clam_mb_v0176.json"
    m.write(path)
    loaded = Manifest.read(path)
    assert loaded.spec == m.spec
    assert loaded.source_node == m.source_node
    assert loaded.source_overlay_files == m.source_overlay_files
    assert loaded.ported_at == m.ported_at
    assert loaded.tool_version == m.tool_version


def test_write_atomic(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    path = tmp_path / "clam_mb_v0176.json"
    m.write(path)
    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []
    # File parseable.
    json.loads(path.read_text())


def test_read_missing_file(tmp_path):
    from automil.registry.manifest import Manifest
    with pytest.raises(FileNotFoundError):
        Manifest.read(tmp_path / "nope.json")


def test_read_malformed_json(tmp_path):
    from automil.registry.manifest import Manifest
    path = tmp_path / "bad.json"
    path.write_text("{this is not json")
    with pytest.raises(ValueError, match=r"manifest|JSON"):
        Manifest.read(path)


def test_read_missing_required_key(tmp_path):
    from automil.registry.manifest import Manifest
    path = tmp_path / "bad.json"
    # Missing "ported_at" key.
    path.write_text(json.dumps({
        "spec": {"name": "x", "kind": "model", "parent": "p",
                 "base_commit": "abc", "primary_value": 0.5, "node_id": "n",
                 "created_at": "2026-05-02"},
        "source_node": "n",
        "source_overlay_files": [],
        "tool_version": "x",
    }))
    with pytest.raises(ValueError, match=r"ported_at|missing"):
        Manifest.read(path)


def test_cross_check_happy(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    module_path = tmp_path / "clam_mb_v0176.py"
    module_path.write_text(
        'clam_mb_v0176 variant.\n\n'
        'Parent: clam_mb\n'
        'Base commit: abc1234\n'
        'Primary_value: 0.8074\n'
        'Node ID: node_0176\n'
        'Mutations: ce_smooth=0.008, sam_lookahead\n'
        '"""\n'
    )
    # Write as proper triple-quoted docstring via module source
    module_path.write_text(
        '"""clam_mb_v0176 variant.\n\n'
        'Parent: clam_mb\n'
        'Base commit: abc1234\n'
        'Primary_value: 0.8074\n'
        'Node ID: node_0176\n'
        'Mutations: ce_smooth=0.008, sam_lookahead\n'
        '"""\n'
    )
    ok, reason = m.cross_check_with_module(module_path)
    assert ok, f"cross_check failed: {reason}"


def test_cross_check_name_mismatch(tmp_path):
    from automil.registry.manifest import Manifest
    from automil.registry.spec import VariantSpec
    spec = VariantSpec(**_spec_kwargs(name="v0177"))
    m2 = Manifest(**_manifest_kwargs(spec=spec))

    module_path = tmp_path / "v0177.py"
    # docstring says name "v9999" -- wrong.
    module_path.write_text(
        '"""v9999 variant.\n\n'
        'Parent: clam_mb\n'
        'Base commit: abc1234\n'
        'Primary_value: 0.8074\n'
        'Node ID: node_0176\n'
        'Mutations: \n'
        '"""\n'
    )
    ok, reason = m2.cross_check_with_module(module_path)
    assert not ok
    assert "v9999" in reason or "name" in reason.lower()


def test_cross_check_no_docstring(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    module_path = tmp_path / "clam_mb_v0176.py"
    module_path.write_text("# no docstring\nx = 1\n")
    ok, reason = m.cross_check_with_module(module_path)
    assert not ok
    assert "docstring" in reason.lower()


def test_cross_check_parent_mismatch(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    module_path = tmp_path / "clam_mb_v0176.py"
    module_path.write_text(
        '"""clam_mb_v0176 variant.\n\n'
        'Parent: ab_mil\n'  # WRONG
        'Base commit: abc1234\n'
        'Primary_value: 0.8074\n'
        'Node ID: node_0176\n'
        'Mutations: \n'
        '"""\n'
    )
    ok, reason = m.cross_check_with_module(module_path)
    assert not ok
    assert "parent" in reason.lower()


def test_cross_check_primary_value_mismatch(tmp_path):
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    module_path = tmp_path / "clam_mb_v0176.py"
    module_path.write_text(
        '"""clam_mb_v0176 variant.\n\n'
        'Parent: clam_mb\n'
        'Base commit: abc1234\n'
        'Primary_value: 0.81\n'  # WRONG (manifest says 0.8074)
        'Node ID: node_0176\n'
        'Mutations: \n'
        '"""\n'
    )
    ok, reason = m.cross_check_with_module(module_path)
    assert not ok
    assert "primary_value" in reason.lower()


def test_manifest_is_frozen():
    from automil.registry.manifest import Manifest
    m = Manifest(**_manifest_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.source_node = "other"  # type: ignore[misc]


def test_manifest_kwargs_helper_excludes_unknown_keys():
    """Helper sanity: extra keys passed to _manifest_kwargs are silently ignored by the helper."""
    kwargs = _manifest_kwargs(extra_ignored_key="x")
    # The extra key is still present in the dict (the helper uses dict.update),
    # so confirm the Manifest constructor would reject it.
    assert "extra_ignored_key" in kwargs
    # This test validates that the helper doesn't silently swallow kwarg typos
    # that WOULD break Manifest() construction; callers must use known keys only.
    # (This is a meta-sanity for the helper itself, not a Manifest test.)
    from automil.registry.manifest import Manifest
    with pytest.raises(TypeError):
        Manifest(**kwargs)


class TestLegacyCompositeLabelFailsClosed:
    """A pre-rename variant doc carrying `Composite:` must FAIL the
    cross-check, not silently skip the D-44 exact-float comparison."""

    def test_legacy_label_is_rejected(self, tmp_path):
        from automil.registry.manifest import Manifest
        m = Manifest(**_manifest_kwargs())
        module_path = tmp_path / "clam_mb_v0176.py"
        module_path.write_text(
            '"""clam_mb_v0176 variant.\n\n'
            'Parent: clam_mb\n'
            'Base commit: abc1234\n'
            'Composite: 0.8074\n'
            'Node ID: node_0176\n'
            '"""\n'
        )
        ok, reason = m.cross_check_with_module(module_path)
        assert ok is False
        assert "retired 'Composite:' label" in reason

    def test_writer_emits_the_pinned_label(self):
        """The generated variant module must carry `Primary_value:` so the
        parser's exact-float cross-check can never become a vacuous pass."""
        import inspect
        from automil.cli.lifecycle import port_variant
        source = inspect.getsource(port_variant._write_variant_module)
        assert "Primary_value:" in source
