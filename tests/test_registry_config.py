"""Coverage for RegistryConfig reader (REG-04 / REG-06 / D-31 / D-33 / D-39)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


def _write_config(tmp_path: Path, body: str) -> Path:
    """Create automil/config.yaml under tmp_path/automil/."""
    adir = tmp_path / "automil"
    adir.mkdir(exist_ok=True)
    (adir / "config.yaml").write_text(body)
    return adir


def test_empty_config_returns_defaults(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "")
    cfg = load_registry_config(adir)
    assert cfg.protected == ()
    assert cfg.mode == "free"
    assert cfg.repro_tolerance == pytest.approx(0.005)
    assert cfg.identity_constraints == ()


def test_empty_registry_section_returns_defaults(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry: {}\n")
    cfg = load_registry_config(adir)
    assert cfg.protected == ()
    assert cfg.mode == "free"
    assert cfg.repro_tolerance == pytest.approx(0.005)


def test_protected_list_returns_tuple(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  protected:\n    - 'a/**'\n    - 'b/foo.py'\n")
    cfg = load_registry_config(adir)
    assert cfg.protected == ("a/**", "b/foo.py")
    assert isinstance(cfg.protected, tuple)


def test_mode_architecture_preserving_accepted(tmp_path):
    """H-4: the mode now REQUIRES a non-empty protected list.

    It used to be accepted bare — and read by no code at all. Since submit's gate
    is written `if reg_cfg.protected and ...`, an empty list short-circuits the
    whole enforcement path to a no-op, so a project could declare that it
    preserves a substrate and get nothing. See
    tests/test_frozen_substrate_enforcement.py.
    """
    from automil.registry.config import load_registry_config
    adir = _write_config(
        tmp_path,
        "registry:\n  mode: architecture-preserving\n  protected:\n    - \"splits.py\"\n",
    )
    cfg = load_registry_config(adir)
    assert cfg.mode == "architecture-preserving"
    assert cfg.protected == ("splits.py",)


def test_mode_architecture_preserving_without_protected_is_rejected(tmp_path):
    import pytest
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  mode: architecture-preserving\n")
    with pytest.raises(ValueError, match="protected"):
        load_registry_config(adir)


def test_mode_free_explicitly(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  mode: free\n")
    cfg = load_registry_config(adir)
    assert cfg.mode == "free"


def test_mode_unknown_value_rejected(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  mode: evil\n")
    with pytest.raises(ValueError, match=r"registry\.mode|free|architecture-preserving"):
        load_registry_config(adir)


def test_repro_tolerance_custom_value(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  repro_tolerance: 0.01\n")
    cfg = load_registry_config(adir)
    assert cfg.repro_tolerance == pytest.approx(0.01)


def test_protected_wrong_type_rejected(tmp_path):
    from automil.registry.config import load_registry_config
    adir = _write_config(tmp_path, "registry:\n  protected: 42\n")
    with pytest.raises(TypeError, match=r"registry\.protected"):
        load_registry_config(adir)


def test_config_yaml_missing_returns_defaults(tmp_path):
    """Useful for fresh init flow — automil_dir exists but config.yaml not yet written."""
    from automil.registry.config import load_registry_config
    adir = tmp_path / "automil"
    adir.mkdir()
    cfg = load_registry_config(adir)
    assert cfg.protected == ()
    assert cfg.mode == "free"


def test_registry_config_is_frozen(tmp_path):
    from automil.registry.config import load_registry_config, RegistryConfig
    adir = _write_config(tmp_path, "")
    cfg = load_registry_config(adir)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.mode = "architecture-preserving"  # type: ignore[misc]


def test_no_autobench_defaults_d49():
    """D-49: framework ships zero protected-file defaults; the consumer
    config.yaml is the source of truth. Grep-guard against autobench paths
    creeping into the framework default."""
    from automil.registry.config import RegistryConfig
    cfg = RegistryConfig()
    assert "benchmarks" not in str(cfg.protected)
    assert "AUTOBENCH" not in str(cfg.protected)
    assert "ccrcc" not in str(cfg.protected)
