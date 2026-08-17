"""Coverage for `automil check` registry extension (REG-04 / REG-05 / D-40 / D-46)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


GOOD_VARIANT = '''"""v0001 variant.

Parent: clam_mb
Base commit: abc1234
Primary_value: 0.5
Node ID: node_0001
Mutations:
"""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="v0001", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_0001",
    created_at="2026-05-02T10:00:00Z",
))
class V0001(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

BAD_IMPORT_VARIANT = '''"""bad variant."""
import _automil_definitely_does_not_exist  # noqa
'''

DOCSTRING_MISMATCH_VARIANT = '''"""v_mismatch variant.

Parent: clam_mb
Base commit: abc1234
Primary_value: 0.81
Node ID: node_0001
Mutations:
"""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="v_mismatch", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_0001",
    created_at="2026-05-02T10:00:00Z",
))
class VMismatch(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup(tmp_path: Path, protected: list[str] | None = None) -> Path:
    _init_git_repo(tmp_path)
    from automil.cli import main
    import os
    os.chdir(tmp_path)
    CliRunner().invoke(main, ["init"])
    if protected is not None:
        adir = tmp_path / "automil"
        cfg = yaml.safe_load((adir / "config.yaml").read_text())
        cfg.setdefault("registry", {})["protected"] = protected
        (adir / "config.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "automil"


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


# --- protected files ---

def test_protected_clean_no_issue(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path, protected=["src/foo.py"])
    monkeypatch.chdir(tmp_path)
    # File doesn't exist OR is clean.
    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "registry.protected paths dirty" not in result.output


def test_protected_dirty_issue_raised(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path, protected=["src/foo.py"])
    monkeypatch.chdir(tmp_path)
    # Create + commit the file.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text("# v1\n")
    subprocess.run(["git", "add", "src/foo.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add foo"], cwd=tmp_path, check=True, capture_output=True)
    # Now modify it (dirty).
    (tmp_path / "src" / "foo.py").write_text("# v2\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "registry.protected paths dirty" in result.output
    assert "src/foo.py" in result.output
    assert "revert-baseline" in result.output


def test_protected_staged_also_fails_d34(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path, protected=["src/foo.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text("# v1\n")
    subprocess.run(["git", "add", "src/foo.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add"], cwd=tmp_path, check=True, capture_output=True)
    # Modify and stage (no commit).
    (tmp_path / "src" / "foo.py").write_text("# v2\n")
    subprocess.run(["git", "add", "src/foo.py"], cwd=tmp_path, check=True)

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "registry.protected paths dirty" in result.output


# --- registry consistency ---

def test_registry_consistency_happy(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Place a clean variant + manifest.
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(GOOD_VARIANT)
    manifest_data = {
        "spec": {
            "name": "v0001", "kind": "model", "parent": "clam_mb",
            "base_commit": "abc1234", "primary_value": 0.5, "node_id": "node_0001",
            "created_at": "2026-05-02T10:00:00Z", "mutations": [],
        },
        "source_node": "node_0001",
        "source_overlay_files": [],
        "ported_at": "2026-05-02T10:00:00Z",
        "tool_version": "automil 0.1.0",
    }
    (v_dir / "v0001.json").write_text(json.dumps(manifest_data, indent=2))

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "failed import" not in result.output
    assert "mismatches docstring" not in result.output


def test_registry_variant_missing_manifest_warns(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(GOOD_VARIANT)
    # NO sibling .json.

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "no sibling manifest" in result.output or "v0001.json" in result.output
    # WARNING, not ISSUE — output should show WARNINGS section.


def test_registry_manifest_mismatch_issue(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v_mismatch.py").write_text(DOCSTRING_MISMATCH_VARIANT)
    manifest_data = {
        "spec": {
            "name": "v_mismatch", "kind": "model", "parent": "clam_mb",
            "base_commit": "abc1234", "primary_value": 0.5, "node_id": "node_0001",
            "created_at": "2026-05-02T10:00:00Z", "mutations": [],
        },
        "source_node": "node_0001",
        "source_overlay_files": [],
        "ported_at": "2026-05-02T10:00:00Z",
        "tool_version": "automil 0.1.0",
    }
    (v_dir / "v_mismatch.json").write_text(json.dumps(manifest_data, indent=2))

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    # docstring primary_value=0.81, manifest=0.5 -> mismatch.
    assert "mismatches docstring" in result.output


def test_registry_failed_import_issue(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(BAD_IMPORT_VARIANT)

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "failed import" in result.output


# --- repro manifest ---

def test_repro_manifest_missing_warns(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "repro_manifest.yaml" in result.output
    assert "verify-repro" in result.output
    # Warning, not issue.


def test_repro_manifest_stale_warns(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Write a repro_manifest first (older).
    (adir / "repro_manifest.yaml").write_text("status: pass\n")
    time.sleep(0.05)
    # Then add a variant module that's newer.
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(GOOD_VARIANT)

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "older than" in result.output


def test_repro_manifest_current_no_warning(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(GOOD_VARIANT)
    time.sleep(0.05)
    (adir / "repro_manifest.yaml").write_text("status: pass\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "older than" not in result.output


# --- regression guards ---

def test_phase0_check_outputs_preserved(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    # Phase 0 output: nvidia-smi line + env whitelist visible.
    assert "nvidia-smi" in result.output.lower()
    assert "env whitelist" in result.output.lower() or "env.passthrough" in result.output.lower()


def test_protected_dirty_includes_suggestion(tmp_path, cli_runner, monkeypatch):
    """Good error message: names the protected path + suggests fix."""
    adir = _setup(tmp_path, protected=["src/bar.py"])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "bar.py").write_text("# v1\n")
    subprocess.run(["git", "add", "src/bar.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "add bar"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src" / "bar.py").write_text("# v2 dirty\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["check"])
    assert "registry.protected paths dirty" in result.output
    assert "src/bar.py" in result.output
    assert "revert-baseline" in result.output
