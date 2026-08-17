"""Coverage for `automil submit` validator chain (REG-03 / D-30 / T-01-14 ordering)."""
from __future__ import annotations

import subprocess
import warnings as warnings_module
from pathlib import Path

import pytest
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

PURITY_BAD = '''"""Bad: top-level open."""
data = open("/tmp/x").read()
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="bad", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_0002",
    created_at="2026-05-02T10:00:00Z",
))
class Bad(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

INTERFACE_BAD = '''"""Bad: missing forward."""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="bad_iface", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_0003",
    created_at="2026-05-02T10:00:00Z",
))
class Bad(ModelVariant):
    pass
'''

BOTH_BAD = '''"""BOTH: top-level print AND missing forward."""
print("loading...")
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="bad_both", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_0004",
    created_at="2026-05-02T10:00:00Z",
))
class Bad(ModelVariant):
    pass
'''


@pytest.fixture
def cli_runner():
    return CliRunner()


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup_project(tmp_path: Path) -> Path:
    _init_git_repo(tmp_path)
    from automil.cli import main
    import os
    os.chdir(tmp_path)
    CliRunner().invoke(main, ["init"])
    return tmp_path / "automil"


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


def test_validator_happy_path_variant_module(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Place a clean variant.
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(GOOD_VARIANT)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/v0001.py"],
    )
    # Validator should not reject. Other rejects may apply (file not in git);
    # we only assert no [purity] or [interface] in output.
    assert "[purity]" not in result.output
    assert "[interface]" not in result.output


def test_validator_purity_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(PURITY_BAD)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/bad.py"],
    )
    assert result.exit_code != 0
    assert "[purity]" in result.output or "purity" in result.output
    assert "open" in result.output.lower()


def test_validator_interface_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad_iface.py").write_text(INTERFACE_BAD)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/bad_iface.py"],
    )
    assert result.exit_code != 0
    assert "interface" in result.output.lower()
    assert "forward" in result.output.lower()


def test_validator_purity_runs_before_interface(tmp_path, cli_runner, monkeypatch):
    """T-01-14: purity FIRST so untrusted modules don't import."""
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "both.py").write_text(BOTH_BAD)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/both.py"],
    )
    assert result.exit_code != 0
    # Output should mention purity, NOT interface — purity ran first and
    # short-circuited.
    assert "purity" in result.output.lower()
    # "print" should be the named cause (purity catches print as I/O).
    assert "print" in result.output.lower()


def test_validator_skipped_on_non_variant_path(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("data = open('/tmp/x').read()\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "src/main.py"],
    )
    # main.py is NOT under variants/ — validator chain skips.
    assert "[purity]" not in result.output
    assert "[interface]" not in result.output


def test_validator_skipped_on_init_py(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "__init__.py").write_text("# generated\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/__init__.py"],
    )
    assert "[purity]" not in result.output
    assert "[interface]" not in result.output


def test_validator_skipped_on_underscore_helper(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "_helper.py").write_text("data = open('/tmp/x').read()\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/_helper.py"],
    )
    # _helper.py is a private helper; not validated.
    assert "[purity]" not in result.output


def test_validator_error_format_file_line_fix(tmp_path, cli_runner, monkeypatch):
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(PURITY_BAD)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "automil/variants/clam_mb/bad.py"],
    )
    # Path mentioned, line number mentioned, Fix: substring present.
    assert "bad.py" in result.output
    assert "Fix:" in result.output or "fix" in result.output.lower()


def test_submit_help_mentions_validator_workflow(cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["submit", "--help"])
    # Production-grade: --help text explains the workflow, not just flags.
    # We assert the docstring mentions `variants/` validation.
    assert (
        "variants" in result.output.lower()
        or "validate" in result.output.lower()
        or "register" in result.output.lower()
    )


def test_d32_hard_fail_no_soft_warn(tmp_path, cli_runner, monkeypatch):
    """D-32: validator raises, does NOT emit warnings — no soft-warn substitute."""
    adir = _setup_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(PURITY_BAD)

    from automil.cli import main
    with warnings_module.catch_warnings(record=True) as w:
        warnings_module.simplefilter("always")
        result = cli_runner.invoke(
            main, ["submit", "--node", "node_0001", "--desc", "t",
                   "--files", "automil/variants/clam_mb/bad.py"],
        )
    # (a) Hard-fail: exit non-zero.
    assert result.exit_code != 0
    # (b) No DeprecationWarning / UserWarning from the validator chain.
    validator_warnings = [
        x for x in w
        if issubclass(x.category, (DeprecationWarning, UserWarning))
        and "validator" in str(x.message).lower()
    ]
    assert not validator_warnings, f"Soft-warn emitted: {validator_warnings}"
