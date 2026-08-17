"""Coverage for `automil refresh-registry` (CLI-08 / D-29)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Fixture variant module templates
# ---------------------------------------------------------------------------

GOOD_MODEL = '''\
"""v_NN variant.

Parent: clam_mb
Base commit: abc1234
Primary_value: 0.5
Node ID: node_NN
Mutations:
"""
from automil.registry import register, VariantSpec, ModelVariant


@register(VariantSpec(
    name="v_NN", kind="model", parent="clam_mb",
    base_commit="abc1234", primary_value=0.5, node_id="node_NN",
    created_at="2026-05-02T10:00:00Z",
))
class V_NN(ModelVariant):
    def forward(self, features, coords=None):
        return None
'''

GOOD_LOSS = '''\
"""l_NN loss."""
from automil.registry import register, VariantSpec, LossVariant


@register(VariantSpec(
    name="l_NN", kind="loss", parent=None,
    base_commit="abc1234", primary_value=0.5, node_id="node_NN",
    created_at="2026-05-02T10:00:00Z",
))
class L_NN(LossVariant):
    def __call__(self, logits, targets, *, instance_logits=None, instance_labels=None):
        return 0.0
'''

BAD_IMPORT = '''\
"""Bad — raises import error."""
import _automil_definitely_does_not_exist
'''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup(tmp_path: Path) -> Path:
    """Setup git repo + automil init; returns automil/ dir."""
    _init_git_repo(tmp_path)
    import os
    os.chdir(tmp_path)
    from automil.cli import main
    CliRunner().invoke(main, ["init"])
    return tmp_path / "automil"


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Clear registry before and after each test to prevent cross-test pollution."""
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


# ---------------------------------------------------------------------------
# Test 1: happy refresh — single variant module, init.py gains import
# ---------------------------------------------------------------------------

def test_happy_refresh_single_variant(tmp_path, cli_runner, monkeypatch):
    """After refresh, clam_mb/__init__.py contains `from . import v0001`."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    body = (
        GOOD_MODEL.replace("v_NN", "v0001").replace("V_NN", "V0001")
        .replace("node_NN", "node_0001")
    )
    (v_dir / "v0001.py").write_text(body)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output

    init_text = (v_dir / "__init__.py").read_text()
    assert "from . import v0001" in init_text


# ---------------------------------------------------------------------------
# Test 2: empty kind dir — init.py contains only the auto-generated header
# ---------------------------------------------------------------------------

def test_empty_kind_dir(tmp_path, cli_runner, monkeypatch):
    """An empty kind dir gets an init.py with AUTO-GENERATED header but no imports."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output

    init_text = (adir / "variants" / "_losses" / "__init__.py").read_text()
    assert "AUTO-GENERATED" in init_text
    assert "from . import" not in init_text


# ---------------------------------------------------------------------------
# Test 3: idempotent re-run — body byte-identical modulo generated-at line
# ---------------------------------------------------------------------------

def test_idempotent_rerun(tmp_path, cli_runner, monkeypatch):
    """Two consecutive refresh-registry calls produce byte-identical __init__.py bodies."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    body = (
        GOOD_MODEL.replace("v_NN", "v0001").replace("V_NN", "V0001")
        .replace("node_NN", "node_0001")
    )
    (v_dir / "v0001.py").write_text(body)

    from automil.cli import main
    cli_runner.invoke(main, ["refresh-registry"])
    first = (v_dir / "__init__.py").read_text()
    # Clear registry so the second scan re-imports cleanly.
    from automil.registry._state import _clear_registry
    _clear_registry()
    cli_runner.invoke(main, ["refresh-registry"])
    second = (v_dir / "__init__.py").read_text()

    def _strip_ts(s: str) -> str:
        return "\n".join(ln for ln in s.splitlines() if not ln.startswith("# generated-at:"))

    assert _strip_ts(first) == _strip_ts(second)


# ---------------------------------------------------------------------------
# Test 4: failed import — default warns and continues (exit 0)
# ---------------------------------------------------------------------------

def test_failed_import_default_warns(tmp_path, cli_runner, monkeypatch):
    """A bad variant module → exit 0, 'Failed imports' in output, bad.py listed."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(BAD_IMPORT)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output
    assert "Failed imports" in result.output or "failed" in result.output.lower()
    assert "bad.py" in result.output


# ---------------------------------------------------------------------------
# Test 5: failed import + --strict hard-fails (exit non-zero)
# ---------------------------------------------------------------------------

def test_failed_import_strict_hard_fails(tmp_path, cli_runner, monkeypatch):
    """A bad variant module + --strict → exit non-zero, 'failed' in output."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "bad.py").write_text(BAD_IMPORT)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry", "--strict"])
    assert result.exit_code != 0
    assert "failed" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 6: three kind dirs walked independently
# ---------------------------------------------------------------------------

def test_three_kinds_walked(tmp_path, cli_runner, monkeypatch):
    """clam_mb/, _losses/, _policies/ all get their __init__.py regenerated."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Model variant in clam_mb/.
    m_dir = adir / "variants" / "clam_mb"
    m_dir.mkdir(parents=True, exist_ok=True)
    body = (
        GOOD_MODEL.replace("v_NN", "v0001").replace("V_NN", "V0001")
        .replace("node_NN", "node_0001")
    )
    (m_dir / "v0001.py").write_text(body)
    # Loss variant in _losses/.
    body_l = (
        GOOD_LOSS.replace("l_NN", "l0001").replace("L_NN", "L0001")
        .replace("node_NN", "node_0002")
    )
    (adir / "variants" / "_losses" / "l0001.py").write_text(body_l)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output

    assert (m_dir / "__init__.py").exists()
    assert (adir / "variants" / "_losses" / "__init__.py").exists()
    assert (adir / "variants" / "_policies" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# Test 7: variants/ missing — ClickException with init suggestion
# ---------------------------------------------------------------------------

def test_variants_dir_missing(tmp_path, cli_runner, monkeypatch):
    """No automil/variants/ → exit non-zero, output mentions init or variants."""
    # Manually create automil/config.yaml so _find_automil_dir resolves.
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "automil").mkdir()
    (tmp_path / "automil" / "config.yaml").write_text("registry: {}\n")

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code != 0
    assert "init" in result.output.lower() or "variants" in result.output


# ---------------------------------------------------------------------------
# Test 8: output format — "imported=N failed=M skipped=K"
# ---------------------------------------------------------------------------

def test_output_format(tmp_path, cli_runner, monkeypatch):
    """Success output includes `imported=N failed=M skipped=K` format."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output
    assert "imported=" in result.output
    assert "failed=" in result.output


# ---------------------------------------------------------------------------
# Test 9: --help mentions workflow ("after adding/renaming variant")
# ---------------------------------------------------------------------------

def test_help_workflow_text(cli_runner):
    """refresh-registry --help mentions 'after adding' or 'renaming' variant workflow."""
    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry", "--help"])
    assert result.exit_code == 0
    out = result.output.lower()
    assert "after" in out or "adding" in out or "renaming" in out


# ---------------------------------------------------------------------------
# Test 10: no .tmp leftovers after refresh
# ---------------------------------------------------------------------------

def test_no_tmp_leftover(tmp_path, cli_runner, monkeypatch):
    """No *.tmp files remain under variants/ after a successful refresh."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    from automil.cli import main
    cli_runner.invoke(main, ["refresh-registry"])
    leftovers = list((adir / "variants").rglob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Test 11: _candidates dir walked (not skipped)
# ---------------------------------------------------------------------------

def test_candidates_dir_walked(tmp_path, cli_runner, monkeypatch):
    """variants/_candidates/ with a loss variant gets its __init__.py regenerated."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    cand_dir = adir / "variants" / "_candidates"
    # cand_dir should exist after init; add a valid module
    body_l = (
        GOOD_LOSS.replace("l_NN", "candidate_l").replace("L_NN", "CandidateL")
        .replace("node_NN", "node_0042")
    )
    (cand_dir / "candidate_l.py").write_text(body_l)
    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output
    assert (cand_dir / "__init__.py").exists()


# ---------------------------------------------------------------------------
# Test 12: per-parent dirs walked — two different model parents
# ---------------------------------------------------------------------------

def test_per_parent_dirs_walked(tmp_path, cli_runner, monkeypatch):
    """Two model parent dirs (clam_mb, ab_mil) both get __init__.py regenerated."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    for parent in ("clam_mb", "ab_mil"):
        d = adir / "variants" / parent
        d.mkdir(parents=True, exist_ok=True)
        body = (
            GOOD_MODEL
            .replace("v_NN", f"v_{parent}")
            .replace("V_NN", f"V{parent.upper()}")
            .replace("node_NN", f"node_{parent}")
            .replace("clam_mb", parent)
        )
        (d / f"v_{parent}.py").write_text(body)

    from automil.cli import main
    result = cli_runner.invoke(main, ["refresh-registry"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "clam_mb" / "__init__.py").exists()
    assert (adir / "variants" / "ab_mil" / "__init__.py").exists()


# ---------------------------------------------------------------------------
# Test 13: registry cleared between runs — duplicate name hits hard-fail
# ---------------------------------------------------------------------------

def test_clears_registry_between_runs(tmp_path, cli_runner, monkeypatch):
    """Second refresh with a duplicate-name variant yields a failed-import entry."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    v_dir = adir / "variants" / "_losses"

    # First module (unique name l0001).
    body1 = (
        GOOD_LOSS.replace("l_NN", "l0001").replace("L_NN", "L0001")
        .replace("node_NN", "node_0001")
    )
    (v_dir / "l0001.py").write_text(body1)

    from automil.cli import main
    result1 = cli_runner.invoke(main, ["refresh-registry"])
    assert result1.exit_code == 0, result1.output

    # Second module reuses the SAME name l0001 → @register should hard-fail.
    body2 = (
        GOOD_LOSS.replace("l_NN", "l0001").replace("L_NN", "L0001_DUP")
        .replace("node_NN", "node_0002")
    )
    (v_dir / "l0001_dup.py").write_text(body2)

    # Registry NOT manually cleared here — refresh-registry must clear it internally.
    result2 = cli_runner.invoke(main, ["refresh-registry"])
    # The duplicate should land in result.failed, producing "failed=1" or "Failed imports".
    assert "failed=1" in result2.output or "Failed imports" in result2.output or result2.exit_code != 0
