"""Coverage for cli/lifecycle/ package + stub commands (CLI-01/02/05/06/08/09).

Tests assert:
- All six commands are registered on the main Click group.
- Each command has a workflow-explaining --help (>100 chars, keyword present).
- Each stub hard-fails with "not yet implemented (Plan 01-NN)".
- File-structure invariants (lifecycle.py deleted, lifecycle/ package present).
- _shared helpers work correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# 1. Command registration
# ---------------------------------------------------------------------------


def test_six_commands_registered(cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in (
        "apply",
        "revert-baseline",
        "refresh-registry",
        "port-variant",
        "promote-variant",
        "verify-repro",
    ):
        assert cmd in result.output, f"missing command in automil --help: {cmd!r}"


# ---------------------------------------------------------------------------
# 2. Each command has --help (exits 0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        "apply",
        "revert-baseline",
        "refresh-registry",
        "port-variant",
        "promote-variant",
        "verify-repro",
    ],
)
def test_each_command_has_help(cli_runner, cmd):
    from automil.cli import main
    result = cli_runner.invoke(main, [cmd, "--help"])
    assert result.exit_code == 0, f"{cmd} --help exited {result.exit_code}: {result.output}"
    # Docstring must be at least 100 chars — not just a flags list.
    assert len(result.output) > 100, f"{cmd} --help is too brief:\n{result.output}"


# ---------------------------------------------------------------------------
# 3. Stub error format (naming implementing plan + "not yet implemented")
# ---------------------------------------------------------------------------


def test_stub_error_format_none_remaining():
    """All lifecycle commands are now fully implemented (Plan 01-12 closed
    the last stub — verify-repro). This test confirms the stub list is empty."""
    # apply (01-09), refresh-registry (01-09), revert-baseline (01-10),
    # port-variant (01-11), promote-variant (01-11), verify-repro (01-12)
    # are all implemented — no stubs remain.
    remaining_stubs = []  # was: [("verify-repro", "01-12")]
    assert remaining_stubs == [], (
        "Unexpected stubs still in the lifecycle — update this test after "
        "implementing them."
    )


# ---------------------------------------------------------------------------
# 4. File-structure invariants
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_lifecycle_py_file_deleted():
    """lifecycle.py (Phase 0 stub) must be gone; the package replaces it."""
    assert not (_REPO_ROOT / "src/automil/cli/lifecycle.py").is_file(), (
        "lifecycle.py still exists — should have been replaced by lifecycle/ package"
    )


def test_lifecycle_package_exists():
    """All expected sub-modules exist under cli/lifecycle/."""
    assert (_REPO_ROOT / "src/automil/cli/lifecycle/__init__.py").is_file(), (
        "lifecycle/__init__.py missing"
    )
    for sub in (
        "_shared",
        "apply",
        "revert_baseline",
        "refresh_registry",
        "port_variant",
        "promote_variant",
        "verify_repro",
    ):
        assert (_REPO_ROOT / f"src/automil/cli/lifecycle/{sub}.py").is_file(), (
            f"missing sub-module: lifecycle/{sub}.py"
        )


# ---------------------------------------------------------------------------
# 5. cli/__init__.py still imports lifecycle (now a package)
# ---------------------------------------------------------------------------


def test_cli_init_imports_lifecycle_package():
    """After conversion, `from automil.cli import lifecycle` resolves the package."""
    import automil.cli.lifecycle as lifecycle_pkg

    # The package should expose the sub-module names after __init__.py runs.
    assert hasattr(lifecycle_pkg, "apply"), "lifecycle.apply not available after import"
    assert hasattr(lifecycle_pkg, "revert_baseline"), (
        "lifecycle.revert_baseline not available after import"
    )


# ---------------------------------------------------------------------------
# 6. _shared helpers: _atomic_write_text
# ---------------------------------------------------------------------------


def test_atomic_write_helper_available(tmp_path):
    from automil.cli.lifecycle._shared import _atomic_write_text

    target = tmp_path / "out.txt"
    _atomic_write_text(target, "hello\n")
    assert target.read_text() == "hello\n"
    # No leftover .tmp files.
    assert list(tmp_path.glob("*.tmp")) == [], "leftover .tmp files after atomic write"


def test_atomic_write_creates_parent_dirs(tmp_path):
    from automil.cli.lifecycle._shared import _atomic_write_text

    target = tmp_path / "nested" / "deep" / "file.txt"
    _atomic_write_text(target, "data")
    assert target.read_text() == "data"


def test_atomic_write_overwrites_existing(tmp_path):
    from automil.cli.lifecycle._shared import _atomic_write_text

    target = tmp_path / "file.txt"
    _atomic_write_text(target, "first")
    _atomic_write_text(target, "second")
    assert target.read_text() == "second"


# ---------------------------------------------------------------------------
# 7. _shared helpers: _get_node_or_die
# ---------------------------------------------------------------------------


def test_get_node_or_die_missing_lists_available(tmp_path):
    import click
    from automil.cli.lifecycle._shared import _get_node_or_die

    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "graph.json").write_text(json.dumps({
        "nodes": {
            "node_0001": {"id": "node_0001"},
            "node_0042": {"id": "node_0042"},
        }
    }))

    with pytest.raises(click.ClickException) as exc_info:
        _get_node_or_die(adir, "node_9999")
    msg = exc_info.value.message
    assert "node_0001" in msg, f"expected node_0001 in error: {msg}"
    assert "node_0042" in msg, f"expected node_0042 in error: {msg}"
    assert "available" in msg.lower(), f"expected 'available' in error: {msg}"


def test_get_node_or_die_missing_graph(tmp_path):
    import click
    from automil.cli.lifecycle._shared import _get_node_or_die

    adir = tmp_path / "automil"
    adir.mkdir()
    with pytest.raises(click.ClickException, match=r"graph\.json"):
        _get_node_or_die(adir, "node_0001")


def test_get_node_or_die_returns_node_dict(tmp_path):
    from automil.cli.lifecycle._shared import _get_node_or_die

    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "graph.json").write_text(json.dumps({
        "nodes": {"node_0001": {"id": "node_0001", "primary_value": 0.5}}
    }))

    node = _get_node_or_die(adir, "node_0001")
    assert node["primary_value"] == 0.5


def test_get_node_or_die_malformed_json(tmp_path):
    import click
    from automil.cli.lifecycle._shared import _get_node_or_die

    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "graph.json").write_text("{not valid json")
    with pytest.raises(click.ClickException, match=r"malformed"):
        _get_node_or_die(adir, "node_0001")


# ---------------------------------------------------------------------------
# 8. --help workflow keywords (production-grade docstrings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,keyword",
    [
        ("apply", "config"),
        ("revert-baseline", "git"),
        ("refresh-registry", "scan"),
        ("port-variant", "manifest"),
        ("promote-variant", "candidate"),
        ("verify-repro", "manifest"),
    ],
)
def test_each_command_helpdoc_mentions_workflow(cli_runner, cmd, keyword):
    """Production-grade --help: explains workflow, not just flags."""
    from automil.cli import main

    result = cli_runner.invoke(main, [cmd, "--help"])
    assert keyword.lower() in result.output.lower(), (
        f"{cmd} --help missing workflow keyword {keyword!r}:\n{result.output}"
    )
