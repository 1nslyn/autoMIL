"""RED stubs for automil --project PATH group option (OPS-04).

Wave-0 Nyquist compliance — all stubs xfail until 13-02 implements the
--project group option and _PROJECT_OVERRIDE bridge in _helpers.py.

IMPORTANT: every test resets automil.cli._helpers._PROJECT_OVERRIDE = None
in teardown via the _reset_project_override autouse fixture to prevent
cross-test bleed (RESEARCH §Pitfall 4).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _reset_project_override(monkeypatch):
    """Reset _PROJECT_OVERRIDE to None after every OPS-04 test.

    Guards against cross-test bleed: if _PROJECT_OVERRIDE is set by a test
    and not cleaned up, subsequent tests in the same session see the wrong
    project root.  Uses monkeypatch.setattr so teardown is automatic.

    The attribute may not exist pre-implementation — guard with hasattr so
    the fixture itself does not fail before OPS-04 is shipped.
    """
    import automil.cli._helpers as _h  # noqa: PLC0415
    if hasattr(_h, "_PROJECT_OVERRIDE"):
        monkeypatch.setattr(_h, "_PROJECT_OVERRIDE", None)
    yield
    # monkeypatch auto-reverts setattr in teardown; no explicit reset needed.


# ---------------------------------------------------------------------------
# OPS-04 RED stubs (Wave 0 — Nyquist compliance)
# All xfail until plan 13-02 implements the --project group option.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="OPS-04 not yet implemented", strict=True)
def test_project_option_project_root(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """--project PATH (PATH = project root) routes discovery outside cwd.

    OPS-04: when cwd is an unrelated directory, --project <project_root>
    must allow _find_automil_dir() to return the correct automil/ path.

    Fails correctly pre-implementation: `--project` is not a recognized option
    (Click returns "No such option: --project"), which is the wrong reason to
    succeed or fail. The test's strict assertion ensures it only goes green when
    the option is registered AND routes project discovery correctly.
    """
    # Create project_a with a valid automil/config.yaml.
    project_a = tmp_path / "project_a"
    (project_a / "automil").mkdir(parents=True)
    (project_a / "automil" / "config.yaml").write_text("run:\n  script: train.py\n")

    # Set cwd to a completely separate, empty directory.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # Invoke status (read-only command) with --project pointing at project root.
    # Do NOT use catch_exceptions=False — pre-impl the option doesn't exist and
    # Click raises UsageError; we want to inspect exit_code and output.
    result = cli_runner.invoke(main, ["--project", str(project_a), "status"])

    # The option must be recognized (not "No such option").
    assert "No such option" not in result.output, (
        f"--project option not yet registered (pre-implementation): {result.output!r}"
    )

    # Must resolve correctly and NOT report "No automil/config.yaml".
    assert "No automil/config.yaml" not in result.output, (
        f"--project override did not resolve project root: {result.output!r}"
    )


@pytest.mark.xfail(reason="OPS-04 not yet implemented", strict=True)
def test_project_option_automil_dir(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """--project PATH (PATH = automil/ dir directly) also resolves correctly.

    OPS-04: the --project option must accept both the project root (containing
    automil/) and the automil/ directory itself as valid PATH arguments.
    """
    project_a = tmp_path / "project_a"
    (project_a / "automil").mkdir(parents=True)
    (project_a / "automil" / "config.yaml").write_text("run:\n  script: train.py\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    # Pass the automil/ dir directly (not the project root).
    result = cli_runner.invoke(
        main,
        ["--project", str(project_a / "automil"), "status"],
    )

    # The option must be recognized (not "No such option").
    assert "No such option" not in result.output, (
        f"--project option not yet registered (pre-implementation): {result.output!r}"
    )

    # Must resolve correctly when PATH points at automil/ directly.
    assert "No automil/config.yaml" not in result.output, (
        f"--project automil/ override did not resolve: {result.output!r}"
    )


def test_project_option_absent_cwd_walk(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch
) -> None:
    """--project absent → cwd walk unchanged (regression for normal invocation).

    OPS-04: adding --project must not break the existing cwd-walk behaviour.
    When --project is absent, _find_automil_dir() must still find the project
    by walking up from cwd, as it does today.

    NOT xfail — this is existing behaviour that must remain correct both
    before and after OPS-04 is implemented.
    """
    # Create automil/ in tmp_path (directly accessible via cwd walk).
    (tmp_path / "automil").mkdir(parents=True)
    (tmp_path / "automil" / "config.yaml").write_text("run:\n  script: train.py\n")
    monkeypatch.chdir(tmp_path)

    # No --project flag; rely on cwd walk.
    result = cli_runner.invoke(main, ["status"], catch_exceptions=False)

    assert "No automil/config.yaml" not in result.output, (
        f"cwd walk regression: expected project to resolve, got: {result.output!r}"
    )
