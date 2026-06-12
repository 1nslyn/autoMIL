"""RED stubs for CFG-02 and CFG-03 — Phase 11 Wave 0.

All tests in this file guard against regressions introduced by Plans 11-02
and 11-03. Tests marked # RED will fail until the production code is fixed;
tests marked # GREEN provide regression coverage and should pass both before
and after the fix.

CFG-02 (ISSUE-022): `automil submit --timeout` default 150 masks orchestrator
  default_timeout_min. Fix: default=None, omit `timeout_min` from spec when
  unset; sentinel `timeout != 150` must become `timeout is not None` (D-03).

CFG-03 (ISSUE-008): No per-node run-command override. Fix: `--override "<args>"`
  written into queue spec; daemon appends override args after base run.command.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest.mock as mock
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


# ---------------------------------------------------------------------------
# Helpers — reuse the git-init pattern from test_cli.py
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> None:
    """Initialize a minimal git repo with an initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True
    )


def _setup_project(tmp_path: Path, runner: CliRunner, monkeypatch) -> Path:
    """Init a minimal automil project and return tmp_path."""
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, f"init failed: {result.output}"
    # Provide a dummy snapshotable file so submit has something to capture
    (tmp_path / "train.py").write_text("# dummy\n")
    subprocess.run(["git", "add", "train.py"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add train"], cwd=tmp_path, capture_output=True, check=True
    )
    # Put a fresh copy (untracked/dirty) so auto-detect picks it up
    (tmp_path / "train.py").write_text("# modified\n")
    return tmp_path


def _read_queue_spec(tmp_path: Path) -> dict:
    """Return the single spec written to automil/orchestrator/queue/."""
    queue_dir = tmp_path / "automil" / "orchestrator" / "queue"
    specs = list(queue_dir.glob("*.json"))
    assert len(specs) == 1, f"Expected 1 queue spec, found {len(specs)}: {specs}"
    return json.loads(specs[0].read_text())


# ---------------------------------------------------------------------------
# CFG-02: timeout_min omit when --timeout not supplied
# ---------------------------------------------------------------------------

class TestCFG02TimeoutOmit:
    def test_submit_without_timeout_omits_timeout_min(
        self, tmp_path, monkeypatch
    ):
        """RED: submit without --timeout must NOT write timeout_min to the spec.

        Fails today because submit.py L29 defaults `--timeout` to 150, so
        `timeout_min: 150` is always written into the spec (L441).
        Fix (D-02): change default to None; omit key when None.
        """
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "no timeout",
             "--files", "train.py", "--mil-model", "test_model"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"submit failed: {result.output}"

        spec = _read_queue_spec(tmp_path)
        # RED: this assertion fails today (timeout_min IS present with value 150)
        assert "timeout_min" not in spec, (
            f"timeout_min must be absent when --timeout is not supplied; "
            f"got timeout_min={spec.get('timeout_min')}"
        )

    def test_submit_with_explicit_timeout_writes_timeout_min(
        self, tmp_path, monkeypatch
    ):
        """GREEN: submit --timeout 90 must write timeout_min: 90 into the spec.

        This passes today and should continue to pass after the CFG-02 fix.
        Provides regression coverage that explicit --timeout is still honored.
        """
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_0002", "--desc", "explicit timeout",
             "--files", "train.py", "--mil-model", "test_model",
             "--timeout", "90"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"submit failed: {result.output}"

        spec = _read_queue_spec(tmp_path)
        assert spec.get("timeout_min") == 90, (
            f"Expected timeout_min=90, got {spec.get('timeout_min')}"
        )


# ---------------------------------------------------------------------------
# CFG-02 / D-03 regression guard: --max-time wins when both flags given
# ---------------------------------------------------------------------------

class TestCFG02MaxTimeInteraction:
    def test_max_time_wins_over_explicit_timeout(
        self, tmp_path, monkeypatch
    ):
        """D-03 regression guard: --max-time must win when --timeout is also given.

        Scenario: --max-time 120 --timeout 99
        Expected: spec timeout_min == ceil(120/60) == 2 (--max-time wins)

        Today the sentinel is `timeout != 150`. Because 99 != 150 is True, the
        --max-time path currently does still log the conflict message — but the
        test is written here explicitly so that after D-03 changes the sentinel
        to `timeout is not None`, this combination is still correctly handled.

        This test MAY pass green today depending on current sentinel behavior;
        it MUST pass green after the fix. Its role is to be the named regression
        guard for the D-03 sentinel change.
        """
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_0003", "--desc", "max-time interaction",
             "--files", "train.py", "--mil-model", "test_model",
             "--max-time", "120", "--timeout", "99"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"submit failed: {result.output}"

        spec = _read_queue_spec(tmp_path)
        # 120 seconds ceil-div to minutes = ceil(120/60) = 2
        assert spec.get("timeout_min") == 2, (
            f"--max-time 120s must produce timeout_min=2 (ceil-div), "
            f"got {spec.get('timeout_min')}"
        )


# ---------------------------------------------------------------------------
# CFG-03: --override written into queue spec
# ---------------------------------------------------------------------------

class TestCFG03OverrideSpec:
    def test_submit_override_written_to_spec(
        self, tmp_path, monkeypatch
    ):
        """RED: submit --override must write the override string into the queue spec.

        Fails today because the `--override` option does not exist in submit.py.
        Fix (D-04): add `--override` option; write to spec under an appropriate
        field name (e.g. `run_command_override` or `override_args`).
        """
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_0004", "--desc", "override test",
             "--files", "train.py", "--mil-model", "test_model",
             "--override", "--seed 42 --lr 1e-4"],
        )
        # RED: today this exits non-zero (unknown option)
        assert result.exit_code == 0, (
            f"submit with --override must succeed; got exit={result.exit_code}, "
            f"output={result.output}"
        )

        spec = _read_queue_spec(tmp_path)
        # The exact field name is at the implementer's discretion (D-04).
        # Accept either run_command_override or override_args.
        override_value = spec.get("run_command_override") or spec.get("override_args")
        assert override_value is not None, (
            f"spec must contain override field (run_command_override or override_args); "
            f"spec keys: {list(spec.keys())}"
        )
        assert "--seed 42" in override_value and "--lr 1e-4" in override_value, (
            f"override field must contain the supplied args; got {override_value!r}"
        )


# ---------------------------------------------------------------------------
# CFG-03: daemon appends override args after base run.command
# ---------------------------------------------------------------------------

class TestCFG03DaemonAppend:
    def test_daemon_appends_override_to_run_command(self, tmp_path):
        """GREEN (after Plan 11-03): daemon appends spec override args after base run.command.

        Verifies D-04 suffix-append semantics: the final cmd list passed to
        subprocess.Popen must equal shlex.split(base_command) + shlex.split(override_string).

        Strategy: construct a minimal ExperimentOrchestrator with all filesystem
        dependencies mocked, patch subprocess.Popen, invoke _launch, and assert
        the cmd list passed to Popen is the expected concatenation.
        Security check: Popen must NOT be called with shell=True (T-11-03-01).
        """
        import shlex
        from unittest.mock import MagicMock, patch, call

        from automil.backends._orchestrator_daemon import ExperimentOrchestrator

        base_command = "python train.py --dataset ovarian"
        override_string = "--seed 99 --lr 5e-4"
        expected_cmd = shlex.split(base_command) + shlex.split(override_string)

        spec = {
            "id": "node_test",
            "base_commit": "deadbeef",
            "overlay_dir": "archive/node_test",
            "overlay_manifest": {},
            "deletions": [],
            "priority": 1,
            "estimated_vram_gb": 0.5,
            "run_command_override": override_string,
            "graph_metadata": {"parent_id": None, "techniques": [], "config_hash": "abc"},
        }

        # Set up minimal directory structure so the daemon constructor doesn't crash.
        automil_dir = tmp_path / "automil"
        automil_dir.mkdir()
        (automil_dir / "config.yaml").write_text(
            f"run:\n  command: \"{base_command}\"\n"
        )
        (automil_dir / "graph.json").write_text('{"nodes": {}, "meta": {"next_id": 1}}')
        # Create orchestrator subdirs the constructor and _launch expect.
        for sub in ("queue", "running", "archive", "completed"):
            (automil_dir / "orchestrator" / sub).mkdir(parents=True, exist_ok=True)

        # Create node archive dir (spec.json will be written there by _launch).
        node_archive = automil_dir / "orchestrator" / "archive" / "node_test"
        node_archive.mkdir(parents=True, exist_ok=True)

        with patch("automil.backends._orchestrator_daemon._find_automil_dir",
                   return_value=automil_dir), \
             patch("automil.backends._orchestrator_daemon._find_git_root",
                   return_value=tmp_path):

            orch = ExperimentOrchestrator(
                project_root=tmp_path,
                automil_dir=automil_dir,
            )

        # Patch runner so worktree creation/overlay are no-ops.
        fake_wt_path = tmp_path / "worktree_node_test"
        fake_wt_path.mkdir()
        orch.runner = MagicMock()
        orch.runner.create_worktree.return_value = fake_wt_path
        orch.runner.apply_overlay.return_value = None

        launched_cmds: list[list] = []
        popen_kwargs: list[dict] = []

        class _FakePopen:
            pid = 12345
            def __init__(self, cmd, **kwargs):
                launched_cmds.append(cmd)
                popen_kwargs.append(kwargs)
            def communicate(self):
                return (b"", b"")

        with patch("automil.backends._orchestrator_daemon.subprocess.Popen",
                   side_effect=_FakePopen):
            try:
                orch._launch(spec, gpu_id=0)
            except Exception:
                # _launch may raise after Popen if running-spec write fails —
                # we only care about the cmd that was passed to Popen.
                pass

        assert len(launched_cmds) >= 1, (
            "subprocess.Popen was never called — _launch did not reach the launch block. "
            "Check that mocked runner.create_worktree returned a valid path."
        )
        actual_cmd = launched_cmds[0]
        assert actual_cmd == expected_cmd, (
            f"Daemon must append shlex.split(override) after shlex.split(run_command).\n"
            f"Expected: {expected_cmd}\n"
            f"Got:      {actual_cmd}"
        )
        # Security guard: Popen must NOT use shell=True (T-11-03-01).
        assert not popen_kwargs[0].get("shell", False), (
            "subprocess.Popen must NOT be called with shell=True (T-11-03-01 command-injection guard)"
        )


# ---------------------------------------------------------------------------
# WR-01: --override shlex validation at CLI submit time
# ---------------------------------------------------------------------------

class TestWR01OverrideShlex:
    def test_submit_override_malformed_quotes_raises_cli_error(
        self, tmp_path, monkeypatch
    ):
        """WR-01: submit --override with unbalanced quotes must fail at CLI time.

        Before the fix, shlex.split() was only called inside the daemon at
        launch time (after the spec was dequeued), routing the node to
        _mark_crashed with no user-visible CLI feedback.

        Fix: shlex.split() is validated in submit.py before writing the spec.
        This test asserts exit_code != 0 and that the error mentions the
        parsing failure, so operators get immediate feedback.
        """
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_wr01", "--desc", "override-shlex-test",
             "--files", "train.py", "--mil-model", "test_model",
             "--override", "'unbalanced"],
        )
        assert result.exit_code != 0, (
            "WR-01: submit with malformed --override must exit non-zero; "
            f"got exit_code={result.exit_code}, output={result.output!r}"
        )
        assert "unbalanced" in result.output.lower() or "parse" in result.output.lower(), (
            f"WR-01: error output must mention parsing problem; got {result.output!r}"
        )
        # Confirm the spec was NOT written to queue/ (spec must not be dequeued)
        queue_dir = tmp_path / "automil" / "orchestrator" / "queue"
        specs = list(queue_dir.glob("*.json")) if queue_dir.exists() else []
        assert len(specs) == 0, (
            f"WR-01: no spec must be written to queue/ when override is invalid; "
            f"found: {specs}"
        )

    def test_submit_override_valid_quotes_succeeds(self, tmp_path, monkeypatch):
        """WR-01 regression guard: valid quoted --override string is accepted."""
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_wr01b", "--desc", "override-valid",
             "--files", "train.py", "--mil-model", "test_model",
             "--override", "--seed 42 --lr 1e-4"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (
            f"WR-01: valid --override must succeed; got {result.output!r}"
        )
        spec = _read_queue_spec(tmp_path)
        assert spec.get("run_command_override") == "--seed 42 --lr 1e-4"

    def test_submit_max_time_zero_raises_cli_error(self, tmp_path, monkeypatch):
        """WR-03: --max-time 0 must be rejected at CLI time (not silently become 1 minute)."""
        runner = CliRunner()
        _setup_project(tmp_path, runner, monkeypatch)

        result = runner.invoke(
            main,
            ["submit", "--node", "node_wr03", "--desc", "max-time-zero",
             "--files", "train.py", "--mil-model", "test_model",
             "--max-time", "0"],
        )
        assert result.exit_code != 0, (
            f"WR-03: --max-time 0 must exit non-zero; got {result.output!r}"
        )
        assert "0" in result.output, (
            f"WR-03: error must mention the bad value; got {result.output!r}"
        )
