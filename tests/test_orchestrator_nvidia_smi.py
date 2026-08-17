"""Coverage for nvidia-smi path pinning + automil check report (CLN-05).

The orchestrator's `query_gpus` shells out to nvidia-smi every poll cycle;
on a shared host a PATH-shim earlier on $PATH could spoof VRAM numbers and
trick the bin-packer (CONCERNS.md §"nvidia-smi invocation has no path
pinning"). Phase 0 / Plan 03 closes this by resolving the absolute path
once at module import via shutil.which() and surfacing the resolved path
through `automil check` (D-18).
"""
from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

import automil.orchestrator as orch_mod


def _reload_with_which(monkeypatch, which_return):
    """Patch shutil.which then reload orchestrator to re-resolve NVIDIA_SMI_PATH."""
    monkeypatch.setattr(
        shutil, "which", lambda name: which_return if name == "nvidia-smi" else None
    )
    return importlib.reload(orch_mod)


@pytest.fixture(autouse=True)
def _restore_orchestrator_module():
    """Reload orchestrator after each test so cross-test state stays clean."""
    yield
    importlib.reload(orch_mod)


def test_path_resolved(monkeypatch):
    """When shutil.which returns a real path, NVIDIA_SMI_PATH equals that path."""
    mod = _reload_with_which(monkeypatch, "/usr/bin/nvidia-smi")
    assert mod.NVIDIA_SMI_PATH == "/usr/bin/nvidia-smi"


def test_path_missing_fallback_warns(monkeypatch, caplog):
    """When shutil.which returns None, NVIDIA_SMI_PATH is the bare string and WARN logs."""
    with caplog.at_level(logging.WARNING, logger="automil.orchestrator"):
        mod = _reload_with_which(monkeypatch, None)
    assert mod.NVIDIA_SMI_PATH == "nvidia-smi"
    # WARN at import time about detection failure
    assert any(
        "nvidia-smi" in rec.message and "PATH" in rec.message
        for rec in caplog.records
    ), f"Expected WARN about nvidia-smi PATH detection failure; got: {[r.message for r in caplog.records]}"


def test_subprocess_uses_pinned_path(monkeypatch):
    """The first positional arg to subprocess.run is NVIDIA_SMI_PATH."""
    mod = _reload_with_which(monkeypatch, "/opt/nvidia/nvidia-smi")
    captured_argv: dict[str, str] = {}

    def fake_run(argv, **kwargs):
        captured_argv["argv0"] = argv[0]
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    mod.query_gpus()
    assert captured_argv["argv0"] == "/opt/nvidia/nvidia-smi"


def test_check_reports_nvidia_smi_path(tmp_path, monkeypatch):
    """`automil check` output contains a `nvidia-smi:` line."""
    from click.testing import CliRunner

    # Set up minimal automil/ skeleton so check runs.
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    # Required orchestrator subdirectories so check doesn't blow up before
    # reaching the nvidia-smi report.
    for d in ("queue", "running", "archive", "completed"):
        (automil_dir / "orchestrator" / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git").mkdir()

    monkeypatch.chdir(tmp_path)
    _reload_with_which(monkeypatch, "/usr/bin/nvidia-smi")

    # Reload the cli.check module so it picks up the freshly-resolved path.
    import automil.cli.check as check_mod
    importlib.reload(check_mod)

    from automil.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["check"])
    # Exit code may be non-zero for unrelated check failures (e.g. missing
    # baseline primary_value or training script); what matters is the nvidia-smi:
    # line appears in stdout (D-18).
    assert "nvidia-smi:" in result.output, (
        f"check output missing nvidia-smi report: {result.output}"
    )
