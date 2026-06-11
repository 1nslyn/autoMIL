"""Wave 1 unit tests for D-195 / RESEARCH.md OQ-5: `automil submit --max-time SECONDS`.

These tests exercise the new flag in isolation. Full integration with the
LocalBackend walltime path is covered by plan 07-09's setup-done gate test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner


def _init_minimal_project(tmp_path: Path) -> Path:
    """Create a minimal project layout sufficient for `automil submit --files train.py`."""
    import subprocess

    (tmp_path / "train.py").write_text(
        "import json, pathlib\n"
        "pathlib.Path('result.json').write_text(json.dumps({'status':'completed','metrics':{},'composite':0.0}))\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    automil_dir = tmp_path / "automil"
    for sub in ("orchestrator/queue", "orchestrator/running", "orchestrator/archive", "orchestrator/completed"):
        (automil_dir / sub).mkdir(parents=True)
    (automil_dir / "config.yaml").write_text(
        "backend:\n  name: local\nfiles:\n  editable: [train.py]\n  readonly: []\n"
        "registry:\n  protected: []\n  editable: [train.py]\n  identity_constraints: []\n"
        "data:\n  features_dir: ''\n  splits_dir: ''\n  metadata: ''\n"
        "encoders: {}\n"
        "metrics:\n  composite: {formula: 'val_auc'}\n  required: [val_auc]\n"
        "training: {}\n"
        "cap:\n  budget_seconds: 21600\n  safety_buffer_seconds: 1800\n"
        "run:\n  mil_model: test_model\n"  # D-12: required for submit
    )
    return tmp_path


def test_max_time_60_seconds_yields_timeout_min_1(tmp_path, monkeypatch):
    """OQ-5: `--max-time 60` writes spec timeout_min == 1 (ceil-div minutes)."""
    project = _init_minimal_project(tmp_path)
    monkeypatch.chdir(project)
    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "submit", "--node", "node_test01", "--desc", "max-time-test",
        "--files", "train.py", "--max-time", "60",
    ])
    assert result.exit_code == 0, f"submit failed: {result.output}\n{result.exception!r}"
    spec_path = project / "automil" / "orchestrator" / "queue" / "node_test01.json"
    assert spec_path.exists(), f"queue file missing; submit output:\n{result.output}"
    spec = json.loads(spec_path.read_text())
    assert spec["timeout_min"] == 1, spec


def test_max_time_121_seconds_ceil_to_3_min(tmp_path, monkeypatch):
    """OQ-5: `--max-time 121` writes timeout_min == 3 (ceil-div: (121+59)//60 == 3)."""
    project = _init_minimal_project(tmp_path)
    monkeypatch.chdir(project)
    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "submit", "--node", "node_test02", "--desc", "max-time-ceil",
        "--files", "train.py", "--max-time", "121",
    ])
    assert result.exit_code == 0, result.output
    spec = json.loads((project / "automil" / "orchestrator" / "queue" / "node_test02.json").read_text())
    assert spec["timeout_min"] == 3, spec


def test_max_time_overrides_timeout_with_warning(tmp_path, monkeypatch):
    """OQ-5: `--max-time 60 --timeout 99` -> timeout_min == 1, warning printed."""
    project = _init_minimal_project(tmp_path)
    monkeypatch.chdir(project)
    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "submit", "--node", "node_test03", "--desc", "override-test",
        "--files", "train.py", "--max-time", "60", "--timeout", "99",
    ])
    assert result.exit_code == 0, result.output
    spec = json.loads((project / "automil" / "orchestrator" / "queue" / "node_test03.json").read_text())
    assert spec["timeout_min"] == 1, spec
    assert "--max-time wins" in result.output, result.output


def test_max_time_negative_rejected(tmp_path, monkeypatch):
    """OQ-5: negative --max-time raises ClickException with a clear message.

    WR-03 update: guard is now <= 0, error message says 'must be > 0 seconds'.
    """
    project = _init_minimal_project(tmp_path)
    monkeypatch.chdir(project)
    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, [
        "submit", "--node", "node_test04", "--desc", "negative",
        "--files", "train.py", "--max-time", "-5",
    ])
    assert result.exit_code != 0
    assert "must be > 0" in result.output, result.output
