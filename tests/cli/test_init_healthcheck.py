"""Integration tests for D-198 clauses 2-3: automil init healthcheck wiring.

Five tests:
  1. test_init_no_healthcheck_flag: --no-healthcheck path skips probe; conservative defaults stamped.
  2. test_init_stamps_gpu_count: fresh init stamps healthcheck_gpu_count into rendered config.yaml.
  3. test_init_recomputes_default_vram_from_results_tsv: results.tsv >=10 rows -> empirical quantile_95.
  4. test_init_uses_conservative_default_below_10_samples: results.tsv 5 rows -> conservative path.
  5. test_init_aborts_on_failed_detection_user_decline: click.confirm declined -> ClickException.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("test\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    return tmp_path


def _read_rendered_config(automil_dir: Path) -> dict:
    return yaml.safe_load((automil_dir / "config.yaml").read_text())


def test_init_no_healthcheck_flag(tmp_path, monkeypatch):
    """D-198 clause 2: --no-healthcheck stamps conservative defaults; no probe runs."""
    project = _init_git_repo(tmp_path)
    monkeypatch.chdir(project)
    from automil.cli import main as cli_main
    runner = CliRunner()
    result = runner.invoke(cli_main, ["init", "--no-healthcheck"])
    assert result.exit_code == 0, result.output
    cfg = _read_rendered_config(project / "automil")
    assert cfg["orchestrator"]["default_vram_estimate_gb"] == 8.0
    assert cfg["orchestrator"]["max_concurrent_per_gpu"] == 4
    assert cfg["hardware"]["accelerator"] == "cpu"
    assert cfg["hardware"]["gpu_count"] == 0


def test_init_stamps_gpu_count(tmp_path, monkeypatch):
    """D-198 clause 2: healthcheck values flow into rendered config.yaml hardware: section."""
    project = _init_git_repo(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fake_run(argv, **kwargs):
        if "mig.mode.current" in str(argv):
            return MagicMock(stdout="Disabled\nDisabled\nDisabled\n", returncode=0, stderr="")
        return MagicMock(stdout="0, 49140\n1, 49140\n2, 49140\n", returncode=0, stderr="")

    from automil.cli import main as cli_main
    runner = CliRunner()
    with patch("subprocess.run", side_effect=_fake_run):
        result = runner.invoke(cli_main, ["init"])
    assert result.exit_code == 0, result.output
    cfg = _read_rendered_config(project / "automil")
    assert cfg["hardware"]["gpu_count"] == 3
    assert cfg["hardware"]["accelerator"] == "cuda"
    assert cfg["hardware"]["min_vram_gb"] >= 47.0


def test_init_recomputes_default_vram_from_results_tsv(tmp_path, monkeypatch):
    """D-198 clause 2 + Pitfall 8 anti-acceptance #2: empirical quantile_95 path.

    Seeds a results.tsv with 30 rows; asserts default_vram_estimate_gb ~= quantile_95.
    """
    project = _init_git_repo(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    automil_dir = project / "automil"
    automil_dir.mkdir()
    # Seed results.tsv per _orchestrator_daemon.py:1289 column order.
    header = "node_id\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tprimary_value\tvram_gb\telapsed_min\tstatus\tdescription\n"
    rows = []
    vram_values = [4.0 + i * 0.5 for i in range(30)]  # 4.0 .. 18.5 in 0.5 steps
    for i, v in enumerate(vram_values):
        rows.append(f"node_{i:04d}\t0.85\t0.80\t0.85\t0.82\t0.83\t{v:.1f}\t10.0\tcompleted\trun_{i}\n")
    (automil_dir / "results.tsv").write_text(header + "".join(rows))
    # Need config.yaml present so --update works.
    (automil_dir / "config.yaml").write_text("placeholder: true\n")

    def _fake_run(argv, **kwargs):
        if "mig.mode.current" in str(argv):
            return MagicMock(stdout="Disabled\n", returncode=0, stderr="")
        return MagicMock(stdout="0, 49140\n", returncode=0, stderr="")

    from automil.cli import main as cli_main
    runner = CliRunner()
    with patch("subprocess.run", side_effect=_fake_run):
        result = runner.invoke(cli_main, ["init", "--update"])
    assert result.exit_code == 0, result.output

    cfg = _read_rendered_config(automil_dir)
    import numpy
    expected = float(numpy.quantile(vram_values, 0.95))
    actual = cfg["orchestrator"]["default_vram_estimate_gb"]
    assert abs(actual - expected) <= 0.05, f"actual={actual} expected~{expected}"


def test_init_uses_conservative_default_below_10_samples(tmp_path, monkeypatch):
    """D-198 clause 2 + Pitfall 8 anti-acceptance #3: <10 rows -> conservative path."""
    project = _init_git_repo(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    automil_dir = project / "automil"
    automil_dir.mkdir()
    header = "node_id\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tprimary_value\tvram_gb\telapsed_min\tstatus\tdescription\n"
    rows = "".join(
        f"node_{i:04d}\t0.85\t0.80\t0.85\t0.82\t0.83\t{4.0 + i:.1f}\t10.0\tcompleted\trun\n"
        for i in range(5)  # only 5 rows
    )
    (automil_dir / "results.tsv").write_text(header + rows)
    (automil_dir / "config.yaml").write_text("placeholder: true\n")

    def _fake_run(argv, **kwargs):
        if "mig.mode.current" in str(argv):
            return MagicMock(stdout="Disabled\n", returncode=0, stderr="")
        return MagicMock(stdout="0, 49140\n", returncode=0, stderr="")

    from automil.cli import main as cli_main
    runner = CliRunner()
    with patch("subprocess.run", side_effect=_fake_run):
        result = runner.invoke(cli_main, ["init", "--update"])
    assert result.exit_code == 0, result.output

    cfg = _read_rendered_config(automil_dir)
    # min_vram = 48.0 (from 49140 MB / 1024); conservative = max(8.0, 48.0/8.0) = 8.0
    assert cfg["orchestrator"]["default_vram_estimate_gb"] == 8.0


def test_init_aborts_on_failed_detection_user_decline(tmp_path, monkeypatch):
    """D-198 clause 3 / STP-03: failed detection + user 'no' -> ClickException."""
    project = _init_git_repo(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")  # signals user expected GPU

    from automil.cli import main as cli_main
    runner = CliRunner()
    # All probes fail; click.confirm sees stdin "no" via runner.invoke(input="n\n").
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi missing")):
        result = runner.invoke(cli_main, ["init"], input="n\n")
    assert result.exit_code != 0
    assert "Healthcheck failed" in result.output or "declined" in result.output
