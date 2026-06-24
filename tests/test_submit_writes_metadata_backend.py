"""Coverage for `automil submit` writing metadata.backend to queue spec (D-76, BCK-01 prereq).

Scenarios:
  1. Default config (no backend.name key) → metadata.backend == "local"
  2. Config with backend.name: "mock_slurm" → metadata.backend == "mock_slurm"
  3. No config.yaml at all → metadata.backend == "local"
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main


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


def _do_submit(tmp_path: Path, cli_runner: CliRunner) -> dict:
    """Run `automil submit` with a single model.py file and return the queue spec dict."""
    (tmp_path / "model.py").write_text("print('changed')\n")
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "test metadata.backend",
         "--files", "model.py", "--mil-model", "test_model"],  # D-12: required
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
    assert len(queue_files) == 1
    return json.loads(queue_files[0].read_text())


class TestSubmitWritesMetadataBackend:
    """D-76: queue/<id>.json must contain metadata.backend at submit time."""

    def test_default_config_yields_local_backend(self, cli_runner, tmp_path, monkeypatch):
        """When config.yaml has no backend.name key, metadata.backend defaults to 'local'."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        spec = _do_submit(tmp_path, cli_runner)

        assert "metadata" in spec, "spec must contain a 'metadata' key"
        assert spec["metadata"]["backend"] == "local", (
            f"Expected 'local', got {spec['metadata']['backend']!r}"
        )

    def test_config_with_backend_name_propagated(self, cli_runner, tmp_path, monkeypatch):
        """When config.yaml has backend.name: 'mock_slurm', metadata.backend == 'mock_slurm'."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Inject backend.name into config.yaml
        cfg_path = tmp_path / "automil" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        cfg["backend"] = {"name": "mock_slurm"}
        cfg_path.write_text(yaml.safe_dump(cfg))

        spec = _do_submit(tmp_path, cli_runner)

        assert "metadata" in spec, "spec must contain a 'metadata' key"
        assert spec["metadata"]["backend"] == "mock_slurm", (
            f"Expected 'mock_slurm', got {spec['metadata']['backend']!r}"
        )

    def test_no_opaque_id_at_submit_time(self, cli_runner, tmp_path, monkeypatch):
        """D-76: opaque_id must NOT be written to queue spec at submit time (daemon writes it)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        spec = _do_submit(tmp_path, cli_runner)

        metadata = spec.get("metadata", {})
        assert "opaque_id" not in metadata, (
            "opaque_id must not appear in queue spec — daemon writes it on launch (D-76)"
        )
