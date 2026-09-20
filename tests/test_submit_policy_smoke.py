"""`registry.policy_smoke`: run a consumer-owned smoke test on every variant
module at submit, so a policy that crashes the trainer is refused for free
instead of charging an attempt (two of the five rehearsal cells lost an
attempt to a policy that failed on the seam's call order).

The framework knows nothing about torch: it runs the declared command with
the module path appended, in a subprocess with a timeout, and refuses on a
non-zero exit or a timeout with the command's own diagnostics.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.registry.config import load_registry_config
from automil.registry.errors import ValidationError
from automil.registry.validators.smoke import run_policy_smoke

VARIANT = '''"""v0001 variant."""
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

OK_SCRIPT = '''import sys, pathlib
pathlib.Path(sys.argv[2]).write_text(sys.argv[1])
'''
FAIL_SCRIPT = '''import sys
print("boom: zero_grad restored weights in place", file=sys.stderr)
sys.exit(1)
'''
HANG_SCRIPT = '''import time
time.sleep(30)
'''


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


def _project(tmp_path: Path, monkeypatch, smoke: dict | None) -> tuple[CliRunner, Path]:
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(main, ["init"]).exit_code == 0
    adir = tmp_path / "automil"
    if smoke is not None:
        config_path = adir / "config.yaml"
        cfg = yaml.safe_load(config_path.read_text()) or {}
        cfg.setdefault("registry", {})["policy_smoke"] = smoke
        config_path.write_text(yaml.safe_dump(cfg))
    v_dir = adir / "variants" / "clam_mb"
    v_dir.mkdir(parents=True, exist_ok=True)
    (v_dir / "v0001.py").write_text(VARIANT)
    return runner, adir


def _submit(runner):
    return runner.invoke(main, ["submit", "--node", "node_0001", "--desc", "t",
                               "--files", "automil/variants/clam_mb/v0001.py"])


class TestRegistryConfig:
    def test_declaration_is_parsed(self, tmp_path):
        adir = tmp_path / "automil"
        adir.mkdir()
        (adir / "config.yaml").write_text(yaml.safe_dump({"registry": {
            "policy_smoke": {"command": ["{python}", "-m", "x"], "timeout_s": 90}}}))
        smoke = load_registry_config(adir).policy_smoke
        assert smoke.command == ("{python}", "-m", "x") and smoke.timeout_s == 90

    def test_absent_declaration_is_none(self, tmp_path):
        adir = tmp_path / "automil"
        adir.mkdir()
        (adir / "config.yaml").write_text(yaml.safe_dump({"registry": {"mode": "free"}}))
        assert load_registry_config(adir).policy_smoke is None

    @pytest.mark.parametrize("bad", [
        {"command": [], "timeout_s": 10},
        {"command": "python -m x", "timeout_s": 10},
        {"command": ["python"], "timeout_s": 0},
        {"command": ["python"]},
    ])
    def test_malformed_declaration_is_refused(self, tmp_path, bad):
        adir = tmp_path / "automil"
        adir.mkdir()
        (adir / "config.yaml").write_text(yaml.safe_dump({"registry": {"policy_smoke": bad}}))
        with pytest.raises((TypeError, ValueError)):
            load_registry_config(adir)


class TestRunPolicySmoke:
    def test_passing_command_gets_the_module_path_and_python(self, tmp_path):
        script = tmp_path / "ok.py"
        script.write_text(OK_SCRIPT)
        marker = tmp_path / "marker.txt"
        module = tmp_path / "policy.py"
        module.write_text("# policy\n")
        smoke = load_registry_config_from({"command": ["{python}", str(script), "{module}", str(marker)],
                                           "timeout_s": 10}, tmp_path)
        run_policy_smoke(module, smoke, cwd=tmp_path)
        assert marker.read_text() == str(module)

    def test_failing_command_is_a_validation_error_carrying_its_stderr(self, tmp_path):
        script = tmp_path / "fail.py"
        script.write_text(FAIL_SCRIPT)
        module = tmp_path / "policy.py"
        module.write_text("# policy\n")
        smoke = load_registry_config_from({"command": ["{python}", str(script)], "timeout_s": 10}, tmp_path)
        with pytest.raises(ValidationError) as excinfo:
            run_policy_smoke(module, smoke, cwd=tmp_path)
        assert excinfo.value.validator_name == "smoke"
        assert "zero_grad restored weights" in excinfo.value.reason

    def test_timeout_is_a_validation_error(self, tmp_path):
        script = tmp_path / "hang.py"
        script.write_text(HANG_SCRIPT)
        module = tmp_path / "policy.py"
        module.write_text("# policy\n")
        smoke = load_registry_config_from({"command": ["{python}", str(script)], "timeout_s": 1}, tmp_path)
        with pytest.raises(ValidationError) as excinfo:
            run_policy_smoke(module, smoke, cwd=tmp_path)
        assert "timed out" in excinfo.value.reason


def load_registry_config_from(smoke: dict, tmp_path: Path):
    adir = tmp_path / f"automil-{abs(hash(str(smoke))) % 10**6}"
    adir.mkdir()
    (adir / "config.yaml").write_text(yaml.safe_dump({"registry": {"policy_smoke": smoke}}))
    return load_registry_config(adir).policy_smoke


class TestSubmitRunsTheSmoke:
    def test_a_failing_smoke_refuses_the_submission(self, tmp_path, monkeypatch):
        script = tmp_path / "fail.py"
        script.write_text(FAIL_SCRIPT)
        runner, adir = _project(tmp_path, monkeypatch,
                                {"command": ["{python}", str(script)], "timeout_s": 10})
        result = _submit(runner)
        assert result.exit_code != 0
        assert "smoke" in result.output and "zero_grad restored weights" in result.output
        assert not (adir / "orchestrator" / "queue" / "node_0001.json").exists()

    def test_a_passing_smoke_lets_the_submission_through(self, tmp_path, monkeypatch):
        script = tmp_path / "ok.py"
        script.write_text(OK_SCRIPT)
        marker = tmp_path / "marker.txt"
        runner, adir = _project(tmp_path, monkeypatch,
                                {"command": ["{python}", str(script), "{module}", str(marker)],
                                 "timeout_s": 10})
        result = _submit(runner)
        assert "smoke" not in result.output
        assert marker.read_text().endswith("automil/variants/clam_mb/v0001.py")

    def test_no_declaration_runs_nothing(self, tmp_path, monkeypatch):
        runner, adir = _project(tmp_path, monkeypatch, None)
        result = _submit(runner)
        assert "smoke" not in result.output
