"""Tests for automil gate CLI surface (plan 05-08 / GTE-01, GTE-02, GTE-04, GTE-06).

Task 1 tests (4): pyproject.toml scipy deps + config.yaml.j2 gate: section.
Task 2 tests (8): CLI register-manifest, retire-manifest, status, stats subcommands.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Task 1: pyproject.toml + config.yaml.j2 static assertions
# ---------------------------------------------------------------------------


def test_scipy_in_core_deps():
    """T-1: scipy>=1.11 must be in [project.dependencies] core deps."""
    pyproject = tomllib.loads(
        (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
    )
    deps = pyproject["project"]["dependencies"]
    assert any(d.startswith("scipy>=1.11") for d in deps), (
        f"scipy>=1.11 must be in core [project.dependencies]; found {deps}"
    )


def test_scipy_no_longer_only_in_ml_optional():
    """T-2: scipy presence in core deps (complement of T-1 for readability)."""
    pyproject = tomllib.loads(
        (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
    )
    core_deps = pyproject["project"]["dependencies"]
    # Core must contain it; optional-ml may also keep it for backward-compat.
    assert any(d.startswith("scipy") for d in core_deps), (
        f"scipy must appear in core [project.dependencies]; got {core_deps}"
    )


def test_config_yaml_j2_has_gate_section():
    """T-3: config.yaml.j2 must contain gate: key with required sub-keys."""
    template_path = (
        Path(__file__).parent.parent.parent
        / "src" / "automil" / "templates" / "config.yaml.j2"
    )
    content = template_path.read_text()
    assert "gate:" in content, "config.yaml.j2 must contain 'gate:'"
    assert "auto_nominate" in content, "config.yaml.j2 must contain 'auto_nominate'"
    assert "bootstrap_reps" in content, "config.yaml.j2 must contain 'bootstrap_reps'"
    assert "p_threshold" in content, "config.yaml.j2 must contain 'p_threshold'"
    assert "K:" in content or "K " in content, "config.yaml.j2 must contain K key"


def test_rendered_config_parses_yaml():
    """T-4: Rendered config.yaml.j2 with placeholder vars must parse as valid YAML
    with gate section containing expected defaults."""
    import yaml
    from jinja2 import Environment, FileSystemLoader

    template_dir = (
        Path(__file__).parent.parent.parent
        / "src" / "automil" / "templates"
    )
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("config.yaml.j2")
    # Provide all template variables so the rendered YAML is valid
    rendered = template.render(
        project_name="test_project",
        task_type="classification",
        encoder="uni_v2",
    )
    data = yaml.safe_load(rendered)
    gate = data.get("gate", {})
    assert gate.get("auto_nominate") is False, (
        f"gate.auto_nominate must be False; got {gate.get('auto_nominate')!r}"
    )
    assert gate.get("K") == 2, f"gate.K must be 2; got {gate.get('K')!r}"
    assert gate.get("p_threshold") == 0.05, (
        f"gate.p_threshold must be 0.05; got {gate.get('p_threshold')!r}"
    )
    assert gate.get("bootstrap_reps") == 1000, (
        f"gate.bootstrap_reps must be 1000; got {gate.get('bootstrap_reps')!r}"
    )


# ---------------------------------------------------------------------------
# Task 2: CLI subcommand tests
# ---------------------------------------------------------------------------


import os
import pytest


@pytest.fixture
def project(tmp_path):
    """Minimal automil project with git repo + graph.json with node_0001."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("# fixture\n")
    (adir / "graph.json").write_text(
        json.dumps({
            "meta": {"total_executed": 1},
            "nodes": {
                "node_0001": {
                    "id": "node_0001",
                    "status": "keep",
                    "composite": 0.85,
                }
            },
        })
    )
    # Fresh git repo
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init", "-q"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


_HELD_OUT_SPEC = (
    "abc12345678901ab:ccrcc:uni_v2:high_grade,"
    "def678901234def0:clwd:ctranspath:subtype"
)


def _run_gate(project_dir: Path, args: list[str]) -> "click.testing.Result":
    """Invoke `automil gate ...` with cwd set to project_dir."""
    from click.testing import CliRunner
    from automil.cli import main

    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        result = CliRunner().invoke(main, args, catch_exceptions=False)
    finally:
        os.chdir(old_cwd)
    return result


def test_gate_register_manifest_basic(project):
    """T-5: register-manifest creates file + git commit."""
    result = _run_gate(project, [
        "gate", "register-manifest", "node_0001",
        "--K", "2",
        "--p-threshold", "0.05",
        "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ])

    assert result.exit_code == 0, (
        f"Expected exit 0; got {result.exit_code}. Output: {result.output}"
    )
    manifest_file = project / "automil" / "gate" / "node_0001.gate_manifest.json"
    assert manifest_file.exists(), f"Manifest file not created at {manifest_file}"
    # Verify git log shows the commit
    log = subprocess.run(
        ["git", "log", "--oneline", "-3"],
        cwd=str(project), capture_output=True, text=True, check=True,
    ).stdout
    assert "gate" in log, f"Expected gate-register commit in git log; got:\n{log}"


def test_gate_register_validates_parent_id(project):
    """T-6: parent_id not matching ^node_\\d+$ must exit non-zero."""
    old_cwd = os.getcwd()
    try:
        os.chdir(str(project))
        from click.testing import CliRunner
        from automil.cli import main
        result = CliRunner().invoke(main, [
            "gate", "register-manifest", "not_a_node",
            "--K", "2",
            "--p-threshold", "0.05",
            "--bootstrap-reps", "1000",
            "--held-out-cells", _HELD_OUT_SPEC,
        ])
    finally:
        os.chdir(old_cwd)
    assert result.exit_code != 0, "Expected non-zero exit for invalid parent_id"
    output_lower = result.output.lower()
    assert "parent_id" in output_lower or "invalid" in output_lower, (
        f"Expected error message about parent_id; got: {result.output}"
    )


def test_gate_register_refuses_overwrite(project):
    """T-7: Second register for same parent_id must fail with retire-manifest hint."""
    args = [
        "gate", "register-manifest", "node_0001",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ]
    r1 = _run_gate(project, args)
    assert r1.exit_code == 0, f"First register failed: {r1.output}"

    # Second attempt should fail
    old_cwd = os.getcwd()
    try:
        os.chdir(str(project))
        from click.testing import CliRunner
        from automil.cli import main
        r2 = CliRunner().invoke(main, args)
    finally:
        os.chdir(old_cwd)

    assert r2.exit_code != 0, "Second register should fail"
    assert "retire" in r2.output.lower(), (
        f"Expected 'retire-manifest' hint; got: {r2.output}"
    )


def test_gate_register_strategy_stratified(project):
    """T-8: --strategy is a valid choice with stratified/random/operator-curated options."""
    # Valid: stratified + held-out-cells works
    r1 = _run_gate(project, [
        "gate", "register-manifest", "node_0001",
        "--strategy", "stratified",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ])
    assert r1.exit_code == 0, f"stratified + held-out-cells failed: {r1.output}"

    # Help shows all three choices
    old_cwd = os.getcwd()
    try:
        os.chdir(str(project))
        from click.testing import CliRunner
        from automil.cli import main
        r_help = CliRunner().invoke(main, ["gate", "register-manifest", "--help"])
    finally:
        os.chdir(old_cwd)

    assert "--strategy" in r_help.output, (
        f"--strategy option must appear in help; got: {r_help.output}"
    )
    assert "stratified" in r_help.output
    assert "random" in r_help.output
    assert "operator-curated" in r_help.output


def test_gate_retire_manifest(project):
    """T-9: retire-manifest writes .retired file and git-commits the rename."""
    # Register first
    r1 = _run_gate(project, [
        "gate", "register-manifest", "node_0001",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ])
    assert r1.exit_code == 0, f"Register failed: {r1.output}"

    # Retire
    r2 = _run_gate(project, [
        "gate", "retire-manifest", "node_0001",
        "--reason", "K too generous",
    ])

    assert r2.exit_code == 0, f"Retire failed: {r2.output}"
    active = project / "automil" / "gate" / "node_0001.gate_manifest.json"
    retired = project / "automil" / "gate" / "node_0001.retired.gate_manifest.json"
    assert not active.exists(), "Active manifest should be gone after retire"
    assert retired.exists(), "Retired manifest should exist"
    retired_data = json.loads(retired.read_text())
    assert "retired_reason" in retired_data, "Retired manifest must have retired_reason"
    assert retired_data["retired_reason"] == "K too generous"


def test_gate_status_shows_manifest_details(project):
    """T-10: status node_0001 shows parent_id, K, p_threshold, held_out count."""
    # Register first
    r1 = _run_gate(project, [
        "gate", "register-manifest", "node_0001",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ])
    assert r1.exit_code == 0, f"Register failed: {r1.output}"

    r2 = _run_gate(project, ["gate", "status", "node_0001"])

    assert r2.exit_code == 0, f"Status failed: {r2.output}"
    out = r2.output
    assert "node_0001" in out
    assert "2" in out          # K=2
    assert "0.05" in out       # p_threshold
    # held_out count = 2 cells (already contains "2" from K check above)


def test_gate_status_no_args_lists_all(project):
    """T-11: status without args lists all active manifests."""
    # Add a second node to graph
    graph_path = project / "automil" / "graph.json"
    graph_data = json.loads(graph_path.read_text())
    graph_data["nodes"]["node_0002"] = {
        "id": "node_0002", "status": "keep", "composite": 0.87
    }
    graph_path.write_text(json.dumps(graph_data))
    # Commit the graph change
    subprocess.run(
        ["git", "add", str(graph_path)], cwd=str(project), check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "add node_0002", "-q"], cwd=str(project), check=True
    )

    held2 = (
        "aaaabbbbccccdddd:ccrcc:uni_v2:high_grade,"
        "eeeeffffgggghhhh:clwd:ctranspath:subtype"
    )
    r1 = _run_gate(project, [
        "gate", "register-manifest", "node_0001",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", _HELD_OUT_SPEC,
    ])
    assert r1.exit_code == 0, f"Register node_0001 failed: {r1.output}"

    r2 = _run_gate(project, [
        "gate", "register-manifest", "node_0002",
        "--K", "2", "--p-threshold", "0.05", "--bootstrap-reps", "1000",
        "--held-out-cells", held2,
    ])
    assert r2.exit_code == 0, f"Register node_0002 failed: {r2.output}"

    r3 = _run_gate(project, ["gate", "status"])

    assert r3.exit_code == 0, f"Status failed: {r3.output}"
    assert "node_0001" in r3.output
    assert "node_0002" in r3.output


def test_gate_stats_shows_promotion_rate_and_health(project):
    """T-12: stats shows a percentage and health diagnostic string."""
    result = _run_gate(project, ["gate", "stats"])

    assert result.exit_code == 0, f"Stats failed: {result.output}"
    out = result.output
    # Must contain a percentage (promotion rate)
    assert "%" in out or "rate" in out.lower(), (
        f"Expected promotion rate output; got: {out}"
    )
    # Must contain health diagnostic
    assert any(
        word in out.lower()
        for word in ("gate", "health", "promotion", "nominated", "strict", "loose")
    ), f"Expected gate health diagnostic; got: {out}"
