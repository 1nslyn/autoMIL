"""Coverage for `automil submit` registry.protected reject (REG-04 / REG-05 / D-33 / D-34)."""
from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


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


def _setup_project(tmp_path: Path, protected: list[str]) -> tuple[Path, Path]:
    _init_git_repo(tmp_path)
    from automil.cli import main
    runner = CliRunner()
    import os
    os.chdir(tmp_path)
    runner.invoke(main, ["init"])

    # Edit config.yaml to set protected.
    adir = tmp_path / "automil"
    cfg_path = adir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("registry", {})["protected"] = protected
    cfg_path.write_text(yaml.safe_dump(cfg))
    return tmp_path, adir


def _campaign_record() -> tuple[dict, dict]:
    cell = {
        "cell_id": "cohort__arm__task",
        "identity": {
            "dataset": "cohort", "task": "task", "encoder": "encoder",
            "arm": "arm", "seed": 42, "protocol_version": "preprint-v1",
        },
        "budget_identity": {"cell_id": "budget-123"},
        "commands": {"discovery": "python train.py --folds 0,1,2"},
    }
    canonical = json.dumps(
        cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    cell["cell_sha256"] = hashlib.sha256(canonical).hexdigest()
    manifest = {
        "schema_version": 4,
        "campaign_id": "campaign-v1",
        "cells": [cell],
    }
    campaign = {
        "campaign_id": "campaign-v1",
        "manifest": "campaign/manifest.json",
        "manifest_sha256": "set-by-caller",
        "cell_id": cell["cell_id"],
        "cell_sha256": cell["cell_sha256"],
        "budget_cell_id": "budget-123",
        "stage": "discovery",
        "protocol_version": "preprint-v1",
    }
    return manifest, campaign


def test_campaign_binding_requires_one_manifest_source_of_truth(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    bound = validate_campaign_binding(
        path,
        campaign,
        base_run_command="python train.py --folds 0,1,2",
        budget_cell_id="budget-123",
    )
    assert bound == campaign


def test_campaign_binding_preserves_legacy_manifest_compatibility(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    manifest["schema_version"] = 3
    manifest["cells"][0].pop("identity")
    cell = manifest["cells"][0]
    unhashed = {key: value for key, value in cell.items() if key != "cell_sha256"}
    canonical = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    cell["cell_sha256"] = hashlib.sha256(canonical).hexdigest()
    campaign["cell_sha256"] = cell["cell_sha256"]
    campaign.pop("protocol_version")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    bound = validate_campaign_binding(
        path,
        campaign,
        base_run_command="python train.py --folds 0,1,2",
        budget_cell_id="budget-123",
    )

    assert bound == campaign


def test_v4_campaign_cannot_omit_protocol_version(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    campaign.pop("protocol_version")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="protocol_version"):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python train.py --folds 0,1,2",
            budget_cell_id="budget-123",
        )


def test_v4_campaign_cannot_downgrade_by_omitting_identity_and_protocol(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    manifest["cells"][0].pop("identity")
    cell = manifest["cells"][0]
    unhashed = {key: value for key, value in cell.items() if key != "cell_sha256"}
    canonical = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    cell["cell_sha256"] = hashlib.sha256(canonical).hexdigest()
    campaign["cell_sha256"] = cell["cell_sha256"]
    campaign.pop("protocol_version")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="protocol_version"):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python train.py --folds 0,1,2",
            budget_cell_id="budget-123",
        )


def test_campaign_cannot_downgrade_by_omitting_schema_and_identity(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    manifest.pop("schema_version")
    manifest["cells"][0].pop("identity")
    cell = manifest["cells"][0]
    unhashed = {key: value for key, value in cell.items() if key != "cell_sha256"}
    canonical = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    cell["cell_sha256"] = hashlib.sha256(canonical).hexdigest()
    campaign["cell_sha256"] = cell["cell_sha256"]
    campaign.pop("protocol_version")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="schema_version"):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python train.py --folds 0,1,2",
            budget_cell_id="budget-123",
        )


def test_campaign_binding_rejects_protocol_version_drift(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    campaign["protocol_version"] = "preprint-v2"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol_version"):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python train.py --folds 0,1,2",
            budget_cell_id="budget-123",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cell_sha256", "0" * 64, "cell hash"),
        ("budget_cell_id", "other", "budget identity"),
        ("stage", "promotion", "stage command"),
    ],
)
def test_campaign_binding_rejects_config_manifest_drift(
    tmp_path, field, value, message,
):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    campaign[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python train.py --folds 0,1,2",
            budget_cell_id="budget-123",
        )


def test_campaign_binding_rejects_command_drift(tmp_path):
    from automil.admissibility import validate_campaign_binding

    manifest, campaign = _campaign_record()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="base run command"):
        validate_campaign_binding(
            path,
            campaign,
            base_run_command="python other.py",
            budget_cell_id="budget-123",
        )


def test_protected_glob_match_rejects(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_project(tmp_path, ["benchmarks/lib/CLAM/**"])
    monkeypatch.chdir(proj)

    # Create the file we'll try to submit.
    target = proj / "benchmarks" / "lib" / "CLAM"
    target.mkdir(parents=True)
    (target / "foo.py").write_text("# whatever\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "test",
               "--files", "benchmarks/lib/CLAM/foo.py"],
    )
    assert result.exit_code != 0, result.output
    assert "Refusing to submit" in result.output
    assert "registry.protected" in result.output
    assert "benchmarks/lib/CLAM" in result.output
    assert "revert-baseline" in result.output


def test_multiple_matched_patterns_named(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_project(
        tmp_path,
        ["benchmarks/**", "src/lib/**"],
    )
    monkeypatch.chdir(proj)

    target = proj / "benchmarks" / "lib"
    target.mkdir(parents=True)
    (target / "x.py").write_text("# x\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "benchmarks/lib/x.py"],
    )
    assert result.exit_code != 0
    assert "benchmarks/**" in result.output


def test_protected_exact_path_rejects(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_project(tmp_path, ["src/foo.py"])
    monkeypatch.chdir(proj)

    (proj / "src").mkdir()
    (proj / "src" / "foo.py").write_text("# foo\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "src/foo.py"],
    )
    assert result.exit_code != 0
    assert "src/foo.py" in result.output


def test_non_matching_path_not_rejected_on_protected(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_project(tmp_path, ["benchmarks/**"])
    monkeypatch.chdir(proj)

    (proj / "src").mkdir()
    (proj / "src" / "main.py").write_text("# main\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "src/main.py"],
    )
    # The submit may still succeed or fail for other reasons (e.g., git
    # tracking). The protected branch must NOT be the cause.
    assert "registry.protected" not in result.output


def test_empty_protected_no_reject(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_project(tmp_path, [])
    monkeypatch.chdir(proj)

    (proj / "any.py").write_text("# any\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "any.py"],
    )
    assert "registry.protected" not in result.output


def test_no_force_flag_d34(tmp_path, cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["submit", "--force", "--node", "x", "--desc", "y"])
    assert result.exit_code != 0
    # Click's default error for unknown flags.
    assert "no such option" in result.output.lower() or "--force" in result.output


def test_submit_help_does_not_mention_force(cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["submit", "--help"])
    assert result.exit_code == 0
    assert "--force" not in result.output
    # Existing flags still listed.
    assert "--node" in result.output
    assert "--desc" in result.output


def test_good_error_message_names_pattern_and_suggests_fix(tmp_path, cli_runner, monkeypatch):
    """Production-grade: error names what + why + how-to-fix."""
    proj, adir = _setup_project(tmp_path, ["benchmarks/lib/**"])
    monkeypatch.chdir(proj)
    target = proj / "benchmarks" / "lib"
    target.mkdir(parents=True)
    (target / "x.py").write_text("# x\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "benchmarks/lib/x.py"],
    )
    # Three required substrings:
    # 1. WHAT: "Refusing to submit"
    assert "Refusing to submit" in result.output
    # 2. WHY: "registry.protected" + named pattern
    assert "registry.protected" in result.output
    assert "benchmarks/lib/" in result.output
    # 3. HOW: suggestion of revert-baseline
    assert "revert-baseline" in result.output


def test_protected_reject_runs_before_path_validation(tmp_path, cli_runner, monkeypatch):
    """Protected-files reject fires BEFORE the existing path-validation guard.

    The existing guard at submit.py line 179 rejects absolute paths.
    If a protected glob matches an absolute path AND the submit runs the
    protected check first, the error should be the protected message, not
    the path-validation message.  We use a glob that matches all paths under
    /etc/ to test this.
    """
    proj, adir = _setup_project(tmp_path, ["/etc/**"])
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main, ["submit", "--node", "node_0001", "--desc", "t",
               "--files", "/etc/passwd"],
    )
    assert result.exit_code != 0
    # With protected matching, protected error message fires first
    # (or at minimum, the path-validation error fires — both are non-zero).
    # The key invariant is exit_code != 0; the exact message depends on
    # whether the protected check or path-validation check runs first.
    # This test verifies the ordering: protected BEFORE path-validation.
    assert "Refusing to submit" in result.output
    # If protected check fires first, the output contains "registry.protected".
    # If path-validation fires first (violation!), the output does NOT contain
    # "registry.protected" and does contain "must be relative".
    # The test enforces that the protected message is present.
    assert "registry.protected" in result.output


def _enable_recipe_only_mode(adir: Path) -> None:
    cfg_path = adir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("registry", {}).update({
        "mode": "architecture-preserving",
        "protected": ["models/**", "evaluate.py"],
        "allowed_override_options": ["--hparams", "--policy-variant"],
        "allowed_variant_kinds": ["policy"],
    })
    cfg.setdefault("files", {})["editable"] = ["recipes/**"]
    cfg.setdefault("run", {})["mil_model"] = "clam_mb"
    cfg_path.write_text(yaml.safe_dump(cfg))


def test_architecture_preserving_explicit_files_cannot_escape_editable(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    monkeypatch.chdir(proj)
    (proj / "train.py").write_text("# unauthorized trainer edit\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "escape",
         "--files", "train.py", "--mil-model", "clam_mb"],
    )
    assert result.exit_code != 0
    assert "files.editable" in result.output
    assert "protected-surface-violation" in result.output


def test_architecture_preserving_submit_persists_train_only_verdict(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    monkeypatch.chdir(proj)
    recipe = proj / "recipes" / "cosine.py"
    recipe.parent.mkdir()
    recipe.write_text("NAME = 'cosine'\n")

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "recipe",
         "--files", "recipes/cosine.py", "--mil-model", "clam_mb"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    spec = json.loads(
        (adir / "orchestrator" / "queue" / "node_0001.json").read_text()
    )
    assert spec["admissibility"]["candidate_class"] == "train-only-source"
    assert spec["admissibility"]["accepted"] is True
    assert spec["admissibility"]["files"] == ["recipes/cosine.py"]
    assert spec["admissibility"]["policy_hash"]


def test_architecture_preserving_hparam_only_submit_is_config_only(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "hp",
         "--override", "--hparams '{\"lr\":0.0001}'", "--mil-model", "clam_mb"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    spec = json.loads(
        (adir / "orchestrator" / "queue" / "node_0001.json").read_text()
    )
    assert spec["overlay_manifest"] == {}
    assert spec["admissibility"]["candidate_class"] == "config-only"
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=proj, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    expected_hash = hashlib.sha256(
        (base_commit + "\nOVERRIDE:--hparams '{\"lr\":0.0001}'").encode()
    ).hexdigest()[:16]
    assert spec["graph_metadata"]["config_hash"] == expected_hash


def test_architecture_preserving_forbidden_command_override_is_rejected(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "identity escape",
         "--override", "--dataset other", "--mil-model", "clam_mb"],
    )
    assert result.exit_code != 0
    assert "allowed_override_options" in result.output


def test_architecture_preserving_policy_module_and_selector_form_one_candidate(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    cfg_path = adir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["files"]["editable"] = ["automil/variants/_policies/*.py"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    policy = adir / "variants" / "_policies" / "identity.py"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text('''
from automil.registry import PolicyVariant, VariantSpec, register
@register(VariantSpec(
    name="identity", kind="policy", parent=None, base_commit="abc",
    composite=0.5, node_id="node_0001",
    created_at="2026-08-02T00:00:00+00:00",
))
class Identity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
''')
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "policy source",
         "--files", "automil/variants/_policies/identity.py",
         "--override", "--policy-variant identity", "--mil-model", "clam_mb"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    spec = json.loads(
        (adir / "orchestrator" / "queue" / "node_0001.json").read_text()
    )
    assert spec["admissibility"]["candidate_class"] == "train-only-source"
    assert spec["run_command_override"] == "--policy-variant identity"
    assert "automil/variants/_policies/identity.py" in spec["overlay_manifest"]


def test_architecture_preserving_policy_source_without_selector_is_rejected(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    cfg_path = adir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["files"]["editable"] = ["automil/variants/_policies/*.py"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    policy = adir / "variants" / "_policies" / "identity.py"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text('''
from automil.registry import PolicyVariant, VariantSpec, register
@register(VariantSpec(
    name="identity", kind="policy", parent=None, base_commit="abc",
    composite=0.5, node_id="node_0001",
    created_at="2026-08-02T00:00:00+00:00",
))
class Identity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
''')
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "inert policy",
         "--files", "automil/variants/_policies/identity.py",
         "--mil-model", "clam_mb"],
    )
    assert result.exit_code != 0
    assert "would execute as a no-op" in result.output


def test_architecture_preserving_policy_selectors_cannot_disagree(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    (adir / "active_variant.json").write_text(json.dumps({
        "model": {"variant": None},
        "loss": {"variant": None},
        "policy": {"variant": "lookahead"},
    }))
    monkeypatch.chdir(proj)

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "ambiguous policy",
         "--override", "--policy-variant cosine", "--mil-model", "clam_mb"],
    )
    assert result.exit_code != 0
    assert "disagrees with active_variant.json" in result.output


def test_architecture_preserving_stale_model_variant_cannot_ride_next_submit(
    tmp_path, cli_runner, monkeypatch,
):
    proj, adir = _setup_project(tmp_path, ["models/**"])
    _enable_recipe_only_mode(adir)
    monkeypatch.chdir(proj)
    (adir / "active_variant.json").write_text(json.dumps({
        "model": {"variant": "clam_as_abmil", "parent": "clam_mb"},
        "loss": {"variant": None},
        "policy": {"variant": None},
    }))

    from automil.cli import main
    result = cli_runner.invoke(
        main,
        ["submit", "--node", "node_0001", "--desc", "variant escape",
         "--override", "--hparams '{\"lr\":0.0001}'", "--mil-model", "clam_mb"],
    )
    assert result.exit_code != 0
    assert "allowed_variant_kinds" in result.output
    assert "model" in result.output
