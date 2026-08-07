"""H-3b × H-4 end to end: the two fixes have to compose, or neither works.

They were designed to be load-bearing for each other and it is worth proving,
because the failure mode is a paper claim rather than a crash:

* **H-4 alone would over-freeze.** Protecting ``pipeline/config.py`` and
  ``run_experiment.py`` — the split entry point, the composite writer and the
  fold/seed dataclasses — takes away the only route an agent had to change a
  learning rate, because the shared transport was the only hyperparameter
  channel that existed. A search that cannot tune is not a search.
* **H-3b alone would under-protect.** Giving every arm an explicit
  ``--hparams`` channel does nothing about the *other* channel: free-mode file
  editing, which reaches ``splits.py`` and the composite writer, and which the
  agent skill actively instructs.

The architecture-preserving campaign now draws the line more narrowly:
**published model/trainer/measurement code is frozen; registered train-only
PolicyVariant source and declared scalar overrides remain open.** These tests
assert both halves against the real ``automil submit`` path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main

#: The paths the roster overlays freeze. Kept in sync by
#: ``test_matches_the_shipped_roster_list`` below rather than by hand.
_SUBSTRATE = [
    "benchmarks/src/autobench/pipeline/splits.py",
    "benchmarks/scripts/run_experiment.py",
    "benchmarks/src/autobench/pipeline/config.py",
    "benchmarks/datasets/tcga/tcga_luad.yaml",
]

#: Published identity surfaces that preserving mode must now refuse.
_IDENTITY_SURFACE = [
    "benchmarks/src/autobench/pipeline/clam/train.py",
    "benchmarks/lib/CLAM/models/model_clam.py",
]

_POLICY = "benchmarks/experiments/tcga_luad/automil/variants/_policies/identity.py"


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _roster_registry_block() -> dict:
    """Read the real list off a shipped roster overlay — never a copy of it."""
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo / "benchmarks/experiments/tcga_luad/automil/config.yaml").read_text()
    )
    return cfg["registry"]


def _roster_editable() -> list[str]:
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo / "benchmarks/experiments/tcga_luad/automil/config.yaml").read_text()
    )
    return cfg["files"]["editable"]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project carrying the roster cohorts' real protected list."""
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(main, ["init"]).exit_code == 0

    adir = tmp_path / "automil"
    cfg = yaml.safe_load((adir / "config.yaml").read_text()) or {}
    cfg["registry"] = _roster_registry_block()
    cfg.setdefault("files", {})["editable"] = _roster_editable()
    cfg["project"] = {**(cfg.get("project") or {}), "name": "tcga_luad"}
    cfg["encoders"] = {**(cfg.get("encoders") or {}), "primary": "uni_v2"}
    cfg["task"] = {**(cfg.get("task") or {}), "name": "kras"}
    cfg.setdefault("cap", {})["mode"] = "wall_clock"
    (adir / "config.yaml").write_text(yaml.safe_dump(cfg))

    for rel in _SUBSTRATE + _IDENTITY_SURFACE:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# original\n")
    policy = tmp_path / _POLICY
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text('''
from automil.registry import PolicyVariant, VariantSpec, register
@register(VariantSpec(
    name="identity", kind="policy", parent=None, base_commit="abc",
    composite=0.5, node_id="n_0001",
    created_at="2026-08-02T00:00:00+00:00",
))
class Identity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
''')
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, capture_output=True, check=True)
    return runner, tmp_path


def _submit(runner, node, rel):
    return runner.invoke(
        main, ["submit", "--node", node, "--desc", "d", "--files", rel],
    )


class TestSubstrateEditsAreRefused:
    @pytest.mark.parametrize("rel", _SUBSTRATE)
    def test_a_protected_file_cannot_be_submitted(self, project, rel):
        runner, root = project
        (root / rel).write_text("# tampered\n")
        result = _submit(runner, "n_0001", rel)
        assert result.exit_code != 0
        assert "protected" in result.output.lower()

    def test_the_refusal_names_the_matching_pattern(self, project):
        runner, root = project
        rel = "benchmarks/src/autobench/pipeline/splits.py"
        (root / rel).write_text("# tampered\n")
        out = _submit(runner, "n_0001", rel).output
        assert "splits.py" in out

    def test_there_is_no_force_escape(self, project):
        """Phase 1 deliberately ships no --force for protected files."""
        runner, root = project
        rel = "benchmarks/src/autobench/pipeline/splits.py"
        (root / rel).write_text("# tampered\n")
        result = runner.invoke(
            main, ["submit", "--node", "n_0001", "--desc", "d", "--files", rel, "--force"],
        )
        assert result.exit_code != 0

    def test_a_dataset_yaml_is_refused(self, project):
        """Labels and feature roots live here; it was in NO list before H-4 —
        not editable, not readonly, not protected — so it submitted silently."""
        runner, root = project
        rel = "benchmarks/datasets/tcga/tcga_luad.yaml"
        (root / rel).write_text("name: tampered\n")
        assert _submit(runner, "n_0001", rel).exit_code != 0


class TestTrainOnlyPolicyStaysOpen:
    """The non-HP source seam stays open without changing model identity."""

    @pytest.mark.parametrize("rel", _IDENTITY_SURFACE)
    def test_architecture_and_training_loop_are_refused(self, project, rel):
        runner, root = project
        (root / rel).write_text("# a new idea\n")
        result = _submit(runner, "n_0001", rel)
        assert result.exit_code != 0
        assert "files.editable" in result.output

    def test_registered_policy_module_is_submittable(self, project):
        runner, root = project
        result = runner.invoke(
            main,
            ["submit", "--node", "n_0001", "--desc", "d", "--files", _POLICY,
             "--override", "--policy-variant identity"],
        )
        assert result.exit_code == 0, result.output


class TestTheListIsTheShippedOne:
    def test_matches_the_shipped_roster_list(self):
        """Guards the fixture: if a roster overlay's protected list changes, this
        test's _SUBSTRATE constants must be re-checked rather than silently
        drifting into testing a list nobody ships."""
        from automil.cli._helpers import _matches_scope

        patterns = list(_roster_registry_block()["protected"])
        for rel in _SUBSTRATE:
            assert _matches_scope(rel, patterns), f"{rel} is no longer protected"
        for rel in _IDENTITY_SURFACE:
            assert not _matches_scope(rel, patterns), (
                f"{rel} unexpectedly entered registry.protected; the hard editable "
                "allowlist is the intended identity-surface boundary"
            )
        assert _matches_scope(_POLICY, _roster_editable())

    def test_every_roster_overlay_ships_the_same_list(self):
        repo = Path(__file__).resolve().parents[1] / "benchmarks" / "experiments"
        lists = {}
        for cohort in ("tcga_luad", "tcga_lgg", "cptac_gbm", "cptac_pdac", "tcga_hnsc"):
            cfg = yaml.safe_load((repo / cohort / "automil" / "config.yaml").read_text())
            lists[cohort] = tuple(cfg["registry"]["protected"])
        assert len(set(lists.values())) == 1, (
            f"roster cohorts freeze different things: "
            f"{ {k: len(v) for k, v in lists.items()} }"
        )
