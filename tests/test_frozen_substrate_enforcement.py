"""H-4 / HASH-0: the frozen substrate must be enforced, not merely documented.

`PLAN.md` §5 makes the frozen data substrate the paper's central rigor claim: the
agent's equal-effort recipe search may change architecture, loss and training
procedure, but not splits, folds, labels or feature extraction. Three defects made
that claim documented-only:

1. **The gate is written ``if reg_cfg.protected and _matches_scope(...)``, so an
   empty tuple silently short-circuits it to a no-op** — and with it
   ``check.py``'s drift detection and ``revert-baseline``. Commit `0b2da55`
   populated `protected` on the three *template* projects and skipped `ccrcc`,
   the only one with a real ``run.command``.
2. **``registry.mode`` was parsed and never read by any code.** A project could
   declare ``architecture-preserving`` and get nothing.
3. **The overlay manifest hash was written and never verified.** ``submit``
   records ``sha256:...`` per file into the queue spec; no consumer checked it
   (``backends/local.py`` wrote empty strings, ``port_variant.py`` read only
   ``.keys()``). An archived overlay edited between submit and launch would run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from automil.admissibility import load_candidate_policy
from automil.registry.config import load_registry_config
from automil.runner import Runner


def _write_config(automil_dir: Path, registry: dict) -> None:
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text(yaml.safe_dump({"registry": registry}))


class TestArchitecturePreservingRequiresAProtectedList:
    """The mode was decorative: parsed, validated, and then read by nothing.

    Requiring a non-empty list makes it self-consistent — a project cannot
    announce that it preserves a substrate while naming nothing to preserve.
    """

    def test_mode_with_a_protected_list_loads(self, tmp_path):
        _write_config(tmp_path / "automil", {
            "mode": "architecture-preserving",
            "protected": ["benchmarks/src/autobench/pipeline/splits.py"],
        })
        cfg = load_registry_config(tmp_path / "automil")
        assert cfg.mode == "architecture-preserving"
        assert cfg.protected == ("benchmarks/src/autobench/pipeline/splits.py",)

    def test_mode_without_a_protected_list_raises(self, tmp_path):
        _write_config(tmp_path / "automil", {"mode": "architecture-preserving"})
        with pytest.raises(ValueError, match="protected"):
            load_registry_config(tmp_path / "automil")

    def test_mode_with_an_empty_protected_list_raises(self, tmp_path):
        _write_config(tmp_path / "automil", {
            "mode": "architecture-preserving", "protected": [],
        })
        with pytest.raises(ValueError, match="protected"):
            load_registry_config(tmp_path / "automil")

    def test_free_mode_with_no_protected_list_is_still_fine(self, tmp_path):
        """The framework ships no defaults (D-33/D-49) and generic projects are
        free by default; this must not become a hard requirement everywhere."""
        _write_config(tmp_path / "automil", {})
        cfg = load_registry_config(tmp_path / "automil")
        assert cfg.mode == "free" and cfg.protected == ()

    def test_missing_config_still_returns_defaults(self, tmp_path):
        assert load_registry_config(tmp_path / "nonexistent").protected == ()


class TestOverlayManifestIsVerified:
    """HASH-0: the digest recorded at submit time must actually be checked."""

    def _overlay(self, tmp_path: Path, content: bytes = b"print('hello')\n"):
        overlay = tmp_path / "archive" / "node-1"
        (overlay / "src").mkdir(parents=True)
        f = overlay / "src" / "train.py"
        f.write_bytes(content)
        manifest = {
            "src/train.py": f"sha256:{hashlib.sha256(content).hexdigest()}",
        }
        return overlay, manifest, f

    def _runner(self, tmp_path: Path) -> Runner:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        return Runner(project, project / "automil")

    def test_matching_digest_applies_normally(self, tmp_path):
        overlay, manifest, _ = self._overlay(tmp_path)
        wt = tmp_path / "wt"
        wt.mkdir()
        self._runner(tmp_path).apply_overlay(wt, overlay, manifest=manifest)
        assert (wt / "src" / "train.py").read_bytes() == b"print('hello')\n"

    def test_tampered_file_is_refused(self, tmp_path):
        overlay, manifest, f = self._overlay(tmp_path)
        f.write_bytes(b"import os; os.system('rm -rf /')\n")  # edited after submit
        wt = tmp_path / "wt"
        wt.mkdir()
        with pytest.raises(ValueError, match="digest"):
            self._runner(tmp_path).apply_overlay(wt, overlay, manifest=manifest)

    def test_nothing_lands_when_verification_fails(self, tmp_path):
        """Verification runs before any copy, so a rejected overlay leaves the
        worktree untouched rather than half-applied."""
        overlay, manifest, f = self._overlay(tmp_path)
        (overlay / "other.py").write_bytes(b"x = 1\n")
        f.write_bytes(b"tampered\n")
        wt = tmp_path / "wt"
        wt.mkdir()
        with pytest.raises(ValueError):
            self._runner(tmp_path).apply_overlay(wt, overlay, manifest=manifest)
        assert not (wt / "other.py").exists()

    def test_unrecorded_files_are_not_verified(self, tmp_path):
        """The archive also holds run artifacts (fold_*_result.json, summary.json)
        that the manifest never claimed. Only claimed files are checked."""
        overlay, manifest, _ = self._overlay(tmp_path)
        (overlay / "fold_0_result.json").write_text("{}")
        wt = tmp_path / "wt"
        wt.mkdir()
        self._runner(tmp_path).apply_overlay(wt, overlay, manifest=manifest)
        assert (wt / "fold_0_result.json").exists()

    def test_no_manifest_means_no_verification(self, tmp_path):
        """Legacy specs carry no manifest; they must keep working."""
        overlay, _, _ = self._overlay(tmp_path)
        wt = tmp_path / "wt"
        wt.mkdir()
        self._runner(tmp_path).apply_overlay(wt, overlay)
        assert (wt / "src" / "train.py").exists()

    def test_a_claimed_but_missing_file_is_refused(self, tmp_path):
        overlay, manifest, f = self._overlay(tmp_path)
        f.unlink()
        wt = tmp_path / "wt"
        wt.mkdir()
        with pytest.raises(ValueError, match="missing|digest"):
            self._runner(tmp_path).apply_overlay(wt, overlay, manifest=manifest)

    def test_malformed_digest_is_refused_rather_than_ignored(self, tmp_path):
        overlay, _, _ = self._overlay(tmp_path)
        wt = tmp_path / "wt"
        wt.mkdir()
        with pytest.raises(ValueError):
            self._runner(tmp_path).apply_overlay(
                wt, overlay, manifest={"src/train.py": "not-a-digest"},
            )


class TestTheLiveBenchmarkProjectIsProtected:
    """0b2da55 populated the three template projects and skipped the live one."""

    #: Discovered rather than listed, so a new cohort overlay cannot be added
    #: without enforcement (NO-OVERLAY added the five roster cohorts).
    PROJECTS = sorted(
        p.name
        for p in (Path(__file__).resolve().parents[1] / "benchmarks" / "experiments").iterdir()
        if (p / "automil" / "config.yaml").exists()
    )

    def test_every_roster_cohort_has_an_overlay(self):
        """NO-OVERLAY: the agentic layer is the paper's headline, and it was
        configured for no preprint cohort at all."""
        roster = {"tcga_luad", "tcga_lgg", "cptac_gbm", "cptac_pdac", "tcga_hnsc"}
        assert roster <= set(self.PROJECTS), f"missing: {sorted(roster - set(self.PROJECTS))}"

    @pytest.mark.parametrize("project", PROJECTS)
    def test_mode_is_architecture_preserving(self, project):
        root = Path(__file__).resolve().parents[1]
        cfg = load_registry_config(root / "benchmarks" / "experiments" / project / "automil")
        assert cfg.mode == "architecture-preserving"

    @pytest.mark.parametrize("project", PROJECTS)
    def test_every_benchmark_project_declares_a_protected_list(self, project):
        root = Path(__file__).resolve().parents[1]
        cfg = load_registry_config(root / "benchmarks" / "experiments" / project / "automil")
        assert cfg.protected, (
            f"{project} ships an empty registry.protected, which short-circuits "
            f"submit.py's gate to a no-op — the frozen substrate would be "
            f"documented but unenforced."
        )

    @pytest.mark.parametrize("project", PROJECTS)
    def test_the_split_and_composite_writers_are_protected(self, project):
        """The two paths that can rewrite the evaluation protocol or fold test
        into the selection signal."""
        root = Path(__file__).resolve().parents[1]
        cfg = load_registry_config(root / "benchmarks" / "experiments" / project / "automil")
        joined = " ".join(cfg.protected)
        assert "splits.py" in joined
        assert "run_experiment.py" in joined

    @pytest.mark.parametrize("project", PROJECTS)
    def test_no_project_marks_a_protected_path_editable(self, project):
        """ccrcc listed run_experiment.py and pipeline/config.py as editable, and
        the no---files branch auto-detects editable files with no readonly check
        at all — so the split entry point was captured silently."""
        root = Path(__file__).resolve().parents[1]
        path = root / "benchmarks" / "experiments" / project / "automil" / "config.yaml"
        raw = yaml.safe_load(path.read_text()) or {}
        editable = set((raw.get("files") or {}).get("editable") or [])
        protected = set((raw.get("registry") or {}).get("protected") or [])
        assert not (editable & protected), (
            f"{project}: {sorted(editable & protected)} are both editable and "
            f"protected — the auto-detect branch would capture them and submit "
            f"would then hard-reject, which is a confusing dead end."
        )

    @pytest.mark.parametrize("project", PROJECTS)
    def test_preserving_mode_exposes_only_the_recipe_module(self, project):
        root = Path(__file__).resolve().parents[1]
        policy = load_candidate_policy(
            root / "benchmarks" / "experiments" / project / "automil"
        )
        assert policy.editable == (
            f"benchmarks/experiments/{project}/automil/variants/_policies/*.py",
        )
        assert policy.allowed_override_options == ("--hparams", "--policy-variant")
        assert policy.allowed_variant_kinds == ("policy",)

        for forbidden in (
            "benchmarks/lib/CLAM/models/model_clam.py",
            "benchmarks/lib/CLAM/utils/core_utils.py",
            "benchmarks/src/autobench/pipeline/clam/train.py",
        ):
            verdict = policy.classify([forbidden])
            assert not verdict.accepted, (project, forbidden)
