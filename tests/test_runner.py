"""Tests for the git worktree overlay runner."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from automil.runner import Runner


@pytest.fixture
def project_repo(tmp_path):
    """Create a minimal git repo simulating an automil project."""
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "train.py").write_text("print('baseline')\n")
    (repo / "prepare.py").write_text("ENCODER_DIMS = {}\n")
    (repo / "config.yaml").write_text("project:\n  name: test\n")
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo, capture_output=True, check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
    )
    return repo


@pytest.fixture
def runner(project_repo):
    return Runner(project_root=project_repo, automil_dir=project_repo / "automil")


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


class TestWorktreeLifecycle:
    def test_create_and_cleanup(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0001")
        assert wt_path.exists()
        assert (wt_path / "train.py").read_text() == "print('baseline')\n"
        assert (wt_path / "prepare.py").exists()
        runner.cleanup_worktree(wt_path)
        assert not wt_path.exists()

    def test_overlay_files(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        overlay_dir = project_repo / "orchestrator" / "archive" / "node_0001"
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "train.py").write_text("print('modified')\n")
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0001")
        runner.apply_overlay(wt_path, overlay_dir)
        assert (wt_path / "train.py").read_text() == "print('modified')\n"
        assert (wt_path / "prepare.py").read_text() == "ENCODER_DIMS = {}\n"
        runner.cleanup_worktree(wt_path)

    def test_overlay_new_file(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        overlay_dir = project_repo / "orchestrator" / "archive" / "node_0002"
        overlay_dir.mkdir(parents=True)
        sub = overlay_dir / "models"
        sub.mkdir()
        (sub / "custom.py").write_text("class Custom: pass\n")
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0002")
        runner.apply_overlay(wt_path, overlay_dir)
        assert (wt_path / "models" / "custom.py").read_text() == "class Custom: pass\n"
        runner.cleanup_worktree(wt_path)

    def test_overlay_keeps_nested_result_json(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        overlay_dir = project_repo / "orchestrator" / "archive" / "node_nested"
        overlay_dir.mkdir(parents=True)
        nested = overlay_dir / "configs"
        nested.mkdir()
        (nested / "result.json").write_text('{"config": true}\n')
        wt_path = runner.create_worktree(base_commit=base, node_id="node_nested")
        runner.apply_overlay(wt_path, overlay_dir)
        assert (wt_path / "configs" / "result.json").read_text() == '{"config": true}\n'
        runner.cleanup_worktree(wt_path)

    def test_overlay_excludes_certify_vault(self, runner, project_repo):
        """Val-firewall (Scope B): apply_overlay must never copy the sealed
        certify/ vault (held-out test) into a worktree. A resubmit overlays the
        parent node's archive dir, which now holds a born-sealed certify/."""
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        overlay_dir = project_repo / "orchestrator" / "archive" / "node_certify"
        certify = overlay_dir / "certify"
        (certify / "results").mkdir(parents=True)
        (certify / "certify.json").write_text('{"held_out": {"test_auc": 0.9}}')
        (certify / "results" / "summary.json").write_text('{"test": {"auc": 0.9}}')
        (certify / "fold_0_result.json").write_text('{"held_out": {"test_auc": 0.9}}')
        (overlay_dir / "train.py").write_text("print('real overlay')\n")
        wt_path = runner.create_worktree(base_commit=base, node_id="node_certify")
        runner.apply_overlay(wt_path, overlay_dir)
        # Non-certify overlay files are still applied.
        assert (wt_path / "train.py").read_text() == "print('real overlay')\n"
        # The sealed test vault is NOT copied into the worktree.
        assert not (wt_path / "certify").exists(), (
            "sealed certify/ (held-out test) must never be copied into a worktree"
        )
        runner.cleanup_worktree(wt_path)

    def test_prune_stale_worktrees(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0003")
        assert wt_path.exists()
        import shutil
        shutil.rmtree(wt_path)
        runner.prune_stale_worktrees()

    def test_worktree_path(self, runner):
        assert runner.worktree_path("node_0001").name == "node_0001"


class TestResultCollection:
    def test_collect_result(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0004")
        result = {"status": "completed", "primary_value": 0.85, "metrics": {"test_auc": 0.87}}
        (wt_path / "result.json").write_text(json.dumps(result))
        archive_dir = project_repo / "orchestrator" / "archive" / "node_0004"
        archive_dir.mkdir(parents=True)
        collected = runner.collect_result(wt_path, archive_dir)
        assert collected is not None
        assert collected["status"] == "completed"
        # Scope B val-firewall: the raw result.json (carries held_out) is copied
        # into the sealed certify/ subdir, never the agent-visible node-archive root.
        assert (archive_dir / "certify" / "result.json").exists()
        assert not (archive_dir / "result.json").exists()
        runner.cleanup_worktree(wt_path)

    def test_missing_result_returns_none(self, runner, project_repo):
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0005")
        archive_dir = project_repo / "orchestrator" / "archive" / "node_0005"
        archive_dir.mkdir(parents=True)
        collected = runner.collect_result(wt_path, archive_dir)
        assert collected is None
        runner.cleanup_worktree(wt_path)

    def test_collect_result_prefers_an_already_sealed_full_payload(self, runner, project_repo):
        """L-3: when a training script wrote via runtime_helpers.write_result_json,
        the sealed certify/result.json already carries the FULL (held_out-included)
        payload and the worktree copy is a stripped val-only sibling. collect_result
        must read the sealed copy back rather than let the stripped worktree copy
        overwrite it -- overwriting would silently discard held_out."""
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0006")
        archive_dir = project_repo / "orchestrator" / "archive" / "node_0006"
        sealed_dir = archive_dir / "certify"
        sealed_dir.mkdir(parents=True)

        full_payload = {
            "status": "completed", "primary_value": 0.845,
            "metrics": {"val_auc": 0.88}, "held_out": {"test_auc": 0.87},
        }
        stripped_payload = {"status": "completed", "primary_value": 0.845, "metrics": {"val_auc": 0.88}}
        (sealed_dir / "result.json").write_text(json.dumps(full_payload))
        (wt_path / "result.json").write_text(json.dumps(stripped_payload))

        collected = runner.collect_result(wt_path, archive_dir)

        assert collected is not None
        assert collected["held_out"] == {"test_auc": 0.87}, (
            "collect_result must recover held_out from the sealed copy"
        )
        still_sealed = json.loads((sealed_dir / "result.json").read_text())
        assert "held_out" in still_sealed, (
            "the stripped worktree copy must not overwrite the already-sealed full payload"
        )
        runner.cleanup_worktree(wt_path)

    def test_collect_result_legacy_worktree_only_path_still_seals_and_returns_full(
        self, runner, project_repo
    ):
        """Regression guard for the pre-L-3 shape: an older script that writes the
        FULL payload straight into the worktree (no sealed copy yet) must still
        get it copied into certify/ and returned intact."""
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0007")
        archive_dir = project_repo / "orchestrator" / "archive" / "node_0007"
        archive_dir.mkdir(parents=True)

        full_payload = {
            "status": "completed", "primary_value": 0.845,
            "metrics": {"val_auc": 0.88}, "held_out": {"test_auc": 0.87},
        }
        (wt_path / "result.json").write_text(json.dumps(full_payload))

        collected = runner.collect_result(wt_path, archive_dir)

        assert collected is not None
        assert collected["held_out"] == {"test_auc": 0.87}
        sealed = json.loads((archive_dir / "certify" / "result.json").read_text())
        assert sealed["held_out"] == {"test_auc": 0.87}
        runner.cleanup_worktree(wt_path)


class TestWorktreeRecovery:
    """create_worktree must survive every orphan shape a crash can leave behind.

    Canary incident 2026-08-15: the wipe-before-recreate path called
    ``shutil.rmtree`` without ``git worktree prune``, so the follow-up
    ``git worktree add`` always failed with exit 128 ("missing but already
    registered worktree") — the documented recovery never once succeeded,
    turning a recoverable race into 18 dead promotion jobs.
    """

    def test_recreate_after_out_of_band_rmtree(self, runner, project_repo):
        """Registration stale, directory gone — the exact exit-128 case."""
        import shutil
        base = _head(project_repo)
        wt = runner.create_worktree(base_commit=base, node_id="node_0042")
        shutil.rmtree(wt)  # no prune: registration left dangling in .git/worktrees
        wt2 = runner.create_worktree(base_commit=base, node_id="node_0042")
        assert wt2 == wt
        assert (wt2 / "train.py").exists()
        runner.cleanup_worktree(wt2)

    def test_recreate_over_live_orphan_dir(self, runner, project_repo):
        """Directory present and still registered (interrupted prior launch)."""
        base = _head(project_repo)
        wt = runner.create_worktree(base_commit=base, node_id="node_0043")
        (wt / "leftover.marker").write_text("stale state from a dead launch\n")
        wt2 = runner.create_worktree(base_commit=base, node_id="node_0043")
        assert wt2 == wt
        assert not (wt2 / "leftover.marker").exists(), (
            "recreate must produce a fresh checkout, not reuse orphan contents"
        )
        assert (wt2 / "train.py").exists()
        runner.cleanup_worktree(wt2)


class TestWorktreeScoping:
    """Two automil projects in one checkout must never share a worktree path.

    Canary incident 2026-08-15: every cell's promotion jobs are named
    ``node_0001..node_0010`` and the runner keyed worktrees on node_id alone
    under one repo-global ``.automil_worktrees/``, so two concurrent promotion
    orchestrators wiped each other's live worktrees deterministically.
    """

    def test_same_node_id_disjoint_across_projects(self, project_repo):
        a = Runner(project_root=project_repo,
                   automil_dir=project_repo / "cellA" / "automil")
        b = Runner(project_root=project_repo,
                   automil_dir=project_repo / "cellB" / "automil")
        assert a.worktree_path("node_0001") != b.worktree_path("node_0001")
        base_dir = project_repo / ".automil_worktrees"
        assert base_dir in a.worktree_path("node_0001").parents
        assert base_dir in b.worktree_path("node_0001").parents

    def test_concurrent_projects_do_not_clobber(self, project_repo):
        base = _head(project_repo)
        a = Runner(project_root=project_repo,
                   automil_dir=project_repo / "cellA" / "automil")
        b = Runner(project_root=project_repo,
                   automil_dir=project_repo / "cellB" / "automil")
        wa = a.create_worktree(base_commit=base, node_id="node_0001")
        (wa / "in_flight.marker").write_text("cell A training here\n")
        wb = b.create_worktree(base_commit=base, node_id="node_0001")
        assert wa != wb
        assert wa.exists() and wb.exists()
        assert (wa / "in_flight.marker").exists(), (
            "cell B's launch wiped cell A's live worktree"
        )
        a.cleanup_worktree(wa)
        b.cleanup_worktree(wb)

    def test_scope_is_stable_across_instances(self, project_repo):
        adir = project_repo / "cellA" / "automil"
        p1 = Runner(project_root=project_repo, automil_dir=adir).worktree_path("node_0009")
        p2 = Runner(project_root=project_repo, automil_dir=adir).worktree_path("node_0009")
        assert p1 == p2
