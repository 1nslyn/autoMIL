"""``automil.runtime_helpers.write_result_json`` splits result.json across the
val-firewall boundary (L-3).

Before this existed, a training script's final result.json (carrying the
sealed ``held_out`` test block) was written straight into the worktree and
sat there -- readable by anything with filesystem access to the project
directory, including the coding agent driving the search -- for the entire
run (worktree creation to cleanup). ``write_result_json`` is the shared
helper training scripts call instead: it writes the FULL payload to the
sealed ``AUTOMIL_RESULTS_DIR`` and a STRIPPED (val-only) sibling into the
worktree, so the worktree copy can never carry ``held_out``/``summary``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


FULL_PAYLOAD = {
    "status": "completed",
    "metrics": {"val_auc": 0.88, "val_bacc": 0.81},
    "held_out": {"test_auc": 0.87, "test_bacc": 0.83},
    "summary": {"folds": [{"val_auc": 0.88}], "test": {"auc": 0.87}},
    "composite": 0.845,
    "elapsed_seconds": 100,
    "peak_vram_mb": 4000,
}


def _no_stray_tmp_files(*dirs: Path) -> bool:
    return not any(list(d.glob("*.tmp")) for d in dirs if d.exists())


class TestSplitWriteWithSealedDir:
    """AUTOMIL_RESULTS_DIR set (absolute) -- the orchestrated case."""

    def test_full_payload_lands_in_the_sealed_dir(self, tmp_path, monkeypatch):
        from automil.runtime_helpers import write_result_json

        sealed_dir = tmp_path / "archive" / "node_0001" / "certify"
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))

        write_result_json(dict(FULL_PAYLOAD), worktree_dir=worktree_dir)

        sealed = json.loads((sealed_dir / "result.json").read_text())
        assert sealed == FULL_PAYLOAD
        assert sealed["held_out"] == {"test_auc": 0.87, "test_bacc": 0.83}

    def test_worktree_copy_is_stripped_of_held_out_and_summary(self, tmp_path, monkeypatch):
        """The core L-3 assertion: the worktree must never carry test."""
        from automil.runtime_helpers import write_result_json

        sealed_dir = tmp_path / "archive" / "node_0001" / "certify"
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))

        write_result_json(dict(FULL_PAYLOAD), worktree_dir=worktree_dir)

        worktree_copy = json.loads((worktree_dir / "result.json").read_text())
        assert "held_out" not in worktree_copy, (
            "an agent reading the live worktree must never see held_out"
        )
        assert "summary" not in worktree_copy
        # The val-facing fields must still be there -- this is a strip, not a wipe.
        assert worktree_copy["status"] == "completed"
        assert worktree_copy["metrics"] == {"val_auc": 0.88, "val_bacc": 0.81}
        assert worktree_copy["composite"] == 0.845

    def test_payload_without_sealed_keys_is_identical_in_both_copies(self, tmp_path, monkeypatch):
        from automil.runtime_helpers import write_result_json

        sealed_dir = tmp_path / "archive" / "node_0002" / "certify"
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))
        payload = {"status": "completed", "metrics": {"val_auc": 0.5}, "composite": 0.5}

        write_result_json(dict(payload), worktree_dir=worktree_dir)

        assert json.loads((sealed_dir / "result.json").read_text()) == payload
        assert json.loads((worktree_dir / "result.json").read_text()) == payload

    def test_input_payload_is_not_mutated(self, tmp_path, monkeypatch):
        """Immutability: the caller's dict must survive the call unchanged."""
        from automil.runtime_helpers import write_result_json

        sealed_dir = tmp_path / "archive" / "node_0003" / "certify"
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))
        payload = dict(FULL_PAYLOAD)
        snapshot = dict(payload)

        write_result_json(payload, worktree_dir=worktree_dir)

        assert payload == snapshot, "write_result_json must not mutate its input"

    def test_no_stray_tempfiles_left_behind(self, tmp_path, monkeypatch):
        from automil.runtime_helpers import write_result_json

        sealed_dir = tmp_path / "archive" / "node_0004" / "certify"
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))

        write_result_json(dict(FULL_PAYLOAD), worktree_dir=worktree_dir)

        assert _no_stray_tmp_files(sealed_dir, worktree_dir)


class TestSplitWriteFallback:
    """AUTOMIL_RESULTS_DIR unset or malformed -- no sealed location exists."""

    def test_unset_results_dir_writes_full_payload_to_the_worktree(self, tmp_path, monkeypatch):
        """No sealed location -- the worktree copy is the only copy, so it must
        carry everything (losing held_out entirely would be worse than the
        pre-fix behaviour, not an improvement on it)."""
        from automil.runtime_helpers import write_result_json

        monkeypatch.delenv("AUTOMIL_RESULTS_DIR", raising=False)
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)

        write_result_json(dict(FULL_PAYLOAD), worktree_dir=worktree_dir)

        written = json.loads((worktree_dir / "result.json").read_text())
        assert written == FULL_PAYLOAD

    def test_relative_results_dir_is_treated_as_unset(self, tmp_path, monkeypatch):
        """Mirrors the T-09-06 precedent in register_sigterm_flush: a relative
        AUTOMIL_RESULTS_DIR is rejected rather than resolved against an unknown
        base, and the safe fallback is the worktree."""
        from automil.runtime_helpers import write_result_json

        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", "relative/certify/path")
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir(parents=True)

        write_result_json(dict(FULL_PAYLOAD), worktree_dir=worktree_dir)

        written = json.loads((worktree_dir / "result.json").read_text())
        assert written == FULL_PAYLOAD
        assert not (tmp_path / "relative").exists()

    def test_defaults_worktree_dir_to_cwd(self, tmp_path, monkeypatch):
        """Training scripts run with cwd == the worktree (daemon sets Popen cwd=);
        the default must match that without requiring an explicit argument."""
        from automil.runtime_helpers import write_result_json

        monkeypatch.delenv("AUTOMIL_RESULTS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)

        write_result_json(dict(FULL_PAYLOAD))

        assert json.loads((tmp_path / "result.json").read_text()) == FULL_PAYLOAD


class TestRoundTripWithCollectResult:
    """Integration: what runner.collect_result sees after write_result_json runs."""

    def test_agent_visible_worktree_file_never_carries_held_out_while_collect_result_recovers_it(
        self, tmp_path, monkeypatch
    ):
        from automil.runtime_helpers import write_result_json
        from automil.runner import Runner
        import subprocess as sp
        import os

        project = tmp_path / "project"
        project.mkdir()
        (project / "train.py").write_text("print('x')\n")
        sp.run(["git", "init"], cwd=project, capture_output=True, check=True)
        sp.run(["git", "add", "."], cwd=project, capture_output=True, check=True)
        sp.run(
            ["git", "commit", "-m", "initial"], cwd=project, capture_output=True, check=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t.com",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t.com"},
        )
        base = sp.run(
            ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True,
        ).stdout.strip()

        runner = Runner(project_root=project, automil_dir=project / "automil")
        wt_path = runner.create_worktree(base_commit=base, node_id="node_0099")
        archive_dir = project / "orchestrator" / "archive" / "node_0099"
        archive_dir.mkdir(parents=True)
        sealed_dir = archive_dir / "certify"
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed_dir))

        # The training script's final write, via the new helper.
        write_result_json(dict(FULL_PAYLOAD), worktree_dir=wt_path)

        # At this instant -- while the worktree still exists on disk, exactly the
        # window L-3 is about -- an agent reading the worktree must see no test.
        on_disk_worktree_copy = json.loads((wt_path / "result.json").read_text())
        assert "held_out" not in on_disk_worktree_copy

        # The daemon's real collection path must still recover the full payload
        # so terminal_writer can route held_out into certify.json as before.
        collected = runner.collect_result(wt_path, archive_dir)
        assert collected is not None
        assert collected["held_out"] == {"test_auc": 0.87, "test_bacc": 0.83}

        runner.cleanup_worktree(wt_path)
