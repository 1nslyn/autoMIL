"""L-3 wiring: the worktree copy of result.json must not carry test metrics.

`automil.runtime_helpers.write_result_json` landed with the split-write contract
and full tests, but **no training script called it** — `run_experiment.py` still
wrote `result.json` straight into its cwd. So in practice the defect was
untouched: the full payload, `held_out` included, sat in
`.automil_worktrees/<node>/result.json` from worktree creation to cleanup, in a
directory with no access control of its own. Anything that could read the project
tree during search — including the agent driving it — could read the sealed test
block off disk without waiting for `automil certify`.

This pins the wiring, not the helper (that is tested framework-side).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _result():
    return {
        "status": "completed",
        "metrics": {"val_auc": 0.81, "val_bacc": 0.79},
        "held_out": {"test_auc": 0.77, "test_bacc": 0.75},
        "primary_value": 0.80,
    }


def test_run_experiment_calls_the_split_writer():
    """Guard the wiring itself: a direct open('result.json','w') would restore
    the leak while every helper test kept passing."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py").read_text()
    assert "write_result_json(result)" in src
    assert 'with open("result.json", "w") as f:\n        json.dump(result' not in src


def test_worktree_copy_is_stripped(tmp_path, monkeypatch):
    from automil.runtime_helpers import write_result_json

    sealed = tmp_path / "certify"
    sealed.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed))
    monkeypatch.chdir(wt)

    write_result_json(_result())

    worktree_payload = json.loads((wt / "result.json").read_text())
    assert "held_out" not in worktree_payload, (
        "test metrics are still readable in the worktree for the whole run"
    )
    assert worktree_payload["metrics"] == {"val_auc": 0.81, "val_bacc": 0.79}
    assert worktree_payload["primary_value"] == 0.80


def test_sealed_copy_keeps_the_test_block(tmp_path, monkeypatch):
    from automil.runtime_helpers import write_result_json

    sealed = tmp_path / "certify"
    sealed.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed))
    monkeypatch.chdir(wt)

    write_result_json(_result())

    assert json.loads((sealed / "result.json").read_text())["held_out"] == {
        "test_auc": 0.77, "test_bacc": 0.75,
    }


def test_standalone_run_without_the_env_var_still_writes(tmp_path, monkeypatch):
    """`run_benchmark.py` (the static grid) runs outside the orchestrator, so
    AUTOMIL_RESULTS_DIR is unset. That path must keep working."""
    from automil.runtime_helpers import write_result_json

    monkeypatch.delenv("AUTOMIL_RESULTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    write_result_json(_result())
    assert (tmp_path / "result.json").exists()
