"""``cmd_submit`` (legacy in-daemon submit path) must not collide ids (L-6).

The supported submission path is ``automil submit`` -> ``cli/submit.py``,
which allocates node ids from ``graph.json`` under ``locked_update``.
``ExperimentOrchestrator.cmd_submit`` is an older, still-reachable path
(``automil.backends._orchestrator_daemon.main()``'s ``submit`` subcommand)
that instead derives the next id from ``gpu_state.json``'s ``"counter"``
field -- a field only the daemon's main tick loop (``_save_state``)
persists. ``cmd_submit`` itself never advances it. Two near-simultaneous
calls on this path therefore read the same stale counter, compute the same
id, and the second call's unconditional ``write_text`` silently overwrites
the first call's queue spec -- the first submission is lost with no error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _make_orch(tmp_path: Path) -> Any:
    from automil.orchestrator import ExperimentOrchestrator

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir(exist_ok=True)
    orch = ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)
    orch.queue_dir.mkdir(parents=True, exist_ok=True)
    return orch


def test_second_near_simultaneous_submit_does_not_overwrite_the_first(tmp_path, capsys):
    """L-6 reproducer: back-to-back auto-id submits must not collide on one id.

    Neither call ever touches gpu_state.json's counter (only the daemon's
    tick loop does), so both compute the same auto-assigned id from the
    same stale counter -- exactly the "near-simultaneous submissions"
    scenario the finding describes.
    """
    orch = _make_orch(tmp_path)
    spec1 = tmp_path / "spec1.json"
    spec1.write_text(json.dumps({"description": "first submission"}))
    spec2 = tmp_path / "spec2.json"
    spec2.write_text(json.dumps({"description": "second submission"}))

    orch.cmd_submit(str(spec1))
    queued_before = list(orch.queue_dir.glob("*.json"))
    assert len(queued_before) == 1
    colliding_id = queued_before[0].stem

    with pytest.raises(SystemExit):
        orch.cmd_submit(str(spec2))

    queued_after = list(orch.queue_dir.glob("*.json"))
    assert [f.stem for f in queued_after] == [colliding_id], (
        "the second submit must not silently create/overwrite onto the same id"
    )
    surviving = json.loads(queued_after[0].read_text())
    assert surviving["description"] == "first submission", (
        "the first submission must survive untouched -- it must not be "
        "clobbered by the colliding second call"
    )


def test_cmd_submit_refuses_a_preset_id_that_already_exists(tmp_path):
    """Isolates the missing existence-guard: a caller-supplied id must be honoured too."""
    orch = _make_orch(tmp_path)
    existing = orch.queue_dir / "0007.json"
    existing.write_text(json.dumps({"id": "0007", "description": "already queued"}))

    colliding_spec = tmp_path / "collide.json"
    colliding_spec.write_text(json.dumps({"id": "0007", "description": "attempted overwrite"}))

    with pytest.raises(SystemExit):
        orch.cmd_submit(str(colliding_spec))

    assert json.loads(existing.read_text())["description"] == "already queued"


def test_cmd_submit_still_succeeds_for_a_fresh_id(tmp_path, capsys):
    """Regression guard: the ordinary (non-colliding) path must keep working."""
    orch = _make_orch(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"description": "ordinary submission"}))

    orch.cmd_submit(str(spec_path))

    queued = list(orch.queue_dir.glob("*.json"))
    assert len(queued) == 1
    assert json.loads(queued[0].read_text())["description"] == "ordinary submission"
    assert "Submitted experiment" in capsys.readouterr().out


def test_cmd_submit_allocates_the_id_under_the_graph_lock_file(tmp_path):
    """The id allocation + write must be serialized on graph.json's sidecar lock.

    Sharing graph.json's lock file (rather than a submit-only lock) is what
    makes this path mutually exclusive with every other id-allocating
    writer in the process, including `cli/submit.py`'s `locked_update`.
    """
    orch = _make_orch(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"description": "lock check"}))

    expected_lock = orch.graph.path.with_suffix(orch.graph.path.suffix + ".lock")
    assert not expected_lock.exists()

    orch.cmd_submit(str(spec_path))

    assert expected_lock.exists(), "cmd_submit must acquire graph.json's sidecar lock"
