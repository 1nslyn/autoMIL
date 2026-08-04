"""H-7: an operator cancel must not be overwritten as a crash.

From the daemon's side a deliberate stop and a real failure are the same
observation — a dead process and no ``result.json``. The cap path already
annotated the running spec before killing (D-124 / Pitfall 4), so cap kills were
distinguishable; ``automil cancel`` wrote its ``cancel_reason`` only onto the
GRAPH node, which is not where the daemon looks. So every operator cancel was
recorded as ``crash``.

That is not cosmetic. It inflates the failure statistics the gate's health
diagnostic reads (``gate/stats.py::diagnose_gate_health``), and it makes an
operator stopping a run indistinguishable from a bug in the training code — the
two things one most wants to tell apart while debugging a campaign.
"""
from __future__ import annotations

import json

import pytest


@pytest.fixture
def orch(tmp_path):
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator

    adir = tmp_path / "automil"
    orch_dir = adir / "orchestrator"
    for sub in ("archive/0001", "completed", "running/local"):
        (orch_dir / sub).mkdir(parents=True, exist_ok=True)
    o = object.__new__(ExperimentOrchestrator)
    o.automil_dir = adir
    o.orch_dir = orch_dir
    o.archive_dir = orch_dir / "archive"
    o.completed_dir = orch_dir / "completed"
    o.running_root = orch_dir / "running"
    o.running_dir = orch_dir / "running" / "local"
    return o


def _running_spec(orch, node_id, **meta):
    (orch.running_dir / f"{node_id}.json").write_text(
        json.dumps({"id": node_id, "metadata": meta})
    )


class TestRecordedCancelReason:
    def test_reads_cli_from_the_running_spec(self, orch):
        _running_spec(orch, "0001", cancel_reason="cli")
        assert orch._recorded_cancel_reason("0001") == "cli"

    def test_reads_cap_from_the_running_spec(self, orch):
        _running_spec(orch, "0001", cancel_reason="cap")
        assert orch._recorded_cancel_reason("0001") == "cap"

    def test_none_when_the_process_died_on_its_own(self, orch):
        _running_spec(orch, "0001")
        assert orch._recorded_cancel_reason("0001") is None

    def test_falls_back_to_the_archived_spec(self, orch):
        """running/ may already have been cleaned by the time this is asked."""
        (orch.archive_dir / "0001" / "spec.json").write_text(
            json.dumps({"id": "0001", "metadata": {"cancel_reason": "cli"}})
        )
        assert orch._recorded_cancel_reason("0001") == "cli"

    def test_cap_predicate_still_works(self, orch):
        """The narrower predicate is kept as a delegate — callers exist."""
        _running_spec(orch, "0001", cancel_reason="cap")
        assert orch._was_cap_killed_completion("0001") is True
        _running_spec(orch, "0002", cancel_reason="cli")
        assert orch._was_cap_killed_completion("0002") is False

    def test_malformed_spec_does_not_raise(self, orch):
        (orch.running_dir / "0001.json").write_text("{not json")
        assert orch._recorded_cancel_reason("0001") is None


class TestCancelAnnotatesTheRunningSpec:
    """The CLI half: without this the daemon has nothing to read."""

    def test_cancel_writes_cancel_reason_before_killing(self, tmp_path):
        import inspect

        from automil.cli import cancel as cancel_mod

        src = inspect.getsource(cancel_mod)
        annotate = src.index('"cancel_reason": "cli"')
        # The annotation must precede the kill, or a fast daemon poll can observe
        # the dead process before the reason is on disk.
        kill = src.index("Direct-kill path")
        assert annotate < kill, (
            "cancel_reason is written after the kill; the daemon can lose the race"
        )
