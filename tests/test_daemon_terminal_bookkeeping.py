"""M-5 / M-6 / M-7: the daemon's terminal and launch paths must keep the graph honest.

Three defects that all share a shape — the daemon writes artifacts to disk and
leaves ``graph.json`` describing a world that no longer exists.

**M-5** — ``_mark_crashed`` and ``_recover_orphans`` write ``archive/result.json``
and ``completed/<id>.json`` but never touch the graph. The node keeps its
pre-launch ``type``/``status``, so ``automil rank`` and ``automil status`` show a
node that will never resolve until somebody happens to run ``reconcile``. On a
daemon-driven campaign that may be never.

**M-7** — ``meta.total_executed`` is the UCB exploration denominator
(``graph.py`` ``potential = primary_value + w·sqrt(log(total)/(1+child_count))``).
The CLI paths maintain it and ``mark_failed`` maintains it, but the daemon's
*success* path (``terminal_writer.write_terminal_state``) never did. A campaign
driven entirely by the daemon therefore explores against a frozen denominator.

**M-6** — the running spec is written *after* ``Popen``, because it carries the
pid. If the daemon dies in that window the node stays queued forever and the
process keeps running, holding its GPU, invisible to orphan recovery. An intent
record written *before* the spawn cannot recover the pid — nothing can — but it
converts "silently stuck forever" into "correctly marked crashed, with the leak
named".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def orch(tmp_path, monkeypatch):
    """An orchestrator with just the directories the terminal paths touch."""
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator
    from automil.graph import ExperimentGraph

    adir = tmp_path / "automil"
    orch_dir = adir / "orchestrator"
    for sub in ("archive", "completed", "queue", "running/local"):
        (orch_dir / sub).mkdir(parents=True, exist_ok=True)

    o = object.__new__(ExperimentOrchestrator)
    o.automil_dir = adir
    o.orch_dir = orch_dir
    o.archive_dir = orch_dir / "archive"
    o.completed_dir = orch_dir / "completed"
    o.queue_dir = orch_dir / "queue"
    o.running_root = orch_dir / "running"
    o.running_dir = orch_dir / "running" / "local"
    o.graph = ExperimentGraph(adir / "graph.json")
    return o


def _proposed(orch, description="idea") -> str:
    nid = orch.graph.add_proposed(parent_id=None, description=description, techniques=[])
    orch.graph.save()
    return nid


def _reload(orch):
    from automil.graph import ExperimentGraph
    return ExperimentGraph(orch.automil_dir / "graph.json")


class TestCrashReachesTheGraph:
    """M-5: a crash that only lands on disk leaves rank/status permanently stale."""

    def test_mark_crashed_updates_the_graph_node(self, orch):
        nid = _proposed(orch)
        orch._mark_crashed(nid, {"id": nid, "description": "idea"}, "boom")
        node = _reload(orch).get_node(nid)
        assert node["status"] == "crash"
        assert node["type"] == "executed"

    def test_mark_crashed_records_the_error(self, orch):
        nid = _proposed(orch)
        orch._mark_crashed(nid, {"id": nid}, "worktree creation failed")
        assert "worktree" in _reload(orch).get_node(nid).get("error", "")

    def test_mark_crashed_still_writes_the_disk_artifacts(self, orch):
        """The graph update is additive — reconcile must still be able to rebuild."""
        nid = _proposed(orch)
        orch._mark_crashed(nid, {"id": nid}, "boom")
        assert (orch.archive_dir / nid / "result.json").exists()
        assert (orch.completed_dir / f"{nid}.json").exists()

    def test_orphan_recovery_updates_the_graph_node(self, orch):
        nid = _proposed(orch)
        (orch.running_dir / f"{nid}.json").write_text(json.dumps({"id": nid}))
        orch._recover_orphans()
        node = _reload(orch).get_node(nid)
        assert node["status"] == "crash"
        assert node["type"] == "executed"

    def test_orphan_recovery_clears_the_running_spec(self, orch):
        nid = _proposed(orch)
        (orch.running_dir / f"{nid}.json").write_text(json.dumps({"id": nid}))
        orch._recover_orphans()
        assert not (orch.running_dir / f"{nid}.json").exists()

    def test_a_node_absent_from_the_graph_does_not_raise(self, orch):
        """Backend.submit paths can produce a spec the graph never saw."""
        orch._mark_crashed("9999", {"id": "9999"}, "boom")
        assert (orch.completed_dir / "9999.json").exists()


class TestUcbDenominatorIsMaintained:
    """M-7: total_executed is the UCB `total`; a frozen one distorts exploration."""

    def test_a_crash_increments_total_executed(self, orch):
        nid = _proposed(orch)
        before = _reload(orch).meta["total_executed"]
        orch._mark_crashed(nid, {"id": nid}, "boom")
        assert _reload(orch).meta["total_executed"] == before + 1

    def test_a_crash_decrements_total_proposed(self, orch):
        nid = _proposed(orch)
        before = _reload(orch).meta["total_proposed"]
        orch._mark_crashed(nid, {"id": nid}, "boom")
        assert _reload(orch).meta["total_proposed"] == before - 1

    def test_reprocessing_the_same_crash_does_not_double_count(self, orch):
        """Orphan recovery and _mark_crashed can both fire for one node."""
        nid = _proposed(orch)
        orch._mark_crashed(nid, {"id": nid}, "boom")
        after_first = _reload(orch).meta["total_executed"]
        orch._mark_crashed(nid, {"id": nid}, "boom again")
        assert _reload(orch).meta["total_executed"] == after_first

    def test_total_proposed_never_goes_negative(self, orch):
        nid = _proposed(orch)
        orch.graph.meta["total_proposed"] = 0
        orch.graph.save()
        orch._mark_crashed(nid, {"id": nid}, "boom")
        assert _reload(orch).meta["total_proposed"] >= 0


class TestLaunchIntentRecord:
    """M-6: the window between Popen and the running-spec write."""

    def test_intent_record_is_written_before_the_spawn(self, orch, tmp_path):
        orch._write_launch_intent({"id": "0001"}, gpu_id=0,
                                  worktree=tmp_path / "wt")
        path = orch.running_dir / "0001.json"
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["metadata"]["launch_phase"] == "launching"
        assert payload["metadata"].get("pid") is None

    def test_intent_record_names_the_worktree(self, orch, tmp_path):
        """Recovery cannot signal a pid it never had; the least it can do is tell
        the operator exactly which worktree the leaked process is running in."""
        orch._write_launch_intent({"id": "0001"}, gpu_id=3, worktree=tmp_path / "wt")
        meta = json.loads((orch.running_dir / "0001.json").read_text())["metadata"]
        assert meta["worktree"].endswith("wt")
        assert meta["gpu"] == 3

    def test_recovery_of_a_launching_record_marks_the_node_crashed(self, orch, tmp_path):
        nid = _proposed(orch)
        orch._write_launch_intent({"id": nid}, gpu_id=0, worktree=tmp_path / "wt")
        orch._recover_orphans()
        assert _reload(orch).get_node(nid)["status"] == "crash"

    def test_recovery_of_a_launching_record_warns_about_the_possible_leak(
        self, orch, tmp_path, caplog
    ):
        nid = _proposed(orch)
        orch._write_launch_intent({"id": nid}, gpu_id=0, worktree=tmp_path / "wt")
        with caplog.at_level("WARNING"):
            orch._recover_orphans()
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "no pid" in joined.lower() or "unrecorded" in joined.lower()
        assert nid in joined

    def test_a_normal_running_record_is_not_reported_as_a_leak(self, orch, caplog):
        nid = _proposed(orch)
        (orch.running_dir / f"{nid}.json").write_text(json.dumps(
            {"id": nid, "metadata": {"pid": 999999, "launch_phase": "running"}}))
        with caplog.at_level("WARNING"):
            orch._recover_orphans()
        assert "no pid" not in " ".join(r.getMessage() for r in caplog.records).lower()


class TestSuccessPathAlsoMaintainsTheDenominator:
    """M-7's other half: the daemon's COMPLETION path, not just its crash paths.

    Every CLI path and ``mark_failed`` maintained ``total_executed``;
    ``terminal_writer.write_terminal_state`` — the daemon's success path — did
    not. A campaign driven entirely by the daemon therefore ran its whole search
    with a frozen UCB denominator.
    """

    def _write(self, orch, node_id, result):
        from automil.terminal_writer import write_terminal_state

        (orch.archive_dir / node_id).mkdir(parents=True, exist_ok=True)
        write_terminal_state(
            node_id=node_id,
            result=result,
            graph=orch.graph,
            completed_dir=orch.completed_dir,
            archive_dir=orch.archive_dir,
            results_tsv_writer=lambda *a, **k: None,
            spec={"id": node_id, "description": "x", "graph_metadata": {}},
            elapsed_s=1.0,
            gpu_id=0,
        )

    def test_a_completion_increments_total_executed(self, orch):
        nid = _proposed(orch)
        before = _reload(orch).meta["total_executed"]
        self._write(orch, nid, {"status": "completed", "primary_value": 0.8,
                                "metrics": {"val_auc": 0.8, "val_bacc": 0.8}})
        assert _reload(orch).meta["total_executed"] == before + 1

    def test_reprocessing_a_completion_does_not_double_count(self, orch):
        nid = _proposed(orch)
        payload = {"status": "completed", "primary_value": 0.8,
                   "metrics": {"val_auc": 0.8, "val_bacc": 0.8}}
        self._write(orch, nid, payload)
        after_first = _reload(orch).meta["total_executed"]
        self._write(orch, nid, payload)
        assert _reload(orch).meta["total_executed"] == after_first
