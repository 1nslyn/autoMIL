"""Fixed batches and the phasing rule, enforced by the framework (protocol v4).

The campaign brief asked agents to open with ~8 attempts on at least five
axes, never spend more than three consecutive attempts on one axis without a
kept result, and reserve the final attempts for pre-registered robustness
neighbours of the champion. All five rehearsal agents complied voluntarily;
nothing checked. A cell now declares the structure in ``cap.phasing`` and
``automil submit`` refuses (free, before the queue write) a submission that
would break it; ``automil propose`` records the axis, the predicted delta
and the role the checks read.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cells.phasing import (
    Attempt,
    PhasingPolicy,
    batch_position,
    cell_attempts,
    phasing_refusal,
)
from automil.cli import main

POLICY = PhasingPolicy(batches=(2, 2, 1), opening_axes_min=2,
                       max_consecutive_per_axis=2, reserve_neighbours_min=1)


def _attempt(node_id, axis, status="discard", role=None):
    return Attempt(node_id=node_id, axis=axis, role=role, status=status,
                   submitted_at=f"2026-09-20T00:00:{int(node_id[-2:]):02d}+00:00")


BIG = PhasingPolicy(batches=(8, 8, 8, 6), opening_axes_min=5,
                    max_consecutive_per_axis=3, reserve_neighbours_min=2)


# ---------------------------------------------------------------------------
# The policy and its rules (pure)
# ---------------------------------------------------------------------------


class TestPhasingPolicy:
    def test_from_config_reads_the_declaration(self):
        cap = {"eval_budget": 30, "phasing": {
            "batches": [8, 8, 8, 6], "opening_axes_min": 5,
            "max_consecutive_per_axis": 3, "reserve_neighbours_min": 2}}
        policy = PhasingPolicy.from_config(cap)
        assert policy.batches == (8, 8, 8, 6)
        assert policy.total == 30
        assert policy.batch_of(1) == 1 and policy.batch_of(8) == 1
        assert policy.batch_of(9) == 2 and policy.batch_of(30) == 4
        assert policy.batch_bounds(3) == (17, 24)

    def test_absent_declaration_means_no_policy(self):
        assert PhasingPolicy.from_config({"eval_budget": 30}) is None
        assert PhasingPolicy.from_config(None) is None

    @pytest.mark.parametrize("bad", [
        {"batches": [8, 8, 8, 5], "opening_axes_min": 5,
         "max_consecutive_per_axis": 3, "reserve_neighbours_min": 2},   # sums to 29
        {"batches": [], "opening_axes_min": 5,
         "max_consecutive_per_axis": 3, "reserve_neighbours_min": 2},
        {"batches": [30], "opening_axes_min": 0,
         "max_consecutive_per_axis": 3, "reserve_neighbours_min": 2},
        {"batches": [30], "opening_axes_min": 5,
         "max_consecutive_per_axis": 3},                              # missing key
        {"batches": [8, "8", 8, 6], "opening_axes_min": 5,
         "max_consecutive_per_axis": 3, "reserve_neighbours_min": 2},
    ])
    def test_malformed_declaration_is_refused(self, bad):
        with pytest.raises(ValueError):
            PhasingPolicy.from_config({"eval_budget": 30, "phasing": bad})


class TestPhasingRefusal:
    def _refusal(self, attempts, *, axis="lr", role=None, parent="node_0001",
                 best="node_0001", in_flight=()):
        return phasing_refusal(
            POLICY, tuple(attempts), axis=axis, role=role, parent_id=parent,
            best_node_id=best, in_flight=frozenset(in_flight),
        )

    def test_first_attempt_needs_only_an_axis(self):
        assert self._refusal([], axis="lr") is None
        assert "axis" in self._refusal([], axis=None)

    def test_a_new_batch_waits_for_the_previous_one(self):
        done = [_attempt("node_0002", "lr"), _attempt("node_0003", "wd")]
        assert "batch 2" in self._refusal(done, axis="lr", in_flight={"node_0002"})
        assert self._refusal(done, axis="lr", in_flight=set()) is None

    def test_in_flight_nodes_of_the_same_batch_do_not_block(self):
        first = [_attempt("node_0002", "lr", status="running")]
        assert self._refusal(first, axis="wd", in_flight={"node_0002"}) is None

    def test_the_opening_batch_must_span_the_declared_axes(self):
        first = [_attempt("node_0002", "lr")]
        assert "distinct axes" in self._refusal(first, axis="lr")
        assert self._refusal(first, axis="wd") is None

    def test_a_prefix_that_can_no_longer_reach_the_axes_quota_is_refused_early(self):
        # Eight-attempt opening batch on five axes: after lr, wd, lr, wd, lr
        # (two axes, three slots left) a sixth on wd leaves at most four.
        prefix = [_attempt(f"node_000{i}", axis) for i, axis in
                  enumerate(("lr", "wd", "lr", "wd", "lr"), start=2)]
        refuse = phasing_refusal(BIG, tuple(prefix), axis="wd", role=None,
                                 parent_id="node_0001", best_node_id="node_0001",
                                 in_flight=frozenset())
        assert refuse is not None and "unreachable" in refuse
        assert phasing_refusal(BIG, tuple(prefix), axis="dropout", role=None,
                               parent_id="node_0001", best_node_id="node_0001",
                               in_flight=frozenset()) is None

    def test_a_final_batch_that_can_no_longer_seat_its_neighbours_is_refused_early(self):
        done = [_attempt(f"node_{i:04d}", ("lr", "wd", "dropout")[i % 3]) for i in range(2, 30)]
        assert len(done) == 28                      # attempt 29 is next; 30 is the last
        refuse = phasing_refusal(BIG, tuple(done), axis="lr", role=None,
                                 parent_id="node_0001", best_node_id="node_0001",
                                 in_flight=frozenset())
        assert refuse is not None and "neighbour" in refuse and "unreachable" in refuse
        assert phasing_refusal(BIG, tuple(done), axis="lr", role="neighbour",
                               parent_id="node_0001", best_node_id="node_0001",
                               in_flight=frozenset()) is None

    def test_consecutive_attempts_on_one_axis_need_a_kept_result(self):
        two_lr = [_attempt("node_0002", "lr"), _attempt("node_0003", "lr")]
        policy = PhasingPolicy(batches=(3, 2), opening_axes_min=1,
                               max_consecutive_per_axis=2, reserve_neighbours_min=1)
        refuse = phasing_refusal(policy, tuple(two_lr), axis="lr", role=None,
                                 parent_id="node_0001", best_node_id="node_0001",
                                 in_flight=frozenset())
        assert "consecutive" in refuse
        kept = [_attempt("node_0002", "lr"), _attempt("node_0003", "lr", status="keep")]
        assert phasing_refusal(policy, tuple(kept), axis="lr", role=None,
                               parent_id="node_0001", best_node_id="node_0001",
                               in_flight=frozenset()) is None
        assert phasing_refusal(policy, tuple(two_lr), axis="wd", role=None,
                               parent_id="node_0001", best_node_id="node_0001",
                               in_flight=frozenset()) is None

    def test_the_final_batch_reserves_neighbours_of_the_champion(self):
        done = [_attempt("node_0002", "lr"), _attempt("node_0003", "wd"),
                _attempt("node_0004", "lr"), _attempt("node_0005", "dropout")]
        assert "neighbour" in self._refusal(done, axis="lr")
        assert self._refusal(done, axis="lr", role="neighbour") is None

    def test_a_neighbour_must_be_a_child_of_the_best_node(self):
        assert "best node" in self._refusal([], axis="lr", role="neighbour",
                                            parent="node_0009", best="node_0001")

    def test_past_the_budget_the_phasing_says_nothing(self):
        done = [_attempt(f"node_{i:04d}", "lr", role="neighbour") for i in range(2, 7)]
        assert self._refusal(done, axis="lr") is None   # the budget gate refuses, not this


def _spec(node_id, cell_id, submitted_at, *, cap_refused=False):
    meta = {"cell_id": cell_id}
    if cap_refused:
        meta["cap_refused"] = True
    return json.dumps({"id": node_id, "submitted_at": submitted_at, "metadata": meta})


class TestCellAttempts:
    def test_attempts_are_the_specs_on_disk_in_submission_order(self, tmp_path):
        adir = tmp_path / "automil"
        queue = adir / "orchestrator" / "queue"
        archive = adir / "orchestrator" / "archive"
        queue.mkdir(parents=True)
        # launched (archived spec), in an order that differs from the node ids
        for node_id, at in (("node_0003", "T01"), ("node_0002", "T02"), ("node_0005", "T03"),
                            ("node_0006", "T04"), ("node_0008", "T05")):
            (archive / node_id).mkdir(parents=True)
            (archive / node_id / "spec.json").write_text(
                _spec(node_id, "c", at, cap_refused=(node_id == "node_0006")))
        (queue / "node_0009.json").write_text(_spec("node_0009", "c", "T06"))   # queued
        (queue / "node_0010.json").write_text(_spec("node_0010", "other", "T07"))
        nodes = {
            "node_0001": {"cell_id": "c", "bootstrapped": True, "status": "keep", "metadata": {}},
            "node_0002": {"cell_id": "c", "status": "keep", "metadata": {"axis": "wd", "role": "neighbour"}},
            "node_0003": {"cell_id": "c", "status": "running", "metadata": {"axis": "lr"}},
            "node_0004": {"cell_id": "c", "status": "pending", "metadata": {"axis": "lr"}},   # proposed only
            "node_0005": {"cell_id": "c", "status": "cancelled", "metadata": {"axis": "lr", "cancel_reason": "cli"}},
            "node_0006": {"cell_id": "c", "status": "cancelled", "metadata": {"axis": "lr", "cancel_reason": "cap"}},
            "node_0007": {"cell_id": "c", "status": "running", "metadata": {"axis": "lr"}},   # phantom: no spec
            "node_0008": {"cell_id": "c", "status": "crash", "metadata": {"axis": "dropout"}},
            "node_0009": {"cell_id": "c", "status": "running", "metadata": {"axis": "lr"}},
        }
        attempts = cell_attempts(adir, nodes, "c")
        assert [a.node_id for a in attempts] == \
            ["node_0003", "node_0002", "node_0005", "node_0008", "node_0009"]
        assert attempts[1] == Attempt(node_id="node_0002", axis="wd", role="neighbour",
                                      status="keep", submitted_at="T02")
        # a launched-then-cancelled attempt was charged and stays counted;
        # the cap-refused spec, the pending proposal and the phantom do not
        assert attempts[2].status == "cancelled"

    def test_no_orchestrator_dir_means_no_attempts(self, tmp_path):
        assert cell_attempts(tmp_path / "automil", {}, "c") == ()


class TestSubmissionLock:
    def test_the_lock_is_exclusive(self, tmp_path):
        import fcntl
        from automil.cells.phasing import submission_lock

        adir = tmp_path / "automil"
        with submission_lock(adir):
            with open(adir / "orchestrator" / "queue" / ".submission.lock", "a+") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open(adir / "orchestrator" / "queue" / ".submission.lock", "a+") as other:
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)   # released


class TestBatchPosition:
    def test_describes_the_next_attempt(self):
        done = [_attempt("node_0002", "lr"), _attempt("node_0003", "wd")]
        line = batch_position(POLICY, tuple(done), frozenset({"node_0003"}))
        assert line == "phasing: 2/5 submitted, next is attempt 3 in batch 2 of 3 (attempts 3-4), 1 in flight"

    def test_says_when_the_budget_is_spent(self):
        done = [_attempt(f"node_{i:04d}", "lr") for i in range(2, 7)]
        assert batch_position(POLICY, tuple(done), frozenset()).endswith("all 5 attempts submitted")


# ---------------------------------------------------------------------------
# The CLI: propose records, submit refuses, cell status reports
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    (path / "model.py").write_text("print('model')\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _phased_project(tmp_path: Path, monkeypatch) -> tuple[CliRunner, Path]:
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, result.output
    adir = tmp_path / "automil"
    config_path = adir / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg["project"] = {**(cfg.get("project") or {}), "name": "test_ds"}
    cfg["encoders"] = {**(cfg.get("encoders") or {}), "primary": "test_enc"}
    cfg["task"] = {**(cfg.get("task") or {}), "name": "test_task"}
    cfg["cap"] = {"mode": "wall_clock", "eval_budget": 5, "phasing": {
        "batches": [2, 2, 1], "opening_axes_min": 2,
        "max_consecutive_per_axis": 2, "reserve_neighbours_min": 1}}
    config_path.write_text(yaml.safe_dump(cfg))

    from automil.graph import ExperimentGraph
    graph = ExperimentGraph(path=str(adir / "graph.json"))
    root = graph.add_executed(parent_id=None, description="baseline", techniques=[],
                              metrics={"primary_value": 0.6, "val_auc": 0.6}, status="keep")
    graph.get_node(root)["primary_value"] = 0.6
    graph.meta["best_node_id"] = root
    graph.save()
    return runner, adir


def _propose(runner, parent, axis, *, role=None, desc=None, delta="0.01", kind="hp"):
    args = ["propose", "--parent", parent, "--desc", desc or f"{axis} {role or ''}".strip(),
            "--kind", kind, "--mil-model", "root"]
    if axis is not None:
        args += ["--axis", axis]
    if delta is not None:
        args += ["--predicted-delta", delta]
    if role:
        args += ["--role", role]
    return runner.invoke(main, args)


def _node_id(result) -> str:
    assert result.exit_code == 0, result.output
    return result.output.split("Added proposal ")[1].split()[0]


def _submit(runner, node, parent):
    return runner.invoke(main, ["submit", "--node", node, "--desc", f"run {node}",
                               "--files", "model.py", "--mil-model", "root",
                               "--parent", parent])


def _complete(adir: Path, node: str, status: str = "discard") -> None:
    """Simulate the daemon launching and finishing ``node``: the spec moves
    from the queue to the node's archive (the record the census walks) and
    the graph node is terminal."""
    queued = adir / "orchestrator" / "queue" / f"{node}.json"
    archive = adir / "orchestrator" / "archive" / node
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "spec.json").write_text(queued.read_text())
    queued.unlink()
    graph = json.loads((adir / "graph.json").read_text())
    graph["nodes"][node]["type"] = "executed"
    graph["nodes"][node]["status"] = status
    (adir / "graph.json").write_text(json.dumps(graph))


class TestProposeRecordsThePhasingFields:
    def test_axis_and_predicted_delta_are_required(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        assert _propose(runner, "node_0001", None).exit_code != 0
        assert _propose(runner, "node_0001", "lr", delta=None).exit_code != 0
        node = _node_id(_propose(runner, "node_0001", "lr"))
        meta = json.loads((adir / "graph.json").read_text())["nodes"][node]["metadata"]
        assert meta["axis"] == "lr" and meta["predicted_delta"] == 0.01
        assert "role" not in meta

    def test_a_neighbour_must_be_proposed_under_the_best_node(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        child = _node_id(_propose(runner, "node_0001", "lr"))
        refused = _propose(runner, child, "lr", role="neighbour")
        assert refused.exit_code != 0 and "best node" in refused.output
        ok = _propose(runner, "node_0001", "lr", role="neighbour")
        meta = json.loads((adir / "graph.json").read_text())["nodes"][_node_id(ok)]["metadata"]
        assert meta["role"] == "neighbour"


class TestSubmitEnforcesThePhasing:
    def test_the_second_batch_waits_for_the_first(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        a1 = _node_id(_propose(runner, "node_0001", "lr"))
        a2 = _node_id(_propose(runner, "node_0001", "wd"))
        assert _submit(runner, a1, "node_0001").exit_code == 0
        assert _submit(runner, a2, "node_0001").exit_code == 0
        a3 = _node_id(_propose(runner, "node_0001", "lr", desc="lr, batch 2"))
        refused = _submit(runner, a3, "node_0001")
        assert refused.exit_code != 0 and "batch 2" in refused.output
        assert not (adir / "orchestrator" / "queue" / f"{a3}.json").exists()
        _complete(adir, a1)
        _complete(adir, a2)
        assert _submit(runner, a3, "node_0001").exit_code == 0

    def test_the_opening_batch_must_span_the_declared_axes(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        a1 = _node_id(_propose(runner, "node_0001", "lr"))
        assert _submit(runner, a1, "node_0001").exit_code == 0
        same_axis = _node_id(_propose(runner, "node_0001", "lr", desc="lr again"))
        refused = _submit(runner, same_axis, "node_0001")
        assert refused.exit_code != 0 and "distinct axes" in refused.output
        other = _node_id(_propose(runner, "node_0001", "wd"))
        assert _submit(runner, other, "node_0001").exit_code == 0

    def test_the_final_batch_reserves_a_neighbour(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        for batch in (("lr", "wd"), ("lr", "dropout")):
            ids = [_node_id(_propose(runner, "node_0001", axis, desc=f"{axis} {batch}"))
                   for axis in batch]
            for node in ids:
                assert _submit(runner, node, "node_0001").exit_code == 0, node
            for node in ids:
                _complete(adir, node)
        plain = _node_id(_propose(runner, "node_0001", "wd", desc="final plain"))
        refused = _submit(runner, plain, "node_0001")
        assert refused.exit_code != 0 and "neighbour" in refused.output
        neighbour = _node_id(_propose(runner, "node_0001", "wd", role="neighbour"))
        assert _submit(runner, neighbour, "node_0001").exit_code == 0

    def test_submission_order_is_what_the_rules_judge(self, tmp_path, monkeypatch):
        """Three lr proposals then a wd one, submitted lr, lr, wd, lr: the last
        lr is the third consecutive lr only in node-id order; in submission
        order it follows wd and is admitted."""
        runner, adir = _phased_project(tmp_path, monkeypatch)
        # batches (2,2,1) with max 2 consecutive: lr, wd | lr, lr would be the
        # 2nd consecutive... use the opening batch: lr then wd, then batch 2.
        lr1 = _node_id(_propose(runner, "node_0001", "lr", desc="lr one"))
        lr2 = _node_id(_propose(runner, "node_0001", "lr", desc="lr two"))
        wd = _node_id(_propose(runner, "node_0001", "wd"))
        assert _submit(runner, lr1, "node_0001").exit_code == 0
        assert _submit(runner, wd, "node_0001").exit_code == 0
        _complete(adir, lr1)
        _complete(adir, wd)
        assert _submit(runner, lr2, "node_0001").exit_code == 0     # attempt 3, batch 2
        status = runner.invoke(main, ["cell", "status"])
        assert "3/5 submitted" in status.output

    def test_a_node_submitted_without_a_proposal_is_refused(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        refused = _submit(runner, "node_0042", "node_0001")
        assert refused.exit_code != 0 and "axis" in refused.output

    def test_cell_status_reports_the_batch_position(self, tmp_path, monkeypatch):
        runner, adir = _phased_project(tmp_path, monkeypatch)
        a1 = _node_id(_propose(runner, "node_0001", "lr"))
        assert _submit(runner, a1, "node_0001").exit_code == 0
        status = runner.invoke(main, ["cell", "status"])
        assert status.exit_code == 0, status.output
        assert "phasing: 1/5 submitted, next is attempt 2 in batch 1 of 3 (attempts 1-2), 1 in flight" in status.output
