"""Unit contracts for the operator CLI ``campaign_operate.py``.

Everything is exercised at the subprocess boundary: the module's ``_capture``,
``_run_or_die`` and ``_popen`` seams are replaced with recorders, so no tmux,
claude, orchestrator or real controller ever runs here. Daemon liveness uses
fixture pid/gpu_state files plus a monkeypatched starttime check — the pid
files themselves are parsed by the daemon's real loader.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "campaign_operate.py"
CELL_NAME = "tcga_luad__kras__uni_v2__clam__s42__preprint-v2"
BUDGET_CELL_ID = "deadbeefdeadbeef"


@pytest.fixture()
def operate():
    spec = importlib.util.spec_from_file_location("campaign_operate_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def make_cell(
    tmp_path: Path, name: str = CELL_NAME, runtime: str = "runtime", port: int = 9581,
) -> Path:
    cell = tmp_path / "campaign" / runtime / name
    adir = cell / "automil"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "config.yaml").write_text(f"activity:\n  exporter_port: {port}\n")
    (adir / "campaign_cell.json").write_text(json.dumps(
        {"budget_identity": {"cell_id": BUDGET_CELL_ID}}
    ))
    return cell


def append_journal(cell: Path, event: dict) -> None:
    journal = cell / "automil" / ".activity.jsonl"
    with journal.open("a") as handle:
        handle.write(json.dumps(event) + "\n")


def session_open_event(session_id: str = "sess-1", observed_at: float = 100.0) -> dict:
    return {
        "event": "session_open", "cell_id": BUDGET_CELL_ID,
        "session_id": session_id, "observed_at": observed_at,
    }


def session_end_event(session_id: str = "sess-1", observed_at: float = 900.0) -> dict:
    return {
        "event": "session_end", "cell_id": BUDGET_CELL_ID,
        "session_id": session_id, "observed_at": observed_at,
        "final_sample_observed_at": observed_at - 1.0,
    }


def write_agent_session(cell: Path, status: str = "finalized", session_id: str = "sess-1") -> None:
    (cell / "agent_session.json").write_text(json.dumps(
        {"status": status, "session": {"session_id": session_id}}
    ))


def write_live_daemon(orch_dir: Path, pid: int = 4242, gpus: list[int] | None = None) -> None:
    orch_dir.mkdir(parents=True, exist_ok=True)
    (orch_dir / "orchestrator.pid").write_text(json.dumps(
        {"pid": pid, "starttime_ticks": 7, "starttime_iso": "2026-08-15T00:00:00"}
    ) + "\n")
    if gpus is not None:
        (orch_dir / "gpu_state.json").write_text(json.dumps({
            "gpus": {str(index): {"running": []} for index in gpus},
            "execution_slots": {
                f"cuda:{index}": {
                    "accelerator": "cuda", "device_index": index, "running": [],
                }
                for index in gpus
            },
        }))


class FakeClock:
    def __init__(self) -> None:
        self.time = 0.0
        self.sleeps: list[float] = []
        self.on_sleep = None

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.time += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def classify(argv: list) -> tuple[str, object]:
    argv = [str(token) for token in argv]
    for index, token in enumerate(argv):
        if token.endswith("campaign_stage.py"):
            return "stage", argv[index + 1]
        if token.endswith("campaign_launch.py"):
            return "launch", argv[index + 1]
        if token == "automil":
            return "automil", tuple(argv[index + 3:])
    return "other", tuple(argv)


class FakeBoundary:
    """Recording stand-in for the module's three subprocess seams."""

    def __init__(self, module, statuses: list[dict]) -> None:
        self.module = module
        self.statuses = list(statuses)
        self.captures: list[list[str]] = []
        self.run_or_die: list[list[str]] = []
        self.popens: list[tuple[list[str], dict, object]] = []
        self.behaviors: dict[object, object] = {}

    def capture(self, argv, env=None):
        self.captures.append(list(argv))
        kind, action = classify(argv)
        assert kind == "stage" and action == "status", (
            f"unexpected captured subprocess in this test: {argv}"
        )
        assert self.statuses, "test ran out of canned status payloads"
        payload = self.statuses.pop(0)
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload), stderr="",
        )

    def run(self, argv, env=None):
        self.run_or_die.append(list(argv))
        behavior = self.behaviors.get(classify(argv))
        if behavior is not None:
            behavior(argv)

    def popen(self, argv, env, stdout, stderr):
        child = FakeChild()
        self.popens.append((list(argv), dict(env), stdout))
        pid_dir = Path(argv[argv.index("--project", 3) + 1]) / "automil" / "orchestrator"
        write_live_daemon(pid_dir, pid=4242)
        return child

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(self.module, "_capture", self.capture)
        monkeypatch.setattr(self.module, "_run_or_die", self.run)
        monkeypatch.setattr(self.module, "_popen", self.popen)


class FakeChild:
    pid = 4242
    returncode: int | None = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


def actions(recorded: list[list[str]]) -> list[tuple[str, object]]:
    return [classify(argv) for argv in recorded]


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def test_stage_argv_uses_pinned_workspace_form(operate, tmp_path):
    cell = tmp_path / "cell"
    argv = operate.stage_argv("freeze-discovery", cell)
    assert argv == [
        "uv", "run", "--project", str(operate.REPO_ROOT),
        "--package", "autobench", "python",
        str(operate.REPO_ROOT / "benchmarks" / "scripts" / "campaign_stage.py"),
        "freeze-discovery", "--cell-root", str(cell),
    ]


def test_automil_argv_pins_workspace_and_project(operate, tmp_path):
    project = tmp_path / "cell" / "promotion"
    argv = operate.automil_argv(project, "orchestrator", "stop")
    assert argv == [
        "uv", "run", "--project", str(operate.REPO_ROOT),
        "automil", "--project", str(project), "orchestrator", "stop",
    ]


def test_launcher_argv_form(operate, tmp_path):
    cell = tmp_path / "cell"
    argv = operate.launcher_argv("launch", cell)
    assert argv == [
        "uv", "run", "--project", str(operate.REPO_ROOT),
        "--package", "autobench", "python",
        str(operate.REPO_ROOT / "benchmarks" / "scripts" / "campaign_launch.py"),
        "launch", "--cell-root", str(cell),
    ]


def test_orchestrator_window_command_sets_gpu_partition_inline(operate, tmp_path):
    cell = tmp_path / "cell"
    command = operate.orchestrator_window_command(cell, 3)
    assert command.startswith("AUTOMIL_VISIBLE_GPUS=3 ")
    assert command.endswith("orchestrator start")
    assert str(cell) in command


def test_baseline_window_command_carries_gpu(operate, tmp_path):
    command = operate.baseline_window_command(tmp_path / "cell", 0)
    assert "run-baseline" in command
    assert command.endswith("--gpu 0")


def test_session_name_uses_distinguishing_tokens(operate, tmp_path):
    assert operate._session_name(tmp_path / CELL_NAME) == "uni_v2-clam"
    assert operate._session_name(tmp_path / "throwaway.cell") == "throwaway_cell"


# ---------------------------------------------------------------------------
# GPU-claim refusal
# ---------------------------------------------------------------------------
def test_gpu_claim_refuses_other_cells_live_partition(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=[1])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, 1)
    assert operate._gpu_claim_conflicts(ours, 0) == []


def test_gpu_claim_covers_sibling_promotion_daemons(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(
        other / "promotion" / "automil" / "orchestrator", pid=1234, gpus=[2],
    )
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    conflicts = operate._gpu_claim_conflicts(ours, 2)
    assert conflicts and "promotion" in conflicts[0]


def test_gpu_claim_covers_twin_runtime_roots(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name=CELL_NAME, runtime="runtime")
    twin = make_cell(tmp_path, name=CELL_NAME, runtime="runtime-canary")
    write_live_daemon(twin / "automil" / "orchestrator", pid=1234, gpus=[0])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, 0)


def test_gpu_claim_same_cell_discovery_promotion_pair_is_exempt(
    operate, tmp_path, monkeypatch,
):
    ours = make_cell(tmp_path)
    write_live_daemon(ours / "automil" / "orchestrator", pid=1111, gpus=[1])
    write_live_daemon(
        ours / "promotion" / "automil" / "orchestrator", pid=2222, gpus=[1],
    )
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, 1) == []


def test_gpu_claim_ignores_dead_daemons(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=[1])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: False)

    assert operate._gpu_claim_conflicts(ours, 1) == []


def test_gpu_claim_treats_legacy_plain_int_pid_file_as_stale(
    operate, tmp_path, monkeypatch,
):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    orch = other / "automil" / "orchestrator"
    orch.mkdir(parents=True)
    (orch / "orchestrator.pid").write_text("1234\n")  # legacy shape
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, 1) == []


def test_gpu_claim_unknown_partition_conflicts_on_every_gpu(
    operate, tmp_path, monkeypatch,
):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=None)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    for gpu in (0, 1, 7):
        conflicts = operate._gpu_claim_conflicts(ours, gpu)
        assert conflicts and "unknown" in conflicts[0]


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------
def test_bind_payload_is_exactly_two_fields_utc_seconds(operate):
    payload = operate._bind_payload(session_open_event("sess-9", observed_at=0.0))
    assert payload == {"session_id": "sess-9", "started_at": "1970-01-01T00:00:00+00:00"}
    fractional = operate._bind_payload(
        session_open_event("sess-9", observed_at=1234567890.6)
    )
    assert fractional["started_at"] == "2009-02-13T23:31:30+00:00"


def test_bind_takes_the_newest_session_open(operate, tmp_path):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event("sess-old", observed_at=10.0))
    append_journal(cell, session_open_event("sess-new", observed_at=20.0))
    event = operate._newest_session_open(cell)
    assert event["session_id"] == "sess-new"


def test_bind_retries_only_on_the_metrics_error_then_releases(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event("sess-1", observed_at=50.0))
    clock = FakeClock()
    monkeypatch.setattr(operate, "_now", clock.now)
    monkeypatch.setattr(operate, "_sleep", clock.sleep)

    calls: list[list[str]] = []

    def fake_capture(argv, env=None):
        calls.append(list(argv))
        if len(calls) <= 2:
            return subprocess.CompletedProcess(
                argv, 2, stdout="",
                stderr=f"campaign-stage error: {operate.METRICS_RETRY_ERROR}\n",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="{}\n", stderr="")

    monkeypatch.setattr(operate, "_capture", fake_capture)
    operate.main(["bind", str(cell), "--timeout-s", "600"])

    assert [classify(argv) for argv in calls] == [
        ("stage", "open-agent-session"),
    ] * 3
    assert clock.sleeps.count(operate.BIND_RETRY_SECONDS) == 2
    out = capsys.readouterr().out
    assert "Session is bound. Begin the discovery loop per your policy." in out.splitlines()
    # The attestation file passed to the controller is the exact 2-field JSON.
    session_arg = Path(calls[0][calls[0].index("--agent-session") + 1])
    assert json.loads(session_arg.read_text()) == {
        "session_id": "sess-1",
        "started_at": datetime.fromtimestamp(50.0, tz=timezone.utc)
        .isoformat(timespec="seconds"),
    }


def test_bind_any_other_refusal_is_fatal_and_verbatim(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    clock = FakeClock()
    monkeypatch.setattr(operate, "_now", clock.now)
    monkeypatch.setattr(operate, "_sleep", clock.sleep)
    refusal = "campaign-stage error: agent session must open during discovery\n"
    calls: list[list[str]] = []

    def fake_capture(argv, env=None):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr=refusal)

    monkeypatch.setattr(operate, "_capture", fake_capture)
    with pytest.raises(SystemExit) as excinfo:
        operate.main(["bind", str(cell)])

    assert excinfo.value.code == 2
    assert len(calls) == 1
    assert refusal in capsys.readouterr().err


def test_bind_times_out_while_metrics_error_persists(
    operate, tmp_path, monkeypatch,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    clock = FakeClock()
    monkeypatch.setattr(operate, "_now", clock.now)
    monkeypatch.setattr(operate, "_sleep", clock.sleep)

    def fake_capture(argv, env=None):
        return subprocess.CompletedProcess(
            argv, 2, stdout="",
            stderr=f"campaign-stage error: {operate.METRICS_RETRY_ERROR}\n",
        )

    monkeypatch.setattr(operate, "_capture", fake_capture)
    with pytest.raises(SystemExit) as excinfo:
        operate.main(["bind", str(cell), "--timeout-s", "45"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# finish — session-end preconditions
# ---------------------------------------------------------------------------
def test_finish_refuses_without_session_end_while_exporter_serves(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    write_agent_session(cell, status="open")
    monkeypatch.setattr(operate, "_port_in_use", lambda port: True)

    with pytest.raises(SystemExit):
        operate.main(["finish", str(cell)])
    err = capsys.readouterr().err
    assert "/exit" in err
    assert "SessionEnd" in err


def test_finish_requires_attest_when_exporter_dead(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    write_agent_session(cell, status="open")
    monkeypatch.setattr(operate, "_port_in_use", lambda port: False)

    with pytest.raises(SystemExit):
        operate.main(["finish", str(cell)])
    err = capsys.readouterr().err
    assert "--attest" in err
    assert "activity close" in err


def test_finish_runs_activity_close_with_operator_attestation(
    operate, tmp_path, monkeypatch,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    write_agent_session(cell, status="finalized")
    monkeypatch.setattr(operate, "_port_in_use", lambda port: False)

    boundary = FakeBoundary(operate, statuses=[{"phase": "winner-frozen"}])
    attest = "runtime died before SessionEnd: power loss"

    def do_close(argv):
        append_journal(cell, session_end_event())

    boundary.behaviors[
        ("automil", ("activity", "close", "--session", "sess-1", "--attest", attest))
    ] = do_close
    boundary.install(monkeypatch)

    operate.main(["finish", str(cell), "--attest", attest])
    assert ("automil", ("activity", "close", "--session", "sess-1", "--attest", attest)) in actions(boundary.run_or_die)


# ---------------------------------------------------------------------------
# finish — phase-based state machine
# ---------------------------------------------------------------------------
def test_finish_winner_frozen_finalized_session_only_reports(
    operate, tmp_path, monkeypatch,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="finalized")

    boundary = FakeBoundary(operate, statuses=[{"phase": "winner-frozen"}])
    boundary.install(monkeypatch)
    operate.main(["finish", str(cell)])

    assert actions(boundary.run_or_die) == [("stage", "status")]


def test_finish_selection_ready_selects_winner_and_finalizes(
    operate, tmp_path, monkeypatch,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")

    boundary = FakeBoundary(operate, statuses=[
        {"phase": "selection-ready"},
        {"phase": "winner-frozen"},
    ])
    boundary.install(monkeypatch)
    operate.main(["finish", str(cell)])

    recorded = actions(boundary.run_or_die)
    assert recorded == [
        ("stage", "select-winner"),
        ("stage", "finalize-agent-session"),
        ("stage", "status"),
    ]
    finalize_argv = boundary.run_or_die[1]
    end_path = Path(finalize_argv[finalize_argv.index("--agent-session") + 1])
    payload = json.loads(end_path.read_text())
    assert set(payload) == {"session_id", "ended_at", "termination_reason", "usage"}
    assert payload["session_id"] == "sess-1"
    assert payload["termination_reason"] == "budget-complete"
    assert payload["usage"] == {
        "status": "unavailable",
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "cost_usd": None,
        "basis": "operator finish: CLI usage summary not captured",
    }
    ended = datetime.fromisoformat(payload["ended_at"])
    assert ended.tzinfo is not None


def test_finish_passes_usage_json_verbatim(operate, tmp_path, monkeypatch):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    usage = {
        "status": "exact", "input_tokens": 10, "output_tokens": 20,
        "cached_input_tokens": 5, "cost_usd": 1.25, "basis": "CLI /usage",
    }
    usage_path = tmp_path / "usage.json"
    usage_path.write_text(json.dumps(usage))

    boundary = FakeBoundary(operate, statuses=[{"phase": "winner-frozen"}])
    boundary.install(monkeypatch)
    operate.main(["finish", str(cell), "--usage-json", str(usage_path)])

    finalize_argv = next(
        argv for argv in boundary.run_or_die
        if classify(argv) == ("stage", "finalize-agent-session")
    )
    end_path = Path(finalize_argv[finalize_argv.index("--agent-session") + 1])
    assert json.loads(end_path.read_text())["usage"] == usage


def test_finish_discovery_phase_freezes_then_walks_the_chain(
    operate, tmp_path, monkeypatch,
):
    """Full walk: discovery → freeze → materialize → promotion (adopt) →
    freeze-promotion → select-winner → finalize, with skips consulted from
    the controller's own status between transitions."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    # Discovery daemon not running (no pid file) — stop is skipped.
    # Promotion project: materialized with one queued job, live daemon to adopt.
    padir = cell / "promotion" / "automil"
    (padir / "cells").mkdir(parents=True)
    (padir / "campaign_cell.json").write_text(json.dumps(
        {"budget_identity": {"cell_id": BUDGET_CELL_ID}}
    ))
    queue = padir / "orchestrator" / "queue"
    queue.mkdir(parents=True)
    (queue / "0001.json").write_text("{}")
    cell_json = padir / "cells" / f"{BUDGET_CELL_ID}.json"
    cell_json.write_text(json.dumps({"consumed_evals": 1, "eval_budget": 2}))
    write_live_daemon(padir / "orchestrator", pid=777)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    boundary = FakeBoundary(operate, statuses=[
        {"phase": "discovery"},
        {"phase": "promotion-ready"},
        {"phase": "promotion"},
        {"phase": "selection-ready"},
        {"phase": "winner-frozen"},
    ])

    def drain(argv=None):
        (queue / "0001.json").unlink()
        cell_json.write_text(json.dumps({"consumed_evals": 2, "eval_budget": 2}))

    def stop_promotion(argv):
        (padir / "orchestrator" / "orchestrator.pid").unlink()

    boundary.behaviors[("automil", ("orchestrator", "stop"))] = stop_promotion
    boundary.install(monkeypatch)
    clock = FakeClock()
    clock.on_sleep = drain
    monkeypatch.setattr(operate, "_now", clock.now)
    monkeypatch.setattr(operate, "_sleep", clock.sleep)

    operate.main(["finish", str(cell)])

    assert actions(boundary.run_or_die) == [
        ("stage", "freeze-discovery"),
        ("stage", "materialize-promotion"),
        ("automil", ("orchestrator", "stop")),
        ("stage", "freeze-promotion"),
        ("stage", "select-winner"),
        ("stage", "finalize-agent-session"),
        ("stage", "status"),
    ]
    # Adopted, never started: the supervised-child seam stayed untouched.
    assert boundary.popens == []


def test_finish_starting_promotion_daemon_requires_explicit_gpu(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    padir = cell / "promotion" / "automil"
    (padir / "cells").mkdir(parents=True)
    (padir / "campaign_cell.json").write_text(json.dumps(
        {"budget_identity": {"cell_id": BUDGET_CELL_ID}}
    ))
    queue = padir / "orchestrator" / "queue"
    queue.mkdir(parents=True)
    (queue / "0001.json").write_text("{}")
    (padir / "cells" / f"{BUDGET_CELL_ID}.json").write_text(
        json.dumps({"consumed_evals": 0, "eval_budget": 1})
    )

    boundary = FakeBoundary(operate, statuses=[{"phase": "promotion"}])
    boundary.install(monkeypatch)
    with pytest.raises(SystemExit):
        operate.main(["finish", str(cell)])
    err = capsys.readouterr().err
    assert "--gpu" in err
    assert "AUTOMIL_VISIBLE_GPUS" in err
    assert boundary.popens == []


def test_finish_starts_supervised_promotion_child_with_gpu_partition(
    operate, tmp_path, monkeypatch,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="finalized")
    padir = cell / "promotion" / "automil"
    (padir / "cells").mkdir(parents=True)
    (padir / "campaign_cell.json").write_text(json.dumps(
        {"budget_identity": {"cell_id": BUDGET_CELL_ID}}
    ))
    queue = padir / "orchestrator" / "queue"
    queue.mkdir(parents=True)
    (queue / "0001.json").write_text("{}")
    cell_json = padir / "cells" / f"{BUDGET_CELL_ID}.json"
    cell_json.write_text(json.dumps({"consumed_evals": 0, "eval_budget": 1}))
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    boundary = FakeBoundary(operate, statuses=[
        {"phase": "promotion"},
        {"phase": "selection-ready"},
        {"phase": "winner-frozen"},
    ])

    def drain():
        if (queue / "0001.json").exists():
            (queue / "0001.json").unlink()
            cell_json.write_text(json.dumps({"consumed_evals": 1, "eval_budget": 1}))

    def stop_promotion(argv):
        pid_file = padir / "orchestrator" / "orchestrator.pid"
        if pid_file.exists():
            pid_file.unlink()

    boundary.behaviors[("automil", ("orchestrator", "stop"))] = stop_promotion
    boundary.install(monkeypatch)
    clock = FakeClock()
    clock.on_sleep = drain
    monkeypatch.setattr(operate, "_now", clock.now)
    monkeypatch.setattr(operate, "_sleep", clock.sleep)

    operate.main(["finish", str(cell), "--gpu", "3"])

    assert len(boundary.popens) == 1
    argv, env, stdout = boundary.popens[0]
    assert classify(argv) == ("automil", ("orchestrator", "start"))
    assert str(cell / "promotion") in argv
    assert env["AUTOMIL_VISIBLE_GPUS"] == "3"
    assert Path(stdout.name) == padir / "orchestrator" / "operate_supervisor.log"
    assert ("stage", "freeze-promotion") in actions(boundary.run_or_die)


def test_finish_reentry_skips_completed_transitions(operate, tmp_path, monkeypatch):
    """A finish re-run on a promotion-frozen cell touches nothing before
    select-winner: phase decides, not history."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="finalized")

    boundary = FakeBoundary(operate, statuses=[
        {"phase": "selection-ready"},
        {"phase": "winner-frozen"},
    ])
    boundary.install(monkeypatch)
    operate.main(["finish", str(cell)])

    recorded = actions(boundary.run_or_die)
    assert ("stage", "freeze-discovery") not in recorded
    assert ("stage", "materialize-promotion") not in recorded
    assert ("stage", "freeze-promotion") not in recorded
    assert recorded[0] == ("stage", "select-winner")


# ---------------------------------------------------------------------------
# Refusal passthrough
# ---------------------------------------------------------------------------
def test_finish_controller_refusal_passes_through_verbatim(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    refusal = (
        "campaign-stage error: discovery requires exactly 30 charged attempts\n"
    )

    boundary = FakeBoundary(operate, statuses=[{"phase": "discovery"}])

    def refuse(argv):
        import sys as _sys
        _sys.stderr.write(refusal)
        raise SystemExit(2)

    boundary.behaviors[("stage", "freeze-discovery")] = refuse
    boundary.install(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        operate.main(["finish", str(cell)])
    assert excinfo.value.code == 2
    assert refusal in capsys.readouterr().err
    # Nothing past the refused transition ran.
    assert actions(boundary.run_or_die)[-1] == ("stage", "freeze-discovery")


def test_status_refusal_passes_through_verbatim(operate, tmp_path, monkeypatch, capsys):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    refusal = "campaign-stage error: stage state belongs to a different campaign\n"

    def fake_capture(argv, env=None):
        return subprocess.CompletedProcess(argv, 2, stdout="", stderr=refusal)

    monkeypatch.setattr(operate, "_capture", fake_capture)
    with pytest.raises(SystemExit) as excinfo:
        operate.main(["finish", str(cell)])
    assert excinfo.value.code == 2
    assert refusal in capsys.readouterr().err
