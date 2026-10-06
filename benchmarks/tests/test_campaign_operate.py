"""Unit contracts for the operator CLI ``campaign_operate.py``.

Everything is exercised at the subprocess boundary: the module's ``_capture``
and ``_run_or_die`` seams are replaced with recorders, so no tmux, claude,
orchestrator or real controller ever runs here. Daemon liveness uses
fixture pid/gpu_state files plus a monkeypatched starttime check — the pid
files themselves are parsed by the daemon's real loader.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from automil.cells.state import Cell, CellStatus, write_cell

from _helpers import full_h100_nvidia_smi, write_reproduction_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "campaign_operate.py"
CELL_NAME = "tcga_luad__kras__uni_v2__clam__s42__preprint-v5"
BUDGET_CELL_ID = "deadbeefdeadbeef"


@pytest.fixture(autouse=True)
def _full_h100_node(monkeypatch):
    """Every test runs on a node of full H100s unless it forges another GPU."""
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        answer = full_h100_nvidia_smi(command)
        return answer if answer is not None else real_run(command, *args, **kwargs)

    monkeypatch.setattr("autobench.campaign_gpu.subprocess.run", run)


RTX_GPU = {"name": "NVIDIA RTX 6000 Ada Generation", "mig": False}


def _forge_node(monkeypatch, name: str, mig_mode: str):
    """nvidia-smi lists four GPUs of one type and MIG mode."""
    real_run = subprocess.run

    def run(command, *args, **kwargs):
        if command and command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(
                command, 0, stdout="".join(
                    f"{index}, {name}, {mig_mode}\n" for index in range(4)
                ), stderr="",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr("autobench.campaign_gpu.subprocess.run", run)


def _forge_mig_slice(monkeypatch):
    _forge_node(monkeypatch, "NVIDIA H100 80GB HBM3", "Enabled")


def _forge_rtx_workstation(monkeypatch):
    _forge_node(monkeypatch, RTX_GPU["name"], "[N/A]")


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


def write_live_daemon(
    orch_dir: Path,
    pid: int = 4242,
    gpus: list[int] | None = None,
    hostname: str | None = None,
    slurm_job_id: str | None = None,
) -> None:
    orch_dir.mkdir(parents=True, exist_ok=True)
    payload = {"pid": pid, "starttime_ticks": 7, "starttime_iso": "2026-08-15T00:00:00"}
    if hostname is not None:
        payload["hostname"] = hostname
    if slurm_job_id is not None:
        payload["slurm_job_id"] = slurm_job_id
    (orch_dir / "orchestrator.pid").write_text(json.dumps(payload) + "\n")
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


def write_budget_cell(adir: Path, consumed: int, budget: int | None) -> Path:
    """A REAL typed budget cell — campaign_operate reads via read_cell now."""
    write_cell(
        Cell(
            cell_id=BUDGET_CELL_ID,
            dataset="dataset",
            encoder="encoder",
            mil_model="model",
            started_at=1.0,
            budget_seconds=10_000,
            safety_buffer_seconds=10,
            status=CellStatus.ACTIVE,
            mode="wall_clock",
            eval_budget=budget,
            consumed_evals=consumed,
            completed_evals=consumed,
        ),
        adir / "cells",
    )
    return adir / "cells" / f"{BUDGET_CELL_ID}.json"


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
    """Recording stand-in for the module's subprocess seams."""

    def __init__(self, module, statuses: list[dict]) -> None:
        self.module = module
        self.statuses = list(statuses)
        self.captures: list[list[str]] = []
        self.run_or_die: list[list[str]] = []
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

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(self.module, "_capture", self.capture)
        monkeypatch.setattr(self.module, "_run_or_die", self.run)


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
    project = tmp_path / "cell"
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
    command = operate.orchestrator_window_command(cell, [3])
    assert command.startswith("AUTOMIL_VISIBLE_GPUS=3 ")
    assert command.endswith("orchestrator start")
    assert str(cell) in command


def test_operate_script_imports_in_a_fresh_interpreter():
    # The shape predictor is loaded by path at import time; a fresh process
    # has nothing else registered in sys.modules to mask a loading mistake
    # (the shape tests register the module during their own collection).
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('campaign_operate', {str(SCRIPT)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "print(module.MAX_CONCURRENT_PER_GPU)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8"


def test_orchestrator_window_command_sets_the_shape_predictor_cap(operate, tmp_path):
    # The daemon packs as many attempts per GPU as the shape predictor
    # assumed when it sized the job, whatever the frozen cell config says.
    command = operate.orchestrator_window_command(tmp_path / "cell", [0])
    assert f" AUTOMIL_MAX_CONCURRENT_PER_GPU={operate.MAX_CONCURRENT_PER_GPU} " in command
    assert operate.MAX_CONCURRENT_PER_GPU == 8


def test_orchestrator_window_command_joins_a_multi_gpu_list(operate, tmp_path):
    cell = tmp_path / "cell"
    command = operate.orchestrator_window_command(cell, [0, 1])
    assert command.startswith("AUTOMIL_VISIBLE_GPUS=0,1 ")


def test_baseline_window_command_carries_gpu(operate, tmp_path):
    command = operate.baseline_window_command(tmp_path / "cell", [0])
    assert "run-baseline" in command
    assert command.endswith("--gpu 0")


def test_baseline_window_command_uses_only_the_first_gpu_index(operate, tmp_path):
    command = operate.baseline_window_command(tmp_path / "cell", [2, 3])
    assert command.endswith("--gpu 2")


# ---------------------------------------------------------------------------
# --gpu parsing: single index or comma list
# ---------------------------------------------------------------------------
def test_parse_gpu_list_accepts_single_index(operate):
    assert operate._parse_gpu_list("0") == [0]


def test_parse_gpu_list_accepts_comma_list(operate):
    assert operate._parse_gpu_list("0,1,2,3") == [0, 1, 2, 3]


def test_parse_gpu_list_rejects_duplicates(operate):
    with pytest.raises(operate.argparse.ArgumentTypeError):
        operate._parse_gpu_list("1,1")


def test_parse_gpu_list_rejects_non_integer(operate):
    with pytest.raises(operate.argparse.ArgumentTypeError):
        operate._parse_gpu_list("a")


def test_parse_gpu_list_rejects_negative(operate):
    with pytest.raises(operate.argparse.ArgumentTypeError):
        operate._parse_gpu_list("-1")


def test_parse_gpu_list_rejects_empty(operate):
    with pytest.raises(operate.argparse.ArgumentTypeError):
        operate._parse_gpu_list("")
    with pytest.raises(operate.argparse.ArgumentTypeError):
        operate._parse_gpu_list("0,,1")


def test_up_gpu_argparse_accepts_comma_list(operate, tmp_path):
    parser = operate.build_parser()
    args = parser.parse_args(["up", str(tmp_path / "cell"), "--gpu", "0,1"])
    assert args.gpu == [0, 1]


def test_up_gpu_argparse_single_index_unchanged(operate, tmp_path):
    parser = operate.build_parser()
    args = parser.parse_args(["up", str(tmp_path / "cell"), "--gpu", "0"])
    assert args.gpu == [0]


def test_up_gpu_argparse_rejects_duplicate(operate, tmp_path, capsys):
    parser = operate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["up", str(tmp_path / "cell"), "--gpu", "1,1"])
    assert "--gpu" in capsys.readouterr().err


def test_up_gpu_argparse_rejects_non_integer(operate, tmp_path, capsys):
    parser = operate.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["up", str(tmp_path / "cell"), "--gpu", "a"])
    assert "--gpu" in capsys.readouterr().err


def test_finish_parses_with_just_the_cell_root(operate, tmp_path):
    """Finish trains nothing, so it takes no GPU: the cell root alone, and the
    two session options that default to none."""
    parser = operate.build_parser()
    args = parser.parse_args(["finish", str(tmp_path / "cell")])
    assert args.cell_root == str(tmp_path / "cell")
    assert args.usage_json is None
    assert args.attest is None


@pytest.mark.parametrize("flag", [["--gpu", "0"], ["--accept-missing"]])
def test_finish_refuses_a_flag_only_a_promotion_run_needed(
    operate, tmp_path, capsys, flag,
):
    """A stale invocation must fail loudly, not be ignored: nothing in finish
    starts a daemon or freezes a partial slate any more."""
    parser = operate.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["finish", str(tmp_path / "cell"), *flag])
    assert excinfo.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_session_name_uses_the_full_sanitized_cell_id(operate, tmp_path):
    assert operate._session_name(tmp_path / CELL_NAME) == CELL_NAME
    assert operate._session_name(tmp_path / "throwaway.cell") == "throwaway_cell"


def test_session_name_distinguishes_cells_sharing_encoder_and_arm(operate, tmp_path):
    """Two cells with the same encoder+arm tokens (different dataset/task/seed)
    must not collapse onto one tmux session name."""
    one = operate._session_name(tmp_path / "tcga_luad__kras__uni_v2__clam__s42__v3")
    other = operate._session_name(tmp_path / "tcga_luad__egfr__uni_v2__clam__s43__v3")
    assert one != other
    for name in (one, other):
        assert ":" not in name
        assert "." not in name


def test_session_name_sanitizes_colons_and_dots(operate, tmp_path):
    name = operate._session_name(tmp_path / "weird:name.with.dots")
    assert name == "weird_name_with_dots"


# ---------------------------------------------------------------------------
# tmux socket isolation (AUTOMIL_TMUX_SOCKET)
# ---------------------------------------------------------------------------
def test_tmux_prepends_dash_l_when_socket_env_var_set(operate, monkeypatch):
    monkeypatch.setenv("AUTOMIL_TMUX_SOCKET", "automil-campaign")
    captured: list[list[str]] = []

    def fake_capture(argv, env=None):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(operate, "_capture", fake_capture)
    operate._tmux("has-session", "-t", "=foo")

    assert captured == [["tmux", "-L", "automil-campaign", "has-session", "-t", "=foo"]]


def test_tmux_unchanged_when_socket_env_var_unset(operate, monkeypatch):
    monkeypatch.delenv("AUTOMIL_TMUX_SOCKET", raising=False)
    captured: list[list[str]] = []

    def fake_capture(argv, env=None):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(operate, "_capture", fake_capture)
    operate._tmux("has-session", "-t", "=foo")

    assert captured == [["tmux", "has-session", "-t", "=foo"]]


def test_tmux_unchanged_when_socket_env_var_empty(operate, monkeypatch):
    monkeypatch.setenv("AUTOMIL_TMUX_SOCKET", "")
    captured: list[list[str]] = []

    def fake_capture(argv, env=None):
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(operate, "_capture", fake_capture)
    operate._tmux("list-windows", "-t", "=foo")

    assert captured == [["tmux", "list-windows", "-t", "=foo"]]


# ---------------------------------------------------------------------------
# daemon liveness — host/job scoping (co-scheduled SLURM jobs on one node)
# ---------------------------------------------------------------------------
def test_daemon_alive_ignores_pid_file_from_another_hostname(
    operate, tmp_path, monkeypatch,
):
    orch = tmp_path / "automil" / "orchestrator"
    write_live_daemon(orch, pid=999, hostname="other-node")
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)
    monkeypatch.setattr(operate.socket, "gethostname", lambda: "this-node")

    assert operate._daemon_alive(orch) is None


def test_daemon_alive_ignores_pid_file_from_another_slurm_job(
    operate, tmp_path, monkeypatch,
):
    orch = tmp_path / "automil" / "orchestrator"
    write_live_daemon(orch, pid=999, slurm_job_id="111")
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)
    monkeypatch.setattr(operate.socket, "gethostname", socket.gethostname)
    monkeypatch.setenv("SLURM_JOB_ID", "222")

    assert operate._daemon_alive(orch) is None


def test_daemon_alive_same_host_and_job_id_is_alive(
    operate, tmp_path, monkeypatch,
):
    orch = tmp_path / "automil" / "orchestrator"
    write_live_daemon(orch, pid=999, hostname="this-node", slurm_job_id="222")
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)
    monkeypatch.setattr(operate.socket, "gethostname", lambda: "this-node")
    monkeypatch.setenv("SLURM_JOB_ID", "222")

    assert operate._daemon_alive(orch) == 999


def test_daemon_alive_workstation_no_job_ids_unaffected(
    operate, tmp_path, monkeypatch,
):
    """Neither the pid file nor our own environment carries a SLURM job id —
    liveness is decided by pid+starttime alone, as before."""
    orch = tmp_path / "automil" / "orchestrator"
    write_live_daemon(orch, pid=999)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    assert operate._daemon_alive(orch) == 999


def test_daemon_alive_old_format_pid_file_unaffected(
    operate, tmp_path, monkeypatch,
):
    """A pid file written before hostname/slurm_job_id existed carries
    neither field — treated as present-or-None, never a mismatch."""
    orch = tmp_path / "automil" / "orchestrator"
    orch.mkdir(parents=True)
    (orch / "orchestrator.pid").write_text(json.dumps(
        {"pid": 999, "starttime_ticks": 7, "starttime_iso": "2026-08-15T00:00:00"}
    ) + "\n")
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)
    monkeypatch.setenv("SLURM_JOB_ID", "222")

    assert operate._daemon_alive(orch) == 999


def test_daemon_alive_still_dead_when_starttime_mismatches(
    operate, tmp_path, monkeypatch,
):
    """Host/job scoping is a new pre-filter, not a replacement for the
    existing pid+starttime cross-check."""
    orch = tmp_path / "automil" / "orchestrator"
    write_live_daemon(orch, pid=999, hostname="this-node", slurm_job_id="222")
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: False)
    monkeypatch.setattr(operate.socket, "gethostname", lambda: "this-node")
    monkeypatch.setenv("SLURM_JOB_ID", "222")

    assert operate._daemon_alive(orch) is None


# ---------------------------------------------------------------------------
# GPU-claim refusal
# ---------------------------------------------------------------------------
def test_gpu_claim_refuses_other_cells_live_partition(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=[1])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, [1])
    assert operate._gpu_claim_conflicts(ours, [0]) == []


def test_gpu_claim_covers_twin_runtime_roots(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name=CELL_NAME, runtime="runtime")
    twin = make_cell(tmp_path, name=CELL_NAME, runtime="runtime-canary")
    write_live_daemon(twin / "automil" / "orchestrator", pid=1234, gpus=[0])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, [0])


def test_gpu_claim_the_cells_own_daemon_is_exempt(
    operate, tmp_path, monkeypatch,
):
    """finish runs next to the cell's own live daemon on the same GPU."""
    ours = make_cell(tmp_path)
    write_live_daemon(ours / "automil" / "orchestrator", pid=1111, gpus=[1])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, [1]) == []


def test_gpu_claim_ignores_dead_daemons(operate, tmp_path, monkeypatch):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=[1])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: False)

    assert operate._gpu_claim_conflicts(ours, [1]) == []


def test_gpu_claim_treats_legacy_plain_int_pid_file_as_stale(
    operate, tmp_path, monkeypatch,
):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    orch = other / "automil" / "orchestrator"
    orch.mkdir(parents=True)
    (orch / "orchestrator.pid").write_text("1234\n")  # legacy shape
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, [1]) == []


def test_gpu_claim_unknown_partition_conflicts_on_every_gpu(
    operate, tmp_path, monkeypatch,
):
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=None)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    for gpu in (0, 1, 7):
        conflicts = operate._gpu_claim_conflicts(ours, [gpu])
        assert conflicts and "unknown" in conflicts[0]


def test_gpu_claim_empty_slot_state_conflicts_on_every_gpu(
    operate, tmp_path, monkeypatch,
):
    """A live daemon whose gpu_state.json parses but records ZERO cuda slots
    (nvidia-smi failure) must claim everything — 'no claims' is never
    inferred from an empty map."""
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    orch = other / "automil" / "orchestrator"
    write_live_daemon(orch, pid=1234, gpus=None)
    (orch / "gpu_state.json").write_text(json.dumps(
        {"gpus": {}, "execution_slots": {}}
    ))
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._claimed_gpus(orch) is None
    for gpu in (0, 1, 7):
        conflicts = operate._gpu_claim_conflicts(ours, [gpu])
        assert conflicts and "unknown" in conflicts[0]


def test_gpu_claim_non_cuda_slot_state_conflicts_on_every_gpu(
    operate, tmp_path, monkeypatch,
):
    """rocm/cpu-shaped slots are not a cuda partition grant: only a readable
    cuda slot map may grant clearance."""
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    orch = other / "automil" / "orchestrator"
    write_live_daemon(orch, pid=1234, gpus=None)
    (orch / "gpu_state.json").write_text(json.dumps({
        "gpus": {},
        "execution_slots": {
            "rocm:0": {"accelerator": "rocm", "device_index": 0, "running": []},
            "cpu": {"accelerator": "cpu", "running": []},
        },
    }))
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._claimed_gpus(orch) is None
    for gpu in (0, 3):
        conflicts = operate._gpu_claim_conflicts(ours, [gpu])
        assert conflicts and "unknown" in conflicts[0]


def test_gpu_claim_scans_every_index_in_a_multi_gpu_request(
    operate, tmp_path, monkeypatch,
):
    """A cell requesting a multi-GPU partition (e.g. --gpu 0,1) must be
    refused if ANY of its requested indexes is claimed elsewhere, and must
    scan the full set — not just the first index."""
    ours = make_cell(tmp_path, name="a__b__uni_v2__clam__s42__v2")
    other = make_cell(tmp_path, name="a__b__uni_v2__abmil__s42__v2", port=9582)
    write_live_daemon(other / "automil" / "orchestrator", pid=1234, gpus=[3])
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    assert operate._gpu_claim_conflicts(ours, [0, 1, 2]) == []
    conflicts = operate._gpu_claim_conflicts(ours, [2, 3])
    assert conflicts and "3" in conflicts[0]


# ---------------------------------------------------------------------------
# nvidia-smi index validation (must scan every requested index, not just one)
# ---------------------------------------------------------------------------
def _fake_nvidia_smi_capture(argv, env=None):
    csv = "0, 24000, 20000, 5\n1, 24000, 22000, 2\n"
    return subprocess.CompletedProcess(argv, 0, stdout=csv, stderr="")


def test_nvidia_smi_report_passes_when_every_requested_index_present(
    operate, monkeypatch, capsys,
):
    monkeypatch.setattr(operate, "_capture", _fake_nvidia_smi_capture)
    operate._nvidia_smi_report([0, 1])
    out = capsys.readouterr().out
    assert "GPU 0" in out
    assert "GPU 1" in out


def _pass_preflight_probes(operate, monkeypatch, tmp_path) -> None:
    """Everything _preflight checks before the GPU passes."""
    repo = tmp_path / "repo"
    (repo / "benchmarks").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    (repo / "benchmarks" / ".env").write_text("")
    write_reproduction_policy(repo)
    monkeypatch.setattr(operate, "REPO_ROOT", repo)
    monkeypatch.setattr(operate, "_protocol_runtime_version", lambda cell: "2.1.226")
    monkeypatch.setattr(operate, "_claude_version_first_token", lambda: "2.1.226")
    monkeypatch.setattr(operate, "project_exporter_port", lambda adir: 9581)
    monkeypatch.setattr(operate, "_exporter_twin_conflicts", lambda cell, port: [])
    monkeypatch.setattr(operate, "_port_in_use", lambda port: False)
    monkeypatch.setattr(operate, "_gpu_claim_conflicts", lambda cell, gpus: [])
    monkeypatch.setattr(operate, "_capture", _fake_nvidia_smi_capture)


def test_preflight_passes_on_full_h100s(operate, tmp_path, monkeypatch, capsys):
    _pass_preflight_probes(operate, monkeypatch, tmp_path)
    operate._preflight(make_cell(tmp_path), [0, 1])
    assert "declared campaign GPU" in capsys.readouterr().out


def test_preflight_passes_a_workstation_cell_on_its_declared_rtx(
    operate, tmp_path, monkeypatch, capsys,
):
    _pass_preflight_probes(operate, monkeypatch, tmp_path)
    write_reproduction_policy(
        operate.REPO_ROOT, runtimes=("runtime-aihub-hnsc-a",), gpu=RTX_GPU,
    )
    _forge_rtx_workstation(monkeypatch)
    operate._preflight(
        make_cell(tmp_path, runtime="runtime-aihub-hnsc-a"), [0, 1, 2],
    )
    assert "declared campaign GPU" in capsys.readouterr().out


def test_preflight_refuses_a_mig_slice(operate, tmp_path, monkeypatch, capsys):
    """Forged violation: a discovery daemon handed MIG slices never starts."""
    _pass_preflight_probes(operate, monkeypatch, tmp_path)
    _forge_mig_slice(monkeypatch)
    with pytest.raises(SystemExit):
        operate._preflight(make_cell(tmp_path), [0])
    assert "MIG enabled" in capsys.readouterr().err


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
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="finalized")

    boundary = FakeBoundary(operate, statuses=[{"phase": "winner-frozen"}])
    boundary.install(monkeypatch)
    operate.main(["finish", str(cell)])

    # The final status is printed from the last read already in hand — no
    # sixth status subprocess.
    assert boundary.run_or_die == []
    out = capsys.readouterr().out
    assert "finish: final status" in out
    assert '"phase": "winner-frozen"' in out


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
    """Full walk: discovery -> stop the idle daemon -> freeze -> select-winner
    -> finalize, each step taken from the controller's own status."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    orch = cell / "automil" / "orchestrator"
    write_live_daemon(orch, pid=777)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    boundary = FakeBoundary(operate, statuses=[
        {"phase": "discovery"},
        {"phase": "selection-ready"},
        {"phase": "winner-frozen"},
    ])

    def stop_daemon(argv):
        (orch / "orchestrator.pid").unlink()

    boundary.behaviors[("automil", ("orchestrator", "stop"))] = stop_daemon
    boundary.install(monkeypatch)

    operate.main(["finish", str(cell)])

    assert actions(boundary.run_or_die) == [
        ("automil", ("orchestrator", "stop")),
        ("stage", "freeze-discovery"),
        ("stage", "select-winner"),
        ("stage", "finalize-agent-session"),
    ]


def test_finish_reentry_skips_completed_transitions(operate, tmp_path, monkeypatch):
    """A finish re-run on a discovery-frozen cell touches nothing before
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

    assert actions(boundary.run_or_die) == [("stage", "select-winner")]


@pytest.mark.parametrize("phase", ["promotion-ready", "promotion"])
def test_finish_refuses_a_phase_the_ladder_no_longer_walks(
    operate, tmp_path, monkeypatch, capsys, phase,
):
    """The promotion phases are gone: a status that still reports one stops
    finish before anything runs, never falling through to the session close."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")

    boundary = FakeBoundary(operate, statuses=[{"phase": phase}])
    boundary.install(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        operate.main(["finish", str(cell)])

    assert excinfo.value.code == 2
    assert f"does not know how to advance phase {phase!r}" in capsys.readouterr().err
    assert boundary.run_or_die == []


# ---------------------------------------------------------------------------
# finish — discovery drain pre-check
# ---------------------------------------------------------------------------
def test_finish_discovery_refuses_while_daemon_still_draining(
    operate, tmp_path, monkeypatch, capsys,
):
    """Stopping a daemon that is still draining would leave finish waiting on
    _wait_daemon_dead's 300s timeout with a misleading error; refuse first."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    orch = cell / "automil" / "orchestrator"
    (orch / "queue").mkdir(parents=True)
    (orch / "queue" / "n1.json").write_text("{}")
    (orch / "running" / "gpu0").mkdir(parents=True)
    (orch / "running" / "gpu0" / "n2.json").write_text("{}")
    write_live_daemon(orch, pid=555)
    monkeypatch.setattr(operate, "is_pid_alive_with_starttime", lambda pid, ticks: True)

    boundary = FakeBoundary(operate, statuses=[{"phase": "discovery"}])
    boundary.install(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        operate.main(["finish", str(cell)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "still draining 2 discovery run(s)" in err
    assert "queued 1, running 1" in err
    assert "watch" in err
    # The daemon was left alone and no transition was attempted.
    assert boundary.run_or_die == []


def test_finish_discovery_dead_daemon_falls_through_to_controller_refusal(
    operate, tmp_path, monkeypatch, capsys,
):
    """With the daemon dead, pending work is the CONTROLLER's refusal to
    issue verbatim — the pre-check must not mask it with drain advice."""
    cell = make_cell(tmp_path)
    append_journal(cell, session_open_event())
    append_journal(cell, session_end_event())
    write_agent_session(cell, status="open")
    orch = cell / "automil" / "orchestrator"
    (orch / "queue").mkdir(parents=True)
    (orch / "queue" / "n1.json").write_text("{}")
    refusal = "campaign-stage error: discovery still has queued/running work\n"

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
    assert actions(boundary.run_or_die)[-1] == ("stage", "freeze-discovery")


# ---------------------------------------------------------------------------
# watch — budget lines through the typed reader
# ---------------------------------------------------------------------------
def test_budget_lines_use_typed_reader_and_report_unreadable_verbatim(
    operate, tmp_path,
):
    adir = tmp_path / "automil"
    (adir / "cells").mkdir(parents=True)
    write_budget_cell(adir, consumed=3, budget=10)
    (adir / "cells" / "obsolete.json").write_text(
        json.dumps({"consumed_evals": 1, "eval_budget": 2})
    )

    lines = operate._budget_lines(adir)

    assert any(
        f"{BUDGET_CELL_ID}: consumed_evals 3/10" in line
        and "completed 3" in line and "status active" in line
        for line in lines
    )
    assert any(
        "obsolete.json: unreadable: " in line and "obsolete cell layout" in line
        for line in lines
    )


def test_watch_reports_how_many_unique_candidates_discovery_holds(
    operate, tmp_path, monkeypatch, capsys,
):
    cell = make_cell(tmp_path)
    boundary = FakeBoundary(operate, statuses=[{
        "phase": "discovery",
        "discovery": {
            "attempts_charged": 12, "attempt_budget": 30,
            "unique_complete_candidates": 7,
        },
    }])
    boundary.install(monkeypatch)

    operate._watch_once(cell)

    assert (
        "phase discovery | attempts_charged 12/30 | unique candidates 7"
        in capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# reuse pins — one probe implementation, one glob census
# ---------------------------------------------------------------------------
def test_probes_are_imported_from_the_audited_launcher(operate):
    import autobench.campaign_launch as launch

    assert operate._port_in_use is launch._port_in_use
    assert operate.claude_cli_version is launch._claude_cli_version


def test_pending_work_counts_matches_controller_census(operate, tmp_path):
    """Drift alarm: campaign_stages._pending_stage_work is too heavy to
    import in the operator script, so its glob logic is duplicated there.
    The two must agree on what counts as pending work."""
    from autobench.campaign_stages import _pending_stage_work

    adir = tmp_path / "automil"
    queue = adir / "orchestrator" / "queue"
    running = adir / "orchestrator" / "running" / "gpu0"
    queue.mkdir(parents=True)
    running.mkdir(parents=True)
    (queue / "a.json").write_text("{}")
    (queue / "decoy.tmp").write_text("")
    (queue / "sub").mkdir()
    (queue / "sub" / "nested.json").write_text("{}")  # queue scan is flat
    (running / "b.json").write_text("{}")  # running scan is recursive

    queued, in_flight = operate._pending_work_counts(adir)
    census = _pending_stage_work(adir)
    assert (queued, in_flight) == (1, 1)
    assert census == ["a.json", "b.json"]
    assert queued + in_flight == len(census)

    empty = tmp_path / "empty" / "automil"
    empty.mkdir(parents=True)
    assert operate._pending_work_counts(empty) == (0, 0)
    assert _pending_stage_work(empty) == []


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
