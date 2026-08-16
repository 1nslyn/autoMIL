#!/usr/bin/env python
"""Sequence, supervise, and report one campaign cell through the audited controllers.

This is the operator front-end for ``docs/tutorials/run_agentic_campaign.md``.
It adds ZERO protocol semantics: every state transition is executed by the
existing audited surfaces — ``campaign_stage.py``, ``campaign_launch.py`` and
``uv run automil ...`` — as subprocesses. This script only sequences them,
supervises long-running phases, and reports. Every controller refusal is
printed verbatim and exits non-zero; never work around one.

Subcommands
    up      preflight + tmux session (windows baseline/orch/agent) + baseline
            run + discovery orchestrator, idempotently
    launch  readiness gate, then send the campaign launcher into the agent
            window (the one genuinely interactive surface)
    bind    operator-shell session binding: wait for session_open in the
            activity journal, run open-agent-session with bounded retry
    watch   loop printing budget consumption, phase and the results tail
    fleet   one status row per cell root under a runtime root
    finish  drive a finished-discovery cell to a finalized winner-frozen state

Daemon liveness is decided with the daemon's OWN pid-file semantics
(``automil.backends.pidfile``), never ad-hoc parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from automil.activity_hooks import project_exporter_port
from automil.backends.pidfile import is_pid_alive_with_starttime, load_pid_file
from automil.cells.state import read_cell

# Read-only PROBES are imported from the audited launcher so the runtime
# version and exporter-port checks have one implementation and one failure
# mode. State TRANSITIONS still run exclusively through subprocesses.
from autobench.campaign_launch import (
    CampaignLaunchError,
    claude_cli_version,
    port_in_use as _port_in_use,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "campaign_stage.py"
LAUNCH_SCRIPT = REPO_ROOT / "benchmarks" / "scripts" / "campaign_launch.py"
ACTIVITY_JOURNAL = ".activity.jsonl"
AGENT_SESSION_FILE = "agent_session.json"
AGENT_PROTOCOL_FILE = "agent_protocol.json"

RELEASE_LINE = "Session is bound. Begin the discovery loop per your policy."
METRICS_RETRY_ERROR = "Claude active-time metrics were not recorded for this session"
BIND_RETRY_SECONDS = 30
POLL_SECONDS = 30
# finish: queue/ and running/ are sampled non-atomically, so a spec mid-move
# between them can be absent from both in one sample; "drained" therefore
# requires two consecutive clean samples this far apart.
DRAIN_CONFIRM_SECONDS = 2
# finish: with queue and running both empty, this many consecutive polls
# without consumed_evals advancing means the ledger can no longer reach its
# budget (e.g. a cap-refused or cancelled spec). Waiting longer would hang
# forever; stop and audit the slate instead.
STALL_POLL_LIMIT = 3
UNAVAILABLE_USAGE = {
    "status": "unavailable",
    "input_tokens": None,
    "output_tokens": None,
    "cached_input_tokens": None,
    "cost_usd": None,
    "basis": "operator finish: CLI usage summary not captured",
}

# Seams for tests: monkeypatch these, never the stdlib.
_sleep = time.sleep
_now = time.monotonic


def _fail(message: str, code: int = 2) -> None:
    print(f"campaign-operate refusal: {message}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Command construction — the exact audited invocation forms
# ---------------------------------------------------------------------------
def stage_argv(action: str, cell_root: Path, *extra: str) -> list[str]:
    """``campaign_stage.py`` invocation, workspace-pinned per the runbook."""
    return [
        "uv", "run", "--project", str(REPO_ROOT), "--package", "autobench",
        "python", str(STAGE_SCRIPT), action, "--cell-root", str(cell_root),
        *extra,
    ]


def launcher_argv(action: str, cell_root: Path) -> list[str]:
    """``campaign_launch.py`` invocation, workspace-pinned per the runbook."""
    return [
        "uv", "run", "--project", str(REPO_ROOT), "--package", "autobench",
        "python", str(LAUNCH_SCRIPT), action, "--cell-root", str(cell_root),
    ]


def automil_argv(project_root: Path, *command: str) -> list[str]:
    """``automil`` invocation against one project root, workspace-pinned."""
    return [
        "uv", "run", "--project", str(REPO_ROOT),
        "automil", "--project", str(project_root), *command,
    ]


def orchestrator_window_command(cell_root: Path, gpu: int) -> str:
    """Foreground discovery-orchestrator command line for the orch window."""
    return (
        f"AUTOMIL_VISIBLE_GPUS={gpu} "
        + shlex.join(automil_argv(cell_root, "orchestrator", "start"))
    )


def baseline_window_command(cell_root: Path, gpu: int) -> str:
    return shlex.join(stage_argv("run-baseline", cell_root, "--gpu", str(gpu)))


def agent_window_command(cell_root: Path) -> str:
    return shlex.join(launcher_argv("launch", cell_root))


# ---------------------------------------------------------------------------
# Subprocess boundary — tests monkeypatch these three
# ---------------------------------------------------------------------------
def _capture(argv: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def _run_or_die(argv: list[str], env: dict | None = None) -> None:
    """Run with output flowing through; any failure exits with that code."""
    completed = subprocess.run(argv, env=env)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


def _popen(argv: list[str], env: dict, stdout, stderr) -> subprocess.Popen:
    return subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr)


def _die_verbatim(completed: subprocess.CompletedProcess) -> None:
    """Fail-closed passthrough: reprint a captured refusal untouched."""
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    sys.exit(completed.returncode or 2)


def _stage_status(cell_root: Path) -> dict:
    completed = _capture(stage_argv("status", cell_root))
    if completed.returncode != 0:
        _die_verbatim(completed)
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        sys.stdout.write(completed.stdout)
        _fail("campaign_stage.py status did not print JSON")
        raise AssertionError("unreachable")


def _try_stage_status(cell_root: Path) -> tuple[dict | None, str]:
    """Non-fatal status read for the report loops (watch/fleet)."""
    completed = _capture(stage_argv("status", cell_root))
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout).strip()
    try:
        return json.loads(completed.stdout), ""
    except json.JSONDecodeError:
        return None, "status output was not JSON"


# ---------------------------------------------------------------------------
# Daemon liveness and GPU claims (daemon's own pid-file semantics)
# ---------------------------------------------------------------------------
def _daemon_alive(orch_dir: Path) -> int | None:
    """Return the live daemon's pid for one orchestrator dir, else None."""
    loaded = load_pid_file(orch_dir / "orchestrator.pid")
    if loaded and is_pid_alive_with_starttime(
        loaded["pid"], loaded["starttime_ticks"]
    ):
        return int(loaded["pid"])
    return None


def _claimed_gpus(orch_dir: Path) -> set[int] | None:
    """GPU indexes a daemon's gpu_state.json records; None = unknown.

    "No claims" is never inferred from absence: a live daemon whose state
    file is missing, unparseable, or yields no cuda slots (empty ``gpus``
    from an nvidia-smi failure, or a rocm/cpu daemon with no
    ``accelerator == "cuda"`` slot) has an UNKNOWN partition and must be
    treated as claiming every GPU. Only a readable cuda slot map may grant
    clearance.
    """
    state_path = orch_dir / "gpu_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    claimed: set[int] = set()
    gpus = state.get("gpus")
    if isinstance(gpus, dict):
        for key in gpus:
            if str(key).isdecimal():
                claimed.add(int(key))
    slots = state.get("execution_slots")
    if isinstance(slots, dict):
        for slot in slots.values():
            if (
                isinstance(slot, dict)
                and slot.get("accelerator") == "cuda"
                and isinstance(slot.get("device_index"), int)
            ):
                claimed.add(slot["device_index"])
    return claimed or None


def _candidate_cell_roots(cell_root: Path) -> list[Path]:
    """Cell roots that could claim this host's GPUs: siblings in this runtime
    root plus every cell under sibling runtime roots (twin roots included)."""
    runtime_root = cell_root.parent
    roots = [runtime_root]
    campaign_dir = runtime_root.parent
    try:
        for sibling in sorted(campaign_dir.iterdir()):
            if sibling == runtime_root or not sibling.is_dir():
                continue
            try:
                if any(
                    (child / "automil").is_dir()
                    for child in sibling.iterdir()
                    if child.is_dir()
                ):
                    roots.append(sibling)
            except OSError:
                continue
    except OSError:
        pass
    cells: list[Path] = []
    for root in roots:
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "automil").is_dir():
                    cells.append(child)
        except OSError:
            continue
    return cells


def _gpu_claim_conflicts(cell_root: Path, gpu: int) -> list[str]:
    """Other cells' LIVE daemons whose partition covers *gpu*.

    Same-cell discovery/promotion pairs are exempt: this cell's own daemons
    on the same GPU are the normal finish-time state.
    """
    cell_root = cell_root.resolve()
    conflicts: list[str] = []
    for candidate in _candidate_cell_roots(cell_root):
        if candidate.resolve() == cell_root:
            continue
        for orch_dir in (
            candidate / "automil" / "orchestrator",
            candidate / "promotion" / "automil" / "orchestrator",
        ):
            pid = _daemon_alive(orch_dir)
            if pid is None:
                continue
            claimed = _claimed_gpus(orch_dir)
            if claimed is None:
                conflicts.append(
                    f"{orch_dir} has a live daemon (pid {pid}) with no "
                    "readable cuda slot map in gpu_state.json (missing, "
                    "unparseable, or no cuda slots) — its GPU partition is "
                    "unknown, so it may claim every GPU"
                )
            elif gpu in claimed:
                conflicts.append(
                    f"{orch_dir} has a live daemon (pid {pid}) claiming "
                    f"GPU(s) {sorted(claimed)}"
                )
    return conflicts


# ---------------------------------------------------------------------------
# Activity journal (read-only; writes only ever happen via controllers)
# ---------------------------------------------------------------------------
def _journal_path(cell_root: Path) -> Path:
    return cell_root / "automil" / ACTIVITY_JOURNAL


def _journal_events(cell_root: Path, strict: bool = True) -> list[dict]:
    path = _journal_path(cell_root)
    try:
        content = path.read_text()
    except OSError:
        return []
    events: list[dict] = []
    for number, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if strict:
                _fail(f"unparseable activity journal line {number} in {path}")
            continue  # lenient polling: a torn in-flight append self-heals
        if isinstance(event, dict):
            events.append(event)
    return events


def _newest_session_open(cell_root: Path, strict: bool = True) -> dict | None:
    opens = [
        event for event in _journal_events(cell_root, strict=strict)
        if event.get("event") == "session_open"
    ]
    return opens[-1] if opens else None


def _session_end_recorded(cell_root: Path, session_id: str) -> bool:
    return any(
        event.get("event") == "session_end"
        and event.get("session_id") == session_id
        for event in _journal_events(cell_root)
    )


def _bind_payload(open_event: dict) -> dict:
    """The exact two-field open-agent-session attestation."""
    started_at = datetime.fromtimestamp(
        float(open_event["observed_at"]), tz=timezone.utc
    ).isoformat(timespec="seconds")
    return {"session_id": open_event["session_id"], "started_at": started_at}


def _has_session_evidence(cell_root: Path) -> bool:
    """This cell already owns a formal session (bound file or journal open)."""
    return (
        (cell_root / AGENT_SESSION_FILE).is_file()
        or _newest_session_open(cell_root, strict=False) is not None
    )


# ---------------------------------------------------------------------------
# tmux (subprocess only; required for the interactive agent window)
# ---------------------------------------------------------------------------
def _session_name(cell_root: Path) -> str:
    """Derive the tmux session name from the cell id's distinguishing tokens
    (encoder + arm, e.g. ``uni_v2-clam``); fall back to the sanitized name."""
    tokens = cell_root.name.split("__")
    raw = "-".join(tokens[2:4]) if len(tokens) >= 4 else cell_root.name
    return raw.replace(":", "_").replace(".", "_")


def _tmux(*args: str) -> subprocess.CompletedProcess:
    return _capture(["tmux", *args])


def _require_tmux() -> None:
    if shutil.which("tmux") is None:
        _fail("tmux is not installed; the orchestrator and the formal "
              "session must survive an SSH disconnect (runbook §3a)")


def _ensure_tmux_layout(name: str) -> None:
    """Create session + baseline/orch/agent windows; skip whatever exists."""
    if _tmux("has-session", "-t", f"={name}").returncode != 0:
        created = _tmux("new-session", "-d", "-s", name, "-n", "baseline")
        if created.returncode != 0:
            _fail(f"tmux new-session failed: {created.stderr.strip()}")
        print(f"tmux: created session {name!r}")
    else:
        print(f"tmux: session {name!r} already exists")
    listed = _tmux("list-windows", "-t", f"={name}", "-F", "#{window_name}")
    if listed.returncode != 0:
        _fail(f"tmux list-windows failed: {listed.stderr.strip()}")
    existing = set(listed.stdout.split())
    for window in ("baseline", "orch", "agent"):
        if window not in existing:
            created = _tmux("new-window", "-d", "-t", f"={name}", "-n", window)
            if created.returncode != 0:
                _fail(f"tmux new-window {window} failed: {created.stderr.strip()}")
            print(f"tmux: created window {window}")


def _send_to_window(name: str, window: str, command: str) -> None:
    sent = _tmux("send-keys", "-t", f"={name}:{window}", command, "Enter")
    if sent.returncode != 0:
        _fail(f"tmux send-keys to {window} failed: {sent.stderr.strip()}")
    print(f"tmux[{name}:{window}] $ {command}")


# ---------------------------------------------------------------------------
# up
# ---------------------------------------------------------------------------
def _cell_root_arg(raw: str) -> Path:
    path = Path(raw).resolve()
    if not (path / "automil" / "config.yaml").is_file():
        _fail(f"{path} is not a materialized cell root (no automil/config.yaml)")
    if not (path / "automil" / "campaign_cell.json").is_file():
        _fail(f"{path} has no campaign cell record (automil/campaign_cell.json)")
    return path


def _protocol_runtime_version(cell_root: Path) -> str:
    protocol_path = cell_root.parent / AGENT_PROTOCOL_FILE
    try:
        protocol = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {protocol_path}: {exc}")
    version = protocol.get("runtime_version")
    if not isinstance(version, str) or not version:
        _fail(f"{protocol_path} declares no runtime_version")
    return version


def _claude_version_first_token() -> str:
    """First token of ``claude --version`` via the launcher's own probe."""
    try:
        return claude_cli_version("claude")
    except CampaignLaunchError as exc:
        _fail(str(exc))
        raise AssertionError("unreachable")


def _exporter_twin_conflicts(cell_root: Path, port: int) -> list[Path]:
    """Other cell roots (any runtime*/ sibling) declaring the same port."""
    twins: list[Path] = []
    for candidate in _candidate_cell_roots(cell_root):
        if candidate.resolve() == cell_root.resolve():
            continue
        try:
            other = project_exporter_port(candidate / "automil")
        except ValueError:
            continue
        if other == port:
            twins.append(candidate)
    return twins


def _nvidia_smi_report(gpu: int) -> None:
    completed = _capture([
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if completed.returncode != 0:
        print("preflight: nvidia-smi unavailable "
              f"({(completed.stderr or completed.stdout).strip()}) — "
              "the orchestrator will refuse if no schedulable GPU exists")
        return
    indexes: set[int] = set()
    print("preflight: GPU free-VRAM report")
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0].isdecimal():
            indexes.add(int(parts[0]))
            marker = "  <- requested" if int(parts[0]) == gpu else ""
            print(
                f"  GPU {parts[0]}: {parts[2]} MiB free of {parts[1]} MiB"
                f"{marker}"
            )
    if indexes and gpu not in indexes:
        _fail(f"--gpu {gpu} is not present on this host "
              f"(nvidia-smi reports {sorted(indexes)})")


def _preflight(cell_root: Path, gpu: int) -> None:
    if not (REPO_ROOT / "pyproject.toml").is_file() or not (
        REPO_ROOT / "benchmarks"
    ).is_dir():
        _fail(f"derived repo root {REPO_ROOT} does not look like the autoMIL "
              "workspace; run the script from a full clone")
    print(f"preflight: repo root {REPO_ROOT}")
    dotenv = REPO_ROOT / "benchmarks" / ".env"
    if not dotenv.is_file():
        _fail(
            f"{dotenv} is missing. Orchestrator-run training executes in a "
            "detached worktree and reads dataset roots from benchmarks/.env; "
            "without it, runs fail later inside training with a missing-path "
            "error. Create it from benchmarks/.env.example first (runbook §3d)."
        )
    print(f"preflight: {dotenv} exists")

    pinned = _protocol_runtime_version(cell_root)
    observed = _claude_version_first_token()
    if observed != pinned:
        _fail(
            f"runtime version drift: `claude --version` reports {observed}, "
            f"the frozen protocol requires {pinned}. Repin the CLI on this "
            "host (runbook §3b); campaign_launch.py would refuse this launch."
        )
    print(f"preflight: pinned claude runtime {pinned} confirmed")

    try:
        port = project_exporter_port(cell_root / "automil")
    except ValueError as exc:
        _fail(str(exc))
    twins = _exporter_twin_conflicts(cell_root, port)
    if _port_in_use(port) and not _has_session_evidence(cell_root):
        listing = "".join(f"\n  twin: {twin}" for twin in twins)
        _fail(
            f"activity exporter port {port} is already serving and this cell "
            "has no session evidence of its own — another cell root sharing "
            "this manifest row is exporting on it. Never run twin roots "
            f"concurrently.{listing}"
        )
    if twins:
        print(
            f"preflight: WARNING — other cell roots declare exporter port "
            f"{port} (same manifest row); never run them concurrently with "
            "this cell:"
        )
        for twin in twins:
            print(f"  twin: {twin}")
    else:
        print(f"preflight: exporter port {port} has no twin conflict")

    conflicts = _gpu_claim_conflicts(cell_root, gpu)
    if conflicts:
        _fail(
            f"GPU {gpu} is claimed by another cell's live orchestrator "
            "daemon; give each concurrent cell a disjoint partition "
            "(runbook §3c):\n  " + "\n  ".join(conflicts)
        )
    print(f"preflight: no other cell's live daemon claims GPU {gpu}")
    _nvidia_smi_report(gpu)


def cmd_up(args: argparse.Namespace) -> None:
    cell_root = _cell_root_arg(args.cell_root)
    _require_tmux()
    _preflight(cell_root, args.gpu)

    name = _session_name(cell_root)
    _ensure_tmux_layout(name)

    status = _stage_status(cell_root)
    if status.get("baseline_registered"):
        print("baseline: already registered — skipping run-baseline")
    else:
        _send_to_window(
            name, "baseline", baseline_window_command(cell_root, args.gpu)
        )
        print("baseline: run-baseline sent (it locks; a duplicate refuses)")

    pid = _daemon_alive(cell_root / "automil" / "orchestrator")
    if pid is not None:
        print(f"orchestrator: discovery daemon already running (pid {pid}) — "
              "skipping start")
    else:
        _send_to_window(
            name, "orch", orchestrator_window_command(cell_root, args.gpu)
        )
    print(
        f"up: done. Attach with `tmux attach -t {name}`. Next: "
        f"`campaign_operate.py launch {cell_root}` once the baseline is "
        "registered, then `bind`."
    )


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------
def cmd_launch(args: argparse.Namespace) -> None:
    cell_root = _cell_root_arg(args.cell_root)
    _require_tmux()
    status = _stage_status(cell_root)
    if not status.get("baseline_registered"):
        _fail(
            "the native baseline is not registered yet — wait for the "
            "baseline window started by `up` (or run campaign_stage.py "
            "run-baseline) before launching the agent session"
        )
    if _daemon_alive(cell_root / "automil" / "orchestrator") is None:
        _fail(
            "the discovery orchestrator is not running — it must scrape this "
            "cell's exporter before the session binds. Re-run `up` (orch "
            "window) first."
        )
    if _has_session_evidence(cell_root):
        _fail(
            "this cell already has session evidence (agent_session.json or a "
            "journal session_open) — a cell never gets a second session, and "
            "re-sending the launcher would type into the live one. If the "
            "runtime died pre-bind, follow the pre-bind recovery in the "
            "runbook appendix instead."
        )
    name = _session_name(cell_root)
    if _tmux("has-session", "-t", f"={name}").returncode != 0:
        _fail(f"tmux session {name!r} does not exist — run `up` first")
    _send_to_window(name, "agent", agent_window_command(cell_root))
    print(
        "launch: campaign launcher sent to the agent window (it re-verifies "
        "the locked protocol and execs the pinned claude). Next: "
        f"`campaign_operate.py bind {cell_root}` from this operator shell."
    )


# ---------------------------------------------------------------------------
# bind
# ---------------------------------------------------------------------------
def cmd_bind(args: argparse.Namespace) -> None:
    cell_root = _cell_root_arg(args.cell_root)
    deadline = _now() + args.timeout_s

    open_event = _newest_session_open(cell_root, strict=False)
    while open_event is None:
        if _now() >= deadline:
            _fail(
                f"no session_open appeared in {_journal_path(cell_root)} "
                f"within {args.timeout_s}s — is the Claude session running "
                "in the agent window?"
            )
        _sleep(2)
        open_event = _newest_session_open(cell_root, strict=False)

    payload = _bind_payload(open_event)
    with tempfile.NamedTemporaryFile(
        "w", prefix="agent_session_start_", suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle)
        handle.write("\n")
        session_path = Path(handle.name)
    print(f"bind: session {payload['session_id']} started_at "
          f"{payload['started_at']} -> {session_path}")

    while True:
        completed = _capture(
            stage_argv(
                "open-agent-session", cell_root,
                "--agent-session", str(session_path),
            )
        )
        if completed.returncode == 0:
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            print(RELEASE_LINE)
            return
        refusal = (completed.stderr or "") + (completed.stdout or "")
        if METRICS_RETRY_ERROR not in refusal:
            _die_verbatim(completed)
        if _now() + BIND_RETRY_SECONDS >= deadline:
            _die_verbatim(completed)
        print(
            "bind: waiting for the first native active-time export "
            f"(retrying in {BIND_RETRY_SECONDS}s): {METRICS_RETRY_ERROR}"
        )
        _sleep(BIND_RETRY_SECONDS)


# ---------------------------------------------------------------------------
# watch / fleet
# ---------------------------------------------------------------------------
def _budget_lines(automil_dir: Path) -> list[str]:
    lines: list[str] = []
    cells_dir = automil_dir / "cells"
    if not cells_dir.is_dir():
        return ["  (no budget cells yet)"]
    for path in sorted(cells_dir.glob("*.json")):
        # read_cell is the typed reader that fails loud on obsolete layouts;
        # a raw parse here would silently report 0/None for them instead.
        try:
            cell = read_cell(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            lines.append(f"  {path.name}: unreadable: {exc}")
            continue
        budget = cell.eval_budget
        lines.append(
            f"  {path.stem}: consumed_evals "
            f"{cell.consumed_evals}/{budget if budget is not None else '-'}"
            f", completed {cell.completed_evals}"
            f", status {cell.status.value}"
        )
    return lines or ["  (no budget cells yet)"]


def _watch_once(cell_root: Path) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"--- {cell_root.name} @ {stamp}")
    status, error = _try_stage_status(cell_root)
    if status is None:
        print(f"  status unavailable: {error}")
    else:
        discovery = status.get("discovery") or {}
        print(
            f"  phase {status.get('phase')} | attempts_charged "
            f"{discovery.get('attempts_charged')}/{discovery.get('attempt_budget')}"
            f" | promoted {discovery.get('promoted_candidates')}"
        )
    for line in _budget_lines(cell_root / "automil"):
        print(line)
    results = cell_root / "automil" / "results.tsv"
    if results.is_file():
        tail = results.read_text().splitlines()[-5:]
        print("  results.tsv tail:")
        for line in tail:
            print(f"    {line}")
    else:
        print("  results.tsv: not written yet")


def cmd_watch(args: argparse.Namespace) -> None:
    cell_root = _cell_root_arg(args.cell_root)
    try:
        while True:
            _watch_once(cell_root)
            _sleep(args.interval)
    except KeyboardInterrupt:
        print("watch: stopped")


def cmd_fleet(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    if not root.is_dir():
        _fail(f"{root} is not a directory")
    rows: list[tuple[str, str, str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (
            child / "automil" / "campaign_cell.json"
        ).is_file():
            continue
        status, error = _try_stage_status(child)
        if status is None:
            detail = (error.splitlines() or ["(no output)"])[-1][:60]
            rows.append((child.name, "error", "-", detail))
            continue
        discovery = status.get("discovery") or {}
        attempts = (
            f"{discovery.get('attempts_charged')}/{discovery.get('attempt_budget')}"
        )
        winner = status.get("winner") or {}
        winner_text = (
            f"{winner.get('kind')}:{winner.get('candidate_id')}"
            if winner else "-"
        )
        rows.append((child.name, str(status.get("phase")), attempts, winner_text))
    if not rows:
        print(f"fleet: no campaign cell roots under {root}")
        return
    width = max(len(row[0]) for row in rows)
    print(f"{'cell'.ljust(width)}  {'phase':16}  {'attempts':9}  winner")
    for name, phase, attempts, winner_text in rows:
        print(f"{name.ljust(width)}  {phase:16}  {attempts:9}  {winner_text}")


# ---------------------------------------------------------------------------
# finish
# ---------------------------------------------------------------------------
def _bound_session_id(cell_root: Path) -> str:
    session_path = cell_root / AGENT_SESSION_FILE
    if session_path.is_file():
        try:
            payload = json.loads(session_path.read_text())
            session_id = payload["session"]["session_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            _fail(f"cannot read the bound session from {session_path}: {exc}")
        if isinstance(session_id, str) and session_id:
            return session_id
        _fail(f"{session_path} carries no session_id")
    open_event = _newest_session_open(cell_root)
    if open_event is None:
        _fail(
            "the activity journal records no session_open — this cell never "
            "started its formal session; there is nothing to finish"
        )
    return str(open_event["session_id"])


def _ensure_session_end_evidence(cell_root: Path, attest: str | None) -> None:
    session_id = _bound_session_id(cell_root)
    if _session_end_recorded(cell_root, session_id):
        print(f"finish: SessionEnd recorded for {session_id}")
        return
    try:
        port = project_exporter_port(cell_root / "automil")
    except ValueError as exc:
        _fail(str(exc))
    close_command = shlex.join(
        automil_argv(cell_root, "activity", "close",
                     "--session", session_id, "--attest", "<cause>")
    )
    if _port_in_use(port):
        _fail(
            "SessionEnd is not recorded and this cell's activity exporter "
            f"(port {port}) is still serving — the Claude session appears to "
            "be live. Runbook step: type /exit inside that Claude session so "
            "SessionEnd captures and persists the final active-time sample, "
            "then re-run finish. (Only if the runtime is actually dead: "
            f"{close_command})"
        )
    if not attest:
        _fail(
            "SessionEnd is not recorded and the exporter is dead (the "
            "runtime died before its SessionEnd hook). Recovery is an "
            "operator-attested close; re-run finish with "
            '--attest "runtime died before SessionEnd: <cause>" and this '
            f"script will run: {close_command}. Refusing to attest on your "
            "behalf."
        )
    _run_or_die(
        automil_argv(cell_root, "activity", "close",
                     "--session", session_id, "--attest", attest)
    )
    if not _session_end_recorded(cell_root, session_id):
        _fail(
            "activity close reported success but the journal still lacks a "
            "session_end event — refusing to continue"
        )
    print(f"finish: operator-attested close recorded for {session_id}")


def _wait_daemon_dead(orch_dir: Path, timeout_s: float, label: str) -> None:
    deadline = _now() + timeout_s
    while _daemon_alive(orch_dir) is not None:
        if _now() >= deadline:
            _fail(f"{label} daemon did not exit within {timeout_s:.0f}s of "
                  "orchestrator stop; investigate before re-running finish")
        _sleep(2)


def _stop_discovery_daemon(cell_root: Path) -> None:
    orch_dir = cell_root / "automil" / "orchestrator"
    pid = _daemon_alive(orch_dir)
    if pid is None:
        print("finish: discovery orchestrator is not running")
        return
    print(f"finish: stopping discovery orchestrator (pid {pid})")
    _run_or_die(automil_argv(cell_root, "orchestrator", "stop"))
    _wait_daemon_dead(orch_dir, 300, "discovery orchestrator")


def _pending_work_counts(adir: Path) -> tuple[int, int]:
    """(queued, running) spec counts under one automil dir's orchestrator.

    Mirrors ``autobench.campaign_stages._pending_stage_work`` — the freeze
    gates' own census. Importing it would drag the full controller into this
    script, so the globs are duplicated here and pinned against the original
    by ``test_pending_work_counts_matches_controller_census``.
    """
    queue = adir / "orchestrator" / "queue"
    running = adir / "orchestrator" / "running"
    queued = len(list(queue.glob("*.json"))) if queue.is_dir() else 0
    in_flight = len(list(running.rglob("*.json"))) if running.is_dir() else 0
    return queued, in_flight


def _promotion_budget(promotion_adir: Path) -> tuple[int, int | None] | None:
    """(consumed_evals, eval_budget) from the typed cell reader; None = unknown.

    ``read_cell`` fails loud on obsolete cell layouts where a raw parse would
    silently report 0. An unreadable ledger prints the verbatim error and
    reports UNKNOWN (never drained) — the stall detector in
    ``_drive_promotion`` bounds how long an unknown ledger can be polled.
    """
    cell = json.loads((promotion_adir / "campaign_cell.json").read_text())
    budget_cell_id = str(cell["budget_identity"]["cell_id"])
    cell_path = promotion_adir / "cells" / f"{budget_cell_id}.json"
    try:
        budget_cell = read_cell(cell_path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"finish: cannot read the promotion budget cell {cell_path}: {exc}")
        return None
    return budget_cell.consumed_evals, budget_cell.eval_budget


def _promotion_drained_once(promotion_adir: Path) -> bool:
    queued, in_flight = _pending_work_counts(promotion_adir)
    if queued or in_flight:
        return False
    budget = _promotion_budget(promotion_adir)
    if budget is None:
        return False
    consumed, cap = budget
    # >= — never ==: an overconsumed ledger must still count as drained, or
    # the poll below would never terminate on it.
    return cap is not None and consumed >= cap


def _promotion_drained(promotion_adir: Path) -> bool:
    """Two consecutive drained samples: queue/ then running/ are read
    non-atomically, so a spec mid-move between the two directories can be
    absent from both in a single sample."""
    if not _promotion_drained_once(promotion_adir):
        return False
    _sleep(DRAIN_CONFIRM_SECONDS)
    return _promotion_drained_once(promotion_adir)


def _promotion_plan_node_ids(promotion_adir: Path) -> list[str]:
    """Promotion node ids from the immutable plan written at materialization."""
    plan_path = promotion_adir / "promotion_plan.json"
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read the immutable promotion plan {plan_path}: {exc}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not all(
        isinstance(job, dict)
        and isinstance(job.get("promotion_node_id"), str)
        and job["promotion_node_id"]
        for job in jobs
    ):
        _fail(f"{plan_path} carries no readable promotion job roster")
    return [job["promotion_node_id"] for job in jobs]


def _require_complete_promotion_slate(
    promotion_adir: Path, accept_missing: bool,
) -> None:
    """Refuse to hand freeze-promotion a partially-measured slate.

    freeze-promotion tolerates a missing result.json by marking the job
    'ineligible', so proceeding on drained counters alone would silently
    freeze whatever happened to finish. Every planned node must have a
    terminal archive result, or the operator must accept the gap explicitly.
    """
    archive_root = promotion_adir / "orchestrator" / "archive"
    missing = [
        node_id for node_id in _promotion_plan_node_ids(promotion_adir)
        if not (archive_root / node_id / "result.json").is_file()
    ]
    if not missing:
        print("finish: every planned promotion job has a terminal result.json")
        return
    if accept_missing:
        print(
            f"finish: WARNING — proceeding with {len(missing)} promotion "
            "job(s) lacking a terminal result.json (--accept-missing); "
            "freeze-promotion will mark them ineligible: "
            + ", ".join(missing)
        )
        return
    _fail(
        "the promotion queue and running set are empty but these planned "
        "promotion jobs have no terminal result.json:\n  "
        + "\n  ".join(missing)
        + "\nfreeze-promotion would silently mark them 'ineligible' and "
        "freeze a partially-measured slate. Re-run the missing jobs first, "
        "or re-run finish with --accept-missing to freeze the slate as-is."
    )


def _supervisor_log_tail(log_path: Path) -> str:
    try:
        return "\n".join(log_path.read_text().splitlines()[-20:])
    except OSError:
        return "(no supervisor log)"


def _drive_promotion(
    cell_root: Path, gpu: int | None, accept_missing: bool,
) -> None:
    promotion_root = cell_root / "promotion"
    promotion_adir = promotion_root / "automil"
    if not (promotion_adir / "campaign_cell.json").is_file():
        _fail(
            f"phase is promotion but {promotion_adir} is not materialized — "
            "the controller state and the tree disagree; investigate"
        )
    orch_dir = promotion_adir / "orchestrator"
    log_path = orch_dir / "operate_supervisor.log"

    if _promotion_drained(promotion_adir):
        print("finish: promotion queue already drained")
        _require_complete_promotion_slate(promotion_adir, accept_missing)
        _stop_promotion_daemon(promotion_root, orch_dir, child=None)
        return

    child: subprocess.Popen | None = None
    log_handle = None
    adopted_pid = _daemon_alive(orch_dir)
    if adopted_pid is not None:
        print(f"finish: adopting the live promotion orchestrator "
              f"(pid {adopted_pid}) — polling only, not starting another")
    else:
        if gpu is None:
            _fail(
                "finish must start a promotion orchestrator but --gpu was "
                "not given. An unset AUTOMIL_VISIBLE_GPUS schedules on EVERY "
                "GPU of this shared host (the daemon refuses malformed "
                "values, not absent ones) — re-run with an explicit --gpu N."
            )
        orch_dir.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        env = {**os.environ, "AUTOMIL_VISIBLE_GPUS": str(gpu)}
        child = _popen(
            automil_argv(promotion_root, "orchestrator", "start"),
            env=env, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        print(
            f"finish: started the promotion orchestrator as a supervised "
            f"foreground child (pid {child.pid}, AUTOMIL_VISIBLE_GPUS={gpu}, "
            f"log {log_path})"
        )
        deadline = _now() + 120
        while _daemon_alive(orch_dir) is None:
            if child.poll() is not None:
                _fail(
                    "the promotion orchestrator exited during startup "
                    f"(rc {child.returncode}); log tail:\n"
                    + _supervisor_log_tail(log_path)
                )
            if _now() >= deadline:
                _fail("the promotion orchestrator wrote no live pid file "
                      f"within 120s; log tail:\n{_supervisor_log_tail(log_path)}")
            _sleep(2)

    try:
        stalled = False
        stall_polls = 0
        last_consumed: int | None = None
        while not _promotion_drained(promotion_adir):
            if child is not None and child.poll() is not None:
                _fail(
                    "the supervised promotion orchestrator exited with work "
                    f"remaining (rc {child.returncode}); log tail:\n"
                    + _supervisor_log_tail(log_path)
                )
            if child is None and _daemon_alive(orch_dir) is None:
                _fail(
                    "the adopted promotion orchestrator died with work "
                    "remaining; re-run finish with --gpu N to start a "
                    "supervised one"
                )
            queued, in_flight = _pending_work_counts(promotion_adir)
            budget = _promotion_budget(promotion_adir)
            consumed, cap = budget if budget is not None else (None, None)
            print(
                "finish: promotion consumed "
                f"{'?' if consumed is None else consumed}/"
                f"{'?' if cap is None else cap}, "
                f"queued {queued}, running {in_flight}"
            )
            # Stall detector: nothing queued, nothing running, and the
            # consumed counter not advancing means the ledger can never
            # reach its budget (cap-refused/cancelled spec). Polling on
            # would hang forever — the one thing this script must not do.
            if queued == 0 and in_flight == 0 and consumed == last_consumed:
                stall_polls += 1
            else:
                stall_polls = 0
            last_consumed = consumed
            if stall_polls >= STALL_POLL_LIMIT:
                stalled = True
                print(
                    "finish: promotion is STALLED — queue and running are "
                    "empty and consumed_evals has not advanced for "
                    f"{STALL_POLL_LIMIT} polls; the budget can no longer "
                    "drain by itself. Auditing the slate instead of waiting."
                )
                break
            _sleep(POLL_SECONDS)
        if not stalled:
            print("finish: promotion queue drained")
        _require_complete_promotion_slate(promotion_adir, accept_missing)
        _stop_promotion_daemon(promotion_root, orch_dir, child)
    finally:
        if log_handle is not None:
            log_handle.close()


def _stop_promotion_daemon(
    promotion_root: Path, orch_dir: Path, child: subprocess.Popen | None,
) -> None:
    if _daemon_alive(orch_dir) is None and child is None:
        print("finish: promotion orchestrator is not running")
        return
    print("finish: stopping the promotion orchestrator")
    _run_or_die(automil_argv(promotion_root, "orchestrator", "stop"))
    if child is not None:
        try:
            child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            _fail("the supervised promotion orchestrator ignored SIGTERM for "
                  "600s; investigate before re-running finish")
    else:
        _wait_daemon_dead(orch_dir, 300, "promotion orchestrator")


def _run_stage(action: str, cell_root: Path) -> None:
    print(f"finish: {action}")
    _run_or_die(stage_argv(action, cell_root))


def _finalize_session(cell_root: Path, usage_json: str | None) -> None:
    session_path = cell_root / AGENT_SESSION_FILE
    try:
        session = json.loads(session_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {session_path}: {exc}")
    if session.get("status") == "finalized":
        print("finish: agent session is already finalized — skipping")
        return
    session_id = (session.get("session") or {}).get("session_id")
    if not isinstance(session_id, str) or not session_id:
        _fail(f"{session_path} carries no session_id")
    if usage_json is not None:
        try:
            usage = json.loads(Path(usage_json).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read --usage-json {usage_json}: {exc}")
    else:
        usage = dict(UNAVAILABLE_USAGE)
    end_payload = {
        "session_id": session_id,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "termination_reason": "budget-complete",
        "usage": usage,
    }
    with tempfile.NamedTemporaryFile(
        "w", prefix="agent_session_end_", suffix=".json", delete=False
    ) as handle:
        json.dump(end_payload, handle, indent=2)
        handle.write("\n")
        end_path = Path(handle.name)
    print(f"finish: finalize-agent-session with {end_path}")
    _run_or_die(
        stage_argv("finalize-agent-session", cell_root,
                   "--agent-session", str(end_path))
    )


def _finish_discovery(cell_root: Path, args: argparse.Namespace) -> None:
    # A live daemon still draining work would otherwise be stopped here, then
    # spend hours finishing while _wait_daemon_dead times out at 300s with a
    # misleading 'investigate' error. Refuse honestly up front. (With the
    # daemon dead, fall through: freeze-discovery issues its own verbatim
    # queued/running-work refusal.)
    queued, in_flight = _pending_work_counts(cell_root / "automil")
    pending = queued + in_flight
    if pending and _daemon_alive(cell_root / "automil" / "orchestrator") is not None:
        _fail(
            f"the discovery orchestrator is still draining {pending} "
            f"discovery run(s) (queued {queued}, running {in_flight}); "
            "use `watch` and re-run finish after they complete"
        )
    _stop_discovery_daemon(cell_root)
    _run_stage("freeze-discovery", cell_root)


def _finish_promotion_ready(cell_root: Path, args: argparse.Namespace) -> None:
    _run_stage("materialize-promotion", cell_root)


def _finish_promotion(cell_root: Path, args: argparse.Namespace) -> None:
    _drive_promotion(cell_root, args.gpu, args.accept_missing)
    _run_stage("freeze-promotion", cell_root)


def _finish_selection_ready(cell_root: Path, args: argparse.Namespace) -> None:
    _run_stage("select-winner", cell_root)


# Phase ladder: iterated in order, re-reading the controller's status once
# per advance. Phase decides every step; history never does.
_FINISH_LADDER = (
    ("discovery", _finish_discovery),
    ("promotion-ready", _finish_promotion_ready),
    ("promotion", _finish_promotion),
    ("selection-ready", _finish_selection_ready),
)


def cmd_finish(args: argparse.Namespace) -> None:
    cell_root = _cell_root_arg(args.cell_root)
    _ensure_session_end_evidence(cell_root, args.attest)

    status = _stage_status(cell_root)
    phase = status.get("phase")
    print(f"finish: phase {phase}")

    for expected_phase, advance in _FINISH_LADDER:
        if phase != expected_phase:
            continue
        advance(cell_root, args)
        status = _stage_status(cell_root)
        phase = status.get("phase")

    if phase == "winner-frozen":
        _finalize_session(cell_root, args.usage_json)
    elif phase == "certified":
        print("finish: cell is already certified — nothing to do")
    else:
        _fail(f"finish does not know how to advance phase {phase!r}")

    print("finish: final status")
    print(json.dumps(status, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser(
        "up", help="preflight + tmux + baseline + discovery orchestrator")
    up.add_argument("cell_root")
    up.add_argument("--gpu", type=int, required=True,
                    help="physical GPU index for this cell's partition")
    up.set_defaults(func=cmd_up)

    launch = sub.add_parser(
        "launch", help="send the campaign launcher into the agent window")
    launch.add_argument("cell_root")
    launch.set_defaults(func=cmd_launch)

    bind = sub.add_parser(
        "bind", help="bind the running session from the operator shell")
    bind.add_argument("cell_root")
    bind.add_argument("--timeout-s", type=float, default=600,
                      help="overall deadline for session_open + binding")
    bind.set_defaults(func=cmd_bind)

    watch = sub.add_parser("watch", help="loop budget/phase/results reporting")
    watch.add_argument("cell_root")
    watch.add_argument("--interval", type=float, default=60)
    watch.set_defaults(func=cmd_watch)

    fleet = sub.add_parser(
        "fleet", help="one status row per cell root under a runtime root")
    fleet.add_argument("root")
    fleet.set_defaults(func=cmd_fleet)

    finish = sub.add_parser(
        "finish",
        help="drive a finished-discovery cell to a finalized frozen winner",
    )
    finish.add_argument("cell_root")
    finish.add_argument(
        "--gpu", type=int, default=None,
        help="physical GPU index; required whenever finish must START a "
             "promotion orchestrator",
    )
    finish.add_argument(
        "--usage-json", default=None,
        help="path to a runtime usage JSON block, passed verbatim into the "
             "session attestation; omitted = honest `unavailable`",
    )
    finish.add_argument(
        "--attest", default=None,
        help="operator attestation for `automil activity close` when the "
             "runtime died before its SessionEnd hook",
    )
    finish.add_argument(
        "--accept-missing", action="store_true",
        help="proceed to freeze-promotion even though planned promotion jobs "
             "lack a terminal result.json (they freeze as 'ineligible'); "
             "without this flag finish refuses and lists the missing nodes",
    )
    finish.set_defaults(func=cmd_finish)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
