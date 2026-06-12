"""cancel command: terminate a running experiment via its backend (CLI-03 / D-66).

Workflow:
  1. Look up node_id in graph.json (hard-fail if unknown).
  2. Hard-fail if node is not in 'running' state.
  3. Read backend name from node.metadata.backend (default 'local' for legacy nodes, D-76).
  4. Read running/<node_id>.json to obtain opaque_id + submitted_at (W-03 fix: NOT
     from graph metadata — opaque_id is only known after the daemon launches the job).
  5. Resolve BackendClass via BACKENDS[backend_name]; instantiate.
  6. Reconstruct JobHandle; call backend.cancel(handle) — fire-and-forget.
  7. Poll up to --timeout seconds for JobState.CANCELLED.
  8. Atomically update graph node: status='cancelled', cancelled_at, cancel_reason='cli'.
  9. Move running/<id>.json to archive/<id>/.
 10. Echo "Cancelled {node_id}."
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root
from automil.cli.lifecycle._shared import _get_node_or_die

logger = logging.getLogger(__name__)


@main.command("cancel")
@click.argument("node_id")
@click.option(
    "--timeout",
    default=30,
    type=int,
    show_default=True,
    help="Seconds to wait for the job to reach CANCELLED state before failing.",
)
def cancel(node_id: str, timeout: int) -> None:
    """Cancel a running experiment by node_id.

    Dispatches through the registered backend (BACKENDS[node.metadata.backend]).
    Reads the job's opaque_id from running/<node_id>.json (written by the daemon
    at launch time — not from graph metadata, which has no PID until launch).

    Polls up to --timeout seconds for the CANCELLED state transition, then
    updates graph.json atomically (status=cancelled, cancelled_at, cancel_reason=cli)
    and archives the running spec file.

    Hard-fails if:
      - node_id is not in graph.json.
      - node is not in 'running' state.
      - running/<node_id>.json does not exist or is missing 'opaque_id'.
      - backend name is not in the BACKENDS registry.
      - the cancel does not complete within --timeout seconds.
    """
    # Lazy imports inside function body to prevent circular imports at CLI load
    # (PATTERNS.md §8 / D-69).
    from automil.backends import BACKENDS, JobHandle, JobState  # noqa: PLC0415
    from automil.backends.local import LocalBackend  # noqa: F401,PLC0415

    adir = _find_automil_dir()

    # Step 1: look up node — hard-fail if unknown.
    node = _get_node_or_die(adir, node_id)

    # Step 2: hard-fail if node is not running.
    state = node.get("status", "")
    if state != "running":
        raise click.ClickException(
            f"Refusing to cancel: node {node_id!r} is in state {state!r}, not 'running'. "
            f"Only running experiments can be cancelled. "
            f"Use `automil status` to verify the current state."
        )

    # Step 3: resolve backend name — D-76 default fallback for legacy nodes.
    backend_name: str = node.get("metadata", {}).get("backend", "local")

    # Step 4 (W-03 fix): read opaque_id + submitted_at from running/<node_id>.json.
    orch_dir = adir / "orchestrator"
    # D-169: per-backend namespace; backend_name resolved at step 3 (D-76 default 'local').
    running_path = orch_dir / "running" / backend_name / f"{node_id}.json"
    if not running_path.exists():
        raise click.ClickException(
            f"Refusing to cancel: no running spec at {running_path}. "
            f"Node may have already finished — try `automil status`."
        )

    try:
        running_spec: dict = json.loads(running_path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Running spec at {running_path} is malformed JSON: {exc}. "
            f"Inspect the file and manage the process manually."
        ) from exc

    opaque_id: str = running_spec.get("opaque_id", "")
    metadata: dict = running_spec.get("metadata", {})
    _pid: int | None = metadata.get("pid")
    _pgid: int | None = metadata.get("pgid")
    _starttime: int | None = metadata.get("starttime_ticks")  # absent on non-Linux

    if not opaque_id and not (_pid and _pgid):
        raise click.ClickException(
            f"Running spec at {running_path} has neither 'opaque_id' nor "
            f"'metadata.pid'/'metadata.pgid' — corrupted state. "
            f"Manage the process manually."
        )

    if not opaque_id:
        # Direct-kill path (OPS-01 / D-01): signal the process group from the CLI using
        # on-disk metadata. The daemon's in-memory self.running is empty in a fresh CLI
        # process, so routing through backend.cancel() + _kill_experiment() is a no-op.
        import signal as _signal  # noqa: PLC0415
        from automil.orchestrator import (  # noqa: PLC0415
            _is_pid_alive_with_starttime,
            _read_proc_starttime,
        )

        def _proc_state(pid: int) -> str | None:
            """Return the single-char process state from /proc/<pid>/stat, or None."""
            try:
                line = Path(f"/proc/{pid}/stat").read_text()
                # Format: <pid> (<comm>) <state> ...  — state is after the closing ')'
                return line.split(")", 1)[1].strip().split()[0]
            except (FileNotFoundError, PermissionError, OSError, IndexError):
                return None

        def _try_reap(pid: int) -> None:
            """Attempt a non-blocking waitpid to reap a zombie child.

            When the CLI process is the parent of the killed process (e.g. in
            in-process test runners), the dead process becomes a zombie until
            wait() is called. os.waitpid with WNOHANG reaps it without blocking;
            this allows os.kill(pid, 0) to subsequently raise ProcessLookupError
            as callers expect. Safe to call even if the process is not a child
            (ChildProcessError is silently ignored).

            WR-02: callers MUST gate this on a starttime cross-check so we never
            reap an unrelated PID-reused child the CLI happens to parent — doing
            so would silently consume that innocent child's exit status and
            desync a still-live job's bookkeeping.
            """
            try:
                os.waitpid(pid, os.WNOHANG)
            except (ChildProcessError, PermissionError, OSError):
                pass

        def _is_alive(pid: int, st: int | None) -> bool:
            """PID-reuse-safe liveness check, zombie-aware.

            A zombie process (state 'Z') has been killed but not yet reaped by
            its parent. Its /proc entry still exists, so os.kill(pid, 0) and
            _is_pid_alive_with_starttime both return True even though the process
            is effectively dead. We treat zombie as dead so the cancel command
            does not spin waiting for a parent it cannot control to call wait().

            Falls back to os.kill(pid, 0) probe on non-Linux (no /proc).
            """
            state = _proc_state(pid)
            if state == "Z":
                # Zombie — process terminated, waiting for parent reap. Treat as dead.
                return False
            if st is not None:
                return _is_pid_alive_with_starttime(pid, st)
            if state is None:
                # Non-Linux: /proc unavailable; fall back to signal-0 probe.
                try:
                    os.kill(pid, 0)
                    return True
                except ProcessLookupError:
                    return False
                except PermissionError:
                    # WR-03: EPERM from kill(pid, 0) means the process EXISTS but
                    # is owned by another user — it is alive, not dead. Mapping it
                    # to dead would falsely flip the graph to 'cancelled' while the
                    # real job keeps running and holding GPU/VRAM.
                    return True
            # state is a non-Z letter ('R', 'S', 'D', etc.) — process is alive.
            return True

        if not _is_alive(_pid, _starttime):
            # Process already gone — skip signalling, proceed to graph update.
            logger.debug("cancel: process %d already gone, skipping kill", _pid)
        else:
            # SIGTERM first (mirror daemon's SIGTERM→grace→SIGKILL pattern).
            try:
                os.killpg(_pgid, _signal.SIGTERM)
                logger.debug("cancel: sent SIGTERM to pgid %d for %s", _pgid, node_id)
            except (ProcessLookupError, PermissionError):
                pass

            # 5-second grace period (mirrors daemon's _pending_sigkill_at pattern).
            _grace_deadline = time.monotonic() + 5.0
            while time.monotonic() < _grace_deadline and _is_alive(_pid, _starttime):
                time.sleep(0.2)

            if _is_alive(_pid, _starttime):
                # Grace elapsed — escalate to SIGKILL.
                try:
                    os.killpg(_pgid, _signal.SIGKILL)
                    logger.debug("cancel: sent SIGKILL to pgid %d for %s", _pgid, node_id)
                except (ProcessLookupError, PermissionError):
                    pass
                # Brief wait for SIGKILL to land.
                for _ in range(10):
                    if not _is_alive(_pid, _starttime):
                        break
                    time.sleep(0.1)

            if _is_alive(_pid, _starttime):
                raise click.ClickException(
                    f"Could not kill process group {_pgid} for node {node_id!r} after "
                    f"SIGTERM + SIGKILL. Manage the process manually."
                )

        # Reap any zombie: when CLI and the killed process share the same parent
        # (e.g. in-process test runners), the dead process lingers as a zombie
        # until wait() is called. _try_reap uses WNOHANG so it never blocks.
        #
        # WR-02: gate the reap on the starttime cross-check so we never reap an
        # unrelated PID-reused child this CLI happens to parent.
        #   - Linux (_starttime set): reap ONLY when /proc still reports the same
        #     starttime for _pid. If _read_proc_starttime(_pid) now differs (or is
        #     None), the original target is gone and _pid may have been recycled
        #     into an innocent child — skip the reap so we never consume its exit
        #     status. (A just-SIGKILLed zombie keeps its original starttime in
        #     /proc until reaped, so the legitimate reap still fires here.)
        #   - Non-Linux (_starttime is None): /proc is unavailable, so there is no
        #     starttime to verify. OPS-01 already accepts the PID-reuse risk on
        #     this path (the kill itself targets the on-disk pid/pgid unverified),
        #     so a best-effort reap of the just-killed _pid is consistent with
        #     that contract and is required to clear the zombie.
        _reap_starttime = _read_proc_starttime(_pid) if _starttime is not None else None
        if _starttime is None or _reap_starttime == _starttime:
            _try_reap(_pid)

        # Fall through to graph update + archive move (Steps 8-10 below — shared path).

    else:
        # opaque_id present: existing backend.cancel() + poll path (Steps 5-7 — UNCHANGED).
        submitted_at: float = running_spec.get("submitted_at", 0.0)
        if isinstance(submitted_at, str):
            # ISO-8601 string → epoch float (some specs write ISO-8601).
            try:
                submitted_at = datetime.fromisoformat(submitted_at).replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except (ValueError, TypeError):
                submitted_at = 0.0

        # Step 5: resolve backend class.
        BackendClass = BACKENDS.get(backend_name)
        if BackendClass is None:
            raise click.ClickException(
                f"Unknown backend {backend_name!r}; available: {sorted(BACKENDS.keys())}. "
                f"Check automil/config.yaml or import the backend module first."
            )

        # Step 6: instantiate backend + reconstruct JobHandle.
        try:
            git_root = _find_git_root()
        except click.ClickException:
            git_root = adir.parent

        backend = BackendClass(project_root=git_root, automil_dir=adir)
        handle = JobHandle(
            node_id=node_id,
            backend=backend_name,
            opaque_id=opaque_id,
            submitted_at=submitted_at,
        )

        # Step 7: fire-and-forget cancel; poll for CANCELLED.
        backend.cancel(handle)
        logger.debug("cancel sent for %s via %s; polling for CANCELLED...", node_id, backend_name)

        deadline = time.monotonic() + timeout
        final_state: JobState | None = None
        while time.monotonic() < deadline:
            try:
                final_state = backend.poll(handle)
            except Exception as exc:  # noqa: BLE001
                logger.debug("poll error during cancel wait: %s", exc)
                final_state = None
            if final_state == JobState.CANCELLED:
                break
            time.sleep(1.0)

        if final_state != JobState.CANCELLED:
            current = final_state.value if final_state is not None else "unknown"
            raise click.ClickException(
                f"Cancel sent but state did not transition to 'cancelled' within "
                f"{timeout}s (current state: {current!r}). "
                f"Inspect the process manually and re-run `automil cancel {node_id}` "
                f"or use `automil status`."
            )

    # Step 8: atomically update graph node via locked_update (CR-01).
    # Routes through graph.cancel() so the terminal transition (a) serializes
    # against the daemon under the same lock every other writer uses, and (b)
    # decrements meta.total_proposed — a running/pending node was counted as
    # proposed (mark_running does NOT decrement), so the raw write previously
    # drifted the proposed counter on every cancel.
    from automil.graph import locked_update  # noqa: PLC0415
    from automil.cli._helpers import _load_technique_map  # noqa: PLC0415

    graph_path = adir / "graph.json"
    if graph_path.exists():
        with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
            node = graph.get_node(node_id)
            if node:
                graph.cancel(node_id)  # decrements total_proposed + sets status
                node.setdefault("metadata", {})["cancelled_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                node["metadata"]["cancel_reason"] = "cli"
            else:
                logger.warning(
                    "cancel: node %s vanished from graph during lock", node_id
                )

    # Step 9: move running/<id>.json to archive/<id>/.
    archive_node_dir = orch_dir / "archive" / node_id
    archive_node_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_node_dir / f"{node_id}_running_spec.json"
    try:
        running_path.rename(dest)
    except OSError as exc:
        logger.warning(
            "cancel: could not move running spec %s → %s: %s",
            running_path, dest, exc,
        )

    # Step 10: confirm.
    click.echo(f"Cancelled {node_id}.")
