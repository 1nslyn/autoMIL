# Phase 13: CLI Lifecycle & Operability — Pattern Map

**Mapped:** 2026-06-12
**Files analyzed:** 9 (4 new, 5 modified)
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/automil/cli/dequeue.py` | command (CLI) | request-response + file-I/O | `src/automil/cli/cancel.py` | exact |
| `src/automil/cli/cancel.py` | command (CLI) | request-response + process-signal | `src/automil/backends/_orchestrator_daemon.py:1656` (kill pattern) | role+flow match |
| `src/automil/cli/submit.py` | command (CLI) | CRUD + graph state-machine | `src/automil/cli/submit.py:495-519` (existing locked_update block) | in-file extension |
| `src/automil/cli/__init__.py` | CLI group entry | request-response | existing `main` group in same file | in-file extension |
| `src/automil/cli/_helpers.py` | utility | request-response | existing `_find_automil_dir` in same file | in-file extension |
| `src/automil/cli/viz.py` | command (CLI) | request-response | `src/automil/viz/server.py:285-295` (host fallback block) | flow match |
| `src/automil/viz/server.py` | server command | request-response | `server.py:285-295` (host block, same file) | in-file extension |
| `tests/test_cli_dequeue.py` | test | — | `tests/test_cli_cancel_resubmit.py` | exact |
| `tests/test_cli_project_option.py` | test | — | `tests/test_cli_cancel_resubmit.py` | exact |
| `tests/test_viz_port_config.py` | test | — | `tests/test_cli_cancel_resubmit.py` | role match |

---

## Pattern Assignments

### `src/automil/cli/dequeue.py` (new command, request-response + file-I/O)

**Analog:** `src/automil/cli/cancel.py`

**Imports pattern** (cancel.py lines 16-30):
```python
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir
from automil.cli.lifecycle._shared import _get_node_or_die

logger = logging.getLogger(__name__)
```

**Command registration pattern** (cancel.py lines 33-42):
```python
@main.command("dequeue")          # <-- change verb to "dequeue"
@click.argument("node_id")
def dequeue(node_id: str) -> None:
    """Dequeue a queued or pending node by node_id.

    Removes orchestrator/queue/<node>.json if present and marks the graph
    node cancelled via graph.cancel() under locked_update.

    Hard-fails if the node is running ('use automil cancel') or already terminal.
    Idempotent: a pending node with no queue spec on disk is still marked cancelled.
    """
    # Lazy imports inside function body (PATTERNS §8 / D-69).
    from automil.graph import locked_update  # noqa: PLC0415
```

**State guard pattern** (mirror of cancel.py lines 70-77):
```python
    adir = _find_automil_dir()
    node = _get_node_or_die(adir, node_id)

    state = node.get("status", "")
    if state == "running":
        raise click.ClickException(
            f"Node {node_id!r} is running. Use `automil cancel {node_id}` to stop it."
        )
    TERMINAL = {"completed", "cancelled", "crashed", "keep", "discard"}
    if state in TERMINAL:
        raise click.ClickException(
            f"Node {node_id!r} is already terminal (status={state!r}). Nothing to dequeue."
        )
```

**Queue file removal + locked_update graph mutation pattern**
(queue path from RESEARCH.md §OPS-02; locked_update from submit.py lines 496-497):
```python
    orch_dir = adir / "orchestrator"
    queue_spec = orch_dir / "queue" / f"{node_id}.json"
    if queue_spec.exists():
        try:
            queue_spec.unlink()
        except OSError as exc:
            raise click.ClickException(
                f"Could not remove queue spec at {queue_spec}: {exc}"
            ) from exc

    graph_path = adir / "graph.json"
    if graph_path.exists():
        from automil.graph import locked_update  # noqa: PLC0415
        from automil.cli._helpers import _load_technique_map  # noqa: PLC0415
        with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
            if graph.get_node(node_id):
                graph.cancel(node_id)

    click.echo(f"Dequeued {node_id}.")
```

**Key difference from cancel.py:** No archive move (queue specs are not moved to archive), no poll loop, no backend instantiation. Uses `locked_update` (unlike cancel.py which uses raw tempfile write — this is a known gap in cancel.py that dequeue must NOT copy).

---

### `src/automil/cli/cancel.py` — OPS-01 direct-kill extension

**Analog for kill pattern:** `src/automil/backends/_orchestrator_daemon.py:145-165` (verified via RESEARCH.md)

**Fix site** (cancel.py lines 100-105 — relax hard-fail):
```python
# BEFORE (hard-fail):
opaque_id: str = running_spec.get("opaque_id", "")
if not opaque_id:
    raise click.ClickException(
        f"Running spec at {running_path} is missing 'opaque_id' — corrupted state. "
        f"Manage the process manually."
    )

# AFTER (add fallback path):
opaque_id: str = running_spec.get("opaque_id", "")
metadata = running_spec.get("metadata", {})
pid: int | None = metadata.get("pid")
pgid: int | None = metadata.get("pgid")
starttime: int | None = metadata.get("starttime_ticks")  # Optional — absent on non-Linux

if not opaque_id and not (pid and pgid):
    raise click.ClickException(
        f"Running spec at {running_path} has neither 'opaque_id' nor "
        f"'metadata.pid'/'metadata.pgid' — corrupted state. "
        f"Manage the process manually."
    )
```

**Direct-kill helper import** (confirmed importable — orchestrator.py lines 38-40):
```python
# Inside the function body (lazy import, D-69):
from automil.orchestrator import _is_pid_alive_with_starttime, _read_proc_starttime  # noqa: PLC0415
```

**Direct-kill branch** (mirrors daemon `_kill_experiment` pattern, SIGTERM→grace→SIGKILL):
```python
if not opaque_id:
    # Direct-kill path: no daemon round-trip.
    import os, signal  # noqa: PLC0415

    # Starttime cross-check: only available on Linux (/proc).
    # When absent, fall back to os.kill(pid, 0) existence check.
    def _is_alive(pid: int, st: int | None) -> bool:
        if st is not None:
            return _is_pid_alive_with_starttime(pid, st)
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    if not _is_alive(pid, starttime):
        # Process already gone — skip to graph update.
        pass
    else:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        # SIGTERM grace period (mirror daemon's 5s), then SIGKILL.
        grace_deadline = time.monotonic() + 5.0
        while time.monotonic() < grace_deadline and _is_alive(pid, starttime):
            time.sleep(0.2)

        if _is_alive(pid, starttime):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            # Brief wait for SIGKILL to land.
            for _ in range(10):
                if not _is_alive(pid, starttime):
                    break
                time.sleep(0.1)

        if _is_alive(pid, starttime):
            raise click.ClickException(
                f"Could not kill process group {pgid} for node {node_id!r}. "
                f"Manage the process manually."
            )

    # Fall through to graph update + archive move (Steps 8-10 — unchanged).
```

**Poll loop replacement for direct-kill path:** Steps 7 (backend.cancel + poll) are skipped entirely when taking the direct-kill branch. The liveness loop above IS the wait. Jump directly to Step 8 (graph update).

---

### `src/automil/cli/submit.py` — OPS-03 else branch

**Fix site** (submit.py lines 495-519 — add else branch after existing if block):
```python
# EXISTING (lines 494-519):
graph_path = adir / "graph.json"
if graph_path.exists():
    from automil.graph import locked_update
    with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
        if not graph.get_node(node):
            # ... add_proposed + mark_running for new nodes (unchanged)
            graph.mark_running(allocated)
        # ADD: else branch for existing proposed/pending node
        else:
            existing = graph.get_node(node)
            if existing and existing.get("type") == "proposed" and existing.get("status") == "pending":
                graph.mark_running(node)
```

`mark_running` (graph.py lines 280-289) is already type/status-guarded — calling it on any other state logs a warning and returns False without mutating the graph. The `else` branch is safe to add unconditionally.

---

### `src/automil/cli/__init__.py` — OPS-04 group-level `--project` option

**Current main group** (lines 11-25):
```python
@click.group()
def main():
    """autoMIL: Autonomous agent-driven MIL model improvement."""
    try:
        from automil.cli._helpers import _find_automil_dir
        from automil.cells.activity import touch_last_action
        touch_last_action(_find_automil_dir())
    except Exception:
        pass
```

**Modified main group pattern** (D-07/D-08):
```python
@click.group()
@click.option(
    "--project",
    "project_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to project root (containing automil/config.yaml) or automil/ dir. "
         "Overrides cwd-based project discovery.",
    is_eager=True,   # resolve before subcommand options
)
def main(project_path: str | None):
    """autoMIL: Autonomous agent-driven MIL model improvement."""
    import automil.cli._helpers as _h  # noqa: PLC0415
    if project_path is not None:
        from pathlib import Path  # noqa: PLC0415
        _h._PROJECT_OVERRIDE = Path(project_path).resolve()
    try:
        from automil.cli._helpers import _find_automil_dir  # noqa: PLC0415
        from automil.cells.activity import touch_last_action  # noqa: PLC0415
        touch_last_action(_find_automil_dir())   # now respects override
    except Exception:
        pass
```

**Command registration block unchanged** — `from automil.cli import dequeue` added in alphabetical order after `cancel`:
```python
from automil.cli import cancel  # noqa: E402,F401
from automil.cli import dequeue  # noqa: E402,F401   ← NEW
from automil.cli import cell    # noqa: E402,F401
```

---

### `src/automil/cli/_helpers.py` — OPS-04 module-global override bridge

**Current `_find_automil_dir`** (lines 18-31 — pure cwd walk, no override hook).

**Modified pattern** (D-08 module-global):
```python
# Add at module level, before _find_automil_dir definition:
_PROJECT_OVERRIDE: "Path | None" = None   # set by main() callback for --project


def _find_automil_dir() -> Path:
    """Walk up from cwd to find a directory containing automil/config.yaml.

    Honors _PROJECT_OVERRIDE when set by the --project group option before
    falling through to the cwd walk.
    """
    if _PROJECT_OVERRIDE is not None:
        # Accept project root (containing automil/config.yaml) or automil/ dir itself.
        for candidate in (_PROJECT_OVERRIDE / "automil", _PROJECT_OVERRIDE):
            if (candidate / "config.yaml").exists():
                # Normalise: always return the automil/ dir.
                return candidate if candidate.name == "automil" else candidate / "automil"
        raise click.ClickException(
            f"--project {_PROJECT_OVERRIDE}: no automil/config.yaml found under "
            f"{_PROJECT_OVERRIDE}. Point --project at the project root or automil/ dir."
        )
    # Existing cwd walk — unchanged:
    p = Path.cwd()
    while p != p.parent:
        candidate = p / "automil" / "config.yaml"
        if candidate.exists():
            return p / "automil"
        p = p.parent
    raise click.ClickException(
        "No automil/config.yaml found. Run 'automil init' in your project root."
    )
```

**Test isolation note:** Every OPS-04 test must reset the override: `monkeypatch.setattr(automil.cli._helpers, "_PROJECT_OVERRIDE", None)` — or use the `monkeypatch` fixture which auto-reverts.

---

### `src/automil/cli/viz.py` — OPS-05 port default None

**Current** (line 17): `@click.option("--port", default=8420, help="Server port")`

**Modified pattern**:
```python
@viz_group.command("start")
@click.option(
    "--port",
    default=None,      # ← was 8420; now None so config fallback fires
    type=int,
    help="Server port (default: viz.port in automil/config.yaml, then 8420).",
)
@click.option(
    "--host", default=None,
    help="Bind address (default: 127.0.0.1; falls back to viz.host in "
         "automil/config.yaml then AUTOMIL_VIZ_HOST env var). Pass 0.0.0.0 "
         "only on trusted networks — the dashboard exposes PIDs and node "
         "descriptions and has no auth.",
)
def viz_start(port: int | None, host: str | None):    # ← type widens to int | None
    """Start the 3D visualization dashboard."""
    adir = _find_automil_dir()
    from automil.viz.server import cmd_start
    cmd_start(port=port, project_root=adir.parent, host=host)
```

---

### `src/automil/viz/server.py` — OPS-05 port resolution in `cmd_start`

**Analog:** host fallback block (server.py lines 285-295) — exact mirror.

**Current signature** (line 265): `def cmd_start(port: int = DEFAULT_PORT, ...)`

**Modified signature**: `def cmd_start(port: int | None = None, ...)`

**Port resolution block** (add immediately after host fallback block at line 295, reusing already-loaded `cfg`):

```python
# server.py lines 285-295 — existing host fallback (DO NOT MODIFY):
if host is None:
    cfg_host: str | None = None
    config_path = automil_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(config_path.read_text()) or {}
            cfg_host = (cfg.get("viz") or {}).get("host")
        except Exception:
            cfg_host = None
    host = cfg_host or os.environ.get("AUTOMIL_VIZ_HOST") or "127.0.0.1"

# ADD: port resolution block immediately after (shares config_path variable):
if port is None:
    cfg_port: int | None = None
    if config_path.exists():
        try:
            # cfg may already be loaded above; if host block ran, reuse it.
            # If not (host was explicit), load config fresh.
            if "cfg" not in dir():
                import yaml as _yaml
                cfg = _yaml.safe_load(config_path.read_text()) or {}
            raw_port = (cfg.get("viz") or {}).get("port")
            if raw_port is not None:
                cfg_port = int(raw_port)
        except Exception:
            cfg_port = None
    port = cfg_port if cfg_port is not None else DEFAULT_PORT
```

**Implementation note:** In practice both blocks are reachable in sequence. The cleanest implementation loads `cfg` once in a shared block before the two conditionals, rather than relying on `"cfg" not in dir()`. The planner should refactor to load config once and use it for both host and port resolution.

---

## Shared Patterns

### Lazy imports inside function body (D-69 / PATTERNS §8)
**Source:** `src/automil/cli/cancel.py` lines 62-63
**Apply to:** `dequeue.py` (all backend/graph imports), OPS-04 `__init__.py` group callback
```python
# Lazy imports inside function body to prevent circular imports at CLI load
# (PATTERNS.md §8 / D-69).
from automil.backends import BACKENDS, JobHandle, JobState  # noqa: PLC0415
```

### `_get_node_or_die` helper
**Source:** `src/automil/cli/lifecycle/_shared.py` lines 63-97
**Apply to:** `dequeue.py` (same node-lookup + available-listing pattern as cancel.py)
```python
from automil.cli.lifecycle._shared import _get_node_or_die
# ...
node = _get_node_or_die(adir, node_id)
```

### `locked_update` for graph mutations
**Source:** `src/automil/cli/submit.py` lines 496-497
**Apply to:** `dequeue.py` graph.cancel() call; OPS-03 submit.py else branch
```python
from automil.graph import locked_update
with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
    graph.cancel(node_id)   # or graph.mark_running(node_id)
```
**Critical:** `cancel.py` uses raw tempfile+os.replace for its graph write (known gap). `dequeue.py` MUST use `locked_update`, not copy cancel.py's graph-write pattern.

### CliRunner test fixture pattern
**Source:** `tests/test_cli_cancel_resubmit.py` lines 29-85, 161-178
**Apply to:** all three new test files
```python
@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()

@pytest.fixture(autouse=True)
def _isolated_backends():
    from automil.backends import BACKENDS
    saved = dict(BACKENDS)
    yield
    BACKENDS.clear()
    BACKENDS.update(saved)

def _make_adir(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir(exist_ok=True)
    adir = tmp_path / "automil"
    orch_dir = adir / "orchestrator"
    for sub in ("queue", "running", "archive"):
        (orch_dir / sub).mkdir(parents=True, exist_ok=True)
    (adir / "config.yaml").write_text("run:\n  script: train.py\n")
    return adir

def _write_graph(adir: Path, nodes: dict[str, Any]) -> None:
    graph = {
        "schema_version": 1,
        "meta": {
            "best_composite": 0.0, "best_node_id": None, "total_executed": 0,
            "total_proposed": 0, "next_id": 10, "baseline_composite": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

# Invoke pattern:
result = cli_runner.invoke(main, ["cancel", node_id, "--timeout", "10"], catch_exceptions=False)
assert result.exit_code == 0, f"failed: {result.output}"
```

### ClickException hard-fail pattern
**Source:** `src/automil/cli/cancel.py` lines 72-77, 87-90, 100-105
**Apply to:** `dequeue.py` state guard, OPS-04 `_find_automil_dir` override fail
```python
raise click.ClickException(
    f"Refusing to cancel: node {node_id!r} is in state {state!r}, not 'running'. "
    f"Only running experiments can be cancelled. "
    f"Use `automil status` to verify the current state."
)
```

### `monkeypatch.chdir` for project discovery in tests
**Source:** `tests/test_cli_cancel_resubmit.py` line 178
**Apply to:** All new test files — cwd must point at tmp_path so `_find_automil_dir()` resolves correctly
```python
monkeypatch.chdir(tmp_path)
```

---

## OPS-01 Real-Process Test Fixture (Anti-Theater Pattern)

**Source:** RESEARCH.md §"Critical Test Fixture Requirements — OPS-01"
**Apply to:** `tests/test_cli_cancel_resubmit.py` new tests `test_cancel_local_direct_kill` and `test_cancel_missing_pid_metadata`

The test MUST spawn a real process and assert it dies — do NOT mock `os.killpg`:

```python
import subprocess
import os
import json
import time

def test_cancel_local_direct_kill(cli_runner, tmp_path, monkeypatch):
    from automil.cli import main
    from automil.orchestrator import _read_proc_starttime

    adir = _make_adir(tmp_path)
    monkeypatch.chdir(tmp_path)
    node_id = "node_0010"

    # 1. Spawn a real child process in its own session.
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    pgid = os.getpgid(proc.pid)
    starttime = _read_proc_starttime(proc.pid)   # None on non-Linux

    # 2. Write running spec with actual pid/pgid/starttime_ticks (daemon shape).
    running_dir = adir / "orchestrator" / "running" / "local"
    running_dir.mkdir(parents=True, exist_ok=True)
    spec = {"id": node_id, "metadata": {"pid": proc.pid, "pgid": pgid}}
    if starttime is not None:
        spec["metadata"]["starttime_ticks"] = starttime
    (running_dir / f"{node_id}.json").write_text(json.dumps(spec))

    # 3. Write graph with status=running.
    _write_graph(adir, {node_id: {
        "id": node_id, "parent_id": None, "type": "proposed", "status": "running",
        "description": "direct kill test", "techniques": [],
        "metadata": {"backend": "local"},
    }})

    result = cli_runner.invoke(main, ["cancel", node_id], catch_exceptions=False)
    assert result.exit_code == 0, f"cancel failed: {result.output}"

    # 4. Assert real process is dead.
    time.sleep(0.3)
    try:
        os.kill(proc.pid, 0)
        assert False, "process still alive after cancel"
    except ProcessLookupError:
        pass   # expected

    # 5. Assert graph updated.
    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][node_id]["status"] == "cancelled"
```

---

## No Analog Found

All files have close analogs. No entries.

---

## Metadata

**Analog search scope:** `src/automil/cli/`, `src/automil/viz/`, `src/automil/graph.py`, `src/automil/orchestrator.py`, `tests/`
**Files read:** 10 source files
**Pattern extraction date:** 2026-06-12
