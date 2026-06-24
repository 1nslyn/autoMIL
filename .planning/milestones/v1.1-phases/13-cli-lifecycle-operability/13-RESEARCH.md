# Phase 13: CLI Lifecycle & Operability — Research

**Researched:** 2026-06-12
**Domain:** Python CLI (Click), process management (os.killpg, /proc), graph state machine, file-system orchestrator layout
**Confidence:** HIGH — all five fixes verified line-by-line against live source; no external library research needed.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** CLI `cancel` must signal the process group directly from the CLI process using
  on-disk metadata (`metadata.pid` / `metadata.pgid` / `metadata.starttime_ticks`), NOT route
  through `backend.cancel()` → `_kill_experiment()`.
- **D-02:** For a `local` running spec lacking top-level `opaque_id`, fall back to
  `metadata.pid` / `metadata.pgid`. Signal with `os.killpg(pgid, SIGTERM)` guarded by
  `_is_pid_alive_with_starttime`. Reuse the existing helpers from `_orchestrator_daemon.py`.
- **D-03:** Loud-fail only when spec has neither `opaque_id` nor `metadata.pid`/`metadata.pgid`.
  Preserve existing graph update and running-spec→archive move. Poll via starttime cross-check.
- **D-04:** New `automil dequeue <node_id>` command in its own module `src/automil/cli/dequeue.py`,
  registered in `cli/__init__.py`. Does NOT overload `cancel`. Removes
  `orchestrator/queue/<node>.json` if present + marks graph node `cancelled` via `graph.cancel()`
  under `locked_update`.
- **D-05:** `dequeue` state guard: accepts `proposed/pending` and `queued`. Hard-fail if
  `running` ("use `automil cancel`") or already terminal. Idempotent: pending with no queue spec
  → still mark `cancelled`.
- **D-06:** In `submit.py` `locked_update` block (~L497), add `else` branch: when target node
  already exists as `type=proposed, status=pending`, call `graph.mark_running(node)` after queue
  spec is written, within the same `locked_update`.
- **D-07:** Add group-level `--project PATH` option on the `main` Click group in
  `src/automil/cli/__init__.py`. Resolve in group callback and bridge through
  `_find_automil_dir()` in `src/automil/cli/_helpers.py`. Accept project root (containing
  `automil/config.yaml`) or the `automil/` dir itself. Hard-fail with clear message if not found.
- **D-08:** Prefer module-level resolved-override in `_helpers.py` set by the group callback
  (recommendation locked). `_find_automil_dir()` stays single source of truth. The `main`
  callback's `touch_last_action(_find_automil_dir())` must respect the override.
- **D-09:** Change `viz start` CLI option `--port` default to `None`. Resolve port in
  `cmd_start()` (server.py:265): explicit `--port` → `viz.port` in config → `8420`
  (`DEFAULT_PORT`). Mirror the existing host fallback block (server.py:285–295).

### Claude's Discretion

- OPS-01 signal escalation: SIGTERM→grace→SIGKILL or single SIGTERM-then-poll. Must remain
  PID-reuse-safe via starttime.
- OPS-04 bridge: module-global vs `ctx.obj` (recommendation: module-global per D-08).
- Whether `dequeue` reuses cancel.py helpers vs. inlines a smaller path — keep it minimal.

### Deferred Ideas (OUT OF SCOPE)

- Remote-backend (SLURM/Ray) cancel-from-CLI parity — those backends handle via `opaque_id`.
- Unified `automil lifecycle`/`automil rm` super-verb — keep `cancel` + `dequeue` explicit.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| OPS-01 | `automil cancel` can cancel daemon-launched local jobs via `metadata.pid`/`metadata.pgid` fallback | Verified: running spec writes pid+pgid+starttime_ticks at daemon:L1035-1039; helpers importable via `automil.orchestrator`; cancel.py hard-fails at L100-105 — that block is the sole fix site |
| OPS-02 | `automil dequeue <node>` removes queue spec + marks graph `cancelled` | Verified: queue path is flat `orchestrator/queue/<node>.json` (daemon:L395); `graph.cancel()` at graph.py:381 is the correct state-machine path; must use `locked_update` |
| OPS-03 | Submitting existing `type=proposed,status=pending` node transitions it to `running` | Verified: submit.py:498 `if not graph.get_node(node):` blocks `mark_running` for existing nodes — one-line `else` branch fix |
| OPS-04 | `--project PATH` group option routes project discovery outside project root | Verified: `_find_automil_dir()` at _helpers.py:18 has no override hook; `main` group at __init__.py:11 is the injection point |
| OPS-05 | `automil viz start` resolves port as explicit→config→8420 | Verified: viz.py:17 hard-defaults to 8420; server.py:265 `cmd_start` has `port: int = DEFAULT_PORT`; host fallback pattern at server.py:285-295 is the exact mirror |
</phase_requirements>

---

## Summary

Phase 13 is five contained CLI fixes. Every change is confined to `src/automil/` (framework purity). No new dependencies. All five issues are verified against the live source tree; the root causes match the CONTEXT.md analysis exactly.

**OPS-01** is the only fix with non-trivial logic: `cancel.py` hard-fails at L100-105 when `opaque_id` is absent. The running spec written by the daemon at `_orchestrator_daemon.py:1028-1041` does contain `metadata.pid`, `metadata.pgid`, and `metadata.starttime_ticks` (when `/proc` is available). The PID-reuse helpers `_read_proc_starttime` and `_is_pid_alive_with_starttime` live at daemon:145/158 and are already re-exported from `automil.orchestrator` — they are importable from CLI code without circular-import risk. The existing cancel poll loop must be replaced with a starttime-based liveness check because `backend.poll()` returns `CANCELLED` only after the daemon's `_handle_completion` fires, which does not happen from a CLI-initiated direct kill.

**OPS-02** introduces a new `dequeue.py` module. Queue specs are stored at the flat path `orchestrator/queue/<node>.json` (not namespaced by backend — unlike running specs which are `running/<backend>/<node>.json`). `graph.cancel()` (graph.py:381) decrements `meta.total_proposed` and sets `status=cancelled` — this is the correct path.

**OPS-03** is a one-line gap: `graph.mark_running()` (graph.py:280) is already type/status-guarded; adding an `else` branch calling it for existing-pending nodes is safe and complete.

**OPS-04** requires a module-level `_PROJECT_OVERRIDE: Path | None = None` in `_helpers.py` set by the group callback, and one conditional prepended to `_find_automil_dir()`. The `main` group callback's `touch_last_action(_find_automil_dir())` runs after the override is set because Click invokes the group callback before any sub-command.

**OPS-05** is a mirror of the existing host fallback: change `--port default=8420` to `default=None` in `viz.py`, then add a port-resolution block after the host block in `cmd_start()`, reading `(cfg.get("viz") or {}).get("port")` and falling back to `DEFAULT_PORT`.

**Primary recommendation:** Implement in five independent tasks; OPS-01 deserves its own plan because the poll-loop replacement has the most logic to test. OPS-02 through OPS-05 can be a single plan (four small changes, one new file).

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Cancel daemon-launched local job | CLI (cancel.py) | Daemon helpers (read-only) | CLI reads on-disk spec; signals directly; helpers imported for PID-reuse guard |
| Dequeue queued node | CLI (dequeue.py) | Graph state machine | File removal is CLI; graph.cancel() under locked_update is state-machine |
| pending→running on submit | CLI (submit.py locked_update block) | Graph (mark_running) | single-line gap in existing locked_update block |
| Out-of-root project discovery | CLI helpers (_helpers.py + __init__.py) | — | one seam, all commands benefit |
| Viz port config fallback | CLI (viz.py + server.py cmd_start) | — | config read already present for host; port is symmetric |

---

## Verified Code Anchors

### OPS-01: cancel.py hard-fail site [VERIFIED: source read]

```python
# cancel.py:100-105 — THE FIX SITE
opaque_id: str = running_spec.get("opaque_id", "")
if not opaque_id:
    raise click.ClickException(          # <-- relax this to fallback path
        f"Running spec at {running_path} is missing 'opaque_id' — corrupted state. "
        ...
    )
```

**What the running spec actually contains (daemon:1028-1041):**

```python
running_spec_meta["pid"] = process.pid
running_spec_meta["pgid"] = recorded_pgid        # os.getpgid(pid), fallback=pid
# starttime_ticks only written when _read_proc_starttime returns non-None
if recorded_starttime is not None:
    running_spec_meta["starttime_ticks"] = recorded_starttime
running_spec_payload["metadata"] = running_spec_meta
```

Key point: `starttime_ticks` is OPTIONAL in the running spec — it is omitted when `/proc` is unavailable (non-Linux test environments). The fix must handle both cases: with and without `starttime_ticks`.

**Helper import path (confirmed via orchestrator.py:38-40):**

```python
from automil.orchestrator import _is_pid_alive_with_starttime, _read_proc_starttime
```

These are re-exported from `automil.orchestrator`, not imported directly from the private `_orchestrator_daemon` module. This is the correct import path for CLI code — no circular import risk.

**Poll loop replacement:** The existing poll loop (cancel.py:143-162) waits for `backend.poll(handle)` → `JobState.CANCELLED`. For a direct-kill path, the daemon's `_handle_completion` does not run from the CLI, so `backend.poll()` will not return `CANCELLED`. The poll loop must be replaced with a starttime-based liveness check: loop until `_is_pid_alive_with_starttime(pid, starttime)` returns False (process gone or PID reused), or until timeout. When `starttime_ticks` is absent from the spec, fall back to `os.kill(pid, 0)` to check liveness (accepts the small PID-reuse risk documented in the daemon's comment).

**Signal escalation recommendation (Claude's discretion):** Mirror the daemon's pattern: SIGTERM first, then after a grace period (5 seconds) send SIGKILL if the process group is still alive. This matches `_kill_experiment`'s `_pending_sigkill_at` mechanism. Single-SIGTERM-then-poll is simpler but leaves zombie pgroups on stuck processes.

### OPS-02: queue path and graph.cancel() [VERIFIED: source read]

Queue path is **flat** (not backend-namespaced):

```python
# daemon:395
self.queue_dir = self.orch_dir / "queue"
# daemon:748
for f in sorted(self.queue_dir.glob("*.json")):
# submit.py:111
queue_conflict = adir / "orchestrator" / "queue" / f"{node}.json"
```

Queue files are `orchestrator/queue/<node>.json`. There is no `queue/<backend>/` variant. The CONTEXT.md note "any backend-namespaced variant, mirroring D-169" does not apply here — D-169 namespacing applies to running specs only.

`graph.cancel()` signature (graph.py:381):

```python
def cancel(self, node_id: str):
    node = self.nodes[node_id]
    node["status"] = "cancelled"
    self.meta["total_proposed"] = max(0, self.meta["total_proposed"] - 1)
```

It does NOT check pre-conditions — the caller must guard that `node_id` exists and is in a cancellable state before calling. The `locked_update` context manager handles serialization.

`cancel.py` uses raw `json.load` + `tempfile`+`os.replace` for its graph write (not `locked_update`). For `dequeue.py`, use `locked_update` (as submit.py does) since it serializes against the daemon's `_handle_completion`.

### OPS-03: submit.py locked_update block [VERIFIED: source read]

```python
# submit.py:495-519
graph_path = adir / "graph.json"
if graph_path.exists():
    from automil.graph import locked_update
    with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
        if not graph.get_node(node):           # <-- existing-pending falls through here
            # ... add_proposed + mark_running for NEW nodes
            graph.mark_running(allocated)
        # MISSING: else branch for existing type=proposed,status=pending
```

The preflight guard (submit.py:91-103) allows existing `type=proposed, status=pending` — it only refuses `type=executed` or `status in {keep, discard, crash, completed, running}`. So an existing pending node reaches the `locked_update` block and silently exits without calling `mark_running`. The fix is:

```python
        else:
            existing = graph.get_node(node)
            if existing and existing.get("type") == "proposed" and existing.get("status") == "pending":
                graph.mark_running(node)
```

`mark_running` (graph.py:280) is already guarded: it logs a warning and returns False if type/status doesn't match, so calling it on any other state is safe.

### OPS-04: _find_automil_dir override bridge [VERIFIED: source read]

Current `_helpers.py:18-31` has no override hook — pure cwd walk. The `main` group callback at `__init__.py:11-25` calls `_find_automil_dir()` inside a `try/except Exception: pass` block.

Module-global approach:

```python
# _helpers.py — add at module level
_PROJECT_OVERRIDE: Path | None = None

def _find_automil_dir() -> Path:
    if _PROJECT_OVERRIDE is not None:
        # Accept project root or automil/ dir itself
        for candidate in (_PROJECT_OVERRIDE / "automil", _PROJECT_OVERRIDE):
            if (candidate / "config.yaml").exists():
                return candidate if candidate.name == "automil" else candidate / "automil"
            # handle case where _PROJECT_OVERRIDE already points at automil/
        raise click.ClickException(...)
    # ... existing cwd walk unchanged
```

Group callback in `__init__.py`:

```python
@click.group()
@click.option("--project", "project_path", default=None, type=click.Path(exists=True), ...)
def main(project_path: str | None):
    import automil.cli._helpers as _h
    if project_path is not None:
        _h._PROJECT_OVERRIDE = Path(project_path).resolve()
    try:
        from automil.cells.activity import touch_last_action
        touch_last_action(_find_automil_dir())   # now uses override
    except Exception:
        pass
```

Threading `_PROJECT_OVERRIDE` as a module-global is safe for CLI invocations (single process, sequential command dispatch). It is NOT safe for test isolation — tests must reset `_h._PROJECT_OVERRIDE = None` in teardown. The test for OPS-04 must monkeypatch this.

### OPS-05: viz port resolution [VERIFIED: source read]

Current state:
- `viz.py:17`: `@click.option("--port", default=8420, ...)` — hard int default, no `None`
- `viz.py:25`: `def viz_start(port: int, host: str | None)` — port always int
- `server.py:265`: `def cmd_start(port: int = DEFAULT_PORT, ...)` — always int
- `server.py:285-295`: host fallback block already reads `(cfg.get("viz") or {}).get("host")`

Fix is symmetric to host:

```python
# viz.py:17
@click.option("--port", default=None, type=int, help="Server port (default: viz.port in config or 8420)")
def viz_start(port: int | None, host: str | None): ...
    cmd_start(port=port, ...)

# server.py cmd_start signature change
def cmd_start(port: int | None = None, ...):
    ...
    # After the host fallback block:
    if port is None:
        cfg_port: int | None = None
        if config_path.exists():
            try:
                import yaml as _yaml
                cfg = _yaml.safe_load(config_path.read_text()) or {}
                cfg_port = (cfg.get("viz") or {}).get("port")
                if cfg_port is not None:
                    cfg_port = int(cfg_port)
            except Exception:
                cfg_port = None
        port = cfg_port if cfg_port is not None else DEFAULT_PORT
```

The config is already loaded in the host fallback block — reuse `cfg` rather than re-reading. The host block is at lines 285-295, so the port block should follow immediately after (sharing the same config load).

---

## Architecture Patterns

### Recommended Project Structure (new/modified files)

```
src/automil/
├── cli/
│   ├── __init__.py          # OPS-04: add --project group option + module-global bridge
│   ├── _helpers.py          # OPS-04: add _PROJECT_OVERRIDE + conditional in _find_automil_dir
│   ├── cancel.py            # OPS-01: relax opaque_id hard-fail; add direct-kill path
│   ├── dequeue.py           # OPS-02: NEW — dequeue command
│   ├── submit.py            # OPS-03: add else branch in locked_update block (~L498)
│   └── viz.py               # OPS-05: --port default=None
└── viz/
    └── server.py            # OPS-05: port resolution in cmd_start
tests/
├── test_cli_cancel_resubmit.py   # OPS-01: add new test(s) — local-direct-kill path
├── test_cli_dequeue.py           # OPS-02: NEW test file
├── test_cli.py                   # OPS-03: add submit-existing-pending test
└── test_viz_port_config.py       # OPS-05: NEW or extend test_viz_promotion_rate.py
```

### Anti-Patterns to Avoid

- **Routing cancel through `backend.cancel()` for local specs:** `LocalBackend.cancel()` delegates to `_kill_experiment()` which requires `self.running` to be populated. A fresh `LocalBackend` instance has an empty `self.running`. The fix must bypass this entirely and use the on-disk spec.
- **Using `backend.poll()` to detect cancellation in the direct-kill path:** After a CLI-side `os.killpg`, the daemon's `_handle_completion` is NOT called. `backend.poll()` reads on-disk state written by the daemon. The only correct liveness check is the starttime cross-check on the pid from the running spec.
- **Raw JSON writes in dequeue.py without `locked_update`:** `cancel.py` uses raw JSON writes (tempfile+os.replace) — this is a known gap. `dequeue.py` must use `locked_update` to serialize against the daemon.
- **Overriding `_find_automil_dir` without resetting in tests:** The module-global `_PROJECT_OVERRIDE` will bleed between tests. Every test for OPS-04 must reset it in teardown or use `monkeypatch.setattr`.
- **Adding `--project` to individual command signatures:** There are ~20 commands. Adding the option to each is high-impact. The group-level approach is the only maintainable path.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PID-reuse detection | Custom /proc reader | `_is_pid_alive_with_starttime` + `_read_proc_starttime` from `automil.orchestrator` | Already handles comm-with-spaces, /proc unavailable, parse errors |
| Process group signalling | Manual PGID lookup | `os.killpg(os.getpgid(pid), signal)` as the daemon already does | `os.getpgid` can fail if proc already exited — must handle `ProcessLookupError` |
| Graph write serialization | Direct file writes | `locked_update` context manager | Serializes against daemon's `_handle_completion`; prevents race conditions |
| State-machine transitions | Direct dict mutations | `graph.cancel()` / `graph.mark_running()` | Keeps `meta.total_proposed` counter consistent |

---

## Common Pitfalls

### Pitfall 1: `starttime_ticks` absent from running spec on non-Linux

**What goes wrong:** The daemon omits `metadata.starttime_ticks` when `/proc` is unavailable (any non-Linux environment including macOS, some containers). The fix in `cancel.py` must not assume the field is present.
**Why it happens:** `_read_proc_starttime` returns `None` when `/proc/<pid>/stat` cannot be read; the daemon only writes `starttime_ticks` conditionally.
**How to avoid:** Check `starttime_ticks` is not None before calling `_is_pid_alive_with_starttime`. When absent, fall back to `os.kill(pid, 0)` (existence-only check, no reuse guard). Document this as a known limitation.
**Warning signs:** Test failures on macOS CI, "missing field" KeyError in cancel.

### Pitfall 2: OPS-01 poll loop never observes CANCELLED for direct-kill

**What goes wrong:** A naive implementation reuses the existing poll loop (`backend.poll(handle)` waiting for `JobState.CANCELLED`). After `os.killpg` from the CLI, the daemon's `_handle_completion` never runs, so `backend.poll()` reads whatever on-disk state existed before the kill — typically `JobState.RUNNING` forever. The poll times out and the cancel "fails" even though the process is dead.
**Why it happens:** `LocalBackend.poll()` reads queue/running/archive specs from disk. The daemon is the only writer of the archive result. A CLI-side kill bypasses the daemon.
**How to avoid:** Replace the poll loop with a starttime-based liveness check in the direct-kill branch. Only use `backend.poll()` for the remote-backend (`opaque_id` present) path.
**Warning signs:** Tests pass because mock backends transition state synchronously; real behavior fails silently.

### Pitfall 3: Test theater — mock kill instead of real child process for OPS-01

**What goes wrong:** The OPS-01 test writes a running spec with a fake PID (e.g. 999999) and patches `os.killpg` to a no-op. The test verifies the graph is updated but never verifies that a real process was killed.
**Why it happens:** It's easier to mock than to spawn a real subprocess.
**How to avoid:** The test must spawn a real child process (`subprocess.Popen(["sleep", "60"])`), write a running spec with its actual pid/pgid/starttime_ticks (using `_read_proc_starttime`), invoke `cancel`, and assert the process is no longer alive (`os.kill(pid, 0)` raises `ProcessLookupError`). See the test architecture section below.

### Pitfall 4: OPS-04 module-global bleeds between tests

**What goes wrong:** One test sets `_h._PROJECT_OVERRIDE = some_path` and a later test in the same session fails because `_find_automil_dir()` returns the wrong directory.
**Why it happens:** Module-global state is shared across tests in the same process.
**How to avoid:** Every test that exercises OPS-04 must use `monkeypatch.setattr(automil.cli._helpers, "_PROJECT_OVERRIDE", None)` in teardown (or use the `monkeypatch` fixture which auto-reverts).

### Pitfall 5: OPS-02 dequeue on a `running` node silently removes the queue file

**What goes wrong:** A node transitions from `queued` → `running` between the graph-status read and the queue-file unlink. The `dequeue` command removes the queue file (which no longer exists or has been claimed by the daemon) and marks the node `cancelled`, but the process is already running.
**Why it happens:** TOCTOU — dequeue checks status then acts on files.
**How to avoid:** The state guard (`node.status == "running"` → hard-fail with "use automil cancel") runs before any file operations, inside the `locked_update` block. Since the daemon's `_handle_completion` also uses `locked_update`, the race window is minimized. Document that there is a small window before the daemon picks up the spec; operators should use `automil status` to verify.

---

## Runtime State Inventory

> Not applicable — this is a greenfield fix phase with no renames or data migrations.

---

## Environment Availability

> All five fixes are file-system and in-process operations. No external tools or services required. Tests use `subprocess.Popen` for the real-process kill test (OPS-01) — `sleep` must be available on the test runner. No network dependencies.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `/proc/<pid>/stat` | OPS-01 PID-reuse guard | Linux only | — | Skip starttime check; use `os.kill(pid,0)` |
| `os.killpg` / `os.getpgid` | OPS-01 direct kill | POSIX (Linux+macOS) | — | None — documented as Linux/POSIX only |
| `sleep` binary | OPS-01 test (real process) | ✓ (standard) | — | `python -c "import time; time.sleep(60)"` |

---

## Validation Architecture

> `workflow.nyquist_validation` is enabled (key present and not false in `.planning/config.json`).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (existing suite) |
| Config file | `pyproject.toml` (workspace root) |
| Quick run command | `uv run pytest tests/test_cli_cancel_resubmit.py tests/test_cli_dequeue.py tests/test_cli.py tests/test_viz_port_config.py -x -v` |
| Full suite command | `uv run pytest tests/ -v` (automil only — NOT combined with benchmarks/) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| OPS-01 | `cancel` kills real local child process via pid/pgid from running spec | unit (real subprocess) | `uv run pytest tests/test_cli_cancel_resubmit.py::test_cancel_local_direct_kill -xvs` | ❌ Wave 0 (new test in existing file) |
| OPS-01 | `cancel` hard-fails when spec has neither `opaque_id` nor `metadata.pid` | unit | `uv run pytest tests/test_cli_cancel_resubmit.py::test_cancel_missing_pid_metadata -xvs` | ❌ Wave 0 |
| OPS-01 | `cancel` still works for `opaque_id`-bearing specs (regression) | unit | `uv run pytest tests/test_cli_cancel_resubmit.py::test_cancel_happy_path -xvs` | ✅ existing |
| OPS-02 | `dequeue` removes queue spec and marks graph `cancelled` | unit | `uv run pytest tests/test_cli_dequeue.py::test_dequeue_removes_queue_spec -xvs` | ❌ Wave 0 (new file) |
| OPS-02 | `dequeue` hard-fails for running node with cross-reference message | unit | `uv run pytest tests/test_cli_dequeue.py::test_dequeue_refuses_running -xvs` | ❌ Wave 0 |
| OPS-02 | `dequeue` marks pending node with no queue spec as cancelled (idempotent) | unit | `uv run pytest tests/test_cli_dequeue.py::test_dequeue_pending_no_spec -xvs` | ❌ Wave 0 |
| OPS-03 | Submit against existing `pending` node transitions it to `running` in graph | unit | `uv run pytest tests/test_cli.py::test_submit_existing_pending_marks_running -xvs` | ❌ Wave 0 |
| OPS-04 | `--project PATH` resolves project root; `_find_automil_dir` returns correct path | unit | `uv run pytest tests/test_cli_project_option.py::test_project_option_project_root -xvs` | ❌ Wave 0 (new file) |
| OPS-04 | `--project PATH` resolves when PATH points at `automil/` dir directly | unit | `uv run pytest tests/test_cli_project_option.py::test_project_option_automil_dir -xvs` | ❌ Wave 0 |
| OPS-04 | `--project` is absent in normal invocation — cwd walk unchanged (regression) | unit | `uv run pytest tests/test_cli_project_option.py::test_project_option_absent_cwd_walk -xvs` | ❌ Wave 0 |
| OPS-05 | `viz start` with no `--port` and no config uses `DEFAULT_PORT` 8420 | unit | `uv run pytest tests/test_viz_port_config.py::test_viz_port_default -xvs` | ❌ Wave 0 (new file) |
| OPS-05 | `viz start` with `viz.port` in config uses config value | unit | `uv run pytest tests/test_viz_port_config.py::test_viz_port_from_config -xvs` | ❌ Wave 0 |
| OPS-05 | Explicit `--port` flag overrides config `viz.port` | unit | `uv run pytest tests/test_viz_port_config.py::test_viz_port_explicit_overrides_config -xvs` | ❌ Wave 0 |

### Critical Test Fixture Requirements

**OPS-01 — real subprocess fixture (anti-theater):**

The test MUST:
1. Spawn a real process: `proc = subprocess.Popen(["sleep", "60"], start_new_session=True)`
2. Read its starttime: `st = _read_proc_starttime(proc.pid)`
3. Write a running spec at `orchestrator/running/local/<node>.json` with `metadata.pid`, `metadata.pgid`, `metadata.starttime_ticks` — the same shape the daemon writes.
4. Set graph node `status=running`, `metadata.backend=local`.
5. Invoke `cancel` via CliRunner.
6. Assert `os.kill(proc.pid, 0)` raises `ProcessLookupError` (process is dead).
7. Assert graph node `status=cancelled`.

Do NOT hand-mock `os.killpg` — the test must demonstrate a real process was killed.

**OPS-01 — missing `opaque_id` + no `metadata.pid` (hard-fail path):**

Write a running spec with only `{"id": node_id}` (no `opaque_id`, no `metadata`). Assert exit non-zero and "corrupted state" / "cannot cancel" in output.

**OPS-02 — dequeue fixture:**

Write a queue spec at `orchestrator/queue/<node>.json` (plain JSON, same shape as submit.py writes). Write graph with `type=proposed, status=pending`. Invoke `dequeue`. Assert file gone + graph `status=cancelled`.

**OPS-03 — existing-pending submit fixture:**

Pre-write graph.json with `type=proposed, status=pending` for the target node. Write a minimal `config.yaml` with `run.mil_model` set (to satisfy the mil_model check). Invoke `submit --node <node> --desc "..." --files train.py --mil-model clam`. Assert graph `status=running` after.

**OPS-04 — project option fixture:**

Create two directories: `project_a/automil/config.yaml` (with valid content) and set cwd to a completely separate temp directory (not under `project_a`). Invoke `automil --project project_a status` via CliRunner. Assert it resolves without "No automil/config.yaml" error. Must reset `_h._PROJECT_OVERRIDE = None` in teardown.

**OPS-05 — viz port fixture:**

Patch `cmd_start` to capture the resolved `port` value without actually starting the server (monkeypatch or capture via side-effect). Write a config.yaml with `viz:\n  port: 9000`. Assert `cmd_start` is called with `port=9000` when `--port` is not passed.

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_cli_cancel_resubmit.py tests/test_cli_dequeue.py tests/test_cli.py -x -v`
- **Per wave merge:** `uv run pytest tests/ -v` (automil suite only)
- **Phase gate:** Full suite green before `/gsd-verify-work`. Run as `uv run pytest tests/ -v` — NOT `uv run pytest` from workspace root (causes rootdir collision with benchmarks/).

### Wave 0 Gaps

- [ ] `tests/test_cli_cancel_resubmit.py` — add `test_cancel_local_direct_kill` and `test_cancel_missing_pid_metadata` to existing file
- [ ] `tests/test_cli_dequeue.py` — new file, covers OPS-02 (3 tests minimum)
- [ ] `tests/test_cli.py` — add `test_submit_existing_pending_marks_running`
- [ ] `tests/test_cli_project_option.py` — new file, covers OPS-04 (3 tests minimum)
- [ ] `tests/test_viz_port_config.py` — new file, covers OPS-05 (3 tests minimum)

---

## Security Domain

> `security_enforcement` enabled (key absent from config = enabled). ASVS level 1.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (OPS-04 PATH) | `click.Path(exists=True)` + resolved Path; hard-fail if no `automil/config.yaml` found |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| OPS-04 path traversal via `--project` | Tampering | Resolve to absolute path; check for `config.yaml` before using; `click.Path(exists=True)` rejects non-existent paths |
| OPS-01 PID reuse → wrong process signalled | Tampering | Starttime cross-check via `_is_pid_alive_with_starttime`; already the established pattern |
| OPS-02 TOCTOU queue-file removal | Tampering | State guard inside `locked_update` minimizes window; document residual risk |

---

## Package Legitimacy Audit

> No new external packages introduced in this phase. All five fixes use Python stdlib (`os`, `signal`, `json`, `pathlib`) and existing project internals. No `npm install` / `pip install` required.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_is_pid_alive_with_starttime` / `_read_proc_starttime` are safely importable from `automil.orchestrator` in CLI code without circular imports | Verified Code Anchors, OPS-01 | If circular import exists, helpers must be extracted to a shared `automil.backends._proc_utils` module |
| A2 | `starttime_ticks` being optional in the running spec is the only non-trivial edge case for OPS-01 | Pitfalls | If other fields (pgid) can also be absent, the fallback chain needs extension |

If table is otherwise empty: All other claims in this research were verified directly against source code.

---

## Open Questions (RESOLVED)

1. **RESOLVED — OPS-01: SIGTERM→grace→SIGKILL vs single SIGTERM**
   - What we know: CONTEXT.md marks this as Claude's discretion; the daemon uses SIGTERM + 5s + SIGKILL; the existing CLI cancel uses single SIGTERM then polls.
   - What's unclear: The CLI direct-kill path has no daemon tick to fire the SIGKILL escalation. A stuck process group (e.g. ignoring SIGTERM) will not be escalated unless the CLI implements it.
   - **RESOLVED:** Implement SIGTERM + 5s grace + SIGKILL in the CLI direct-kill path, mirroring the daemon. Implemented in plan 13-02/T1 (Part B).

2. **RESOLVED — OPS-02: does `graph.cancel()` require the node to already exist?**
   - What we know: `graph.cancel()` at graph.py:381 does `node = self.nodes[node_id]` — it will raise `KeyError` if node absent.
   - What's unclear: Whether a queued node without a graph entry is a real scenario (submit.py always creates graph entry).
   - **RESOLVED:** Yes — `dequeue` uses `_get_node_or_die` (same as cancel.py) before entering `locked_update`; inside `locked_update`, call `graph.cancel()` safely. Implemented in plan 13-03/T1.

---

## Sources

### Primary (HIGH confidence — verified against live source)
- `src/automil/cli/cancel.py` — full cancel flow; OPS-01 hard-fail site at L100-105
- `src/automil/backends/_orchestrator_daemon.py:145,158,1028-1041,395,1656` — helpers, spec write, queue_dir, _kill_experiment
- `src/automil/cli/submit.py:491-519` — locked_update block; OPS-03 gap at L498
- `src/automil/cli/__init__.py:11-25` — main group + touch_last_action
- `src/automil/cli/_helpers.py:18-31` — _find_automil_dir cwd walk
- `src/automil/cli/viz.py:16-29` — --port option
- `src/automil/viz/server.py:44,265-295` — DEFAULT_PORT, cmd_start signature, host fallback
- `src/automil/graph.py:280-289,381-384` — mark_running, cancel
- `src/automil/orchestrator.py:38-40` — confirms helper re-exports
- `tests/test_cli_cancel_resubmit.py` — existing cancel test patterns and fixture shapes
- `tests/test_orchestrator_pid_starttime.py` — confirms import path for helpers

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — pure stdlib + existing project internals
- Architecture: HIGH — all five fix sites verified line-by-line
- Pitfalls: HIGH — OPS-01 poll-loop trap is a real failure mode confirmed by code inspection; others observed from existing patterns

**Research date:** 2026-06-12
**Valid until:** Stable — no external dependencies; valid until source is modified
