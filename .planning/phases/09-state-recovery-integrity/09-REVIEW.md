---
phase: 09-state-recovery-integrity
reviewed: 2026-06-11T06:20:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - src/automil/terminal_writer.py
  - src/automil/backends/_orchestrator_daemon.py
  - src/automil/cells/state.py
  - src/automil/cells/registry.py
  - src/automil/cells/reconcile.py
  - src/automil/cells/migrate.py
  - src/automil/cli/submit.py
  - src/automil/cli/propose.py
  - src/automil/cli/cells.py
  - src/automil/cli/cell.py
  - src/automil/cli/reconcile.py
  - src/automil/cli/__init__.py
  - src/automil/graph.py
  - src/automil/runtime_helpers.py
  - src/automil/schemas/result.schema.json
findings:
  critical: 4
  warning: 5
  info: 2
  total: 11
status: issues_found
---

# Phase 9: Code Review Report

**Reviewed:** 2026-06-11T06:20:00Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 9 ships a significant refactor: the four-artifact terminal-state write is now centralized in `terminal_writer.write_terminal_state`, the SIGTERM flush handler writes to `AUTOMIL_RESULTS_DIR`, `_handle_timeout` uses a main-PID-first + grace approach, the `cells/` package gains `migrate.py` for budget-cell re-keying, and `reconcile --from-archive` provides a manual graph-refresh path.

The architecture is sound and the design decisions are defensible. However the review surfaces **one ship-stopping regression** (self.graph never initialized — every experiment completion raises AttributeError in production), **one incorrect behavior** in the reconcile CLI (raw result.json statuses written directly into graph nodes), **one unhandled OSError** that can lock up the timeout path forever, and **one signal-handler data-integrity gap**. The remaining findings are warnings and quality items.

---

## Critical Issues

### CR-01: `self.graph` never initialized — `_handle_completion` raises `AttributeError` on every completion

**File:** `src/automil/backends/_orchestrator_daemon.py:1203`

**Issue:** `_handle_completion` calls `write_terminal_state(graph=self.graph, ...)` unconditionally, but `self.graph` is never assigned anywhere in `ExperimentOrchestrator.__init__`, `run()`, or any other method. Every experiment completion in production raises `AttributeError: 'ExperimentOrchestrator' object has no attribute 'graph'`. The exception is swallowed by `tick()`'s outer `except Exception` (line 1739), so the graph node never transitions to `executed`, `completed/<node>.json` is not written, `results.tsv` gets no row, and the worktree is not cleaned up. VRAM is released when the process exits (Popen reaps it), but the running-spec file (`running/local/<node>.json`) is never deleted, so `_recover_orphans()` on next daemon restart will re-process every completed node.

Pre-Phase-9 code guarded all graph accesses with `hasattr(self, 'graph')` (verified in commit `5bd627b`). Phase 9 removed that guard by introducing `write_terminal_state` without ensuring `self.graph` exists.

Tests in `test_tick_cells.py` inject `orch.graph = graph` externally (lines 331, 425), masking the missing initialization.

**Fix:**
```python
# In ExperimentOrchestrator.__init__, after self.runner = Runner(...):
from automil.graph import ExperimentGraph
self.graph = ExperimentGraph(
    path=self.automil_dir / "graph.json",
    technique_map=None,  # loaded fresh inside locked_update per call
)
```

Alternatively, `write_terminal_state` could accept `graph_path` and `technique_map` directly instead of a live `ExperimentGraph` instance, since it only uses `graph.path` and `graph._technique_map` — not the loaded node data.

---

### CR-02: `_handle_timeout` SIGKILL path missing `OSError` catch — hangs experiment permanently on `EPERM`

**File:** `src/automil/backends/_orchestrator_daemon.py:1493-1499`

**Issue:** After `time.sleep(grace)`, the SIGKILL block is:

```python
try:
    os.killpg(os.getpgid(pid), signal.SIGKILL)
except ProcessLookupError:
    pass
```

`os.getpgid(pid)` can raise `OSError(EPERM)` (errno 1) — not `ProcessLookupError` (ESRCH, errno 3) — when the calling process lacks permission to query the PID's process group (different session, container boundary, or kernel hardening). This `OSError` is not caught and propagates through `_handle_completion` up to `tick()`'s outer logger. Because the exception fires **before** `self._timed_out[exp_id] = True` and `self._handle_completion(...)` execute (lines 1498–1499), the experiment stays in `self.running` forever. Every subsequent tick calls `_check_running`, finds `now > exp.timeout_at`, calls `_handle_timeout` again, sleeps `grace` seconds, hits `EPERM` again — an infinite blocking loop that serializes all other scheduling.

Both `_escalate_to_sigkill` (line 1088) and `_kill_experiment` (line 1540) already catch `OSError` correctly. This is an oversight in the newly written `_handle_timeout`.

**Fix:**
```python
if exp.process.poll() is None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        logger.warning(
            "_handle_timeout: killpg(%d, SIGKILL) failed: %s", pid, exc
        )
self._timed_out[exp_id] = True          # must run regardless of killpg outcome
self._handle_completion(exp_id, returncode=-9)
```

---

### CR-03: `reconcile --from-archive` writes raw `result.json` statuses into graph nodes

**File:** `src/automil/cli/reconcile.py:85`

**Issue:**
```python
gnode["status"] = payload.get("status", gnode.get("status"))
```

`payload` is read directly from `archive/<node>/result.json`, whose `status` enum includes `"completed"` and `"budget_killed"` (per `result.schema.json`). Graph nodes use the mutually exclusive set `{keep, discard, crash, partial, running, pending, cancelled}`. Writing `"completed"` or `"budget_killed"` into a graph node corrupts graph semantics:

- `_reevaluate_descendants` at `graph.py:354` only re-evaluates children whose status is in `("keep", "discard")` — nodes set to `"completed"` are silently skipped.
- `recompute_best` at `graph.py:467` only counts `status == "keep"` nodes — a `"completed"` node is never elected best even if it has the highest composite.
- `rank_proposals` and UCB scoring work off parent composites — a parent with `status="completed"` will propagate incorrect potential scores.

The `--from-archive` feature is intended for manual graph repair; corrupting node statuses defeats that purpose entirely.

**Fix:**
```python
raw_status = payload.get("status")
if raw_status is not None:
    # Map result.json statuses to graph node statuses
    _STATUS_MAP = {
        "completed": None,  # compute keep/discard from composite vs parent
        "budget_killed": None,  # same — treat like completed
        "crash": "crash",
        "partial": "partial",
        "cancelled": "cancelled",
    }
    mapped = _STATUS_MAP.get(raw_status, raw_status)
    if mapped is None:
        # Compute keep/discard
        parent_id = gnode.get("parent_id")
        parent = g.get_node(parent_id) if parent_id else None
        p_comp = parent.get("composite", 0.0) if parent else 0.0
        composite = payload.get("composite", 0.0)
        mapped = "keep" if composite > p_comp else "discard"
    gnode["status"] = mapped
```

---

### CR-04: `runtime_helpers.py` SIGTERM handler writes `result.json` non-atomically

**File:** `src/automil/runtime_helpers.py:67`

**Issue:**
```python
(target / "result.json").write_text(json.dumps(payload, indent=2))
```

This runs inside a SIGTERM signal handler. The process can receive SIGKILL from the daemon's `_handle_timeout` grace expiry at any moment during the write, leaving a partial (torn) `result.json`. The daemon's `_collect_or_synthesize_result` tries to read `result.json` first; a partial file will fail JSON parsing and fall through to log-heuristic synthesis, discarding all fold results computed by the handler.

`_atomic_write_json` exists in `terminal_writer.py` for exactly this purpose, but `runtime_helpers` doesn't use it (and importing it would create a circular dependency from `runtime_helpers` → `terminal_writer`). The atomic pattern is: `mkstemp` + `os.fdopen` write + `os.replace`.

Note: `sys.exit(0)` immediately after the write means the process will not stay alive long, but the SIGKILL from the grace timer can arrive between the `write_text` open and close system calls.

**Fix:**
```python
import os
import tempfile

def _atomic_write_json_local(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

# Replace line 67:
_atomic_write_json_local(target / "result.json", payload)
```

---

## Warnings

### WR-01: `_handle_timeout` `time.sleep(grace)` blocks the entire daemon tick loop

**File:** `src/automil/backends/_orchestrator_daemon.py:1492`

**Issue:** `time.sleep(grace)` (default 10 seconds, configurable) executes synchronously inside `_check_running` → `_handle_timeout`. During the sleep, **all** other experiment polling is suspended. With `max_concurrent_per_gpu=8` and multiple GPUs, multiple experiments could time out simultaneously; each timeout serializes at `grace` seconds, stacking to `N × grace` seconds of total blocking. During this window, completions are not reaped, new experiments are not scheduled, and cell budget accrual is paused.

The current design is documented as intentional ("main-PID-first approach"), but there is no cap on the blocking duration in the tick loop, no comment explaining the worst-case blocking behavior, and no operator-visible warning when grace is set high.

**Fix (minimal):** Cap `grace` to a safe maximum and log a warning when it is high:
```python
MAX_GRACE = 30
grace = min(int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10)), MAX_GRACE)
if grace > 15:
    logger.warning("timeout_grace_seconds=%ds will block the tick loop for that duration", grace)
```

**Fix (robust, non-blocking):** Record a SIGTERM-sent timestamp and convert the grace window to an async deadline similar to `_pending_sigkill_at`, checked on subsequent ticks. This matches the pattern already used for cap-driven cancels.

---

### WR-02: `_recover_orphans` only scans `running/local/` — SLURM and Ray orphans are never reaped

**File:** `src/automil/backends/_orchestrator_daemon.py:635-637`

**Issue:**
```python
if not self.running_dir.exists():   # self.running_dir = running/local/
    return
for f in self.running_dir.glob("*.json"):
```

`_recover_orphans` only iterates `running/local/`. SLURM and Ray running specs live in `running/slurm/` and `running/ray/`. After a daemon restart, SLURM and Ray experiments whose processes are still alive (or whose VRAM is held) are not reaped. Their running specs remain indefinitely, preventing future `_handle_completion` detection and blocking the D-168 namespacing guard (which checks for flat files, not backend subdirs, but still signals operator confusion).

`_read_backend_name_for_node` already iterates `("local", "slurm", "ray")` subdirs (lines 1124–1131) — `_recover_orphans` should do the same.

**Fix:**
```python
def _recover_orphans(self):
    for backend_subdir in ("local", "slurm", "ray"):
        subdir = self.running_root / backend_subdir
        if not subdir.exists():
            continue
        for f in subdir.glob("*.json"):
            try:
                spec = json.loads(f.read_text())
                node_id = spec.get("id", f.stem)
                # ... rest of existing orphan handling ...
```

---

### WR-03: `submit.py` running-spec conflict check uses flat path — misses all running experiments post-D-169

**File:** `src/automil/cli/submit.py:99-106`

**Issue:**
```python
for subdir in ("queue", "running"):
    conflict = adir / "orchestrator" / subdir / f"{node}.json"
```

Since D-169 (Phase 6), running specs are namespaced under `running/<backend>/`. The flat path `orchestrator/running/<node>.json` never exists. The guard that prevents resubmitting a currently-running node is **permanently disabled** for all experiments. A resubmit would overwrite `archive/<node>/result.json` and corrupt the completed node's record.

**Fix:**
```python
for subdir in ("queue",):
    conflict = adir / "orchestrator" / subdir / f"{node}.json"
    if conflict.exists():
        raise click.ClickException(...)

# Check all backend subdirs for running specs:
running_root = adir / "orchestrator" / "running"
for backend_dir in running_root.iterdir() if running_root.exists() else []:
    if backend_dir.is_dir():
        conflict = backend_dir / f"{node}.json"
        if conflict.exists():
            raise click.ClickException(
                f"Refusing to submit: {node} is currently running in "
                f"orchestrator/running/{backend_dir.name}/. ..."
            )
```

---

### WR-04: `_tick_cells` cap-cancel annotation uses `self.running_dir` (local only) — SLURM/Ray cap-kills not detected

**File:** `src/automil/backends/_orchestrator_daemon.py:1009`

**Issue:**
```python
running_spec_path = self.running_dir / f"{handle.node_id}.json"
```

`self.running_dir` is `running/local/`. For experiments running on SLURM or Ray backends, the running spec lives in `running/slurm/` or `running/ray/`. The `if running_spec_path.exists()` check at line 1010 silently passes with `False`, the `cancel_reason='cap'` annotation is never written, and `_was_cap_killed_completion` returns `False` for all SLURM/Ray cap-triggered cancels. They are processed as standard completions without budget-kill reconciliation.

**Fix:**
```python
backend_name = self._read_backend_name_for_node(handle.node_id)
running_spec_path = self._backend_running_dir(backend_name) / f"{handle.node_id}.json"
```

---

### WR-05: `migrate_cells` merge discards `agent_active` budget when merging into a `wall_clock` cell

**File:** `src/automil/cells/migrate.py:75-84`

**Issue:** The mode-aware merge branches on `cell.mode` (the old cell being iterated), but the resulting merged object always keeps `existing`'s mode. When `cell.mode == "agent_active"` but `existing.mode == "wall_clock"`:

```python
if cell.mode == "agent_active":
    merged_consumed = cell.consumed_active_seconds + existing.consumed_active_seconds
    merged = dataclasses.replace(existing, consumed_active_seconds=merged_consumed)
    # merged.mode is still "wall_clock" — consumed_active_seconds is silently ignored
```

The `wall_clock` billing path in `consumed_seconds` (state.py:142–144) only uses `now - started_at` and ignores `consumed_active_seconds`. The summed budget is written to disk but never read, so the agent_active budget is **silently discarded** in this scenario. The inverse case (cell is wall_clock, existing is agent_active) is also problematic: the wall_clock branch adjusts `started_at` but the budget effectively continues to bill on existing's agent_active accumulator without incorporating the old cell's elapsed wall-clock time.

This affects operators who change `mode` between submits or run mixed-mode configurations. The spec (T-09-08) says "sum consumed_active_seconds" — it does not address the mode-mismatch case, which should be documented or guarded.

**Fix (minimal):** Log a warning and skip the merge if modes differ:
```python
if cell.mode != existing.mode:
    logger.warning(
        "migrate_cells: skipping merge of %s (mode=%s) into %s (mode=%s) — "
        "mode mismatch; manual review required.",
        cell.cell_id[:8], cell.mode, existing.cell_id[:8], existing.mode,
    )
    summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "skip"})
    continue
```

---

## Info

### IN-01: `_recover_orphans` artifact writes are non-atomic and bypass `terminal_writer`

**File:** `src/automil/backends/_orchestrator_daemon.py:648-653`

**Issue:** `_recover_orphans` writes `archive/result.json` and `completed/<node>.json` directly via `write_text` (non-atomic). This is inconsistent with the D-10 sole-writer invariant (terminal_writer owns these artifacts). In the startup path this is low-risk since the daemon is single-process during recovery, but a crash mid-write during orphan recovery would leave a 0-byte or partial `result.json` that later causes JSON parse errors in `_collect_or_synthesize_result`. Also, `_recover_orphans` does not update the graph node — the node stays `running` in graph.json until the next `automil reconcile`.

**Fix:** Either route through `write_terminal_state` (requires `self.graph` to be initialized first per CR-01), or use `_atomic_write_json` from `terminal_writer` for the two artifact writes.

---

### IN-02: `result.schema.json` `"budget_killed"` enum value is never produced by current code paths

**File:** `src/automil/schemas/result.schema.json:16`

**Issue:** The schema declares `"budget_killed"` as a valid `status` enum value:
```json
"enum": ["completed", "crash", "budget_killed", "cancelled", "partial"]
```
However, `reconcile_budget_kill` (the only budget-kill reconciliation path) does not produce `status="budget_killed"` — it returns `aggregate_folds()` which produces `"completed"`, `"partial"`, or `"crash"`. The `budget_killed` flag is conveyed via `metadata.budget_killed=True`, not via the `status` field. No training script in the codebase produces `status="budget_killed"`. The enum value is dead schema surface that could mislead consumers writing training scripts to use it.

**Fix:** Either remove `"budget_killed"` from the enum and document that budget_killed state is signaled via `metadata.budget_killed`, or add a note to the schema description clarifying that `status="budget_killed"` is reserved for future use.

---

_Reviewed: 2026-06-11T06:20:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
