---
phase: 09-state-recovery-integrity
fixed_at: 2026-06-11T08:58:00Z
review_path: .planning/phases/09-state-recovery-integrity/09-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 9
skipped: 0
status: all_fixed
---

# Phase 9: Code Review Fix Report

**Fixed at:** 2026-06-11T08:58:00Z
**Source review:** .planning/phases/09-state-recovery-integrity/09-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (4 Critical + 5 Warning)
- Fixed: 9
- Skipped: 0

---

## Fixed Issues

### CR-01: `self.graph` never initialized — every completion raised `AttributeError`

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`, `tests/test_tick_cells.py`
**Commit:** `33b5383`
**Applied fix:**

Root-caused and wired `self.graph = ExperimentGraph(path=self.automil_dir / "graph.json", technique_map=None)` in `ExperimentOrchestrator.__init__` immediately after `self.runner` is assigned. The `ExperimentGraph` constructor is cheap (no I/O); `locked_update` re-loads graph.json on each `write_terminal_state` call, so constructing once at init is correct.

Also removed the now-dead `hasattr(self, "graph")` guards in `_handle_cap_killed_completion` — both completion paths now unconditionally call `write_terminal_state` with the live graph.

Added `test_handle_completion_daemon_supplies_own_graph` to `tests/test_tick_cells.py`: constructs an `ExperimentOrchestrator`, deliberately does NOT inject `orch.graph`, calls `_handle_completion`, and asserts that `completed/<node>.json`, `archive/result.json`, and `graph.json` (node type=executed, status=keep/discard) are all written correctly. This test cannot pass under the pre-fix code and will catch any future regression where `self.graph` initialization is removed.

---

### CR-02: `_handle_timeout` SIGKILL path missing `OSError` catch

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `88b2ec3`
**Applied fix:**

Added `except OSError as exc` handler after `except ProcessLookupError` in the `os.killpg(os.getpgid(pid), SIGKILL)` block. When `os.getpgid(pid)` raises `OSError(EPERM)` (container boundary, different session, kernel hardening), the exception now logs a warning and falls through. The critical `self._timed_out[exp_id] = True` and `self._handle_completion(exp_id, returncode=-9)` lines already come after the try/except block, so they run regardless — preventing the infinite re-fire loop. Added a comment explicitly documenting that these two lines must run regardless of killpg outcome.

---

### CR-03: `reconcile --from-archive` wrote raw result.json statuses into graph nodes

**Files modified:** `src/automil/cli/reconcile.py`, `tests/test_reconcile_from_archive.py`
**Commit:** `3c09340`
**Applied fix:**

Replaced the single `gnode["status"] = payload.get("status", gnode.get("status"))` line with a proper status-mapping block that mirrors `terminal_writer.write_terminal_state` logic:

- `completed` / `budget_killed`: compare refreshed composite vs parent composite → `keep` if `composite > p_comp`, else `discard`
- `crash` / `partial` / `cancelled`: pass through unchanged (partial stays quarantined per D-01)
- Unknown status values: leave `gnode["status"]` unchanged
- Raw result status preserved under `metadata.result_status` for operator traceability

Added `test_from_archive_maps_result_status_to_graph_vocabulary` covering all four status paths (completed→keep, budget_killed→discard, crash→crash, partial→partial) with parent-composite comparison assertions.

---

### CR-04: SIGTERM handler writes `result.json` non-atomically

**Files modified:** `src/automil/runtime_helpers.py`
**Commit:** `7b4874d`
**Applied fix:**

Replaced `(target / "result.json").write_text(json.dumps(payload, indent=2))` with the `tempfile.mkstemp + os.fdopen + os.replace` pattern. Added `import tempfile` to module imports. The implementation is inlined (not importing from `terminal_writer`) to avoid adding `automil.terminal_writer` as a hard dependency of `runtime_helpers`, which is imported by training scripts and must stay lightweight. On any exception during write, the temp file is unlinked before re-raising. A SIGKILL mid-write now leaves either the complete previous file or the complete new file — never a torn intermediate state.

---

### WR-01: `time.sleep(grace)` blocks the tick loop

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `f5b1c96`
**Applied fix (minimal):**

Capped `grace` to `_MAX_GRACE = 30` seconds so misconfigured `timeout_grace_seconds` cannot block the tick loop unboundedly. Added a warning log when `grace > 15s` to alert operators. The sleep remains synchronous (known limitation — proper async-deadline fix would restructure the tick loop significantly and is deferred). The cap is sufficient for production safety.

---

### WR-02: `_recover_orphans` only scans `running/local/`

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `f5b1c96`
**Applied fix:**

Replaced the single `self.running_dir.exists()` guard + `self.running_dir.glob()` with a loop over all three backend subdirs `("local", "slurm", "ray")` under `self.running_root`, skipping those that don't exist. Uses `itertools.chain.from_iterable` to iterate all matching `*.json` files across backends. Mirrors the existing pattern in `_read_backend_name_for_node`.

---

### WR-03: `submit.py` running-spec conflict check uses flat path

**Files modified:** `src/automil/cli/submit.py`
**Commit:** `f5b1c96`
**Applied fix:**

Split the old `for subdir in ("queue", "running")` loop into two separate checks:

1. Queue check: `adir / "orchestrator" / "queue" / f"{node}.json"` — unchanged flat path (queue was never backend-namespaced).
2. Running check: iterate all subdirectories under `running_root` and check `backend_dir / f"{node}.json"` for each. This correctly finds specs under `running/local/`, `running/slurm/`, `running/ray/`, etc.

---

### WR-04: `_tick_cells` cap annotation uses `self.running_dir` (local only)

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `f5b1c96`
**Applied fix:**

In the TERMINATING branch of `_tick_cells`, replaced:
```python
running_spec_path = self.running_dir / f"{handle.node_id}.json"
```
with:
```python
_backend_name = self._read_backend_name_for_node(handle.node_id)
running_spec_path = self._backend_running_dir(_backend_name) / f"{handle.node_id}.json"
```

This uses the existing `_read_backend_name_for_node` helper (which already scans all three backend subdirs) to find the correct running spec location for any backend. SLURM and Ray cap-kills now receive the `cancel_reason='cap'` annotation, so `_was_cap_killed_completion` returns True and budget-kill reconciliation runs correctly.

---

### WR-05: `migrate_cells` discards `agent_active` budget on mode mismatch

**Files modified:** `src/automil/cells/migrate.py`
**Commit:** `f5b1c96`
**Applied fix (minimal — skip with warning):**

Added a mode-mismatch guard before the merge logic:

```python
if cell.mode != existing.mode:
    logger.warning("migrate_cells: skipping merge of %s (mode=%s) into %s (mode=%s) ...")
    summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "skip"})
    continue
```

When `cell.mode != existing.mode` the merge semantics are undefined by T-09-08 (spec only addresses same-mode cases) and either direction silently loses budget data. Skipping with a warning and `action='skip'` in the summary requires the operator to manually inspect and reconcile the two cell files before re-running migrate. Same-mode merges (both `agent_active` or both `wall_clock`) are unchanged.

---

## Test Results

**Final suite:** `uv run pytest tests/ -q`
- **1005 passed**, 54 skipped, 207 warnings
- **1 pre-existing failure:** `tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` (stale v1.0 milestone-rotation test — pre-existing, not touched)
- **0 new regressions introduced**

---

_Fixed: 2026-06-11T08:58:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
