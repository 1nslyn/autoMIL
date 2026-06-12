---
phase: "09"
plan: "06"
subsystem: terminal-state-writer
tags:
  - rec-02
  - rec-01
  - terminal-writer
  - partial-quarantine
  - reconcile-from-archive
dependency_graph:
  requires:
    - "09-03"
    - "09-04"
    - "09-05"
  provides:
    - "terminal_writer.write_terminal_state"
    - "graph.best_node partial quarantine"
    - "reconcile --from-archive"
  affects:
    - "src/automil/terminal_writer.py"
    - "src/automil/backends/_orchestrator_daemon.py"
    - "src/automil/cli/reconcile.py"
    - "src/automil/graph.py"
    - "src/automil/cells/reconcile.py"
tech_stack:
  added:
    - "src/automil/terminal_writer.py — standalone four-artifact terminal-state writer (REC-02/D-09/D-10)"
  patterns:
    - "tempfile+os.replace atomic write for all four artifacts"
    - "locked_update context manager for all graph mutations in write path"
    - "callable injection for TSV writer to preserve daemon method binding"
key_files:
  created:
    - "src/automil/terminal_writer.py"
  modified:
    - "src/automil/backends/_orchestrator_daemon.py"
    - "src/automil/cli/reconcile.py"
    - "src/automil/graph.py"
    - "src/automil/cells/reconcile.py"
    - "tests/test_terminal_writer.py"
    - "tests/test_terminal_writer_consistency.py"
    - "tests/cells/test_reconcile_full.py"
    - "tests/cells/test_cap_fires_with_partial_fold_recovery.py"
decisions:
  - "D-09: write_terminal_state writes graph → completed/<node>.json → archive result.json → results.tsv in fixed order"
  - "D-10: terminal_writer is sole archive result.json writer; removed write_text from reconcile_budget_kill"
  - "D-11: reconcile --from-archive NODE_OR_ALL opt-in refresh; skips running nodes (Pitfall 3 guard)"
  - "D-01: graph.py best_node and _reevaluate_descendants skip status=partial nodes"
  - "crash status maps to graph_status=crash (not discard); only completed/budget_killed/cancelled compare composite"
  - "result.metadata propagated to gnode metadata for budget_killed flag preservation"
metrics:
  duration_minutes: 35
  completed_date: "2026-06-11"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
  files_modified: 8
---

# Phase 09 Plan 06: Terminal Writer + Partial Quarantine Summary

**One-liner:** Single terminal-state writer (`write_terminal_state`) consolidates all four artifact writes in fixed order with D-01 partial quarantine in `best_node` and `_reevaluate_descendants`.

## What Was Built

### Task 1: terminal_writer.py + graph.py D-01 quarantine (commit 97637c3)

Created `src/automil/terminal_writer.py` — a standalone module implementing `write_terminal_state()` that owns all four terminal artifacts:

1. **Graph node** (via `locked_update`) — atomically updates the graph with correct status mapping:
   - `partial` → `"partial"` (D-01 quarantine, never keep/discard)
   - `crash` → `"crash"` (failure, not a discard)
   - `completed`/`budget_killed` → `"keep"` or `"discard"` based on composite vs parent
   - Propagates `result.metadata` (e.g. `budget_killed`) to `gnode.metadata`
2. **`completed/<node>.json`** — atomic tempfile+os.replace write
3. **Archive `result.json`** — atomic tempfile+os.replace write (sole writer, D-10)
4. **`results.tsv`** — delegated to callable (daemon's `_append_results_tsv` bound method)

Added D-01 partial quarantine guards to `graph.py`:
- `best_node()`: returns `None` if `meta.best_node_id` points to a `partial` node
- `_reevaluate_descendants()`: skips `status=partial` children (not keep/discard candidates)

Implemented full test bodies for `test_terminal_writer.py` and `test_terminal_writer_consistency.py` (replaced RED stubs).

### Task 2: Daemon refactor + reconcile --from-archive (commit 22ea03c)

**`_orchestrator_daemon.py`:**
- `_handle_completion`: Removed direct `completed/<node>.json` write and `_append_results_tsv` call; delegates to `write_terminal_state` (Pitfall 2/6 guard)
- `_handle_cap_killed_completion`: Removed direct `gnode["status"]`/`gnode["composite"]` dict mutation and `self.graph.save()` calls (Pitfall 1 guard); added `elapsed_s`/`gpu_id`/`spec` parameters; delegates to `write_terminal_state`

**`cells/reconcile.py` — D-10 sole-writer fix:**
- Removed `(node_archive / "result.json").write_text(...)` from `reconcile_budget_kill`
- `mkdir` kept (archive dirs must exist before terminal_writer writes)
- `grep write_text` returns no match in `reconcile_budget_kill` body

**`cli/reconcile.py` — D-11 --from-archive:**
- Added `--from-archive NODE_OR_ALL` Click option
- New branch uses `locked_update` for safe concurrent writes
- Pitfall 3 guard: skip nodes with `status=running`
- Default reconcile path (no `--from-archive`) unchanged

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] ExperimentGraph has `_technique_map` not `technique_map`**
- **Found during:** Task 1 test run
- **Issue:** `AttributeError: 'ExperimentGraph' object has no attribute 'technique_map'` — plan spec referenced `graph.technique_map` but the attribute is `_technique_map`
- **Fix:** `getattr(graph, "_technique_map", None)` in terminal_writer.py
- **Files modified:** `src/automil/terminal_writer.py`
- **Commit:** 97637c3

**2. [Rule 1 - Bug] crash status was mapped to discard (zero composite edge case)**
- **Found during:** Task 2 full suite run
- **Issue:** Zero-fold cap-kill produces `status="crash"`, `composite=0.0`. With `parent_id=None`, `p_comp=0.0`, so `0.0 > 0.0` → `"discard"`. Tests expected `"crash"`.
- **Fix:** Added explicit `crash → graph_status="crash"` mapping in terminal_writer. Only `completed`/`budget_killed`/`cancelled` go through composite comparison.
- **Files modified:** `src/automil/terminal_writer.py`
- **Commit:** 22ea03c

**3. [Rule 1 - Bug] `metadata.budget_killed` not propagated to graph node**
- **Found during:** Task 2 full suite run
- **Issue:** `gnode.get("metadata", {}).get("budget_killed")` was `None` after cap-kill completion
- **Fix:** Added `if result.get("metadata"): gnode.setdefault("metadata", {}).update(result["metadata"])` in terminal_writer
- **Files modified:** `src/automil/terminal_writer.py`
- **Commit:** 22ea03c

**4. [Rule 1 - Bug] Tests asserted old `reconcile_budget_kill` disk-write behavior**
- **Found during:** Task 2 full suite run (5 test failures)
- **Issue:** `test_reconcile_full.py` and `test_cap_fires_with_partial_fold_recovery.py` asserted that `reconcile_budget_kill` writes `result.json` to disk — behavior intentionally removed per D-10
- **Fix:** Updated tests to only check payload return value, not disk file existence (which is now terminal_writer's responsibility)
- **Files modified:** `tests/cells/test_reconcile_full.py`, `tests/cells/test_cap_fires_with_partial_fold_recovery.py`
- **Commit:** 22ea03c

## Test Results

All target tests GREEN:
- `test_terminal_writer.py` — 3/3 passed
- `test_terminal_writer_consistency.py` — 1/1 passed
- `test_reconcile_from_archive.py` — 3/3 passed
- `test_partial_fold_recovery.py` — 3/3 passed

Full suite: **993 passed, 51 skipped** (non-acceptance)
Pre-existing acceptance failure: `test_d208_clause_11_state_roadmap_complete` — workstation-data-gated, unchanged.

## Known Stubs

None — all data paths are wired.

## Threat Flags

No new threat surface introduced. Terminal writer follows existing locked_update + atomic write patterns. Reconcile --from-archive skips running nodes (Pitfall 3 guard). Both boundaries (archive result.json ingestion, --from-archive operator update) were pre-planned in the threat model.

## Self-Check: PASSED

- `src/automil/terminal_writer.py` — FOUND
- `src/automil/graph.py` (partial quarantine) — FOUND (verified by test_partial_fold_recovery GREEN)
- `src/automil/cli/reconcile.py` (--from-archive) — FOUND (verified by test_reconcile_from_archive GREEN)
- `src/automil/cells/reconcile.py` (write_text removed) — CONFIRMED CLEAN
- Commit 97637c3 — FOUND
- Commit 22ea03c — FOUND
