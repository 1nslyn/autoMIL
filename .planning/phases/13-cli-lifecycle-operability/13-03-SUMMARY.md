---
phase: 13-cli-lifecycle-operability
plan: "03"
subsystem: cli
tags: [ops-02, ops-03, dequeue, submit, graph-state-machine, locked_update]
dependency_graph:
  requires: [13-01]
  provides: [dequeue-command, submit-pending-to-running]
  affects: [src/automil/cli/dequeue.py, src/automil/cli/__init__.py, src/automil/cli/submit.py]
tech_stack:
  added: []
  patterns: [locked_update-for-graph-mutations, _get_node_or_die-before-cancel, flat-queue-path]
key_files:
  created:
    - src/automil/cli/dequeue.py
  modified:
    - src/automil/cli/__init__.py
    - src/automil/cli/submit.py
    - tests/test_cli_dequeue.py
    - tests/test_cli.py
decisions:
  - "OPS-02 queue path is flat orchestrator/queue/<node>.json — D-169 backend namespacing applies to running specs only"
  - "dequeue uses locked_update (not raw tempfile write) to serialize graph.cancel() against daemon"
  - "_get_node_or_die called before locked_update to prevent graph.cancel KeyError on missing node"
  - "OPS-03 else branch calls mark_running inside same locked_update block; mark_running is already type/status-guarded so safe for any existing state"
  - "Removed logger.debug call (submit.py has no module-level logger); informational log not worth adding import"
metrics:
  duration_minutes: 8
  completed: "2026-06-12"
  tasks_completed: 2
  files_changed: 5
---

# Phase 13 Plan 03: OPS-02 dequeue + OPS-03 submit pending→running Summary

**One-liner:** New `automil dequeue` command removes queue spec + cancels graph node via `locked_update`; submit's `locked_update` block gains `else` branch calling `mark_running` for existing pending proposals.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create dequeue.py and register it (OPS-02) | 4ef3850 | src/automil/cli/dequeue.py (new), src/automil/cli/__init__.py, tests/test_cli_dequeue.py |
| 2 | Fix submit.py pending→running transition (OPS-03) | 67caca6 | src/automil/cli/submit.py, tests/test_cli.py |

## What Was Built

### OPS-02: `automil dequeue <node>` (src/automil/cli/dequeue.py)

New command module following the cancel.py pattern with these key differences from cancel.py:

- Uses `locked_update` for graph write (cancel.py uses raw tempfile+os.replace — a known gap; dequeue gets the correct pattern)
- No archive move, no backend instantiation, no poll loop — queue dequeue is simpler than cancel
- Queue spec path is flat: `orchestrator/queue/<node>.json` (D-169 backend namespacing is for running specs only)
- `_get_node_or_die` called BEFORE entering `locked_update` to prevent `graph.cancel()` KeyError (graph.py:381 does unguarded `self.nodes[node_id]`)
- State guard: hard-fails for `running` nodes (cross-references `automil cancel`), hard-fails for terminal states
- Idempotent: pending node with no queue spec on disk still gets marked cancelled (clears orphaned proposals)
- Registered in cli/__init__.py alphabetically between cancel and cell

### OPS-03: submit.py pending→running transition (src/automil/cli/submit.py)

Surgical one-`else`-branch fix inside the existing `locked_update` block (~L519):

- The `if not graph.get_node(node):` branch creates new nodes and calls `mark_running`
- Added `else:` branch: when node already exists as `type=proposed, status=pending`, calls `graph.mark_running(node)`
- `mark_running` is already type/status-guarded at graph.py:280 — logs warning and returns False for any other state — so the else branch is unconditionally safe
- Closes the gap where existing pending proposals stayed stuck in pending state after submit, making cancellation and portfolio accounting inconsistent

## Verification

```
uv run pytest tests/test_cli_dequeue.py tests/test_cli.py -v
# 4 OPS-02 tests PASSED, 24 test_cli.py tests PASSED (including OPS-03)

uv run pytest tests/ -q --tb=no
# 1050 passed, 53 skipped, 3 xfailed, 1 failed (pre-existing clause_11 only)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] submit.py has no module-level logger**

- **Found during:** Task 2 (OPS-03 test failure: `NameError: name 'logger' is not defined`)
- **Issue:** Plan specified `logger.debug(...)` but submit.py has no `import logging` / `logger = logging.getLogger(__name__)` at module level
- **Fix:** Removed the `logger.debug` call rather than adding a new import (minimal-impact, debug log is purely informational)
- **Files modified:** src/automil/cli/submit.py
- **Commit:** 67caca6

## Known Stubs

None — both OPS-02 and OPS-03 are fully wired with real behavior.

## Threat Flags

No new security-relevant surface beyond what is in the plan's threat model:
- T-13-03-01: TOCTOU on queue-file removal — state guard runs before file operations; documented
- T-13-03-02: node_id path traversal — mitigated by `_get_node_or_die` graph-key validation

## Self-Check: PASSED

- [x] src/automil/cli/dequeue.py exists: FOUND
- [x] src/automil/cli/__init__.py contains `from automil.cli import dequeue`: FOUND
- [x] submit.py else branch with `graph.mark_running(node)`: FOUND
- [x] Commit 4ef3850 exists: FOUND
- [x] Commit 67caca6 exists: FOUND
- [x] 4 OPS-02 dequeue tests GREEN
- [x] 24 test_cli.py tests GREEN (including OPS-03)
- [x] Full framework suite: 1050 passed, 1 pre-existing clause_11 failure only
