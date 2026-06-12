---
phase: "09"
plan: "01"
subsystem: tests
tags: [tdd, wave-0, red-stubs, rec-01, rec-02, rec-03, rec-04]
dependency_graph:
  requires: []
  provides:
    - "Wave-0 RED test scaffolds for all four REC requirements"
    - "Nyquist compliance gate for Plans 02–06"
  affects:
    - tests/test_sigterm_flush.py
    - tests/test_collect_or_synthesize.py
    - tests/test_handle_timeout.py
    - tests/test_partial_fold_recovery.py
    - tests/test_crashed_canonicalization.py
    - tests/test_terminal_writer.py
    - tests/test_terminal_writer_consistency.py
    - tests/test_reconcile_from_archive.py
    - tests/test_mil_model_normalization.py
    - tests/cells/test_migrate.py
    - tests/test_result_schema_validation.py
    - tests/test_submit_cell_identity.py
tech_stack:
  added: []
  patterns:
    - "pytest.fail() for RED stubs (not pytest.skip())"
    - "_fresh_handler() capture pattern for SIGTERM handler testing without killing test process"
    - "MagicMock(spec=ExperimentOrchestrator) for daemon unit tests"
    - "ExperimentGraph.nodes dict mutation for direct node insertion in graph tests"
key_files:
  created:
    - tests/test_sigterm_flush.py
    - tests/test_collect_or_synthesize.py
    - tests/test_handle_timeout.py
    - tests/test_partial_fold_recovery.py
    - tests/test_crashed_canonicalization.py
    - tests/test_terminal_writer.py
    - tests/test_terminal_writer_consistency.py
    - tests/test_reconcile_from_archive.py
    - tests/test_mil_model_normalization.py
    - tests/cells/test_migrate.py
  modified:
    - tests/test_result_schema_validation.py
    - tests/test_submit_cell_identity.py
decisions:
  - "Used pytest.fail() instead of pytest.raises(ImportError) for terminal_writer stubs — pytest.raises(ImportError) passes when ImportError is raised, making tests GREEN instead of RED"
  - "test_sigterm_flush.py uses _fresh_handler() capture pattern (patches automil.runtime_helpers.signal) to invoke the handler directly without killing the pytest process via SIGTERM"
  - "test_partial_fold_recovery.py: two quarantine tests are GREEN because existing graph.recompute_best() already only considers status='keep' nodes — they serve as regression guards"
  - "tests/test_crashed_canonicalization.py chosen as filename (not test_aggregate_folds.py) to avoid collision with tests/cells/test_aggregate_folds.py"
  - "ExperimentOrchestrator (not OrchestratorDaemon) is the actual daemon class name in _orchestrator_daemon.py"
metrics:
  duration: "~25 minutes"
  completed: "2026-06-11"
  tasks_completed: 2
  files_created: 10
  files_modified: 2
---

# Phase 09 Plan 01: Wave-0 Test Scaffolds Summary

**One-liner:** Wave-0 RED test stubs for all 4 REC requirements (D-01..D-15) — 10 new files + 2 extensions, 27 new RED functions, existing 970 tests remain GREEN.

## What Was Built

This plan creates the Nyquist-compliance gate for Phase 9: all implementation tasks in Plans 02–06 must land code that turns these RED tests GREEN. No implementation code was added.

### Task 1: REC-01 and REC-03 test stubs (commit b090b5a)

| File | Tests | RED | GREEN | Purpose |
|------|-------|-----|-------|---------|
| `tests/test_sigterm_flush.py` | 2 | 1 | 1 | D-02 AUTOMIL_RESULTS_DIR write target |
| `tests/test_collect_or_synthesize.py` | 3 | 3 | 0 | D-03 fold-first + D-05/D-06 canonicalization |
| `tests/test_handle_timeout.py` | 3 | 2 | 1 | D-04 main-PID-first SIGTERM + configurable grace |
| `tests/test_partial_fold_recovery.py` | 3 | 0 | 3 | D-01 quarantine (2 GREEN regression guards + 1 baseline) |
| `tests/test_crashed_canonicalization.py` | 2 | 2 | 0 | D-06 `crashed`→`crash` in `_crashed_payload` |

### Task 2: REC-02 and REC-04 test stubs (commit 30ac28a)

| File | Tests | RED | GREEN | Purpose |
|------|-------|-----|-------|---------|
| `tests/test_terminal_writer.py` | 3 | 3 | 0 | D-09/D-10 four-artifact write + fixed order |
| `tests/test_terminal_writer_consistency.py` | 1 | 1 | 0 | rank/TSV agreement post-write |
| `tests/test_reconcile_from_archive.py` | 3 | 2 | 1 | D-11 --from-archive opt-in refresh |
| `tests/test_mil_model_normalization.py` | 3 | 3 | 0 | D-14 normalize_mil_model |
| `tests/cells/test_migrate.py` | 4 | 4 | 0 | D-15 budget-merge + legacy compat shim |
| `tests/test_result_schema_validation.py` (extend) | +3 | 1 | 2 | D-07 partial status + D-05 termination_reason |
| `tests/test_submit_cell_identity.py` (extend) | +4 | 4 | 0 | D-12/D-13/D-14 --mil-model flag |

## Verification Results

```
Full suite: 977 passed, 28 failed, 53 skipped
  - 27 new RED stubs (expected — implementation pending in Plans 02–06)
  - 1 pre-existing acceptance failure (test_d208_clause_11_state_roadmap_complete)
  - 970 pre-existing tests unchanged (baseline preserved)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Daemon class name is ExperimentOrchestrator, not OrchestratorDaemon**
- **Found during:** Task 1 — test_collect_or_synthesize.py, test_handle_timeout.py
- **Issue:** The plan's action spec referenced `OrchestratorDaemon` but the actual class in `_orchestrator_daemon.py` is `ExperimentOrchestrator`
- **Fix:** Used `ExperimentOrchestrator` throughout all daemon stub tests
- **Files modified:** test_collect_or_synthesize.py, test_handle_timeout.py

**2. [Rule 1 - Bug] graph.graph["nodes"] attribute does not exist**
- **Found during:** Task 1 — test_partial_fold_recovery.py
- **Issue:** ExperimentGraph uses `self._data` internally; nodes are accessible via `graph.nodes` property
- **Fix:** Used `graph.nodes["node_id"] = ...` for direct node insertion
- **Files modified:** test_partial_fold_recovery.py

**3. [Rule 1 - Bug] best_node is a method, not a property**
- **Found during:** Task 1 — test_partial_fold_recovery.py
- **Issue:** `graph.best_node` without `()` returns the function object (not a dict)
- **Fix:** Used `graph.best_node()` (method call)
- **Files modified:** test_partial_fold_recovery.py

**4. [Rule 1 - Bug] pytest.raises(ImportError) makes tests GREEN, not RED**
- **Found during:** Task 2 — test_terminal_writer.py
- **Issue:** The plan's spec said "use pytest.raises(ImportError) — this makes the test RED." This is incorrect: pytest.raises() passes when the expected exception is raised, so tests would be GREEN until the module ships (at which point they'd fail)
- **Fix:** Used `try: import... except ImportError: pytest.fail("RED: ...")` pattern so tests FAIL when the module is absent
- **Files modified:** test_terminal_writer.py, test_terminal_writer_consistency.py

**5. [Rule 1 - Bug] SIGTERM to test process kills pytest**
- **Found during:** Task 1 — test_sigterm_flush.py
- **Issue:** Sending `os.kill(os.getpid(), signal.SIGTERM)` in the test body triggers the installed handler which calls `sys.exit(0)` — killing the entire pytest process before assertions can run
- **Fix:** Created `_fresh_handler()` helper that captures the handler function via patching `automil.runtime_helpers.signal`, then invokes it directly with `pytest.raises(SystemExit)` to suppress the exit
- **Files modified:** test_sigterm_flush.py

**6. [Rule 1 - Bug] _collect_or_synthesize_result takes (returncode, wt_path), not log_text**
- **Found during:** Task 1 — test_collect_or_synthesize.py initial draft
- **Issue:** The actual method signature is `(self, node_id, archive, returncode, wt_path)` and reads `run.log` from archive dir; the plan spec implied a `log_text` parameter
- **Fix:** Used actual signature; wrote `run.log` files into the archive dir for OOM/timeout tests
- **Files modified:** test_collect_or_synthesize.py

### Design Notes

**D-01 quarantine tests are GREEN (intentional regression guards):**
`test_partial_status_excluded_from_best_node` and `test_partial_status_excluded_from_keep_discard` pass because `ExperimentGraph.recompute_best()` already only selects nodes with `status="keep"` — a partial node (which never gets "keep") is already excluded by the current filter. These tests serve as regression guards: if Plan 06 breaks this property, they turn RED. This is acceptable per the plan's "confirm already-working behavior" framing for these two tests.

**test_termination_reason_is_optional and test_crashed_drift_value_fails_validation are GREEN:**
`termination_reason` passes validation today because `additionalProperties: true` allows any key. `crashed` fails today because it was never in the enum. Both serve as regression guards — they confirm existing schema behavior that must not regress.

## Known Stubs

All test bodies with `pytest.fail("RED: ...")` are intentional stubs — they will be fleshed out when the implementing plan (02–06) ships the production code. The stub body structure is preserved in comments within each test function for the implementer.

## Threat Flags

None — this plan adds only test files; no new network endpoints, auth paths, file access patterns, or schema changes were introduced.

## Self-Check: PASSED

All 11 created/modified files found on disk. Both task commits verified:
- b090b5a: test(09-01): add Wave-0 RED stubs for REC-01 and REC-03
- 30ac28a: test(09-01): add Wave-0 RED stubs for REC-02 and REC-04
- cd932c4: docs(09-01): complete Wave-0 test stubs plan

Full suite result: 977 passed, 28 failed (27 new RED stubs + 1 pre-existing), 53 skipped.
