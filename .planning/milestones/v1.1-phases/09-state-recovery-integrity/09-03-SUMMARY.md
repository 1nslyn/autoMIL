---
phase: "09"
plan: "03"
subsystem: schemas
tags: [rec-03, schema, status-vocabulary, d-05, d-06, d-07]
dependency_graph:
  requires: ["09-01"]
  provides: ["result.schema.json accepts partial + termination_reason", "_crashed_payload emits canonical crash"]
  affects: ["09-04", "09-05", "09-06"]
tech_stack:
  added: []
  patterns: ["tight status enum", "additive schema evolution", "status canonicalization constant"]
key_files:
  created: []
  modified:
    - src/automil/schemas/result.schema.json
    - src/automil/cells/reconcile.py
    - tests/cells/test_aggregate_folds.py
    - tests/cells/test_reconcile_full.py
    - tests/test_tick_cells.py
decisions:
  - "D-07 no version bump: additive-only change (adds one enum value + one optional property); additionalProperties:true already set; DBT-01/Phase 14 owns breaking schema changes"
  - "D-06 canonicalization constant _STATUS_CANON added at module level for machine-greppable documentation of the drift fix"
  - "Rule 1 auto-fix: updated 4 test assertions in test_aggregate_folds.py, test_reconcile_full.py, test_tick_cells.py that were asserting the drift value 'crashed' instead of canonical 'crash'"
metrics:
  duration_minutes: 5
  completed_date: "2026-06-11"
  tasks_completed: 2
  files_changed: 5
---

# Phase 09 Plan 03: Schema Vocabulary Fix Summary

**One-liner:** Added `partial` to the result.schema.json status enum and optional `termination_reason` field (D-05/D-07), and canonicalized `_crashed_payload` from `"crashed"` to `"crash"` (D-06/REC-03).

## What Was Built

### Task 1 — result.schema.json: additive schema update (commit 3f67d9e)

Two additive changes to `src/automil/schemas/result.schema.json`:

1. **Status enum extended:** `["completed", "crash", "budget_killed", "cancelled"]` → `["completed", "crash", "budget_killed", "cancelled", "partial"]`
2. **Optional `termination_reason` property added:** Free-form string (not an enum) documenting why a run terminated. Known values: `timeout`, `oom`, `sigterm`, `sigkill`, `unknown`. Consumers may extend.

No schema version bump was made per D-07. Reasoning documented in the `termination_reason` description field: the change is additive-only, `additionalProperties: true` is already set, and all existing payloads (`completed`/`crash`/`budget_killed`/`cancelled`) continue to validate. Breaking-change schema versioning is tracked in DBT-01 / Phase 14.

### Task 2 — reconcile.py: D-06 crashed→crash canonicalization (commit 12e691b)

Two changes to `src/automil/cells/reconcile.py`:

1. **Module-level `_STATUS_CANON` constant** added after imports — documents the canonicalization rule machine-greppably for future readers.
2. **`_crashed_payload` now emits `"status": "crash"`** — was `"crashed"` (not in the tight enum). Added inline comment `# D-06: canonical value (was "crashed")`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated 4 test assertions from drift value "crashed" → canonical "crash"**

- **Found during:** Task 2 verification (full suite run)
- **Issue:** Three test files contained assertions expecting the old drift value `"crashed"` that was never in the schema enum. These tests were written before D-06 was defined. After `_crashed_payload` was fixed, these tests failed:
  - `tests/cells/test_aggregate_folds.py::test_zero_folds_returns_crashed_status` — line 92
  - `tests/cells/test_reconcile_full.py::test_reconcile_budget_kill_zero_folds_writes_crashed_result_json` — lines 123 and 131
  - `tests/test_tick_cells.py::test_handle_completion_with_cap_cancel_zero_folds_marks_crash` — line 455 (contradicted its own line 462 which already used `"crash"`)
- **Fix:** Updated all 4 assertions to expect `"crash"`. Updated docstrings and comments to reference D-06.
- **Files modified:** `tests/cells/test_aggregate_folds.py`, `tests/cells/test_reconcile_full.py`, `tests/test_tick_cells.py`
- **Commit:** 12e691b (included in Task 2 commit)

## Test Results

### Primary targets (all GREEN after this plan)

| Test | Before | After |
|------|--------|-------|
| `test_result_schema_validation.py::test_partial_status_validates` | FAIL (RED) | PASS |
| `test_result_schema_validation.py::test_termination_reason_is_optional` | PASS (already) | PASS |
| `test_result_schema_validation.py::test_crashed_drift_value_fails_validation` | PASS (already) | PASS |
| `test_crashed_canonicalization.py::test_crashed_payload_returns_crash_not_crashed` | FAIL (RED) | PASS |
| `test_crashed_canonicalization.py::test_crashed_payload_has_no_crashed_key` | FAIL (RED) | PASS |

### Auto-fixed collateral tests (all GREEN after fix)

| Test | Action |
|------|--------|
| `cells/test_aggregate_folds.py::test_zero_folds_returns_crashed_status` | Updated drift assertion |
| `cells/test_reconcile_full.py::test_reconcile_budget_kill_zero_folds_writes_crashed_result_json` | Updated 2 drift assertions |
| `test_tick_cells.py::test_handle_completion_with_cap_cancel_zero_folds_marks_crash` | Updated contradictory assertion |

### Pre-existing failures (unrelated to this plan — not touched)

The following failures existed before this plan and are owned by future plans:

| Test | Owner Plan |
|------|-----------|
| `tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` | Pre-existing (unrelated) |
| `tests/cells/test_migrate.py` (3 tests) | Plan 05 (REC-04) |
| `tests/test_collect_or_synthesize.py` (3 tests) | Plan 04 (REC-01) |
| `tests/test_handle_timeout.py` (2 tests) | Plan 04 (REC-01) |
| `tests/test_reconcile_from_archive.py` (2 tests) | Plan 05 (REC-02) |
| `tests/test_sigterm_flush.py` (1 test) | Plan 04 (REC-01) |
| `tests/test_submit_cell_identity.py::TestMilModelCellIdentity` (4 tests) | Plan 02 (REC-04) |
| `tests/test_terminal_writer.py` (3 tests) | Plan 06 (REC-02) |
| `tests/test_terminal_writer_consistency.py` (1 test) | Plan 06 (REC-02) |

No new failures were introduced. Plan-scope failures decreased from 3 to 0.

## Known Stubs

None — no stubs or placeholder values in the two files modified by this plan.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries were introduced beyond what the plan's threat model covers. The `termination_reason` field is a log annotation (T-09-03: accept, not a control-flow value).

## Self-Check

Files modified:
- `src/automil/schemas/result.schema.json` — ✅ contains `"partial"` in enum and `termination_reason` property
- `src/automil/cells/reconcile.py` — ✅ contains `_STATUS_CANON` and `"crash"` in `_crashed_payload`

Commits:
- `3f67d9e` — Task 1 schema changes
- `12e691b` — Task 2 reconcile.py fix + test corrections
