---
phase: "09"
plan: "02"
subsystem: cells
tags: [cell-identity, mil-model, normalization, backward-compat, rec-04]
dependency_graph:
  requires: [09-01]
  provides: [normalize_mil_model, Cell.mil_model, make_cell_id(mil_model), read_cell-shim, get_or_create_cell(mil_model)]
  affects: [09-05, 10-xx]
tech_stack:
  added: []
  patterns: [dataclass-field-rename, backward-compat-shim, underscore-normalize]
key_files:
  created: []
  modified:
    - src/automil/cells/state.py
    - src/automil/cells/registry.py
    - src/automil/cli/cell.py
    - src/automil/cli/submit.py
    - tests/cells/conftest.py
    - tests/cells/test_cell_registry.py
    - tests/gate/test_evaluate.py
    - tests/test_cli_cell.py
    - tests/test_submit_cell_refusal.py
decisions:
  - "normalize_mil_model replaces underscores with spaces before whitespace-collapsing so CLAM_SB==clam_sb==' clam sb ' (D-14 extension beyond spec: test drove this)"
  - "submit.py call-site updated to mil_model= immediately (Rule 1 fix) rather than waiting for Plan 05, to prevent pre-existing test regressions"
metrics:
  duration: "~16 minutes"
  completed: "2026-06-11T08:44:09Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 9
requirements:
  - REC-04
---

# Phase 9 Plan 02: Cell Identity mil_model Rename Summary

Budget-cell identity layer renamed from `parent_id` to `mil_model` across `state.py`, `registry.py`, and all call-sites. `normalize_mil_model` and `read_cell` backward-compat shim shipped together with the rename per RESEARCH.md Pitfall 4.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Rename Cell.parent_id→mil_model, add normalize_mil_model, read_cell shim | f79e961 | src/automil/cells/state.py |
| 2 | Update registry.py and all call-sites to mil_model | 5ba0e2d | registry.py, cli/cell.py, cli/submit.py, 5 test files |

## What Was Built

`Cell.parent_id` field renamed to `Cell.mil_model` throughout the cell-identity layer (D-13, REC-04). Graph parent lineage stays in `graph.json`; cell budget identity is now keyed on `(dataset, encoder, mil_model)`.

**`normalize_mil_model(raw: str) -> str`** (D-14): strips, lowercases, replaces underscores with spaces, collapses internal whitespace. Ensures `CLAM_SB`, `clam_sb`, and `' clam sb '` all map to the same budget cell (`'clam sb'`).

**`read_cell` compat shim** (D-15 / Pitfall 4): `if "parent_id" in data and "mil_model" not in data: data["mil_model"] = data.pop("parent_id")`. Ships with the rename so Plan 05's `automil cells migrate` can read old cells.

**`get_or_create_cell`** signature updated: `parent_id: str` → `mil_model: str`. Cell construction and log messages updated.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] normalize_mil_model must also replace underscores with spaces**
- **Found during:** Task 1 verification
- **Issue:** The PATTERNS.md spec (`" ".join(raw.strip().lower().split())`) only collapses whitespace; `clam_sb` and `clam sb` remained distinct. The test `test_clam_sb_variants_collapse` requires all three variants (`CLAM_SB`, `clam_sb`, `' clam sb '`) to normalize to the same value.
- **Fix:** Added `.replace("_", " ")` before whitespace collapsing. Matches D-14's intent ("CLAM_SB and clam_sb collapse to one cell") and the test specification.
- **Files modified:** `src/automil/cells/state.py`
- **Commit:** f79e961

**2. [Rule 1 - Bug] submit.py call-site fixed immediately to prevent pre-existing test regressions**
- **Found during:** Task 2 verification
- **Issue:** The plan notes "submit.py tests will be RED until Plan 05" and only expects `test_submit_cell_identity.py` to fail. However, `test_submit_cell_refusal.py`, `test_submit_writes_metadata_backend.py`, `test_integration.py`, and `tests/gate/test_evaluate.py` were also broken because they invoke submit via CLI and rely on `get_or_create_cell` succeeding. These are pre-existing passing tests, not Plan 01 stubs.
- **Fix:** Updated `submit.py` call-site from `parent_id=_parent_for_cell` to `mil_model=_parent_for_cell`, and updated error message text. Also fixed `cli/cell.py` header/display to use `mil_model`.
- **Files modified:** `src/automil/cli/submit.py`, `src/automil/cli/cell.py`
- **Commit:** 5ba0e2d

**3. [Rule 1 - Bug] Test helper files using Cell(parent_id=...) updated**
- **Found during:** Task 2 verification
- **Issue:** `tests/test_cli_cell.py`, `tests/test_submit_cell_refusal.py`, `tests/cells/conftest.py`, `tests/cells/test_cell_registry.py`, and `tests/gate/test_evaluate.py` all had local `_make_cell()` helpers or direct `Cell(parent_id=...)` constructors.
- **Fix:** Renamed all `parent_id=` keyword args to `mil_model=` in test helpers and Cell constructors. JSON key assertion in `test_submit_cell_refusal.py` updated from `data["parent_id"]` to `data["mil_model"]`.
- **Files modified:** 5 test files
- **Commit:** 5ba0e2d

## Verification Results

**Plan-specified targets:**
```
tests/test_mil_model_normalization.py::test_clam_sb_variants_collapse PASSED
tests/test_mil_model_normalization.py::test_normalization_strips_leading_trailing PASSED
tests/test_mil_model_normalization.py::test_normalization_collapses_internal_whitespace PASSED
tests/cells/test_migrate.py::test_legacy_cell_loads PASSED
tests/cells/test_cell_registry.py — all 12 tests PASSED
```

**Pre-existing suite (145 tests, ignoring known Plan 01 RED stubs):** 145 passed, 3 RED stubs remain (migrate_cells — Plan 05's job).

**Known remaining RED stubs (Plan 01 / future plans):**
- `tests/cells/test_migrate.py::test_agent_active_merge_sums_consumed` — Plan 05
- `tests/cells/test_migrate.py::test_wall_clock_merge_keeps_earliest_started_at` — Plan 05
- `tests/cells/test_migrate.py::test_dry_run_does_not_write` — Plan 05
- `tests/test_terminal_writer.py` — Plan 06
- `tests/test_result_schema_validation.py::test_partial_status_validates` — Plan 03
- `tests/test_sigterm_flush.py` — Plan 04
- `tests/test_collect_or_synthesize.py` — Plan 04
- `tests/test_crashed_canonicalization.py` — Plan 03
- `tests/test_handle_timeout.py` — Plan 04
- `tests/test_reconcile_from_archive.py` — Plan 03
- `tests/acceptance/test_phase8_acceptance.py` (2 failures) — pre-existing before this milestone (REQUIREMENTS.md replaced with v1.1 content)

## Known Stubs

None — no stubs introduced by this plan. The `mil_model` field in submit.py still uses `_parent_for_cell` as the value (old `parent or "root"` logic) — this is intentional scaffolding that Plan 05 will replace with the `--mil-model` flag + resolution chain.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries introduced. `normalize_mil_model` output is used only as input to `sha256(...)[:16]` hex digest (T-09-01 mitigated: model name never used as filesystem path). `read_cell` shim operates on in-memory dict before constructing a Cell (T-09-02 mitigated: no code evaluation).

## Self-Check: PASSED

- SUMMARY.md: FOUND at `.planning/phases/09-state-recovery-integrity/09-02-SUMMARY.md`
- Commit f79e961 (Task 1): FOUND in git log
- Commit 5ba0e2d (Task 2): FOUND in git log
- 2 pre-existing acceptance failures confirmed pre-existing before this plan (stash verification)
- No new regressions: 961 passing (excl. RED stubs), 2 pre-existing failures
