---
phase: "09"
plan: "05"
subsystem: cells/cli
tags: [REC-04, D-12, D-13, D-14, D-15, mil_model, budget-cell, migration]
dependency_graph:
  requires: ["09-02"]
  provides: ["submit --mil-model", "propose --mil-model", "automil cells migrate", "D-12 resolution chain"]
  affects: ["09-06", "10-*"]
tech_stack:
  added: ["cells/migrate.py", "cli/cells.py"]
  patterns: ["D-12 three-step resolution chain", "mode-aware budget merge", "atomic write-before-delete migration"]
key_files:
  created:
    - src/automil/cells/migrate.py
    - src/automil/cli/cells.py
  modified:
    - src/automil/cli/submit.py
    - src/automil/cli/propose.py
    - src/automil/cli/__init__.py
    - tests/test_submit_cell_identity.py
    - tests/test_submit_cell_refusal.py
    - tests/test_submit_writes_metadata_backend.py
    - tests/test_cli.py
    - tests/test_integration.py
    - tests/cli/test_submit_max_time.py
    - tests/acceptance/test_final_phase8_acceptance.py
    - tests/skills/test_setup_dry_run_gate.py
decisions:
  - "D-12 resolution chain on submit: flag → run.mil_model config → propose node metadata → ClickException"
  - "normalize_mil_model applied before get_or_create_cell; clam_sb normalizes to 'clam sb' (underscore→space, D-14 Plan 02 extension)"
  - "Test assertions for 'clam_sb' updated to 'clam sb' to match Plan 02's underscore-as-word-separator normalization"
  - "D-15 migration: mode-aware merge; agent_active sums consumed_active_seconds; wall_clock keeps min(started_at)"
  - "submit --mil-model now required: 11 test files updated to pass --mil-model or set run.mil_model in config"
metrics:
  duration: "~15 minutes"
  completed_date: "2026-06-11"
  tasks_completed: 2
  files_changed: 13
---

# Phase 09 Plan 05: REC-04 CLI + Migration Summary

Wire the budget-cell identity changes into the CLI (D-12/D-13/D-14) and ship the `automil cells migrate` migration helper (D-15), making `--mil-model` required on submit with a three-step inference chain.

## Tasks Completed

### Task 1: Add --mil-model to submit.py and propose.py

**Commit:** `7a7431c`

**submit.py changes:**
- Added `@click.option("--mil-model", default=None, ...)` and `mil_model: str | None` parameter
- Replaced `_parent_for_cell` (old parent-id keying) with D-12 three-step resolution chain:
  1. `--mil-model` flag
  2. `run.mil_model` from `config.yaml`
  3. Propose-time node metadata (`graph_json["nodes"][node]["metadata"]["mil_model"]`)
  4. `ClickException` if none found
- Applies `normalize_mil_model()` before calling `get_or_create_cell`

**propose.py changes:**
- Added `@click.option("--mil-model", default=None, ...)` and `mil_model: str | None` parameter
- After `graph.add_proposed()`, stores `normalize_mil_model(mil_model)` in `gnode["metadata"]["mil_model"]`
- This allows submit to inherit the model via the metadata fallback step

**Test fixes (Rule 2 — D-12 now enforces required):**
- `_submit_and_read_cell` helper: added `--mil-model test_model`
- `_submit_node` helper in `test_submit_cell_refusal.py`: added `mil_model="root"` default
- `test_missing_identity_falls_back_to_unknown_with_warning`: added `--mil-model test_model`
- Normalization assertion: `"clam_sb"` → `"clam sb"` (Plan 02 underscore→space rule, D-14)

All 4 `TestMilModelCellIdentity` tests GREEN. All 7 existing `TestSubmitCellIdentity` + `TestSubmitCellLayer` tests still GREEN.

### Task 2: Create cells/migrate.py + cli/cells.py + register in cli/__init__.py

**Commit:** `020899e`

**cells/migrate.py (new):**
- `migrate_cells(cells_dir, mil_model, dry_run=False) -> list[dict]`
- Iterates `cells_dir/*.json` with malformed-file guard (mirrors `registry.py:list_cells`)
- Computes `new_id = make_cell_id(cell.dataset, cell.encoder, mil_model_norm)` for each cell
- **Merge case** (new_path exists): mode-aware merge without double-counting (T-09-08):
  - `agent_active`: sum `consumed_active_seconds`
  - `wall_clock`: keep earliest `started_at`
  - Write new merged cell first (atomic), then unlink old (T-09-09)
- **Rename case** (new_path absent): `dataclasses.replace(cell, cell_id=new_id, mil_model=mil_model_norm)`, write-before-delete
- **Skip case** (new_path == path): already correctly keyed
- `dry_run=True`: returns summaries, no writes/deletes

**cli/cells.py (new):**
- `@main.group() def cells()` — "Budget-cell management commands."
- `@cells.command("migrate")` with `--mil-model` (required) and `--dry-run` (flag)
- Resolves `_find_automil_dir()`, calls `migrate_cells`, prints per-cell action + total

**cli/__init__.py:**
- Added `from automil.cli import cells  # noqa: E402,F401  (REC-04 / D-15)` (alphabetic, between `cell` and `check`)

**Test propagation (Rule 2 — D-12 required in all submit calls):**
- `test_cli.py`: 5 submit invocations updated with `--mil-model test_model`
- `test_integration.py`: 3 submit invocations updated
- `test_submit_max_time.py`: added `run.mil_model: test_model` to config YAML
- `test_submit_writes_metadata_backend.py`: `_do_submit` helper updated
- `test_final_phase8_acceptance.py`: iris acceptance test updated with `--mil-model sklearn_iris`
- `test_setup_dry_run_gate.py`: `_run_gate` helper updated

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical] Test suite updated for D-12 --mil-model enforcement**
- **Found during:** Task 1 verification
- **Issue:** 11 test files called `automil submit` without `--mil-model` or `run.mil_model` in config; D-12 enforcement made these fail with ClickException
- **Fix:** Added `--mil-model test_model` (or `run.mil_model: test_model` in config) to all affected test submit invocations
- **Files modified:** test_cli.py, test_integration.py, test_submit_cell_identity.py, test_submit_cell_refusal.py, test_submit_writes_metadata_backend.py, test_submit_max_time.py, test_final_phase8_acceptance.py, test_setup_dry_run_gate.py
- **Commits:** 7a7431c, 020899e

**2. [Rule 1 - Bug] Normalization assertion corrected for Plan 02 underscore rule**
- **Found during:** Task 1 first run
- **Issue:** `test_explicit_mil_model_flag_keys_cell` asserted `cell["mil_model"] == "clam_sb"` but `normalize_mil_model("clam_sb")` produces `"clam sb"` (Plan 02 added underscore→space treatment)
- **Fix:** Updated assertion to `"clam sb"`
- **Files modified:** tests/test_submit_cell_identity.py
- **Commit:** 7a7431c

### Bonus Fix

`test_d208_clause_08_final_acceptance_gate` (previously failing because it invoked `test_subgate_b_sklearn_iris_end_to_end` which called submit without `--mil-model`) now **passes** after the acceptance test fix.

## Test Results

```
uv run pytest tests/test_submit_cell_identity.py tests/cells/test_migrate.py tests/test_mil_model_normalization.py -v
14 passed in 1.43s

uv run pytest tests/ -q
7 failed, 998 passed, 53 skipped
```

The 7 remaining failures are all pre-existing RED stubs for Plans 06 (terminal_writer, reconcile --from-archive) and the DEC-01 REQUIREMENTS.md check — confirmed unchanged from before this plan.

## CLI Smoke Test

```
uv run automil cells --help   # exits 0
uv run automil cells migrate --help   # exits 0
```

## Known Stubs

None — all plan objectives fully implemented.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes at trust boundaries beyond what the plan's threat model covers (T-09-08, T-09-09, T-09-10, T-09-SC all mitigated or accepted as documented in the plan).

## Self-Check: PASSED

| Item | Status |
|------|--------|
| src/automil/cells/migrate.py | FOUND |
| src/automil/cli/cells.py | FOUND |
| 09-05-SUMMARY.md | FOUND |
| commit 7a7431c | FOUND |
| commit 020899e | FOUND |
| tests/test_submit_cell_identity.py (7 tests) | GREEN |
| tests/cells/test_migrate.py (4 tests) | GREEN |
| tests/test_mil_model_normalization.py (3 tests) | GREEN |
| Full suite regression (7 pre-existing failures) | UNCHANGED |
