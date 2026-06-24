---
phase: "10"
plan: "02"
subsystem: "cli/lifecycle/apply"
tags: [APL-03, A1-fix, D-01, D-05, variant-application, loud-fail, overlay]
dependency_graph:
  requires: ["10-01"]
  provides: ["_classify_variant_route", "applied_variant.json-write", "AUTOMIL_VARIANT_MODEL-env-injection"]
  affects: ["10-04"]
tech_stack:
  added: []
  patterns: ["lazy-import inside function to avoid circular deps", "atomic write via _atomic_write_text", "scan_variants for registry population"]
key_files:
  created: []
  modified:
    - src/automil/cli/lifecycle/apply.py
    - tests/test_lifecycle_apply.py
decisions:
  - "Lazy-import scan_variants + LOSS_VARIANTS inside _classify_variant_route to avoid circular import at module load time"
  - "Policy variants also classified symmetrically with loss variants (POLICY_VARIANTS checked)"
  - "applied_variant.json written AFTER config.yaml write (not before); classification fires before both"
  - "AUTOMIL_VARIANT_MODEL env injection is best-effort (logs warning on failure, does not abort apply)"
  - "Two mandatory tests added to test_lifecycle_apply.py (tests 15 + 16)"
metrics:
  duration_minutes: 15
  completed_date: "2026-06-11"
  tasks_completed: 2
  files_changed: 2
---

# Phase 10 Plan 02: APL-03 Loud-Fail + A1 Fix Summary

**One-liner:** `_classify_variant_route` raises `ClickException` before config mutation for registered `LossVariant`/`PolicyVariant` callables (APL-03); `applied_variant.json` written to `archive/<node_id>/` for worktree propagation (A1 fix).

## What Was Built

### Task 1: `_classify_variant_route` (APL-03, D-05)

Added `_classify_variant_route(selection, variants_root)` as a module-level function in `src/automil/cli/lifecycle/apply.py`.

**Classification logic:**
- If `selection["loss"]["variant"]` is not None: call `scan_variants(variants_root)` then check `LOSS_VARIANTS`. If found (registered `LossVariant` subclass), raise `ClickException` with message containing "requires loop opening" and "ISSUE-007 / RTA".
- If `selection["policy"]["variant"]` is not None: same check against `POLICY_VARIANTS`.
- Model variants: never raise — seam-expressible via `_make_clam_args`.
- String bag_loss selectors (e.g. "svm"): not in `LOSS_VARIANTS` → no raise (Pitfall 5).

**Insertion point in `apply()`:** After `_derive_variant_selection` + empty-selection check, BEFORE `raw_yaml = yaml.safe_load(...)` — guarantees classification fires before ANY config mutation (D-05, Pitfall 3).

**Import strategy:** Lazy imports (`from automil.registry.scanner import scan_variants`, `from automil.registry._state import LOSS_VARIANTS, POLICY_VARIANTS`) inside the function body to avoid circular imports at module load time.

### Task 2: `applied_variant.json` write + env injection (A1 fix, D-01)

Added two writes at the end of `apply()`, AFTER the config.yaml atomic write:

1. **`applied_variant.json`** written to `automil/orchestrator/archive/<node_id>/applied_variant.json` using `_atomic_write_text`. The content is the raw `selection` dict (`json.dumps(selection, indent=2)`):
   ```json
   {
     "model": {"variant": "classifier_v0", "parent": "baseline"},
     "loss": {"variant": null},
     "policy": {"variant": null}
   }
   ```
   The orchestrator's existing `apply_overlay` copies all non-metadata files from `archive/<node_id>/` into the worktree — so `applied_variant.json` lands at `<worktree>/applied_variant.json` without any additional wiring.

2. **`AUTOMIL_VARIANT_MODEL` env injection** into the queue spec if `automil/orchestrator/queue/<node_id>.json` exists. Reads the spec, sets `spec["env"]["AUTOMIL_VARIANT_MODEL"] = selection["model"]["variant"] or ""`, re-writes atomically. Best-effort (logs warning on `JSONDecodeError`/`OSError`, does not abort apply).

## Tests

| Test | File | Status |
|------|------|--------|
| `test_loss_variant_raises_click_exception_at_apply_time` | test_apl03_loud_fail.py | GREEN (was RED) |
| `test_model_variant_does_not_raise` | test_apl03_loud_fail.py | GREEN (was RED) |
| `test_string_selector_loss_does_not_raise` | test_apl03_loud_fail.py | GREEN (was RED) |
| `test_error_raised_before_config_mutation` | test_apl03_loud_fail.py | GREEN (was RED) |
| `test_apply_writes_applied_variant_json` | test_lifecycle_apply.py | GREEN (NEW, MANDATORY) |
| `test_apply_injects_env_var_into_queue_spec` | test_lifecycle_apply.py | GREEN (NEW, MANDATORY) |
| 14 pre-existing apply tests | test_lifecycle_apply.py | GREEN (no regressions) |
| Full suite (1013 tests) | tests/ | 3 pre-existing failures, 0 new regressions |

**Pre-existing failures (expected, not caused by this plan):**
- `test_d208_clause_11_state_roadmap_complete` — acceptance milestone gate (roadmap not complete yet)
- `test_iris_dispatches_classifier_v0_when_variant_set` — plan 10-04 scope (iris train.py dispatch)
- `test_iris_applied_variant_reaches_worktree_at_runtime` — plan 10-04 stub (explicitly noted in plan as remaining RED)

## Commits

| Hash | Message | Files |
|------|---------|-------|
| `477e019` | feat(10-02): add _classify_variant_route to apply.py (APL-03, D-05) | apply.py |
| `fcfe2f3` | feat(10-02): write applied_variant.json to archive + inject env var (A1 fix, D-01) | apply.py, test_lifecycle_apply.py |

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

**Clarification on policy variant classification:** The plan specified checking `POLICY_VARIANTS` symmetrically with `LOSS_VARIANTS`. This was implemented as written. The test suite has no concrete registered `PolicyVariant` subclasses, so the path is covered by the classification logic but not exercised by a dedicated test (the plan only mandated 4 APL-03 tests, which are all GREEN).

## Known Stubs

- `test_iris_applied_variant_reaches_worktree_at_runtime` in `tests/test_apl01_iris_dispatch.py` remains RED (pytest.fail stub). Resolves in plan 10-04 when iris `train.py` is fixed to read `applied_variant.json`.

## Threat Surface Scan

No new network endpoints, auth paths, or trust boundary crossings introduced. `applied_variant.json` is written to `automil/orchestrator/archive/<node_id>/` — a path derived from `adir` (validated by `_find_automil_dir`) + a fixed prefix + `node_id` (from graph, trusted). No user-controlled path component. T-10-02 (variant name path construction) is mitigated by the fact that `_classify_variant_route` uses names only for dict lookup, not Path construction.

## Self-Check: PASSED

- `src/automil/cli/lifecycle/apply.py` modified — FOUND
- `tests/test_lifecycle_apply.py` modified — FOUND
- Commit `477e019` — FOUND (`git log --oneline | grep 477e019`)
- Commit `fcfe2f3` — FOUND
- `_classify_variant_route` in apply.py — FOUND (line 23)
- `applied_variant.json` write in apply.py — FOUND (line 207)
- `AUTOMIL_VARIANT_MODEL` injection in apply.py — FOUND (line 222)
- All 20 tests GREEN — CONFIRMED
