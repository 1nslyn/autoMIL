---
phase: "10"
plan: "01"
subsystem: variant-application-integrity
tags: [wave-0, test-stubs, APL-01, APL-02, APL-03, RED-tests, nyquist]
dependency_graph:
  requires: []
  provides:
    - tests/test_apl01_iris_dispatch.py
    - tests/test_apl03_loud_fail.py
    - benchmarks/tests/test_variant_dispatch_clam.py
    - benchmarks/tests/test_apl02_real_run.py
    - benchmarks/src/autobench/pipeline/variant_dispatch.py
  affects:
    - pyproject.toml
    - benchmarks/pyproject.toml
tech_stack:
  added: []
  patterns:
    - autouse _clear_registry fixture for registry singleton pollution prevention (T-10-01)
    - pytest.mark.workstation for workstation-gated real-run tests
    - pytest.fail() with explicit plan-reference messages for intentional RED stubs
key_files:
  created:
    - tests/test_apl01_iris_dispatch.py
    - tests/test_apl03_loud_fail.py
    - benchmarks/tests/test_variant_dispatch_clam.py
    - benchmarks/tests/test_apl02_real_run.py
    - benchmarks/src/autobench/pipeline/variant_dispatch.py
  modified:
    - pyproject.toml
    - benchmarks/pyproject.toml
decisions:
  - "autouse _clear_registry fixture pattern (not manual teardown) for all registry tests"
  - "pytest.fail() with explicit plan-reference messages (not NotImplementedError) for A1-closure tests that require multiple plans"
  - "workstation marker registered in both pyproject.toml files to suppress PytestUnknownMarkWarning"
metrics:
  duration: "~15 minutes"
  completed: "2026-06-11"
  tasks_completed: 2
  tasks_total: 2
  files_created: 5
  files_modified: 2
---

# Phase 10 Plan 01: Wave-0 Test Scaffolding Summary

Wave-0 test scaffolding: 5 new files (4 RED test files + 1 importable stub module) establishing Nyquist compliance before any implementation begins. All new tests fail RED for the correct reason (missing implementations), not for wrong reasons (syntax errors or import failures).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | APL-01 + APL-03 framework-side test stubs | c84b9df | tests/test_apl01_iris_dispatch.py, tests/test_apl03_loud_fail.py |
| 2 | APL-02 variant_dispatch stub + autobench test stubs | adb89cf | benchmarks/src/autobench/pipeline/variant_dispatch.py, benchmarks/tests/test_variant_dispatch_clam.py, benchmarks/tests/test_apl02_real_run.py |

## Test Status After Plan 01

| File | Tests | Status | Reason for RED |
|------|-------|--------|---------------|
| tests/test_apl01_iris_dispatch.py | 3 | 1 GREEN (baseline), 2 RED | Missing dispatch branch (Plan 10-04) + applied_variant.json (Plan 10-02) |
| tests/test_apl03_loud_fail.py | 4 | 4 RED | _classify_variant_route not yet in apply.py (Plan 10-02) |
| benchmarks/tests/test_variant_dispatch_clam.py | 5 | 5 RED | apply_model_variant_to_exp_cfg raises NotImplementedError (Plan 10-03) |
| benchmarks/tests/test_apl02_real_run.py | 1 | SKIPPED (workstation) | AUTOBENCH_CCRCC_ROOT not set in CI |
| tests/test_lifecycle_apply.py | 14 | 14 GREEN | Regression guard — unaffected by this plan |

## Decisions Made

1. **autouse fixture for registry teardown:** Used `@pytest.fixture(autouse=True)` calling `_clear_registry()` both before and after each test. This prevents T-10-01 state pollution without requiring every test to have explicit teardown boilerplate.

2. **pytest.fail() for A1-closure tests:** `test_iris_applied_variant_reaches_worktree_at_runtime` and the APL-03 tests use `pytest.fail()` with explicit plan-reference messages rather than relying on ImportError. This makes the RED reason unambiguous — a human reading the output immediately knows which plan must be implemented.

3. **workstation marker registration:** Added `workstation` marker to both `pyproject.toml` (workspace root) and `benchmarks/pyproject.toml` to suppress `PytestUnknownMarkWarning`. This is a Rule 2 fix — incorrect marker configuration causes spurious warnings that mask real issues.

4. **variant_dispatch.py top-level imports:** The stub validates the full import chain (`automil.registry.scanner`, `automil.registry._state`, `automil.registry.spec`) at import time — ensuring the module fails loudly on import if the registry chain is broken, not silently at call time.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Config] Registered workstation pytest marker in both pyproject.toml files**
- **Found during:** Task 2 verification
- **Issue:** `pytest.mark.workstation` used in `test_apl02_real_run.py` triggered `PytestUnknownMarkWarning` because neither `pyproject.toml` (workspace root) nor `benchmarks/pyproject.toml` declared the marker.
- **Fix:** Added `"workstation: requires workstation with data + GPU; skipped in CI"` to the `markers` list in both `pyproject.toml` files.
- **Files modified:** `pyproject.toml`, `benchmarks/pyproject.toml`
- **Commit:** adb89cf

## Known Stubs

| File | Stub | Reason |
|------|------|--------|
| benchmarks/src/autobench/pipeline/variant_dispatch.py | `apply_model_variant_to_exp_cfg` raises `NotImplementedError` | Plan 10-03 implements the body |
| tests/test_apl01_iris_dispatch.py | `test_iris_dispatches_classifier_v0_when_variant_set` asserts `variant_dispatched` key in result.json | Plan 10-04 adds dispatch to iris train.py |
| tests/test_apl01_iris_dispatch.py | `test_iris_applied_variant_reaches_worktree_at_runtime` calls `pytest.fail()` | Plans 10-02 + 10-04 required |
| tests/test_apl03_loud_fail.py | All 4 tests call `pytest.fail()` when `_classify_variant_route` absent | Plan 10-02 adds `_classify_variant_route` to apply.py |
| benchmarks/tests/test_apl02_real_run.py | `test_real_clam_run_composite_differs_from_baseline` calls `pytest.fail()` | Plan 10-03 wires run_experiment.py dispatch |

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. All new files are test code or a stub module. No new threat flags.

## Self-Check: PASSED

Files exist:
- FOUND: tests/test_apl01_iris_dispatch.py
- FOUND: tests/test_apl03_loud_fail.py
- FOUND: benchmarks/tests/test_variant_dispatch_clam.py
- FOUND: benchmarks/tests/test_apl02_real_run.py
- FOUND: benchmarks/src/autobench/pipeline/variant_dispatch.py

Commits exist:
- FOUND: c84b9df (Task 1)
- FOUND: adb89cf (Task 2)

Test verification:
- tests/test_apl01_iris_dispatch.py: 1 passed, 2 failed (correct RED)
- tests/test_apl03_loud_fail.py: 4 failed (correct RED)
- benchmarks/tests/test_variant_dispatch_clam.py: 5 failed (correct RED: NotImplementedError)
- benchmarks/tests/test_apl02_real_run.py: 1 skipped (workstation-gated)
- tests/test_lifecycle_apply.py: 14 passed (regression guard green)
