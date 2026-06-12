---
phase: 12-scheduling-overlay-isolation
plan: "01"
subsystem: tests
tags: [sch-01, sch-02, wave-0, red-stubs, nyquist, tdd]
dependency_graph:
  requires: []
  provides:
    - tests/test_scheduling_policy.py (7 RED xfail stubs for SCH-01)
    - tests/test_editable_overlay_guard.py (5 RED xfail stubs for SCH-02)
  affects:
    - 12-02-PLAN.md (scheduling policy implementation — tests drive it)
    - 12-03-PLAN.md (editable overlay guard implementation — tests drive it)
tech_stack:
  added: []
  patterns:
    - FakeDaemon SimpleNamespace fixture to avoid full daemon construction
    - PropertyMock dispatch gate for attribute-read verification
    - xfail(strict=True) RED stubs with future-state assertions
key_files:
  created:
    - tests/test_scheduling_policy.py
    - tests/test_editable_overlay_guard.py
  modified: []
decisions:
  - "FakeDaemon SimpleNamespace avoids filesystem-heavy ExperimentOrchestrator init"
  - "test_best_fit_picks_tightest uses PropertyMock to detect scheduling_policy reads (dispatch gate)"
  - "test_round_robin_cursor_wraps asserts _rr_cursor==3 after 3 calls (cursor advancement gate)"
  - "test_unknown_policy_fallback patches logger to assert warning call (security V5 gate)"
  - "test_opt_in_injection_prepends_pythonpath invokes _apply_editable_overlay_guard (future method)"
  - "test_editable_overlay_guard.py contains zero autobench/AUTOBENCH_/benchmarks/ refs (D-206)"
metrics:
  duration_seconds: 305
  completed_date: "2026-06-11"
  tasks_completed: 3
  files_created: 2
---

# Phase 12 Plan 01: Wave-0 RED Test Stubs Summary

**One-liner:** Wave-0 Nyquist test stubs — 7 xfail for SCH-01 policy dispatch + 5 xfail for SCH-02 editable-overlay guard using FakeDaemon fixtures and PropertyMock dispatch gates.

## What Was Built

Two new test files containing 12 total RED stubs that will drive the Wave-2 implementations:

### tests/test_scheduling_policy.py (7 stubs — SCH-01)

| Test | Gate mechanism |
|------|---------------|
| `test_best_fit_picks_tightest` | PropertyMock on `scheduling_policy` attr — fails until dispatch reads it |
| `test_least_loaded_picks_emptiest` | Result assertion — fails until least_loaded branch exists |
| `test_round_robin_cycles_eligible` | Result sequence — fails until round_robin branch exists |
| `test_round_robin_cursor_wraps` | `_rr_cursor == 3` after 3 calls — fails until cursor is incremented |
| `test_policy_hot_reload` | `scheduling_policy` attr update after `_reload_orchestrator_config` |
| `test_unknown_policy_fallback` | `logger.warning` call assertion — fails until dispatch warns unknown |
| `test_cursor_not_reset_on_policy_change` | `_rr_cursor` unchanged after policy switch hot-reload |

**Fixture pattern:** `_fake_daemon()` builds a `SimpleNamespace` with the exact attrs `_find_best_gpu` reads (`gpu_allocations`, `running`, `max_per_gpu`, `safety_margin_gb`, `scheduling_policy`, `_rr_cursor`). `query_gpus` is patched via `unittest.mock.patch`. `ExperimentOrchestrator._find_best_gpu` and `_reload_orchestrator_config` are called as unbound methods against the FakeDaemon — no real filesystem needed.

### tests/test_editable_overlay_guard.py (5 stubs — SCH-02)

| Test | What it gates |
|------|--------------|
| `test_check_warns_missing_guard` | Warning emitted when editable overlap + no guard |
| `test_check_no_warn_when_guard_enabled` | Warning suppressed when `editable_overlay_guard: true` |
| `test_check_no_warn_when_consumer_guard_present` | Warning suppressed when `sys.path.insert` in run script |
| `test_opt_in_injection_prepends_pythonpath` | `_apply_editable_overlay_guard` prepends worktree src to PYTHONPATH |
| `test_check_suppresses_when_no_editable_overlap` | No warning when editable root has no path overlap |

**Fixture pattern:** `tmp_path` for fake project layout, `unittest.mock.patch` for `_collect_editable_source_roots`. All tests use `editable_overlay_guard: true` only when explicitly testing opt-in behavior — default-OFF is preserved, keeping D-199 invariant tests GREEN.

## Verification Results

```
tests/test_scheduling_policy.py          7 xfailed
tests/test_editable_overlay_guard.py     5 xfailed
tests/test_orchestrator_env_whitelist.py 12 passed  (D-199 invariants — ZERO REGRESSIONS)
tests/test_framework_purity.py            3 passed  (D-206 purity gate — ZERO REGRESSIONS)
Total: 15 passed, 12 xfailed
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Class name was OrchestratorDaemon, actual is ExperimentOrchestrator**
- **Found during:** Task 1 first test run (ImportError)
- **Issue:** Plan referred to `OrchestratorDaemon` but the class in `_orchestrator_daemon.py` is `ExperimentOrchestrator`
- **Fix:** Updated all references in `test_scheduling_policy.py` from `OrchestratorDaemon` to `ExperimentOrchestrator`
- **Files modified:** `tests/test_scheduling_policy.py`
- **Commit:** 59077b6 (included in same task commit)

**2. [Rule 1 - Bug] Three tests were XPASS before dispatch implemented**
- **Found during:** Task 1 first test run (3 FAILED due to `strict=True` + unexpected PASS)
- **Issue:** `test_best_fit_picks_tightest`, `test_round_robin_cursor_wraps`, and `test_unknown_policy_fallback` asserted behaviors that already match current code (current best-fit always returns GPU 0). With `xfail(strict=True)` an unexpected pass is a failure.
- **Fix:** Strengthened each test with a dispatch-gate assertion that only holds after SCH-01 is implemented:
  - `test_best_fit_picks_tightest`: added `PropertyMock` on `scheduling_policy` to detect reads
  - `test_round_robin_cursor_wraps`: added `_rr_cursor == 3` assertion (cursor not incremented by current code)
  - `test_unknown_policy_fallback`: added `logger.warning` mock assertion (no warning in current code)
- **Files modified:** `tests/test_scheduling_policy.py`
- **Commit:** 59077b6

## Known Stubs

By design — this is a Wave-0 plan. All 12 tests are intentional RED stubs. The production implementation is deferred to plans 12-02 (SCH-01) and 12-03 (SCH-02).

`_apply_editable_overlay_guard` referenced in `test_opt_in_injection_prepends_pythonpath` is a method that does not yet exist on `ExperimentOrchestrator`. This is the expected xfail condition.

`_collect_editable_source_roots` referenced in `test_editable_overlay_guard.py` is a function that does not yet exist in `automil.cli.check`. This is the expected xfail condition.

## Threat Flags

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Test files only — no production surface changes.

## Self-Check: PASSED

- [x] `tests/test_scheduling_policy.py` exists
- [x] `tests/test_editable_overlay_guard.py` exists
- [x] Commit `59077b6` exists (test_scheduling_policy.py)
- [x] Commit `a952677` exists (test_editable_overlay_guard.py)
- [x] 7 XFAIL + 5 XFAIL + 15 PASS — zero failures, zero errors
- [x] Zero autobench/AUTOBENCH_/benchmarks/ refs in either new test file
