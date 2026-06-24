---
phase: 12
slug: scheduling-overlay-isolation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 12 — Validation Strategy

> Derived from 12-RESEARCH.md §Validation Architecture.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Quick run** | `uv run pytest tests/test_scheduling_policy.py tests/test_editable_overlay_guard.py tests/test_orchestrator_env_whitelist.py tests/test_framework_purity.py -x` |
| **Framework suite** | `uv run pytest tests/ -q` (benchmark tree run SEPARATELY — rootdir collision) |

## Per-Requirement Verification Map

| Req | Behavior | Type | Command | File |
|-----|----------|------|---------|------|
| SCH-01 | `best_fit` picks tightest-fit GPU (current behavior preserved) | unit | `pytest tests/test_scheduling_policy.py::test_best_fit_picks_tightest` | ❌ W0 |
| SCH-01 | `least_loaded` picks emptiest GPU | unit | `::test_least_loaded_picks_emptiest` | ❌ W0 |
| SCH-01 | `round_robin` cycles eligible GPUs in index order | unit | `::test_round_robin_cycles_eligible` | ❌ W0 |
| SCH-01 | `round_robin` cursor wraps across calls | unit | `::test_round_robin_cursor_wraps` | ❌ W0 |
| SCH-01 | `scheduling_policy` hot-reloaded by `_reload_orchestrator_config` | unit | `::test_policy_hot_reload` | ❌ W0 |
| SCH-01 | unknown policy string falls back to `best_fit` | unit | `::test_unknown_policy_fallback` | ❌ W0 |
| SCH-01 | `_rr_cursor` not reset on policy hot-reload | unit | `::test_cursor_not_reset_on_policy_change` | ❌ W0 |
| SCH-02 | `automil check` warns when editable src overlaps overlay + no guard | unit | `pytest tests/test_editable_overlay_guard.py::test_check_warns_missing_guard` | ❌ W0 |
| SCH-02 | `check` suppresses warning when `editable_overlay_guard: true` | unit | `::test_check_no_warn_when_guard_enabled` | ❌ W0 |
| SCH-02 | `check` suppresses warning when consumer run-script has `sys.path.insert` guard | unit | `::test_check_no_warn_when_consumer_guard_present` | ❌ W0 |
| SCH-02 | opt-in injection prepends worktree editable root to PYTHONPATH when flag true | unit | `::test_opt_in_injection_prepends_pythonpath` | ❌ W0 |
| SCH-02 | **D-199 invariant: injection is NO-OP by default** (env-whitelist tests still pass) | unit | `pytest tests/test_orchestrator_env_whitelist.py` | ✅ exists |
| SCH-02 | **framework purity: no autobench/benchmarks refs in new src/automil/ code** | lint | `pytest tests/test_framework_purity.py` | ✅ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements
- [ ] `tests/test_scheduling_policy.py` — SCH-01 (7 tests)
- [ ] `tests/test_editable_overlay_guard.py` — SCH-02 (5 new tests)
- (existing `test_orchestrator_env_whitelist.py` + `test_framework_purity.py` guard the D-199 invariants SCH-02 must not break)

## Manual-Only Verifications
*None — both fixes are CI-testable with mocked GPUs / fake site-packages (no real multi-GPU hardware needed; placement logic is unit-tested on mock GPU candidate lists).*

## Validation Sign-Off
- [ ] All tasks have `<automated>` verify or Wave 0 deps
- [ ] D-199 invariant tests (env-whitelist + framework-purity) stay GREEN
- [ ] `nyquist_compliant: true` set when complete

**Approval:** pending
