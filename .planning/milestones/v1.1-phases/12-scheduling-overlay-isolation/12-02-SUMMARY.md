---
phase: 12-scheduling-overlay-isolation
plan: "02"
subsystem: orchestrator
tags: [scheduling, gpu, round-robin, least-loaded, best-fit, hot-reload, config]

requires:
  - phase: 12-01
    provides: Wave-0 RED stubs for SCH-01 (7 xfail tests in test_scheduling_policy.py)

provides:
  - SCHEDULING_POLICY='best_fit' module-level constant in _orchestrator_daemon.py
  - self.scheduling_policy and self._rr_cursor initialized in ExperimentOrchestrator.__init__
  - _find_best_gpu dispatches on scheduling_policy (best_fit / round_robin / least_loaded / unknown-fallback)
  - _reload_orchestrator_config hot-reloads scheduling_policy without resetting _rr_cursor
  - scheduling_policy key in config.yaml.j2 orchestrator section (default best_fit)
  - All 7 SCH-01 tests GREEN; env-whitelist + framework-purity GREEN

affects:
  - 12-03 (SCH-02 editable overlay guard — same daemon file, different method)
  - Any future plan that modifies _find_best_gpu or _reload_orchestrator_config

tech-stack:
  added: []
  patterns:
    - "Strategy dispatch via if/elif in _find_best_gpu (no strategy class hierarchy — appropriate for 3 branches)"
    - "Monotonic _rr_cursor never reset on policy change (sticky across hot-reloads)"
    - "Hot-reload pattern: orch_cfg.get(key, self.attr) + logger.info on change"

key-files:
  created: []
  modified:
    - src/automil/backends/_orchestrator_daemon.py
    - src/automil/templates/config.yaml.j2
    - tests/test_scheduling_policy.py
    - tests/test_framework_purity.py

key-decisions:
  - "Round-robin cycles among ELIGIBLE candidates only (post-VRAM + concurrency filter), sorted by gpu_index — avoids pointing cursor at a blocked GPU"
  - "_rr_cursor is a plain int instance attr; daemon is single-threaded so no locking needed"
  - "Unknown policy strings fall back to best_fit with logger.warning (never eval'd or used as import path — V5 input validation)"
  - "_rr_cursor NOT reset on policy change during hot-reload (stale cursor is harmless; reset would disrupt in-flight fairness)"
  - "Allowlist line numbers updated in test_framework_purity.py after line drift (+1 daemon, +1 template) caused by new constant and new config key"

patterns-established:
  - "Policy dispatch: if/elif inside the scheduling function, else branch handles unknown + emits warning"
  - "Config hot-reload: read with fallback to current value, log only on actual change, never touch cursor state"

requirements-completed:
  - SCH-01

duration: 8min
completed: 2026-06-11
---

# Phase 12 Plan 02: SCH-01 GPU Scheduling-Policy Knob Summary

**`orchestrator.scheduling_policy` knob (best_fit | round_robin | least_loaded) dispatched in `_find_best_gpu`, hot-reloaded via `_reload_orchestrator_config`, with sticky `_rr_cursor` — zero behavior change at default `best_fit`**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-11T20:57:00Z
- **Completed:** 2026-06-11T21:05:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `SCHEDULING_POLICY = "best_fit"` constant and two new instance attrs (`scheduling_policy`, `_rr_cursor`) to `ExperimentOrchestrator.__init__`
- Refactored `_find_best_gpu` from a single-strategy sort into a three-branch policy dispatch; best_fit behavior preserved exactly (no behavior change at default)
- Added scheduling_policy hot-reload block in `_reload_orchestrator_config` mirroring the existing `max_per_gpu` pattern; `_rr_cursor` intentionally not reset
- Added `scheduling_policy: "best_fit"` to `config.yaml.j2` orchestrator section with inline option comment
- Stripped `xfail` markers from all 7 SCH-01 stubs; added missing attrs (`default_vram`, `default_timeout`, `poll_interval`) to `_fake_daemon` so hot-reload tests can exercise `_reload_orchestrator_config` against a minimal fake
- Fixed allowlist line-number drift in `test_framework_purity.py` caused by the new constant (+1 in daemon) and new config key (+1 in template)

## Task Commits

1. **Task 1: Implement scheduling_policy knob + _find_best_gpu dispatch** — `d342cfd` (feat)
2. **Task 2: Add scheduling_policy to config.yaml.j2 + regression check** — `bc313ba` (feat)

## Files Created/Modified

- `src/automil/backends/_orchestrator_daemon.py` — SCHEDULING_POLICY constant; scheduling_policy + _rr_cursor in __init__; _find_best_gpu policy dispatch; _reload_orchestrator_config hot-reload block
- `src/automil/templates/config.yaml.j2` — scheduling_policy key in orchestrator: section
- `tests/test_scheduling_policy.py` — xfail markers stripped; _fake_daemon extended with hot-reload attrs
- `tests/test_framework_purity.py` — _ALLOWLIST line numbers updated after drift

## Decisions Made

- **Eligible-only round-robin:** cursor cycles through the post-filter candidate list (sorted by gpu_index), not the full GPU index space. This prevents pointing at a VRAM-full or concurrency-capped GPU.
- **No _rr_cursor reset on hot-reload:** switching policy to best_fit and back to round_robin would silently restart the cycle; leaving the cursor sticky is correct per 12-RESEARCH.md Pitfall 1.
- **_fake_daemon extension:** the test stubs' fake object lacked `default_vram`, `default_timeout`, and `poll_interval` — attrs read by `_reload_orchestrator_config` before reaching the scheduling_policy block. Extended the helper rather than mocking those paths; cleaner and exercises more of the real code path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extended _fake_daemon with missing hot-reload attrs**
- **Found during:** Task 1 (running test_policy_hot_reload after stripping xfail)
- **Issue:** `_reload_orchestrator_config` reads `self.default_vram`, `self.default_timeout`, `self.poll_interval` before reaching the scheduling_policy block; the minimal fake didn't have these, causing `AttributeError`
- **Fix:** Added the three missing attrs to `_fake_daemon` with sensible defaults matching module constants
- **Files modified:** tests/test_scheduling_policy.py
- **Verification:** test_policy_hot_reload and test_cursor_not_reset_on_policy_change both PASSED
- **Committed in:** d342cfd (Task 1 commit)

**2. [Rule 1 - Bug] Fixed allowlist line-number drift in test_framework_purity.py**
- **Found during:** Task 2 (regression run after adding scheduling_policy to config.yaml.j2)
- **Issue:** Adding `SCHEDULING_POLICY = "best_fit"` shifted daemon line 54→55; adding scheduling_policy key shifted config.yaml.j2 lines 109→110 and 135→136 — causing two purity test failures
- **Fix:** Updated three _ALLOWLIST keys to their new line numbers; added explanatory comments noting the 12-02 shift
- **Files modified:** tests/test_framework_purity.py
- **Verification:** test_framework_purity_no_autobench_refs and test_allowlist_anchors_still_present PASSED
- **Committed in:** bc313ba (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs — missing fake attrs + allowlist drift)
**Impact on plan:** Both fixes necessary for correctness. No scope creep; no new files added.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. The scheduling_policy string is validated defensively (unknown value falls back to best_fit with logger.warning — never eval'd). The `_rr_cursor` is a monotonic Python int (arbitrary-precision; modulo keeps index in bounds — T-12-03 accepted per threat model).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- SCH-01 complete. `orchestrator.scheduling_policy` is live for operators to opt in to round_robin or least_loaded.
- Plan 12-03 (SCH-02 editable overlay guard) is unblocked — it touches `_launch` and `check.py`, not `_find_best_gpu`.
- Full suite `uv run pytest tests/ -v` should be run at phase gate after 12-03 completes.

---
*Phase: 12-scheduling-overlay-isolation*
*Completed: 2026-06-11*

## Self-Check: PASSED

- `src/automil/backends/_orchestrator_daemon.py` — exists, contains SCHEDULING_POLICY, scheduling_policy, _rr_cursor, round_robin, least_loaded dispatch
- `src/automil/templates/config.yaml.j2` — exists, contains scheduling_policy
- `tests/test_scheduling_policy.py` — exists, no xfail markers remaining
- `tests/test_framework_purity.py` — exists, allowlist updated
- Commits d342cfd and bc313ba present in git log
- 7 SCH-01 tests PASSED; 12 env-whitelist tests PASSED; 3 framework-purity tests PASSED
