---
phase: 12-scheduling-overlay-isolation
plan: "03"
subsystem: cli/check + daemon
tags: [sch-02, editable-guard, check-warning, pythonpath, d199, framework-purity]
dependency_graph:
  requires: ["12-01", "12-02"]
  provides: ["_collect_editable_source_roots", "_apply_editable_overlay_guard", "SCH-02"]
  affects: ["src/automil/cli/check.py", "src/automil/backends/_orchestrator_daemon.py"]
tech_stack:
  added: ["site (stdlib) — editable source root discovery"]
  patterns: ["opt-in PYTHONPATH injection (post-_build_subprocess_env post-processing)", "xfail stub activation by removing decorator", "allowlist line-drift update"]
key_files:
  created: []
  modified:
    - src/automil/cli/check.py
    - src/automil/backends/_orchestrator_daemon.py
    - tests/test_editable_overlay_guard.py
    - tests/test_framework_purity.py
    - tests/test_scheduling_policy.py
decisions:
  - "_apply_editable_overlay_guard added as a named method (not inline block) because test_opt_in_injection_prepends_pythonpath calls ExperimentOrchestrator._apply_editable_overlay_guard directly"
  - "_collect_editable_source_roots imported at module level in daemon (not via local import) so test patch target automil.backends._orchestrator_daemon._collect_editable_source_roots resolves correctly"
  - "xfail(strict=True) markers removed from all 5 SCH-02 test stubs; run_script_content stub string fixed (removed embedded sys.path.insert substring that falsely triggered has_consumer_guard=True)"
metrics:
  duration: "~20 minutes"
  completed: "2026-06-12"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
---

# Phase 12 Plan 03: SCH-02 Editable Overlay Guard Summary

**One-liner:** Opt-in `editable_overlay_guard` PYTHONPATH injection in `_launch` + `automil check` warning when editable package sources are overlaid without a worktree import guard (D-199 invariant preserved: default OFF).

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | automil check SCH-02 warning + `_collect_editable_source_roots` | b2633e9 | src/automil/cli/check.py, tests/test_editable_overlay_guard.py |
| 2 | Daemon opt-in PYTHONPATH injection in `_launch` | f1327b4 | src/automil/backends/_orchestrator_daemon.py, tests/test_framework_purity.py, tests/test_scheduling_policy.py |

## What Was Built

### Task 1 — `check.py`

**`_collect_editable_source_roots() -> list[str]`** (new helper):
- Scans `site.getsitepackages()` + `site.getusersitepackages()` for three patterns: `_editable_impl_*.pth`, `__editable__*.pth`, `*.egg-link`
- Reads each file's text content; if content is a valid existing directory, appends to roots
- Catches `OSError` on read; skips that file
- Zero consumer-specific references (D-206 purity enforced)

**SCH-02 warning block** in `check()` (after `files.editable` check):
- Calls `_collect_editable_source_roots()` to detect editable roots in the active venv
- Checks `has_consumer_guard`: `"sys.path.insert"` in run script text (same heuristic as `"result.json"` check)
- Checks `overlay_guard_enabled`: `orchestrator.editable_overlay_guard` from config (default False)
- For each editable root that overlaps a `files.editable` path: emits warning containing "editable" and "worktree" iff no guard present
- Warning includes the fix suggestion: add `sys.path.insert` to run script OR set `orchestrator.editable_overlay_guard: true`

### Task 2 — `_orchestrator_daemon.py`

**Module-level import:** `from automil.cli.check import _collect_editable_source_roots` — placed at module level so `automil.backends._orchestrator_daemon._collect_editable_source_roots` is patchable by tests.

**`__init__` addition** (after `_rr_cursor`):
```python
self.editable_overlay_guard: bool = bool(orch_cfg.get("editable_overlay_guard", False))
```

**`_reload_orchestrator_config` addition** (after scheduling_policy hot-reload):
```python
new_guard = bool(orch_cfg.get("editable_overlay_guard", self.editable_overlay_guard))
if new_guard != self.editable_overlay_guard:
    logger.info("Config reload: editable_overlay_guard %r -> %r", ...)
    self.editable_overlay_guard = new_guard
```

**`_apply_editable_overlay_guard(self, env, wt_path)` method** (new, before `_launch`):
- Guards on `self.editable_overlay_guard` — returns immediately if False (default OFF, D-199 invariant)
- Calls `_collect_editable_source_roots()` at launch time (not cached — Pitfall 6 avoidance)
- For each editable root: computes `root_p.relative_to(self.project_root)`; silently skips roots outside `project_root` (`ValueError` caught)
- Checks `wt_path / rel` exists as a directory before prepending
- Prepends to `env["PYTHONPATH"]` using `":"` join; preserves existing PYTHONPATH

**`_launch` post-processing call** (after `env = self._build_subprocess_env(...)`):
```python
self._apply_editable_overlay_guard(env=env, wt_path=wt_path)
```
`_build_subprocess_env` signature is **unchanged**.

## Test Results

| Test Suite | Count | Result |
|-----------|-------|--------|
| test_editable_overlay_guard.py (SCH-02) | 5 | ALL GREEN |
| test_orchestrator_env_whitelist.py (D-199) | 9 | ALL GREEN |
| test_framework_purity.py (D-206) | 3 | ALL GREEN |
| test_scheduling_policy.py (SCH-01, regression) | 7 | ALL GREEN |

**D-199 invariant preserved:** `test_pythonpath_not_auto_injected_phase8` and `test_autobench_root_not_auto_injected_phase8` remain GREEN — `editable_overlay_guard` defaults to `False`, so PYTHONPATH is NOT force-set in default config.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test stub `_build_check_warnings` helper: `run_script_content` false-positive guard detection**
- **Found during:** Task 1 test execution
- **Issue:** The xfail stub test helper used `run_script_content="# no sys.path.insert here"` — this string contains the literal `"sys.path.insert"` substring, causing `has_consumer_guard=True` which suppressed the warning. The test then asserted warnings were present, causing the test to fail even after implementation.
- **Fix:** Changed the stub run_script_content to `"# no guard here"` (no embedded substring). Also changed capitalized "Worktree" to lowercase "worktree" in the stub warning string to match the test assertion `"worktree" in w`.
- **Files modified:** `tests/test_editable_overlay_guard.py`
- **Commit:** b2633e9

**2. [Rule 3 - Blocking] Test called `_apply_editable_overlay_guard` as a named method**
- **Found during:** Task 2 — reading test stub `test_opt_in_injection_prepends_pythonpath`
- **Issue:** The plan said "injection block in `_launch`" but the test at line 241 calls `ExperimentOrchestrator._apply_editable_overlay_guard(fake, env=env, wt_path=wt_path)` directly — requiring it to be a named method, not inline code.
- **Fix:** Implemented as `_apply_editable_overlay_guard(self, env, wt_path)` method (not inline). Called from `_launch` as `self._apply_editable_overlay_guard(env=env, wt_path=wt_path)`.
- **Files modified:** `src/automil/backends/_orchestrator_daemon.py`
- **Commit:** f1327b4

**3. [Rule 1 - Bug] `test_scheduling_policy.py` `_fake_daemon` missing `editable_overlay_guard` attr**
- **Found during:** Task 2 full suite run
- **Issue:** `_reload_orchestrator_config` uses `self.editable_overlay_guard` as default in `orch_cfg.get(...)` call. The hot-reload tests use a `SimpleNamespace` mock that didn't have this attr, causing `AttributeError`.
- **Fix:** Added `editable_overlay_guard=False` to `_fake_daemon` SimpleNamespace.
- **Files modified:** `tests/test_scheduling_policy.py`
- **Commit:** f1327b4

**4. [Rule 1 - Bug] Framework purity allowlist line drift in daemon file**
- **Found during:** Task 2 purity check
- **Issue:** Adding `from automil.cli.check import _collect_editable_source_roots` import line shifted the allowlisted D-199 comment from line 55 to line 56.
- **Fix:** Updated `_ALLOWLIST` key from `_orchestrator_daemon.py:55` to `_orchestrator_daemon.py:56`; updated comment to note "12-03" as the cause.
- **Files modified:** `tests/test_framework_purity.py`
- **Commit:** f1327b4

**5. [Rule 1 - Bug] `_collect_editable_source_roots` import approach**
- **Found during:** Task 2 — test patches `automil.backends._orchestrator_daemon._collect_editable_source_roots`
- **Issue:** The plan specified a local import inside the `if self.editable_overlay_guard:` block. But the test patches the module-level attribute `automil.backends._orchestrator_daemon._collect_editable_source_roots`. A local import inside a function does `from automil.cli.check import _collect_editable_source_roots` — this would bind the local name from `automil.cli.check`, not from `automil.backends._orchestrator_daemon`. The test patch would have no effect.
- **Fix:** Import at module level so the name lives in `_orchestrator_daemon`'s module namespace and can be patched.
- **Files modified:** `src/automil/backends/_orchestrator_daemon.py`
- **Commit:** f1327b4

## Pre-existing Failures (Out of Scope)

Two acceptance tests (`test_d208_clause_07_framework_purity_grep_gate`, `test_d208_clause_11_state_roadmap_complete`) were already failing before this plan's execution — they check v1.0 milestone REQUIREMENTS.md rows (`DEC-01 | Phase 8 | Complete`) which don't exist in the v1.1 REQUIREMENTS.md. These are pre-existing failures unrelated to SCH-02.

## Security Review (T-12-04 / T-12-05)

- T-12-04 (Elevation of Privilege via PYTHONPATH): Guard is OFF by default; injection only fires when operator explicitly sets `editable_overlay_guard: true`; paths derived from `project_root`-relative subpaths only; `ValueError` catch prevents external-root injection.
- T-12-05 (Tampering via .pth content): `wt_candidate.is_dir()` ensures only real directories are prepended; Path arithmetic prevents shell injection (no subprocess call on path).

## Self-Check: PASSED

- `src/automil/cli/check.py` exists and contains `_collect_editable_source_roots`
- `src/automil/backends/_orchestrator_daemon.py` exists and contains `_apply_editable_overlay_guard`
- Commit b2633e9 exists (Task 1)
- Commit f1327b4 exists (Task 2)
- All 5 SCH-02 tests GREEN
- D-199 whitelist tests GREEN
- Framework purity GREEN
- Zero autobench/AUTOBENCH_/benchmarks/ refs in new code (grep confirms)
