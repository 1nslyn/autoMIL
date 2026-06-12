---
phase: 12-scheduling-overlay-isolation
verified: 2026-06-12T10:25:00Z
status: passed
score: 2/2 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
deferred: []
human_verification: []
---

# Phase 12: Scheduling Overlay Isolation Verification Report

**Phase Goal:** Configurable GPU placement policy so compute-bound jobs don't over-stack one GPU; daemon guards editable-installed consumer packages so experiments import from their worktree overlay (with automil check warning when missing).
**Verified:** 2026-06-12T10:25:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SCH-01: `_find_best_gpu` dispatches on `self.scheduling_policy` (best_fit/round_robin/least_loaded); best_fit default preserved; round_robin uses `_rr_cursor` not reset on hot-reload; unknown policy falls back to best_fit + warning; knob in `__init__` + `_reload_orchestrator_config`; present in config.yaml.j2 | VERIFIED | `_orchestrator_daemon.py:45,441-442,774-792,1806-1814`; `config.yaml.j2:67` |
| 2 | SCH-02: `automil check` warns when editable src root overlaps overlay and no guard present (generic `_editable_impl_*.pth` scan); opt-in injection (`editable_overlay_guard`, default OFF) is post-processing in `_launch` AFTER `_build_subprocess_env` returns; D-199 non-regression: PYTHONPATH/AUTOBENCH_ROOT NOT auto-injected | VERIFIED | `check.py:15-49,203-239`; `_orchestrator_daemon.py:863-898,952-961`; 27/27 tests pass |

**Score:** 2/2 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/automil/backends/_orchestrator_daemon.py` | SCH-01 dispatch + SCH-02 guard method + init attrs + hot-reload | VERIFIED | `SCHEDULING_POLICY="best_fit"` at line 45; `self.scheduling_policy` + `self._rr_cursor` at lines 441-442; `self.editable_overlay_guard=False` at lines 446-448; `_find_best_gpu` dispatch at lines 774-792; `_apply_editable_overlay_guard` method at lines 863-898; `_reload_orchestrator_config` hot-reloads both at lines 1806-1821 |
| `src/automil/cli/check.py` | `_collect_editable_source_roots()` helper + SCH-02 warning block | VERIFIED | Helper at lines 15-49; warning block at lines 203-239; warning text contains "editable" and "Worktree" (line 234); three `.pth` patterns scanned; `OSError` caught on read |
| `src/automil/templates/config.yaml.j2` | `scheduling_policy: "best_fit"` in orchestrator section | VERIFIED | Line 67: `scheduling_policy: "best_fit"   # best_fit \| round_robin \| least_loaded` |
| `tests/test_scheduling_policy.py` | 7 GREEN tests for SCH-01 | VERIFIED | All 7 pass (not xfail) |
| `tests/test_editable_overlay_guard.py` | 5 GREEN tests for SCH-02 | VERIFIED | All 5 pass (not xfail) |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_find_best_gpu` | `self.scheduling_policy` | `if/elif` dispatch | WIRED | Line 774: `policy = self.scheduling_policy`; branches at 775, 779, 785 |
| `_reload_orchestrator_config` | `self.scheduling_policy` | `orch_cfg.get("scheduling_policy", self.scheduling_policy)` | WIRED | Lines 1806-1813; `_rr_cursor` explicitly NOT reset (line 1814 comment) |
| `_launch` | `_apply_editable_overlay_guard` | post-processing call after `_build_subprocess_env` | WIRED | Line 952: `env = self._build_subprocess_env(...)`; line 961: `self._apply_editable_overlay_guard(env=env, wt_path=wt_path)`; line 963: `log_path = archive / "run.log"` — order confirmed |
| `_apply_editable_overlay_guard` | `env["PYTHONPATH"]` | `if self.editable_overlay_guard:` guard | WIRED | Line 879: early return when False (default); line 894: prepend only when True + wt_candidate exists |
| `_collect_editable_source_roots` | `automil.backends._orchestrator_daemon` module namespace | module-level import at line 31 | WIRED | `from automil.cli.check import _collect_editable_source_roots` at line 31 — patchable by tests |
| `check()` warning | `_collect_editable_source_roots()` | called at line 205 | WIRED | Warning fires iff overlap AND NOT `has_consumer_guard` AND NOT `overlay_guard_enabled` |

---

## Data-Flow Trace (Level 4)

Not applicable — phase delivers configuration knobs and a diagnostic warning, not a data-rendering component.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| SCH-01: 7 policy dispatch tests pass | `uv run pytest tests/test_scheduling_policy.py -v` | 7 passed | PASS |
| SCH-02: 5 editable guard tests pass | `uv run pytest tests/test_editable_overlay_guard.py -v` | 5 passed | PASS |
| D-199 non-regression: PYTHONPATH/AUTOBENCH_ROOT not auto-injected | `uv run pytest tests/test_orchestrator_env_whitelist.py -v` | 13 passed (includes `test_pythonpath_not_auto_injected_phase8` and `test_autobench_root_not_auto_injected_phase8`) | PASS |
| D-206 framework purity: no consumer refs in new code | `uv run pytest tests/test_framework_purity.py -v` | 3 passed | PASS |
| Full gate (all four files) | `uv run pytest tests/test_scheduling_policy.py tests/test_editable_overlay_guard.py tests/test_orchestrator_env_whitelist.py tests/test_framework_purity.py -v` | 27 passed in 0.87s | PASS |

---

## Probe Execution

No probe scripts declared for this phase.

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SCH-01 | 12-02-PLAN.md | Configurable GPU placement policy (best_fit/round_robin/least_loaded) with hot-reload | SATISFIED | `SCHEDULING_POLICY` constant at daemon:45; dispatch at daemon:774-792; hot-reload at daemon:1806-1813; `_rr_cursor` sticky across reloads (daemon:1814); config.yaml.j2:67; 7 tests GREEN |
| SCH-02 | 12-03-PLAN.md | Editable overlay guard: `automil check` warning + opt-in daemon PYTHONPATH injection | SATISFIED | `_collect_editable_source_roots()` at check.py:15-49; warning block at check.py:203-239; `_apply_editable_overlay_guard` method at daemon:863-898; post-processing call at daemon:961; default OFF confirmed (line 447); D-199 invariant preserved |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/automil/backends/_orchestrator_daemon.py` | 56 | `AUTOBENCH_*_ROOT` in a comment | Info | Pre-existing D-199/DEC-01 comment in the `_SYSTEM_ENV_WHITELIST_LITERAL` block; present before Phase 12; in the framework-purity allowlist; not new code introduced by this phase |

No TBD, FIXME, or XXX markers found in Phase 12 modified files. No stubs. No hardcoded empty returns in new code paths.

---

## Human Verification Required

None. All SCH-01 and SCH-02 behaviors are verified programmatically via the test suite.

---

## SCH-01 Detail: Critical Sub-Criteria Verification

| Sub-criterion | Evidence | Status |
|---------------|----------|--------|
| `_find_best_gpu` dispatches on `self.scheduling_policy` | `policy = self.scheduling_policy` at line 774; `if policy == "least_loaded":` at 775, `elif policy == "round_robin":` at 779, `else:` at 785 | VERIFIED |
| `best_fit` is the default | `SCHEDULING_POLICY = "best_fit"` at line 45; `orch_cfg.get("scheduling_policy", SCHEDULING_POLICY)` at line 441 | VERIFIED |
| `round_robin` cycles via `_rr_cursor` | Lines 781-783: sort by index, `candidates[self._rr_cursor % len(candidates)][0]`, `self._rr_cursor += 1` | VERIFIED |
| `_rr_cursor` NOT reset on hot-reload | Line 1814 comment: `# NOTE: self._rr_cursor is NOT reset on policy change`; no reset assignment in `_reload_orchestrator_config` | VERIFIED |
| Unknown policy → best_fit fallback + warning | Lines 786-789: `if policy != "best_fit": logger.warning("Unknown scheduling_policy %r; falling back to best_fit", policy)` | VERIFIED |
| Knob hot-reloaded in `_reload_orchestrator_config` | Lines 1806-1813: `orch_cfg.get("scheduling_policy", self.scheduling_policy)` with `logger.info` on change | VERIFIED |
| Present in `config.yaml.j2` | Line 67: `scheduling_policy: "best_fit"   # best_fit \| round_robin \| least_loaded` | VERIFIED |

---

## SCH-02 Detail: Critical Sub-Criteria Verification

| Sub-criterion | Evidence | Status |
|---------------|----------|--------|
| `automil check` warns on editable overlap + no guard | check.py:232-238: `warnings.append(f"files.editable includes paths under editable-installed package source root '{root}'. Worktree overlays...shadowed...Fix: add sys.path.insert...")` | VERIFIED |
| Warning suppressed when `editable_overlay_guard: true` | check.py:231: `if overlap and not has_consumer_guard and not overlay_guard_enabled:` | VERIFIED |
| Warning suppressed when `sys.path.insert` in run script | check.py:212: `has_consumer_guard = "sys.path.insert" in run_script_path.read_text()` | VERIFIED |
| Generic `_editable_impl_*.pth` scan — NO consumer names | check.py:38: patterns are `"_editable_impl_*.pth"`, `"__editable__*.pth"`, `"*.egg-link"` — no autobench/benchmarks refs | VERIFIED |
| Injection is post-processing AFTER `_build_subprocess_env` | daemon:952-961: `env = self._build_subprocess_env(...)` then `self._apply_editable_overlay_guard(env=env, wt_path=wt_path)` then `log_path = archive / "run.log"` at 963 | VERIFIED |
| `_build_subprocess_env` signature unchanged | Method at daemon:804; call at daemon:952-957 — no new parameters | VERIFIED |
| Default OFF (D-199 non-regression) | daemon:447: `orch_cfg.get("editable_overlay_guard", False)`; daemon:879: `if not self.editable_overlay_guard: return` | VERIFIED |
| `test_pythonpath_not_auto_injected_phase8` GREEN | Test suite output: PASSED | VERIFIED |
| `test_autobench_root_not_auto_injected_phase8` GREEN | Test suite output: PASSED | VERIFIED |
| Zero consumer refs in new `src/automil/` code | `grep autobench\|AUTOBENCH_\|benchmarks/ src/automil/cli/check.py` → 0 results; daemon line 56 is a pre-existing allowlisted comment | VERIFIED |

---

## Gaps Summary

No gaps. All must-haves verified against the real codebase. Phase goal achieved.

---

_Verified: 2026-06-12T10:25:00Z_
_Verifier: Claude (gsd-verifier)_
