---
phase: 12-scheduling-overlay-isolation
fixed_at: 2026-06-12T06:35:00Z
review_path: .planning/phases/12-scheduling-overlay-isolation/12-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 1
status: partial
---

# Phase 12: Code Review Fix Report

**Fixed at:** 2026-06-12T06:35:00Z
**Source review:** `.planning/phases/12-scheduling-overlay-isolation/12-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope (critical + warning): 4
- Fixed: 4
- Skipped: 1 (IN-01 — intentional per task instructions; code comment added instead)
- Stale acceptance test also fixed: 1

---

## Fixed Issues

### CR-01: `site.getsitepackages()` unguarded — crashes all launches under old virtualenv

**Files modified:** `src/automil/cli/check.py`, `tests/test_editable_overlay_guard.py`
**Commit:** `34a0ea4`
**Applied fix:** Wrapped `site.getsitepackages()` in `try/except AttributeError` with
a `site_dirs = []` fallback and an explanatory comment about old virtualenv on SLURM/HPC.
Added `test_collect_editable_source_roots_handles_missing_getsitepackages` to
`test_editable_overlay_guard.py` which monkeypatches `getsitepackages` to raise
`AttributeError` and asserts the function returns a list without crashing.

---

### WR-01: PYTHONPATH duplication when daemon inherits PYTHONPATH from environment

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `f0f0df6`
**Applied fix:** In `_apply_editable_overlay_guard`, split the existing PYTHONPATH into
parts and compute `new_parts = [p for p in prepends if p not in existing_parts]`. Only
update `env["PYTHONPATH"]` and emit the debug log when `new_parts` is non-empty, so
duplicate entries are prevented and the log is suppressed when no new isolation is added.

---

### WR-02: `_rr_cursor` non-reset comment cites planning artifact `12-RESEARCH.md`

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `80e7b6f`
**Applied fix:** Replaced the single-line comment referencing `12-RESEARCH.md Pitfall 1`
with a self-contained 5-line explanation covering: (1) why resetting on policy change
would cause GPU re-visits, (2) why `cursor % len(candidates)` is always valid across
topology changes, and (3) the note that Python int has no overflow.

---

### WR-03: `check()` overlap detection false-positives for non-Python run scripts

**Files modified:** `src/automil/cli/check.py`
**Commit:** `48a2b00`
**Applied fix:** Changed the `elif run_script_path.exists():` branch to
`elif run_script_path.exists() and run_script_path.suffix == ".py":` with a comment
explaining that shell wrappers never contain `sys.path.insert` and would always produce
a false positive. Non-Python scripts fall through to `has_consumer_guard = False` with
a note that content cannot be inspected.

---

### Stale Acceptance Test (Phase 12 exposed): clause-07 hard-coded daemon line `:54`

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`,
`tests/test_framework_purity.py`, `tests/acceptance/test_phase8_acceptance.py`
**Commit:** `2972682`
**Applied fix (combined with IN-01 comment):**
- Added a 6-line inline comment block on the `_collect_editable_source_roots` import
  in the daemon (IN-01 deferred — see below), which shifted the allowlisted
  `AUTOBENCH_*_ROOT` comment from line 56 to line **62**.
- Updated `_ALLOWLIST` key in `test_framework_purity.py` from `:56` to `:62` with
  a drift-tracking note.
- Updated clause-07 assertion in `test_phase8_acceptance.py` from the stale `:54` to
  `:62` (the current value matching `_ALLOWLIST`), and added a code comment flagging
  the line-number brittleness for Phase 14 / DBT-03 anchor cleanup.

Final daemon allowlist line number: **`:62`**

---

## Skipped Issues

### IN-01: Module-level import of `_collect_editable_source_roots` couples daemon to CLI layer

**File:** `src/automil/backends/_orchestrator_daemon.py:31`
**Reason:** Intentionally not moved per task instructions — "leave it; add a brief code
comment noting it could move to a neutral module later." Moving it would shift daemon
line numbers and re-break the allowlist/acceptance anchors. A 6-line inline comment was
added explaining the layering concern and deferring the refactor to Phase 14 / DBT-03.
The allowlist and clause-07 were updated to account for the line shift caused by the
comment addition.
**Original issue:** `_collect_editable_source_roots` is in `automil.cli.check` but has
no CLI dependency; importing it from backends couples the layers and is a fragile
invariant against future circular imports.

---

## Test Gate Results

All required tests pass:

| Test file | Result |
|-----------|--------|
| `tests/test_editable_overlay_guard.py` | 6 passed (5 original + 1 new CR-01 test) |
| `tests/test_scheduling_policy.py` | 7 passed |
| `tests/test_orchestrator_env_whitelist.py` | 13 passed |
| `tests/test_framework_purity.py` | 3 passed |
| `tests/acceptance/test_phase8_acceptance.py::test_d208_clause_07_framework_purity_grep_gate` | PASSED |

Full framework suite (excluding benchmarks): **1035 passed, 4 failed, 54 skipped**

Pre-existing failures (not caused by these fixes):
- `test_d208_clause_11_state_roadmap_complete` — v1.0 milestone-rotation stale test (do not touch)
- `test_apl01_iris_dispatch.py` (3 tests) — `sklearn` not installed in CI venv; pre-existing

---

_Fixed: 2026-06-12T06:35:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
