---
phase: 12-scheduling-overlay-isolation
reviewed: 2026-06-12T06:27:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/automil/backends/_orchestrator_daemon.py
  - src/automil/cli/check.py
findings:
  critical: 1
  warning: 3
  info: 1
  total: 5
status: issues_found
---

# Phase 12: Code Review Report

**Reviewed:** 2026-06-12T06:27:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 12 adds SCH-01 (multi-policy `_find_best_gpu`: best_fit / round_robin / least_loaded
with hot-reload) and SCH-02 (opt-in editable-install PYTHONPATH guard: `_apply_editable_overlay_guard`
in `_launch`, `_collect_editable_source_roots` in `check.py`).

The scheduling policy dispatch (SCH-01) is structurally correct: `_rr_cursor` is accessed only
from the single daemon-loop thread (no race), the empty-candidates guard fires before the modulo
(`if not candidates: return None` at line 771), and the cursor growing unbounded is safe since
Python `int % len(candidates)` always yields a valid index. Hot-reload does not reset the cursor,
which is the documented intent.

The critical issue is in SCH-02: `_collect_editable_source_roots` calls `site.getsitepackages()`
without an `AttributeError` guard. On SLURM/HPC clusters that use the old `virtualenv` package
(still common), that attribute is monkey-patched away. When `editable_overlay_guard: true` is
set, this crashes every `_launch` call in that environment, silently marking all experiments as
crashed (the exception propagates past the call site at line 961 and is swallowed by `tick()`'s
outer `except Exception`). Framework purity is clean — no autobench/AUTOBENCH_/benchmarks refs
in code paths.

---

## Critical Issues

### CR-01: `site.getsitepackages()` unguarded — crashes all launches under old `virtualenv`

**File:** `src/automil/cli/check.py:29`
**Also:** `src/automil/backends/_orchestrator_daemon.py:961` (call site)

**Issue:** `_collect_editable_source_roots()` calls `site.getsitepackages()` directly with no
`AttributeError` guard. The old `virtualenv` package (widely deployed on SLURM/HPC clusters)
monkey-patches the `site` module and removes `getsitepackages`, causing `AttributeError` at call
time. The call chain when `editable_overlay_guard: true`:

```
tick()
  -> _launch()               # line 961 — outside the try/except at 965
    -> _apply_editable_overlay_guard()
      -> _collect_editable_source_roots()
        -> site.getsitepackages()   # AttributeError
```

The exception propagates uncaught through `_launch` (the `try` at line 965 only wraps
`subprocess.Popen`, not line 961) and is caught by `tick()`'s outer `except Exception` at
line 1898. The experiment never launches, is never marked crashed, and is silently dequeued
(queue file was already unlinked at line 913 before the crash). The experiment is permanently
lost for that daemon lifetime. Default-OFF means production users are safe today; the bug
activates exactly when the feature is opted in on the target environment (HPC cluster).

`check()` at line 205 also calls `_collect_editable_source_roots()` unconditionally, so
`automil check` also crashes under old virtualenv.

**Fix:**

```python
def _collect_editable_source_roots() -> list[str]:
    roots: list[str] = []
    try:
        site_dirs = list(site.getsitepackages())
    except AttributeError:
        # Old virtualenv monkey-patches away getsitepackages; fall back gracefully.
        site_dirs = []
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)
    # ... rest unchanged
```

---

## Warnings

### WR-01: PYTHONPATH duplication when daemon inherits PYTHONPATH from environment

**File:** `src/automil/backends/_orchestrator_daemon.py:892-894`

**Issue:** `_build_subprocess_env` includes `"PYTHONPATH"` in `_SYSTEM_ENV_WHITELIST_LITERAL`
(line 60), so the orchestrator's `os.environ["PYTHONPATH"]` is copied into `env["PYTHONPATH"]`
verbatim. `_apply_editable_overlay_guard` then prepends the worktree path to `env["PYTHONPATH"]`.
If the orchestrator's own `PYTHONPATH` already contains the worktree src path (common in
development: the operator activates the same venv, the worktree src is already on path), the
result is a duplicated entry. While Python itself handles duplicates silently, this also means
`_apply_editable_overlay_guard`'s debug log ("prepended N path(s)") fires even when no new
isolation is added, masking whether the guard actually did anything useful.

**Fix:** Deduplicate before joining:

```python
if prepends:
    existing_pp = env.get("PYTHONPATH", "")
    existing_parts = existing_pp.split(":") if existing_pp else []
    # prepend only paths not already in the existing PYTHONPATH
    new_parts = [p for p in prepends if p not in existing_parts]
    parts = new_parts + existing_parts
    if new_parts:
        env["PYTHONPATH"] = ":".join(parts)
        logger.debug(
            "editable_overlay_guard: prepended %d path(s) to PYTHONPATH for wt=%s",
            len(new_parts), wt_path,
        )
```

---

### WR-02: `_find_best_gpu` round_robin cursor not reset on policy hot-reload — undocumented skip risk

**File:** `src/automil/backends/_orchestrator_daemon.py:1814`

**Issue:** The comment says "NOT reset on policy change (see 12-RESEARCH.md Pitfall 1)" but the
actual risk of NOT resetting on a **`round_robin` -> `round_robin`** reload with a changed GPU
topology (e.g. a GPU is taken offline, reducing eligible-candidate count from 4 to 2) is that
`_rr_cursor` may be at value `7`, which mod `2` = `1` — still valid. So the cursor arithmetic
is always safe. However, when the policy changes away from `round_robin` and back (e.g.
`best_fit` -> `round_robin`), the cursor resumes from wherever it was. This is documented
intent per the comment, but the comment cites a planning doc (`12-RESEARCH.md`) that will not
exist in the shipped artifact. The rationale should be inline so future readers don't wonder
whether the non-reset is intentional or an oversight.

**Fix:** Expand the inline comment to self-document the invariant:

```python
# _rr_cursor is intentionally NOT reset on policy change or on topology change.
# Rationale: (1) resetting on policy change would cause re-visits of recently-used
# GPUs when an operator briefly flips policy and reverts; (2) cursor mod len(candidates)
# is always a valid index regardless of how many candidates are currently eligible,
# so correctness is preserved across topology changes. The counter grows unbounded
# but Python int has no overflow.
```

---

### WR-03: `check()` overlap detection false-negatives for non-Python run scripts

**File:** `src/automil/cli/check.py:212`

**Issue:** The consumer-guard heuristic `"sys.path.insert" in run_script_path.read_text()`
only works if `run_script` is a Python file. If an operator uses a shell wrapper script
(`run.script: run.sh`) that calls `python train.py`, the check reads the `.sh` file, never
finds `sys.path.insert`, returns `has_consumer_guard = False`, and fires the editable-overlap
warning spuriously. This is a false positive (not a false negative): the operator gets a
warning for a problem that doesn't exist if their train.py does the sys.path fix. The
inverse is also possible: a shell script that sets PYTHONPATH inline won't be recognized.

**Fix:** Guard by file extension before reading:

```python
elif run_script_path.exists() and run_script_path.suffix == ".py":
    has_consumer_guard = "sys.path.insert" in run_script_path.read_text()
else:
    has_consumer_guard = False  # non-Python script; can't inspect
```

Consider also reading referenced Python files from shell scripts, or add a note that the
check is Python-script-only.

---

## Info

### IN-01: Module-level import of `_collect_editable_source_roots` couples daemon to CLI layer

**File:** `src/automil/backends/_orchestrator_daemon.py:31`

**Issue:** `from automil.cli.check import _collect_editable_source_roots` is a module-level
import. This is a layering concern: the backends layer now imports from the cli layer at
module load time. The function itself has no CLI dependency (it only uses `site` and
`pathlib`), so it is in the wrong module. The current arrangement works because
`automil.cli.orchestrator` does not import `_orchestrator_daemon` directly (no circular
import, confirmed by import test), but this is a fragile invariant — any future CLI submodule
that imports the daemon will create a cycle.

**Fix:** Move `_collect_editable_source_roots` to `automil/utils.py` (or a new
`automil/editable.py`), import it from there in both `check.py` and `_orchestrator_daemon.py`.
This resolves the layering and makes the function independently testable without instantiating
CLI infrastructure.

---

_Reviewed: 2026-06-12T06:27:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
