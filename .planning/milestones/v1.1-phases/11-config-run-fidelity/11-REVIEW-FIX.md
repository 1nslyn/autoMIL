---
phase: 11-config-run-fidelity
fixed_at: 2026-06-11T16:30:00-04:00
review_path: .planning/phases/11-config-run-fidelity/11-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 4
skipped: 1
status: partial
---

# Phase 11: Code Review Fix Report

**Fixed at:** 2026-06-11T16:30:00-04:00
**Source review:** `.planning/phases/11-config-run-fidelity/11-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (3 Warning + 2 Info)
- Fixed: 4 (WR-01, WR-02, WR-03, IN-01)
- Skipped: 1 (IN-02 — code is correct, documentation observation only)

---

## Fixed Issues

### WR-01: `shlex.split(override)` validates at CLI submit time

**Files modified:** `src/automil/cli/submit.py`, `tests/test_cfg_run_fidelity.py`
**Commits:** `a817b0d` (production fix), `3fe5e3b` (tests)
**Applied fix:** Added `import shlex` to submit.py and a `try/except ValueError`
block that validates `shlex.split(override)` before writing the spec to queue/.
On unbalanced quotes, raises `click.ClickException(f"--override contains unbalanced
quotes and cannot be parsed: {exc}")`. The spec is never written when validation
fails. Added `TestWR01OverrideShlex` class to `test_cfg_run_fidelity.py` with
three tests: malformed quotes exits non-zero with no spec written, valid quotes
succeeds and writes `run_command_override`, and `--max-time 0` is rejected (WR-03
regression guard bundled here).

---

### WR-02: Backend-aware path in `_was_cap_killed_completion` and cleanup sites

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`, `tests/test_tick_cells.py`
**Commit:** `a68cebe`
**Applied fix:** Fixed all three `self.running_dir` read/unlink sites plus IN-01
(`_read_fold_count_for_node`) to use backend-aware resolution:

**Backend-aware resolution used:**
```python
_backend = self._read_backend_name_for_node(node_id)
path = self._backend_running_dir(_backend) / f"{node_id}.json"
```

`_read_backend_name_for_node` iterates `running/local/`, `running/slurm/`,
`running/ray/` and falls back to `archive/<node>/spec.json` — mirroring exactly
how `_tick_cells` WRITES the annotation. This is the same pattern already used
at L1044-1045 (the WRITE side fixed in Phase 9 WR-04).

**Four sites fixed:**
1. `_was_cap_killed_completion` (L1280) — now reads from correct backend dir
2. `_handle_completion` cleanup (L1252) — now unlinks correct backend dir
3. `_handle_cap_killed_completion` cleanup (L1343) — now unlinks correct backend dir
4. `_read_fold_count_for_node` (L1136, IN-01) — now reads from correct backend dir first

**Tests added to `test_tick_cells.py`:**
- `test_was_cap_killed_completion_detects_slurm_annotation`: writes
  `running/slurm/<node>.json` with `cancel_reason='cap'`, asserts method returns
  `True` without any `running/local/<node>.json` present (proves the backend-aware
  path is used, not `self.running_dir`)
- `test_was_cap_killed_completion_local_still_works`: regression guard confirming
  local-backend annotations still detected after the fix

**Closes Phase-9 WR-04 residue** (the writer was fixed in Phase 9; the three
reader/cleanup sites were not updated until this commit).

---

### WR-03: Reject `--max-time <= 0` with ClickException

**Files modified:** `src/automil/cli/submit.py`, `tests/cli/test_submit_max_time.py`, `tests/test_cfg_run_fidelity.py`
**Commits:** `b2c1a4c` (production fix), `7b41058` (existing test update), `3fe5e3b` (new test)
**Applied fix:** Changed guard from `max_time_seconds < 0` to `max_time_seconds <= 0`
and updated error message from `"must be non-negative seconds"` to `"must be > 0
seconds, got {max_time_seconds}"`. Zero seconds is semantically nonsensical for a
training timeout. Also updated `test_max_time_negative_rejected` in
`tests/cli/test_submit_max_time.py` to assert `"must be > 0"` (was `"must be
non-negative"`) so the pre-existing test does not regress.

---

### IN-01: `_read_fold_count_for_node` uses backend-aware path

**Files modified:** `src/automil/backends/_orchestrator_daemon.py`
**Commit:** `a68cebe` (bundled with WR-02)
**Applied fix:** The first lookup path in `_read_fold_count_for_node` was
`self.running_dir / f"{node_id}.json"` (= `running/local/`). For SLURM/Ray nodes
the running spec lives in `running/<backend>/` so the lookup silently fell through
to the archive-spec fallback. Fixed to use `_read_backend_name_for_node()` +
`_backend_running_dir()` as the primary lookup, same as the other sites fixed for
WR-02.

---

## Skipped Issues

### IN-02: CFG-01 dict-comprehension filter excludes `no_wandb` unnecessarily

**File:** `benchmarks/scripts/run_experiment.py:157-163`
**Reason:** No code change needed. The REVIEW.md explicitly states "This is a
documentation observation, not a bug: the code is correct." The `args.no_wandb`
boolean is consumed separately from the `_train_overrides` dict-comprehension
(line 230), which is the correct pattern. No fix was applied.

---

## Test Results

**Main suite (`uv run pytest tests/ -q`):**
- 1022 passed, 54 skipped, 1 failed
- Only failure: `tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` (pre-existing, expected)
- Note: `tests/test_apl01_iris_dispatch.py` (3 tests) fail in the isolated worktree venv due to `sklearn` not being a `automil` dependency, but pass in the main repo venv (confirmed: 3 passed when run from `/home/jma/Documents/yinshuol/autoMIL/`)

**Benchmarks suite (`uv run pytest benchmarks/tests/ -q`):**
- 292 passed, 1 skipped, 0 failed

**No new regressions introduced.**

---

_Fixed: 2026-06-11T16:30:00-04:00_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
