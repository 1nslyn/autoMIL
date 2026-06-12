---
phase: 11-config-run-fidelity
reviewed: 2026-06-11T16:10:00-04:00
depth: standard
files_reviewed: 3
files_reviewed_list:
  - src/automil/cli/submit.py
  - src/automil/backends/_orchestrator_daemon.py
  - benchmarks/scripts/run_experiment.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 11: Code Review Report

**Reviewed:** 2026-06-11T16:10:00-04:00
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 11 implements three config-fidelity fixes: CFG-01 (run_experiment.py None-default
overrides), CFG-02 (submit --timeout default→None + --max-time sentinel change), and
CFG-03 (daemon suffix-append of per-node run_command_override). The core logic for all
three is correct. No critical bugs were found. Three warnings follow, one of which is a
pre-existing carry-over that Phase 11 explicitly touched but did not close.

---

## Warnings

### WR-01: `shlex.split(override_str)` raises `ValueError` on malformed override — daemon crashes the launch silently but correctly; user gets no CLI-time feedback

**File:** `src/automil/backends/_orchestrator_daemon.py:908`
**Also:** `src/automil/cli/submit.py:458-459`

**Issue:** `shlex.split(override)` at daemon launch time raises `ValueError: No closing
quotation` when `override_str` contains unbalanced quotes (e.g. `--desc 'foo bar`). This
IS caught by the `except Exception` block at line 917 and routes through `_mark_crashed`,
so the daemon does not crash. However: (a) the error surfaces at run time, not at
`automil submit` time, meaning the user gets a crash-status experiment instead of an
immediate CLI error; (b) the spec is already dequeued before Popen is called (line
848-851), so any subsequent retry would require a full re-submit. There is no validation
of `override` at submit time.

**Fix:** Add a `shlex.split()` validation step in `submit.py` before writing the spec,
raising `ClickException` on `ValueError`:
```python
# In submit() around line 458, before spec["run_command_override"] = override
if override is not None:
    try:
        shlex.split(override)
    except ValueError as exc:
        raise click.ClickException(
            f"--override contains unbalanced quotes and cannot be parsed: {exc}"
        ) from exc
    spec["run_command_override"] = override
```

---

### WR-02: `_was_cap_killed_completion` and completion-cleanup paths use `self.running_dir` (alias for `running/local/`) — always misses SLURM/Ray cap-killed annotations

**File:** `src/automil/backends/_orchestrator_daemon.py:1280, 1252, 1343`

**Issue:** Three uses of `self.running_dir` (the `running/local/` alias) in
post-completion logic were supposed to use the per-backend helper per CLAUDE.md
("New code MUST call `_backend_running_dir(backend_name)` instead"). Specifically:

- **Line 1280** (`_was_cap_killed_completion`): reads `self.running_dir / f"{node_id}.json"` to check for `cancel_reason='cap'`. For a SLURM or Ray job the annotation was written to `running/slurm/<node>.json` by `_tick_cells`, so this path always returns `False` → `_handle_cap_killed_completion` is never called for non-local backends → SLURM/Ray cap-triggered cancels fall through to the normal completion path and are never reconciled as budget-kills.
- **Lines 1252 + 1343** (`_handle_completion` and `_handle_cap_killed_completion` running-spec cleanup): always tries to unlink `running/local/<node>.json` regardless of backend. For SLURM jobs the file lives in `running/slurm/` and is never cleaned up.

These are pre-existing Phase 9 WR-04 partial fixes — `_tick_cells` was corrected to use
`_backend_running_dir`, but the three consumer sites were not updated. Phase 11 touched
the daemon file (CFG-03 append block at line 903-908) and thus had the opportunity to
close these.

**Fix for line 1280:**
```python
def _was_cap_killed_completion(self, node_id: str) -> bool:
    backend_name = self._read_backend_name_for_node(node_id)
    for _spec_path in (
        self._backend_running_dir(backend_name) / f"{node_id}.json",
        self.archive_dir / node_id / "spec.json",
    ):
        ...
```

**Fix for lines 1252 + 1343:** replace `self.running_dir` with:
```python
_backend_name = self._read_backend_name_for_node(node_id)
running_spec = self._backend_running_dir(_backend_name) / f"{node_id}.json"
```

---

### WR-03: `--max-time 0` silently becomes `timeout_min=1` with no user warning

**File:** `src/automil/cli/submit.py:60-64`

**Issue:** The guard at line 60 allows `max_time_seconds == 0` (`< 0` check, zero
passes). `max(1, (0 + 59) // 60)` → `1`. So `--max-time 0` silently queues a 1-minute
job instead of rejecting the obviously-wrong input or at least warning. Zero seconds is
almost certainly a user error (omitted value, scripting bug). The current D-195 spec says
"non-negative seconds" but zero is semantically nonsensical for a training timeout.

**Fix:** Change the guard to `<= 0` (reject zero), or at minimum warn:
```python
if max_time_seconds <= 0:
    raise click.ClickException(
        f"--max-time must be > 0 seconds, got {max_time_seconds}"
    )
```
This is a minor semantic tightening; if zero is intentionally allowed (e.g. test
harnesses passing 0 as a sentinel), document it.

---

## Info

### IN-01: `_read_fold_count_for_node` reads from `self.running_dir` (running/local/) only — fold count silently defaults to 5 for SLURM/Ray nodes

**File:** `src/automil/backends/_orchestrator_daemon.py:1136`

**Issue:** The first lookup path is `self.running_dir / f"{node_id}.json"` which only
covers local-backend nodes. SLURM/Ray running specs live in `running/slurm/` etc. The
second fallback reads `archive/<node>/spec.json` which does exist after _launch writes it,
so this usually recovers correctly — but only after the running-spec lookup fails
silently. This is a latent inconsistency rather than a correctness bug in practice
(archive spec fallback saves it), but is worth noting since the method is called inside
`_handle_cap_killed_completion` for correct fold reconciliation.

---

### IN-02: CFG-01 dict-comprehension filter in `run_experiment.py` excludes `no_wandb` unnecessarily — but `no_wandb` is `store_true`, never None

**File:** `benchmarks/scripts/run_experiment.py:157-163`

**Issue:** The `_train_overrides` dict-comprehension filters `{k: v ... if v is not None}`
but `args.no_wandb` (line 230) is passed directly without going through the filter —
which is correct since it's consumed as a boolean flag, not a TrainConfig field. However,
the comment at line 62 says "default=None so dataclass defaults are honored when not
supplied (CFG-01 / D-01)" and `--no_wandb` uses `store_true` (never None), so the
pattern is consistent. This is a documentation observation, not a bug: the code is
correct.

---

_Reviewed: 2026-06-11T16:10:00-04:00_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
