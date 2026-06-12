---
phase: 13-cli-lifecycle-operability
plan: "02"
subsystem: cli/cancel
tags: [ops-01, cancel, process-management, pid-reuse, zombie, sigterm, sigkill]
dependency_graph:
  requires: [13-01]
  provides: [OPS-01-impl]
  affects: [src/automil/cli/cancel.py]
tech_stack:
  added: []
  patterns:
    - "zombie-aware liveness: /proc/<pid>/stat state field + os.waitpid(WNOHANG)"
    - "SIGTERM → 5s grace → SIGKILL escalation mirroring daemon pattern"
    - "lazy import of automil.orchestrator helpers inside function body (D-69)"
key_files:
  created: []
  modified:
    - src/automil/cli/cancel.py
    - tests/test_cli_cancel_resubmit.py
decisions:
  - "Zombie-aware _is_alive: read /proc/<pid>/stat state field; treat 'Z' as dead to avoid grace-loop spinning on already-killed process"
  - "os.waitpid(WNOHANG) after kill to reap zombie children when cancel and target share the same parent process (in-process test runners); ChildProcessError silently ignored when not a child"
  - "Removed @pytest.mark.xfail from test_cancel_local_direct_kill and test_cancel_no_starttime_ticks — both turn GREEN with real subprocess kill"
  - "Non-Linux fallback: when /proc unavailable (_proc_state returns None), use os.kill(pid, 0) for liveness without starttime cross-check (documented residual PID-reuse risk)"
metrics:
  duration_minutes: 12
  completed_date: "2026-06-12"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 2
---

# Phase 13 Plan 02: OPS-01 cancel.py direct-kill branch — Summary

**One-liner:** Direct-kill branch in cancel.py sends os.killpg(SIGTERM→SIGKILL) using on-disk pid/pgid metadata, zombie-aware via /proc state + WNOHANG reaping.

## What Was Built

Extended `src/automil/cli/cancel.py` with a direct-kill path (OPS-01) that fires when the
running spec has no top-level `opaque_id` but does carry `metadata.pid` / `metadata.pgid`.
This path is the correct fix for daemon-launched local jobs: the daemon's `_kill_experiment`
resolves targets via the in-memory `self.running` map, which is always empty in a fresh CLI
process, making the old routing a guaranteed no-op.

**Three-part implementation:**

**Part A — Relaxed opaque_id hard-fail (D-01/D-02/D-03)**
Replaced the unconditional `raise` on missing `opaque_id` with a conditional: loud-fail only
when the spec has NEITHER `opaque_id` NOR `metadata.pid`/`metadata.pgid`. Reads
`metadata.starttime_ticks` (optional; absent on non-Linux).

**Part B — Direct-kill branch**
When `not opaque_id`: reads pid/pgid/starttime from the spec and signals the process group
directly. Uses a zombie-aware `_is_alive` helper that reads `/proc/<pid>/stat` state field —
treating `Z` (zombie) as dead so the grace loop does not spin on an already-killed process.
Signal escalation: SIGTERM → 5s grace with 200ms poll → SIGKILL if still alive → 10 × 100ms
post-SIGKILL poll → hard-fail if still alive after all.

**Part C — Zombie reaping after kill**
After the signal sequence, calls `os.waitpid(pid, os.WNOHANG)` to reap the child if the CLI
process is the parent (which is the case when CliRunner runs in-process, as in tests). This
converts the zombie state into a fully-gone PID so `os.kill(pid, 0)` raises `ProcessLookupError`
as test assertions and callers expect. `ChildProcessError` is silently swallowed for the case
where cancel is not the parent (production daemon-launched jobs).

**opaque_id path (Steps 5-7) preserved unchanged** in the `else` branch.

**Test markers removed:** `@pytest.mark.xfail` stripped from `test_cancel_local_direct_kill`
and `test_cancel_no_starttime_ticks` — both now pass with real subprocess verification.

## Tasks

| # | Name | Commit | Files |
|---|------|--------|-------|
| 1 | Extend cancel.py with direct-kill branch (OPS-01) | 034b552 | src/automil/cli/cancel.py, tests/test_cli_cancel_resubmit.py |

## Verification

```
uv run pytest tests/test_cli_cancel_resubmit.py -v
```

Results:
- `test_cancel_happy_path` PASSED (opaque_id regression clean)
- `test_cancel_unknown_node` PASSED
- `test_cancel_terminal_node` PASSED
- `test_cancel_missing_running_spec` PASSED
- `test_cancel_timeout` PASSED
- `test_resubmit_happy_path` PASSED
- `test_cancel_local_direct_kill` **PASSED** (real sleep process killed; ProcessLookupError confirmed)
- `test_cancel_missing_pid_metadata` PASSED (corrupted-state hard-fail)
- `test_cancel_no_starttime_ticks` **PASSED** (pid/pgid-only kill without starttime_ticks)

Full suite: `uv run pytest tests/ -q` → 1045 passed, 53 skipped, 8 xfailed, **1 pre-existing failure** (`test_d208_clause_11_state_roadmap_complete`) unrelated to this plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Zombie-state false-positive in _is_alive**

- **Found during:** Task 1 verification
- **Issue:** `_is_pid_alive_with_starttime` reads `/proc/<pid>/stat` which exists for zombie
  processes (state `Z`). After `os.killpg(SIGTERM)`, the child becomes a zombie (killed but
  not reaped by parent). `_is_alive` returned `True`, the grace loop spun until timeout, then
  `SIGKILL` was sent — but even after SIGKILL the zombie remained (SIGKILL does not force-reap;
  only `wait()` clears it). The cancel command raised a hard-fail error.
- **Fix:** `_proc_state(pid)` reads the state character from `/proc/<pid>/stat`; `_is_alive`
  returns `False` immediately when state is `Z`. Added `_try_reap(pid)` using
  `os.waitpid(pid, os.WNOHANG)` after the kill sequence to reap the zombie when cancel is the
  parent process, so `os.kill(pid, 0)` subsequently raises `ProcessLookupError` as the test
  and callers expect.
- **Files modified:** src/automil/cli/cancel.py
- **Commit:** 034b552

## Threat Surface Scan

T-13-02-01 (PID reuse) mitigated: `_is_pid_alive_with_starttime` from `automil.orchestrator`
cross-checks starttime_ticks when present. Zombie-state detection (state `Z`) and
`os.waitpid(WNOHANG)` are local to cancel.py and do not touch the shared helper.

No new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Known Stubs

None — all kill paths are fully wired.

## Self-Check

- [x] `src/automil/cli/cancel.py` exists and contains `os.killpg`
- [x] `tests/test_cli_cancel_resubmit.py` xfail markers removed; 9 tests pass
- [x] Commit 034b552 exists in git log
- [x] Framework suite: no new failures vs baseline
