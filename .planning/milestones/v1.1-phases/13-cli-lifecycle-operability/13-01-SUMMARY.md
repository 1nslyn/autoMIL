---
phase: 13-cli-lifecycle-operability
plan: "01"
subsystem: tests
tags: [tdd, red-stubs, ops, cancel, dequeue, submit, viz, project-option]
dependency_graph:
  requires: []
  provides:
    - "OPS-01 RED xfail stubs in tests/test_cli_cancel_resubmit.py"
    - "OPS-02 RED xfail stubs in tests/test_cli_dequeue.py (new file)"
    - "OPS-03 RED xfail stub in tests/test_cli.py"
    - "OPS-04 RED xfail stubs in tests/test_cli_project_option.py (new file)"
    - "OPS-05 RED xfail stub in tests/test_viz_port_config.py (new file)"
  affects:
    - "13-02-PLAN.md (OPS-01/04/05 implementation picks up these stubs)"
    - "13-03-PLAN.md (OPS-02/03 implementation picks up these stubs)"
tech_stack:
  added: []
  patterns:
    - "Wave-0 Nyquist RED stubs (xfail strict)"
    - "Anti-theater real subprocess for OPS-01 (subprocess.Popen + os.kill liveness check)"
    - "Conditional _PROJECT_OVERRIDE autouse teardown fixture for OPS-04"
    - "unittest.mock.patch of cmd_start for OPS-05 port capture"
key_files:
  created:
    - tests/test_cli_dequeue.py
    - tests/test_cli_project_option.py
    - tests/test_viz_port_config.py
  modified:
    - tests/test_cli_cancel_resubmit.py
    - tests/test_cli.py
decisions:
  - "test_cancel_missing_pid_metadata: xfail removed — existing cancel.py already hard-fails correctly when spec has neither opaque_id nor metadata.pid/pgid; this is a regression guard, not a RED stub"
  - "test_project_option_absent_cwd_walk: not xfail — existing cwd-walk behaviour already correct; regression guard only"
  - "test_viz_port_default + test_viz_port_explicit_overrides_config: not xfail — current hard-coded default (8420) and pass-through of explicit --port already correct; only test_viz_port_from_config is the true RED stub for OPS-05"
  - "test_dequeue_unknown_node assertion: added 'No such command not in output' guard so test xfails for the right reason (node-not-found logic absent) not the wrong reason (command not registered)"
  - "OPS-04 assertions: added 'No such option not in output' guard so xfail fires on missing option registration, not on an irrelevant Click usage error"
metrics:
  duration_minutes: 11
  completed_date: "2026-06-12T11:29:56Z"
  tasks_completed: 2
  files_changed: 5
---

# Phase 13 Plan 01: Wave-0 RED Stubs Summary

Wave-0 Nyquist compliance: 10 xfail stubs + 4 passing regression guards across 5 test files, establishing the test scaffolding for all 5 OPS requirements before any implementation touches production code.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | OPS-01 RED stubs in test_cli_cancel_resubmit.py | 182d042 | tests/test_cli_cancel_resubmit.py |
| 2 | OPS-02/03/04/05 RED stubs (3 new files + 1 extension) | b7b45aa | tests/test_cli_dequeue.py, tests/test_cli.py, tests/test_cli_project_option.py, tests/test_viz_port_config.py |

## Stub Inventory

| Test | File | Requirement | Status |
|------|------|-------------|--------|
| test_cancel_local_direct_kill | test_cli_cancel_resubmit.py | OPS-01 | XFAIL (strict) |
| test_cancel_missing_pid_metadata | test_cli_cancel_resubmit.py | OPS-01 | PASSED (already correct) |
| test_cancel_no_starttime_ticks | test_cli_cancel_resubmit.py | OPS-01 | XFAIL (strict) |
| test_dequeue_removes_queue_spec | test_cli_dequeue.py | OPS-02 | XFAIL (strict) |
| test_dequeue_refuses_running | test_cli_dequeue.py | OPS-02 | XFAIL (strict) |
| test_dequeue_pending_no_spec | test_cli_dequeue.py | OPS-02 | XFAIL (strict) |
| test_dequeue_unknown_node | test_cli_dequeue.py | OPS-02 | XFAIL (strict) |
| test_submit_existing_pending_marks_running | test_cli.py | OPS-03 | XFAIL (strict) |
| test_project_option_project_root | test_cli_project_option.py | OPS-04 | XFAIL (strict) |
| test_project_option_automil_dir | test_cli_project_option.py | OPS-04 | XFAIL (strict) |
| test_project_option_absent_cwd_walk | test_cli_project_option.py | OPS-04 | PASSED (regression guard) |
| test_viz_port_default | test_viz_port_config.py | OPS-05 | PASSED (regression guard) |
| test_viz_port_from_config | test_viz_port_config.py | OPS-05 | XFAIL (strict) |
| test_viz_port_explicit_overrides_config | test_viz_port_config.py | OPS-05 | PASSED (regression guard) |

## Anti-Theater Compliance

OPS-01 `test_cancel_local_direct_kill` and `test_cancel_no_starttime_ticks`:
- Spawn a REAL `subprocess.Popen(["sleep", "60"], start_new_session=True)`
- Write running spec with actual `proc.pid`, `os.getpgid(proc.pid)`, and optionally `_read_proc_starttime(proc.pid)`
- Assert `os.kill(proc.pid, 0)` raises `ProcessLookupError` (not mocked)
- `try/finally proc.kill(); proc.wait()` teardown prevents zombie processes even while xfail

grep confirmation: `grep -n "subprocess.Popen" tests/test_cli_cancel_resubmit.py` → lines 496, 616.

## Deviations from Plan

### Calibrations (not bugs — correct behavior already existed)

**1. [Rule 1 - Bug/Design] test_cancel_missing_pid_metadata: xfail removed**
- **Found during:** Task 1 verification (XPASS strict)
- **Issue:** The plan called for an xfail stub, but current cancel.py:100-105 already hard-fails with "corrupted state / Manage the process manually" when spec has neither `opaque_id` nor `metadata`. The assertion (`"manage" in output`) passes with existing code.
- **Fix:** Removed `@pytest.mark.xfail` — this is a regression guard, not a RED stub. OPS-01 changes relax the hard-fail for specs WITH metadata.pid/pgid (the fix path); the hard-fail for specs WITHOUT any identifiable handle remains correct.
- **Files modified:** tests/test_cli_cancel_resubmit.py

**2. [Rule 1 - Bug/Design] OPS-05: test_viz_port_default and test_viz_port_explicit_overrides_config: xfail removed**
- **Found during:** Task 2 verification (XPASS strict)
- **Issue:** Current `viz.py` hard-codes `--port default=8420` and passes explicit `--port` directly to `cmd_start`. Both behaviors are already correct. Only `test_viz_port_from_config` is the true RED stub (reading `viz.port` from config is not yet implemented).
- **Fix:** Removed `@pytest.mark.xfail` from the two passing cases; kept xfail only on `test_viz_port_from_config`.

**3. [Rule 1 - Bug/Design] OPS-04: test_project_option_absent_cwd_walk: xfail removed**
- **Found during:** Task 2 verification
- **Issue:** Existing cwd-walk via `_find_automil_dir()` already works correctly when `automil/config.yaml` is in cwd. This is a regression guard, not a RED stub.
- **Fix:** Removed `@pytest.mark.xfail`.

**4. [Rule 2 - Anti-theater] Assertion strengthening for dequeue_unknown_node and OPS-04 tests**
- **Found during:** Task 2 verification (XPASS strict)
- **Issue:** `test_dequeue_unknown_node` was XPASS because `exit_code != 0` was satisfied by "No such command 'dequeue'" (wrong reason). OPS-04 tests were XPASS because "No automil/config.yaml" was not in the "No such option: --project" error output (wrong reason).
- **Fix:** Added "No such command not in output" guard for dequeue; added "No such option not in output" guard for OPS-04 stubs. Tests now fail for the RIGHT reason.

## Regression Gate

Full framework suite after plan completion:

```
uv run pytest tests/ -q --tb=no --ignore=tests/acceptance
1032 passed, 51 skipped, 10 xfailed
```

Baseline before plan: `1028 passed, 51 skipped` (no xfailed).

Delta: +4 passing (regression guards), +10 xfailed (RED stubs). Zero regressions.

## Known Stubs

All 10 xfail stubs are intentional Wave-0 RED stubs. They will go green when:
- Plans 13-02 implements OPS-01 (cancel pid/pgid fallback), OPS-04 (--project option), OPS-05 (port from config)
- Plan 13-03 implements OPS-02 (dequeue command) and OPS-03 (submit pending→running)

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| tests/test_cli_cancel_resubmit.py exists | FOUND |
| tests/test_cli_dequeue.py exists | FOUND |
| tests/test_cli.py exists | FOUND |
| tests/test_cli_project_option.py exists | FOUND |
| tests/test_viz_port_config.py exists | FOUND |
| commit 182d042 (OPS-01 stubs) | FOUND |
| commit b7b45aa (OPS-02..05 stubs) | FOUND |
| subprocess.Popen in test_cli_cancel_resubmit.py | lines 496, 616 |
| Full suite: 1032 passed, 51 skipped, 10 xfailed | PASSED |
