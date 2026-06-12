---
phase: 11-config-run-fidelity
plan: 03
subsystem: cli, orchestrator
tags: [cfg-02, cfg-03, click, submit, timeout, override, daemon, security]

requires:
  - phase: 11-config-run-fidelity
    plan: 01
    provides: RED test stubs for CFG-02 and CFG-03

provides:
  - submit --timeout defaults to None; timeout_min omitted from spec when unset (D-02)
  - D-03 sentinel fix: timeout is not None replaces timeout != 150 (max-time interaction guard)
  - submit --override option writing run_command_override into queue spec (D-04, CFG-03)
  - daemon suffix-appends shlex.split(override) after base run.command; no shell=True (T-11-03-01)

affects:
  - src/automil/cli/submit.py
  - src/automil/backends/_orchestrator_daemon.py
  - tests/test_cfg_run_fidelity.py

tech-stack:
  added: []
  patterns:
    - conditional spec field write (if timeout is not None: spec["timeout_min"] = timeout)
    - list append override: cmd = cmd + shlex.split(override_str)
    - Popen-mock pattern for daemon launch unit testing

key-files:
  modified:
    - src/automil/cli/submit.py
    - src/automil/backends/_orchestrator_daemon.py
    - tests/test_cfg_run_fidelity.py

decisions:
  - id: D-02
    summary: submit --timeout default changed to None; timeout_min omitted from queue spec when not supplied so daemon falls back to orchestrator.default_timeout_min
  - id: D-03
    summary: sentinel changed from timeout != 150 to timeout is not None — max-time-wins path preserved correctly for both one-flag and two-flag invocations
  - id: D-04
    summary: run_command_override field used; suffix-append as list (cmd + shlex.split(override_str)); shell=False preserved

metrics:
  duration_minutes: 8
  completed_date: "2026-06-11"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 3
  commits: 2
---

# Phase 11 Plan 03: CFG-02 + CFG-03 Summary

**One-liner:** None-default --timeout with conditional spec omit (D-02/D-03) and --override suffix-append via shlex list concat in daemon (D-04, CFG-03).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | CFG-02 — None-default --timeout, D-03 sentinel, --override option in submit.py | 82a1d9b | src/automil/cli/submit.py |
| 2 | CFG-03 — daemon suffix-append + GREEN Popen-mock test | 1d3e1b5 | src/automil/backends/_orchestrator_daemon.py, tests/test_cfg_run_fidelity.py |

## What Was Built

### Task 1: submit.py — CFG-02 + CFG-03 CLI surface (82a1d9b)

Three targeted edits to `src/automil/cli/submit.py`:

1. **--timeout default → None (D-02):** Changed `@click.option("--timeout", default=150, ...)` to `default=None, type=int`. Updated function signature `timeout: int` → `timeout: int | None`.

2. **D-03 sentinel fix:** Changed `if timeout != 150:` to `if timeout is not None:` in the `--max-time` interaction block (~L60). The ceil-div translation (`max(1, (max_time_seconds + 59) // 60)`) is untouched. With the new None default: when only `--max-time` is given, sentinel is False (correct — no echo), then `timeout = translated`. When both are given, sentinel fires (echo message), timeout = translated (max-time wins). Both paths verified GREEN.

3. **Conditional timeout_min write (D-02):** Removed `"timeout_min": timeout` from the spec dict literal. Added `if timeout is not None: spec["timeout_min"] = timeout` after dict construction. Daemon's `spec.get("timeout_min", self.default_timeout)` at L918 now falls back to `orchestrator.default_timeout_min` from config.yaml when the key is absent.

4. **--override option added (D-04, CFG-03):** New Click option `--override` with `default=None`. Added `override: str | None` to function signature. Added `if override is not None: spec["run_command_override"] = override` after spec construction.

### Task 2: _orchestrator_daemon.py + test update — CFG-03 daemon (1d3e1b5)

**Daemon launch block** (`src/automil/backends/_orchestrator_daemon.py` ~L899):

After `cmd = shlex.split(self.run_command)` (or the `sys.executable` branch), added:
```python
override_str = spec.get("run_command_override")
if override_str:
    cmd = cmd + shlex.split(override_str)
```

This is list append (not string concatenation), and `subprocess.Popen` retains no `shell=True` keyword. `shlex.split` tokenizes metacharacters (`;`, `|`, `&`) as literal argument tokens, preventing accidental shell injection even from user typos (T-11-03-01 defense-in-depth).

**Test stub upgraded to GREEN** (`tests/test_cfg_run_fidelity.py`): `test_daemon_appends_override_to_run_command` replaced the `pytest.fail()` RED stub with a Popen-mock test that:
- Constructs a minimal `ExperimentOrchestrator` with filesystem mocks and a config.yaml setting `run.command`
- Patches `subprocess.Popen` to capture the `cmd` argument
- Asserts `launched_cmds[0] == shlex.split(base) + shlex.split(override)`
- Asserts `shell=False` in Popen kwargs (T-11-03-01 security guard)

## Verification Results

```
uv run pytest tests/test_cfg_run_fidelity.py -v
5 passed in 1.15s

uv run pytest tests/ -q
1021 passed, 1 failed (pre-existing), 53 skipped in 229.80s
```

The single failure (`test_d208_clause_11_state_roadmap_complete`) is pre-existing and unrelated to this plan — it checks DEC-01 REQUIREMENTS.md status from Phase 8 and fails identically on the baseline commit before these changes.

Security check: `grep -n "shell=True" src/automil/backends/_orchestrator_daemon.py` returns nothing in executable code (only in a comment).

## Deviations from Plan

None — plan executed exactly as written.

The test stub update (RED → GREEN Popen-mock) was the expected completion of the plan's TDD cycle. The RED stub contained an explicit `pytest.fail()` with instructions to "replace with a Popen-mock call" — this was implemented as directed.

## Known Stubs

None. All CFG-02 and CFG-03 behaviors are fully wired:
- submit.py omits timeout_min when unset and writes run_command_override when provided
- daemon reads and appends run_command_override after base run.command

## Threat Flags

No new threat surface beyond what was in the plan's threat model. The override append is contained within the existing Popen call block. No new network endpoints, auth paths, or schema changes introduced.

## Self-Check

| Claim | Verified |
|-------|---------|
| src/automil/cli/submit.py modified | FOUND |
| src/automil/backends/_orchestrator_daemon.py modified | FOUND |
| tests/test_cfg_run_fidelity.py modified | FOUND |
| Commit 82a1d9b exists | FOUND |
| Commit 1d3e1b5 exists | FOUND |
| 5/5 CFG tests GREEN | PASSED |
| No shell=True in daemon | CONFIRMED |

## Self-Check: PASSED
