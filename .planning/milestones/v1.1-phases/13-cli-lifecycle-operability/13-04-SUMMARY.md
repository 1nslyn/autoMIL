---
phase: 13-cli-lifecycle-operability
plan: "04"
subsystem: cli
tags: [ops-04, ops-05, project-discovery, viz-port, click-group-option, config-fallback]
dependency_graph:
  requires: [13-01, 13-03]
  provides: [OPS-04, OPS-05]
  affects: [src/automil/cli/__init__.py, src/automil/cli/_helpers.py, src/automil/cli/viz.py, src/automil/viz/server.py]
tech_stack:
  added: []
  patterns:
    - module-global override bridge for Click group option (D-08)
    - CLI-layer port resolution before cmd_start call (mirrors host fallback pattern)
    - click.Path(exists=True) + is_eager=True for group-level path validation
key_files:
  created: []
  modified:
    - src/automil/cli/_helpers.py
    - src/automil/cli/__init__.py
    - src/automil/cli/viz.py
    - src/automil/viz/server.py
    - tests/test_cli_project_option.py
    - tests/test_viz_port_config.py
decisions:
  - "OPS-04 bridge: module-global _PROJECT_OVERRIDE in _helpers.py (D-08) — avoids threading pass_context through 21 commands"
  - "OPS-05 port resolution in viz_start (CLI layer) not cmd_start, because test mocks cmd_start and verifies resolved port passed to it"
  - "server.py cmd_start also gets combined host+port resolution block for robustness of direct callers"
metrics:
  duration_minutes: 12
  completed_date: "2026-06-12"
  tasks_completed: 2
  files_modified: 6
---

# Phase 13 Plan 04: OPS-04 + OPS-05 Summary

**One-liner:** Group-level `--project PATH` routing via module-global bridge in `_find_automil_dir`, plus `viz start` port config fallback (explicit > viz.port config > 8420).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | OPS-04: _helpers.py module-global + __init__.py group option | af0e4f7 | _helpers.py, __init__.py, test_cli_project_option.py |
| 2 | OPS-05: viz.py port=None + server.py port resolution block | 9de4aa0 | viz.py, server.py, test_viz_port_config.py |

## What Was Built

### OPS-04: `--project PATH` Group Option

Added a module-level `_PROJECT_OVERRIDE: Path | None = None` to `src/automil/cli/_helpers.py`.
`_find_automil_dir()` checks this override as the very first step, before the cwd walk:

- Accepts project root (containing `automil/`) or the `automil/` dir itself
- Hard-fails with a clear `ClickException` if neither contains `automil/config.yaml`
- cwd walk is completely unchanged when `_PROJECT_OVERRIDE is None`

The `main` Click group in `__init__.py` gained a `--project` option with:
- `type=click.Path(exists=True)` — Click validates path existence at the CLI boundary (ASVS V5.1)
- `is_eager=True` — processed before any subcommand options
- `Path(project_path).resolve()` — converts to absolute path, eliminates `..` traversal
- Override is set BEFORE `touch_last_action(_find_automil_dir())` so activity stamping targets the correct overlay

The `from automil.cli import dequeue` import from plan 13-03 was preserved intact.

### OPS-05: viz Port Config Fallback

Resolution order (explicit > config > default):

1. `--port 7777` (explicit CLI flag) → `port=7777`
2. `viz.port: 9000` in `automil/config.yaml` → `port=9000`
3. Neither set → `port=DEFAULT_PORT` (8420)

Resolution happens in `viz_start` (CLI layer) using the already-fetched `adir`, before calling
`cmd_start`. This is necessary because tests mock `cmd_start` and assert on the resolved port
that `cmd_start` receives — if resolution were only in `cmd_start`, the mock would capture `None`.

`server.py cmd_start` also received a combined host+port resolution block (loads config once
for both `host` and `port`), which makes direct callers of `cmd_start` robust too.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] OPS-05 port resolution moved from cmd_start to viz_start**

- **Found during:** Task 2 verification (test_viz_port_default failed: `got port=None`)
- **Issue:** The plan specified port resolution in `server.py cmd_start`. But all 3 OPS-05 tests
  mock `automil.viz.server.cmd_start` entirely (replaced by `fake_cmd_start`) and assert on the
  port value received by the mock. If resolution is only inside `cmd_start`, the mock never sees
  a resolved port — it only sees `None` (what `viz_start` passes after `--port default=None`).
- **Fix:** Move port resolution into `viz_start` (CLI layer) using `adir` (already fetched).
  `server.py cmd_start` still has a combined host+port block for robustness of direct callers,
  but the CLI path resolves port before the call.
- **Files modified:** `src/automil/cli/viz.py` (primary fix), `src/automil/viz/server.py` (kept for direct-call robustness)
- **Commits:** 9de4aa0

## Test Results

```
tests/test_cli_project_option.py::test_project_option_project_root       PASSED
tests/test_cli_project_option.py::test_project_option_automil_dir        PASSED
tests/test_cli_project_option.py::test_project_option_absent_cwd_walk    PASSED
tests/test_viz_port_config.py::test_viz_port_default                     PASSED
tests/test_viz_port_config.py::test_viz_port_from_config                 PASSED
tests/test_viz_port_config.py::test_viz_port_explicit_overrides_config   PASSED
```

Full suite: `1 failed, 1053 passed, 53 skipped` — the 1 failure is the pre-existing `clause_11` acceptance gate (Phase 8), unrelated to this plan.

## Framework Purity (D-206)

`grep -r "autobench|AUTOBENCH_|benchmarks" src/automil/ | grep -v .pyc` returns 5 results, all pre-existing comments/documentation in files not touched by this plan. Zero new references introduced.

## Known Stubs

None — both features are fully wired with no placeholders.

## Threat Flags

None — no new network endpoints or auth paths introduced. `click.Path(exists=True)` on `--project` closes T-13-04-01. Port config is loopback-safe (T-13-04-02: accept disposition unchanged).

## Self-Check: PASSED

- `af0e4f7` exists: `git log --oneline --all | grep af0e4f7` ✓
- `9de4aa0` exists: `git log --oneline --all | grep 9de4aa0` ✓
- `src/automil/cli/_helpers.py` contains `_PROJECT_OVERRIDE` ✓
- `src/automil/cli/__init__.py` contains `--project` option and `_h._PROJECT_OVERRIDE` ✓
- `src/automil/cli/viz.py` contains `default=None` and `int | None` ✓
- `src/automil/viz/server.py` contains `DEFAULT_PORT` in port resolution block ✓
- `from automil.cli import dequeue` preserved in `__init__.py` ✓
