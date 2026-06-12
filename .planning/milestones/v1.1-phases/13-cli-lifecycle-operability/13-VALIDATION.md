---
phase: 13
slug: cli-lifecycle-operability
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-12
---

# Phase 13 — Validation Strategy

> Derived from 13-RESEARCH.md §Validation Architecture. Per-phase validation contract.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing suite) |
| **Config file** | `pyproject.toml` (workspace root) |
| **Quick run** | `uv run pytest tests/test_cli_cancel_resubmit.py tests/test_cli_dequeue.py tests/test_cli.py tests/test_cli_project_option.py tests/test_viz_port_config.py -x` |
| **Framework suite** | `uv run pytest tests/ -q` (benchmark tree run SEPARATELY — rootdir collision, see lessons 2026-06-11) |
| **Estimated runtime** | quick <30s |

## Sampling Rate
- After every task commit: quick command
- After every wave: framework suite (`tests/` only, NOT combined with `benchmarks/`)
- Before `/gsd-verify-work`: framework suite green

## Per-Requirement Verification Map

| Req | Behavior | Type | Command | File |
|-----|----------|------|---------|------|
| OPS-01 | `cancel` kills a REAL local child process via `metadata.pid`/`pgid` from the running spec (anti-theater: spawns `sleep`, asserts `ProcessLookupError` after) | unit (real subprocess) | `pytest tests/test_cli_cancel_resubmit.py::test_cancel_local_direct_kill` | ❌ W0 |
| OPS-01 | `cancel` observes termination via starttime liveness (NOT `backend.poll()` — daemon `_handle_completion` never fires for CLI-side kill) | unit | `::test_cancel_local_direct_kill` (same) | ❌ W0 |
| OPS-01 | `cancel` hard-fails when spec has neither `opaque_id` nor `metadata.pid`/`pgid` (corrupted state) | unit | `::test_cancel_missing_pid_metadata` | ❌ W0 |
| OPS-01 | `starttime_ticks` absent (non-Linux spec) still cancels (pid/pgid sufficient; reuse guard degrades gracefully) | unit | `::test_cancel_no_starttime_ticks` | ❌ W0 |
| OPS-01 | `opaque_id`-bearing specs still cancel (regression — remote/legacy path unchanged) | regression | `::test_cancel_happy_path` | ✅ exists |
| OPS-02 | `dequeue` removes `orchestrator/queue/<node>.json` (FLAT path, not backend-namespaced) and marks graph node `cancelled` via `locked_update` | unit | `pytest tests/test_cli_dequeue.py::test_dequeue_removes_queue_spec` | ❌ W0 |
| OPS-02 | `dequeue` hard-fails for a `running` node with a cross-reference message ("use `automil cancel`") | unit | `::test_dequeue_refuses_running` | ❌ W0 |
| OPS-02 | `dequeue` on a `pending` node with NO queue spec still marks it `cancelled` (idempotent, clears orphan) | unit | `::test_dequeue_pending_no_spec` | ❌ W0 |
| OPS-02 | `dequeue` hard-fails for unknown node id (`_get_node_or_die` before `graph.cancel()` to avoid KeyError) | unit | `::test_dequeue_unknown_node` | ❌ W0 |
| OPS-03 | Submit against an existing `type=proposed,status=pending` node transitions it to `running` (`graph.mark_running` in new `else` branch) | unit | `pytest tests/test_cli.py::test_submit_existing_pending_marks_running` | ❌ W0 |
| OPS-04 | `--project PATH` (PATH = project root) routes discovery; `_find_automil_dir()` returns the override from outside cwd | unit | `pytest tests/test_cli_project_option.py::test_project_option_project_root` | ❌ W0 |
| OPS-04 | `--project PATH` (PATH = `automil/` dir directly) also resolves | unit | `::test_project_option_automil_dir` | ❌ W0 |
| OPS-04 | `--project` absent → cwd walk unchanged (regression); `_PROJECT_OVERRIDE` reset to None in teardown | regression | `::test_project_option_absent_cwd_walk` | ❌ W0 |
| OPS-05 | `viz start` with no `--port` and no config → `DEFAULT_PORT` 8420 | unit | `pytest tests/test_viz_port_config.py::test_viz_port_default` | ❌ W0 |
| OPS-05 | `viz start` with `viz.port` in config → config value | unit | `::test_viz_port_from_config` | ❌ W0 |
| OPS-05 | explicit `--port` overrides config `viz.port` (resolution order: flag → config → default) | unit | `::test_viz_port_explicit_overrides_config` | ❌ W0 |
| all | existing CLI + viz tests stay GREEN | regression | framework suite | ✅ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements
- [ ] `tests/test_cli_cancel_resubmit.py` — OPS-01 (add direct-kill, missing-pid, no-starttime tests to existing file)
- [ ] `tests/test_cli_dequeue.py` — OPS-02 (new file, 4 tests)
- [ ] `tests/test_cli.py` — OPS-03 (add existing-pending→running test)
- [ ] `tests/test_cli_project_option.py` — OPS-04 (new file, 3 tests; `_PROJECT_OVERRIDE` teardown reset MANDATORY)
- [ ] `tests/test_viz_port_config.py` — OPS-05 (new file, 3 tests)
- [ ] shared fixtures: fake daemon-shape running spec (pid/pgid/starttime), temp graph.json, temp config.yaml with `viz.port` / `run.mil_model`

## Critical Anti-Theater Constraints (from RESEARCH §Pitfall 3)
- **OPS-01:** the test MUST spawn a real `subprocess.Popen(["sleep", "60"], start_new_session=True)`, write the running spec the way the daemon does, invoke `cancel` via CliRunner, then assert `os.kill(proc.pid, 0)` raises `ProcessLookupError`. Do NOT hand-mock `os.killpg` — a mocked kill gives false-green on broken wiring.
- **OPS-02/03:** drive the REAL producers — write the queue spec / graph node in the daemon/submit shape; assert through `graph.json` state, not the test's own scaffolding.

## Manual-Only Verifications
*None — all five OPS behaviors are CI-testable with mocked processes/files (real subprocess for OPS-01; no GPU/cluster/hardware needed).*

## Validation Sign-Off
- [ ] All tasks have `<automated>` verify or Wave 0 deps
- [ ] OPS-01 uses a REAL subprocess (anti-theater), not a mocked kill
- [ ] OPS-04 tests reset `_PROJECT_OVERRIDE` in teardown (no cross-test bleed)
- [ ] Framework suite (`tests/` only) stays GREEN
- [ ] `nyquist_compliant: true` set when complete

**Approval:** pending
