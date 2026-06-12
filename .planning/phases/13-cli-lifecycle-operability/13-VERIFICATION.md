---
phase: 13-cli-lifecycle-operability
verified: 2026-06-12T11:26:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  note: Initial verification — no prior VERIFICATION.md existed.
carried_forward:
  - item: "tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete"
    status: failing (pre-existing, unrelated to Phase 13)
    reason: >-
      Stale v1.0 acceptance test asserts v1.0 DEC-01..07 'Complete' rows in
      .planning/REQUIREMENTS.md, which is now the v1.1 file. Confirmed reproduced
      in isolation — references no OPS-01..05 code path. Slated for Phase 14 / DBT.
    not_a_phase_13_failure: true
---

# Phase 13: CLI Lifecycle & Operability Verification Report

**Phase Goal:** Operators can drive the full node lifecycle from the CLI — cancel a daemon-launched running job, cleanly dequeue a queued/pending node, have an existing pending proposal transition to `running` on submit, target an overlay from outside its project root, and reliably reach the viz dashboard via config-driven port resolution — without manual file surgery.
**Verified:** 2026-06-12T11:26:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (per ROADMAP Success Criteria)

| # | Criterion (OPS) | Delivered? | Evidence (file:line + test) |
|---|-----------------|-----------|------------------------------|
| 1 | **OPS-01** `automil cancel <node>` terminates a daemon-launched LOCAL job by resolving `metadata.pid`/`metadata.pgid` from the running spec and signalling the process group directly (NOT via the daemon's empty in-memory map); node → `cancelled`. | ✓ VERIFIED | `cli/cancel.py:101-104` reads `opaque_id`/`metadata.pid`/`metadata.pgid` from `running/<backend>/<node>.json`; `:114-268` direct-kill branch (`not opaque_id`) signals the group via `_signal_group` (`os.killpg`, `:212`) with SIGTERM→5s grace→SIGKILL (`:225-245`), starttime PID-reuse guard (`_is_alive`/`_read_proc_starttime`, `:153-184,264`), zombie-aware. Comment `:115-117` explicitly states the daemon `self.running` map is empty in a fresh CLI process so it is bypassed. Graph flip via `locked_update`→`graph.cancel()` `:329-351` (NOT a raw write — the prior raw-write concern is closed). **Test:** `tests/test_cli_cancel_resubmit.py::test_cancel_local_direct_kill` (`:478`) — real `subprocess.Popen(["sleep","60"], start_new_session=True)` (`:495`), asserts `os.kill(proc.pid,0)` raises `ProcessLookupError` (`:536-537`). PASSED. |
| 2 | **OPS-02** `automil dequeue <node>` removes the queue spec AND marks the graph node `cancelled` via `locked_update`; no orphaned pending proposal; state-guarded (refuses running). | ✓ VERIFIED | `cli/dequeue.py` registered command `:30-32`; positive state guard `DEQUEUEABLE_STATES={"pending","queued"}` `:27`, out-of-lock fast-fail `:55-65`, authoritative in-lock re-check + queue-spec `unlink()` + `graph.cancel()` ALL under one `locked_update` `:87-115` (WR-04 TOCTOU narrowing documented `:73-86`); idempotent for orphaned pending (marks cancelled even with no queue spec). Registered in `cli/__init__.py:49`. **Test:** `tests/test_cli_dequeue.py` (4 tests). PASSED. |
| 3 | **OPS-03** `automil submit` against an existing `type=proposed, status=pending` node calls `graph.mark_running` after writing the queue spec → node `running`. | ✓ VERIFIED | `cli/submit.py:480-481` writes queue spec, then `:520-535` else-branch on existing node calls `graph.mark_running(node)` guarded on `type=="proposed" and status=="pending"` (mark_running itself is state-guarded), all under `locked_update` `:494-497`. **Test:** `tests/test_cli.py::test_submit_existing_pending_marks_running` (`:584`). PASSED. |
| 4 | **OPS-04** Every CLI command accepts `--project PATH` (group-level on `main`) routing discovery via `_find_automil_dir`'s `_PROJECT_OVERRIDE`. | ✓ VERIFIED | Group-level option on `main` in `cli/__init__.py:11-24` (`is_eager=True`); callback sets `_h._PROJECT_OVERRIDE = Path(project_path).resolve()` `:33-35`. `_helpers.py:21` declares override; `_find_automil_dir` honors it first (`:32-42`) accepting both project root and `automil/` dir, with a clear ClickException if no config found. **Test:** `tests/test_cli_project_option.py` (3 tests; resets `_PROJECT_OVERRIDE` in teardown per VALIDATION sign-off). PASSED. |
| 5 | **OPS-05** `automil viz start` without `--port` resolves explicit → `viz.port` config → `8420`. | ✓ VERIFIED | `cli/viz.py:30-50`: `--port` default None; when None reads `viz.port` from `automil/config.yaml` (`:37-48`) falling back to `DEFAULT_PORT`; resolution done at CLI layer so `cmd_start` receives a resolved int. `viz/server.py:44` `DEFAULT_PORT = 8420`. Mirrors the existing `viz.host` fallback pattern (`:23-29`). **Test:** `tests/test_viz_port_config.py` (4 tests). PASSED. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/automil/cli/cancel.py` | Direct-kill via on-disk pid/pgid + locked graph flip | ✓ VERIFIED | 367 lines; substantive direct-kill branch + signal/grace/escalation + PID-reuse guard. |
| `src/automil/cli/dequeue.py` | New dequeue command, locked_update, state-guarded | ✓ VERIFIED | 117 lines; positive guard + in-lock unlink + graph.cancel. |
| `src/automil/cli/submit.py` | OPS-03 pending→running else branch | ✓ VERIFIED | else branch `:520-535` calls mark_running. |
| `src/automil/cli/__init__.py` | `--project` group option + override bridge | ✓ VERIFIED | option `:11-24`, callback `:33-35`; dequeue registered `:49`. |
| `src/automil/cli/_helpers.py` | `_PROJECT_OVERRIDE` honored by `_find_automil_dir` | ✓ VERIFIED | `:21` declaration, `:32-42` resolution. |
| `src/automil/cli/viz.py` | viz.port → 8420 fallback | ✓ VERIFIED | `:37-49` resolution. |
| `src/automil/viz/server.py` | `DEFAULT_PORT = 8420` | ✓ VERIFIED | `:44`. |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| `cancel.py` | OS process group | `os.killpg(target_pgid, sig)` `:212` | WIRED — real signal, not mocked. |
| `cancel.py` / `dequeue.py` | `graph.json` | `locked_update` → `graph.cancel()` | WIRED — serialized terminal transition + total_proposed decrement. |
| `submit.py` | `graph.json` | `locked_update` → `graph.mark_running(node)` `:535` | WIRED. |
| `main` callback | `_find_automil_dir` | `_h._PROJECT_OVERRIDE` module bridge | WIRED. |
| `viz.py` | `viz/server.cmd_start` | resolved `port` int | WIRED. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 13 targeted suites | `uv run pytest tests/test_cli_cancel_resubmit.py tests/test_cli_dequeue.py tests/test_cli.py tests/test_cli_project_option.py tests/test_viz_port_config.py -q` | **44 passed in 7.31s** | ✓ PASS |
| OPS-01 real kill (anti-theater) | named test spawns real `sleep 60`, asserts `ProcessLookupError` | exercised, green | ✓ PASS |
| Backend isolation purity | `uv run python scripts/check_backend_isolation.py src/automil` | `OK: no backend isolation violations` (exit 0) | ✓ PASS |
| Carried-forward clause_11 (control) | run in isolation | fails reading v1.1 REQUIREMENTS.md for v1.0 DEC rows — no OPS path | ✓ confirmed unrelated |

### Requirements Coverage

| Requirement | Source Plan | Status | Evidence |
|-------------|-------------|--------|----------|
| OPS-01 | 13-02 | ✓ SATISFIED | cancel.py direct-kill + real-subprocess test |
| OPS-02 | 13-03 | ✓ SATISFIED | dequeue.py + test_cli_dequeue.py |
| OPS-03 | 13-03 | ✓ SATISFIED | submit.py else branch + test_cli.py |
| OPS-04 | 13-04 | ✓ SATISFIED | --project option + test_cli_project_option.py |
| OPS-05 | 13-04 | ✓ SATISFIED | viz.py port resolution + test_viz_port_config.py |

No orphaned requirements: all five OPS IDs mapped to Phase 13 in REQUIREMENTS.md are claimed by plans 13-02/03/04.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `dequeue.py` | 73-86 | "Residual TOCTOU (documented)" | ℹ️ Info | Daemon queue pickup not under graph lock; documented as narrowed-not-closed with safe degradation (worst case: refuse with "started running"). Not a debt marker requiring follow-up; no TODO/FIXME/XXX. |

No `TODO`/`FIXME`/`XXX`/`PLACEHOLDER` debt markers, no stub returns, no hardcoded-empty data in the Phase 13 modified files.

### Human Verification Required

None. 13-VALIDATION.md §Manual-Only Verifications explicitly records "None — all five OPS behaviors are CI-testable (real subprocess for OPS-01; no GPU/cluster/hardware needed)." No deferred `<human-check>` blocks in any 13-0*-PLAN.md.

### Known Carried-Forward Item (not a Phase 13 failure)

`tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` fails in isolation: it asserts v1.0 `| DEC-01 | Phase 8 | Complete |` rows in `.planning/REQUIREMENTS.md`, which is now the v1.1 Bug Fixing file. The failure references no OPS-01..05 code path and is slated for Phase 14 / DBT. The orchestrator-reported full-suite result (`1054 passed, 1 failed, 53 skipped`) is consistent with this single pre-existing failure. It does NOT count against Phase 13's goal.

### Gaps Summary

No gaps. All five ROADMAP success criteria are delivered by substantive, wired code and exercised by passing automated tests (44/44 targeted). The OPS-01 test is genuine anti-theater (real subprocess + real signal, no mocked kill). Backend isolation purity holds.

---

_Verified: 2026-06-12T11:26:00Z_
_Verifier: Claude (gsd-verifier)_
