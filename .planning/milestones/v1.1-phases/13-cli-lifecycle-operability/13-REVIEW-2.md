---
phase: 13-cli-lifecycle-operability
reviewed: 2026-06-12T11:22:00Z
depth: standard
mode: re-review (iteration 2)
files_reviewed: 8
files_reviewed_list:
  - src/automil/cli/cancel.py
  - src/automil/cli/dequeue.py
  - src/automil/viz/server.py
  - src/automil/cli/__init__.py
  - src/automil/cli/_helpers.py
  - src/automil/cli/submit.py
  - tests/test_viz_port_config.py
  - scripts/check_backend_isolation.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 13: Code Review Report (Iteration 2 — Fix Verification)

**Reviewed:** 2026-06-12T11:22:00Z
**Depth:** standard (re-review)
**Files Reviewed:** 8
**Status:** clean

## Summary

All 7 iteration-1 findings (1 Critical + 6 Warnings) are GENUINELY closed in the
current `milestone/v1.1-bug-fixing` code. No new Critical or Warning regressions
were introduced by the fixes. The BCK-04 allowlist entry for `cli/cancel.py` is
accurate. Framework purity (D-206) is CLEAN — zero `autobench`/`AUTOBENCH_`/
`benchmarks` references in any reviewed `src/automil/` file.

**Empirical verification:**
- `scripts/check_backend_isolation.py src/automil` → `OK: no backend isolation violations`
- `pytest test_cli_cancel_resubmit.py test_cli_dequeue.py test_viz_port_config.py` → 17 passed

## Per-Finding Closure Table

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| CR-01 | cancel.py raw unlocked graph write + total_proposed drift | CLOSED | cancel.py:329-351 now uses `locked_update` + `graph.cancel(node_id)` (decrements `total_proposed` per graph.py:381-384). Raw tempfile block gone. `import os` hoisted to module top (line 20); no `_os` alias remains. `cancelled_at` is tz-aware (`datetime.now(timezone.utc)`, line 344-346). `graph.cancel()` is the sole graph-mutation path. |
| WR-01 | killpg targets unvalidated on-disk `_pgid` | CLOSED | `_signal_group` (cancel.py:186-218) re-probes `_is_alive(_pid, _starttime)` then derives `target_pgid = os.getpgid(_pid)` when `_starttime is not None`, falling back to on-disk `_pgid` only when starttime is None (non-Linux best-effort). `os.getpgid` wrapped in `try/except ProcessLookupError` (207-210) so a process exiting between probe and getpgid is treated as dead. |
| WR-02 | `_try_reap` can reap PID-reused child | CLOSED | Reap is gated (cancel.py:264-266): `_reap_starttime = _read_proc_starttime(_pid) if _starttime is not None else None`, then `if _starttime is None or _reap_starttime == _starttime`. PID-reuse hole closed on Linux; non-Linux (`_starttime is None`) intentionally still reaps per OPS-01 accepted risk. NOT self-contradictory — `test_cancel_no_starttime_ticks` (real subprocess, no starttime) still passes because the None-branch short-circuits to allow the reap. |
| WR-03 | non-Linux `os.kill(pid,0)` maps EPERM→dead | CLOSED | cancel.py:175-182: `ProcessLookupError → False` (dead), `PermissionError → True` (alive, owned by other user). Correct. |
| WR-04 | dequeue TOCTOU — unlink/guard outside lock | CLOSED | dequeue.py:88-115: running re-check, type/status re-check, AND `queue_spec.unlink()` all inside the single `locked_update` block against a freshly re-read `locked_node`. Out-of-lock guard (55-65) is only a fast-fail. Residual daemon-pickup TOCTOU is documented (75-86) and accepted — not re-flagged. |
| WR-05 | dequeue cancels terminal/non-proposed states | CLOSED | dequeue.py:27 `DEQUEUEABLE_STATES = {"pending","queued"}`; positive guard (61, 101) requires `type=="proposed" AND status in DEQUEUEABLE_STATES`. crash/oom/timeout/partial/registered now refused. Idempotent pending-with-no-queue-spec still works (unlink gated on `queue_spec.exists()`, cancel still fires). |
| WR-06 | viz port server-side resolution untested / main() shim hard-codes port | CLOSED | New `test_server_cmd_start_resolves_config_port` (test_viz_port_config.py:127-176) calls `cmd_start(port=None,...)` directly with NO cmd_start mock, intercepting at `web.TCPSite` to capture the resolved port=9001 — genuinely exercises server.py's `port is None` config fallback (server.py:302-307). server.py `main()` (408-425) now passes `port=None` when no `--port` flag so the legacy shim shares config resolution. |
| BCK-04 | allowlist `cli/cancel.py` | CLOSED/ACCURATE | check_backend_isolation.py:73 allowlists `Path("cli/cancel.py")` with rationale (lines 21-34) accurately describing sanctioned OPS-01 direct-kill of daemon-launched local jobs, starttime-guarded. Lint passes clean. |

## Regression Scan (fix-introduced defects)

| Concern | Verdict |
|---------|---------|
| WR-04 unlink when node vanished mid-lock | SAFE. Vanished node → warning logged, no unlink/cancel (dequeue.py:91-92). Consistent with daemon having consumed it. |
| WR-04 unlink when graph.json absent | SAFE. `_get_node_or_die` already proved graph.json exists; `if graph_path.exists()` is latent dead-defensive (pre-existing IN-04, info-only). No new defect. |
| WR-01 process exits between liveness probe and `os.getpgid(_pid)` | SAFE. ProcessLookupError caught (cancel.py:208-210) → treated as dead, returns. |
| WR-06 server.py main() fold changes Click path | SAFE. Click `cli/viz.py` resolves and passes its own port; `cmd_start` only falls back when `port is None`. No behavior change for the supported Click entry point. |
| Framework purity D-206 | CLEAN. grep for autobench/AUTOBENCH_/benchmarks across all 6 src files → zero hits. |
| `graph.cancel()` as sole mutation path in cancel.py | CONFIRMED. The only writer in the post-kill block; metadata fields set on the live locked graph node, persisted by `locked_update` auto-save. |

## New Findings

None. No new Critical or Warning issues. (The pre-existing
`tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11` failure is
out of scope per the review mandate and was not evaluated.)

---

_Reviewed: 2026-06-12T11:22:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard (re-review, iteration 2)_
