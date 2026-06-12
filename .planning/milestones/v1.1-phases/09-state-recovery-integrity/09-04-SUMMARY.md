---
phase: 09-state-recovery-integrity
plan: "04"
subsystem: orchestrator
tags: [signal-handling, partial-fold-recovery, status-canonicalization, process-management]

# Dependency graph
requires:
  - phase: 09-01
    provides: Wave-0 RED test stubs for test_sigterm_flush, test_handle_timeout, test_collect_or_synthesize
  - phase: 09-03
    provides: result.schema.json updated with partial status + termination_reason (D-07)

provides:
  - D-02 fix: SIGTERM flush handler writes to AUTOMIL_RESULTS_DIR, not Path.cwd()
  - D-03 fix: _collect_or_synthesize_result tries fold aggregation before log-heuristic synthesis
  - D-04 fix: _handle_timeout sends SIGTERM to main PID first, then SIGKILL pgid after grace window
  - D-05/D-06 fix: oom/timeout synthesis paths produce status=crash + termination_reason (not status=oom/timeout)
  - termination_reason=sigterm annotated on SIGTERM-flush payloads

affects:
  - 09-06 (terminal_writer — plan 06 builds on corrected _handle_completion + _handle_timeout chain)
  - 09-05 (CLI/migrate — no dependency on these fixes)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "D-02: AUTOMIL_RESULTS_DIR env var lookup with is_absolute() safety guard before Path use"
    - "D-03: archive.glob('fold_*_result.json') probe + aggregate_folds before log-heuristic synthesis"
    - "D-04: os.kill(pid, SIGTERM) then time.sleep(grace) then os.killpg(pgid, SIGKILL) with ProcessLookupError guards"
    - "D-05/D-06: status tight-enum enforcement — oom/timeout moved to termination_reason, status=crash"
    - "Atomic tempfile+replace write pattern for synthesized result.json in _collect_or_synthesize_result"

key-files:
  created: []
  modified:
    - src/automil/runtime_helpers.py
    - src/automil/backends/_orchestrator_daemon.py

key-decisions:
  - "D-02: AUTOMIL_RESULTS_DIR env var used as write target; relative values rejected with cwd fallback (T-09-06 security)"
  - "D-03: fold-first probe uses _read_fold_count_for_node (existing helper); falls into log-heuristic synthesis only when no fold files exist"
  - "D-04: main-PID-first is LOCAL BACKEND ONLY — SLURM/Ray bypass _handle_timeout entirely; no backend-abstraction seam added"
  - "D-05/D-06 in _collect_or_synthesize_result synthesis path: oom -> crash+termination_reason=oom, timeout -> crash+termination_reason=timeout"
  - "termination_reason=sigterm added to SIGTERM-flush payload (D-05) for observability; does not affect keep/discard (Plan 06 scope)"

patterns-established:
  - "Partial-fold payloads: aggregate_folds(archive, expected) produces status=partial; no further status override at write site"
  - "Synthesis path: build status first, then build payload dict — never use status='oom' or status='timeout' as top-level status"
  - "ProcessLookupError caught on both os.kill and os.killpg calls (T-09-05 PID-reuse defense)"

requirements-completed:
  - REC-01

# Metrics
duration: 8min
completed: 2026-06-11
---

# Phase 09 Plan 04: Partial-Fold Recovery (REC-01) Summary

**SIGTERM flush writes to AUTOMIL_RESULTS_DIR, `_collect_or_synthesize_result` aggregates completed folds before synthesizing crash, and `_handle_timeout` sends SIGTERM to the main PID first with a configurable grace window before SIGKILL — mid-run kills now recover fold means instead of reporting composite=0.0**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-06-11T09:05:00Z
- **Completed:** 2026-06-11T09:13:02Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Fixed D-02 bug: `_handler` in `register_sigterm_flush` now reads `AUTOMIL_RESULTS_DIR` env var and writes `result.json` to the archive dir, not `Path.cwd()` (the worktree). Added `is_absolute()` guard (T-09-06 security).
- Fixed D-03 gap: `_collect_or_synthesize_result` now probes `archive.glob("fold_*_result.json")` and calls `aggregate_folds(archive, expected)` before falling through to log-heuristic synthesis. Fold result written atomically via tempfile+replace.
- Fixed D-04 regression: `_handle_timeout` SIGTERMs only the main PID first (letting the Python flush handler run), then sleeps `timeout_grace_seconds` (default 10, config-driven), then SIGKILLs the full process group.
- Fixed D-05/D-06 drift: OOM and timeout synthesis paths now emit `status="crash"` + `termination_reason="oom"/"timeout"` instead of the non-enum values `"oom"` / `"timeout"`.
- SIGTERM-flush payloads carry `termination_reason="sigterm"` (D-05).

## Task Commits

1. **Task 1: Fix runtime_helpers.py SIGTERM flush (D-02)** — `ff02fa1` (fix)
2. **Task 2: Rewrite _handle_timeout (D-04) + fix _collect_or_synthesize_result (D-03/D-05/D-06)** — `5bd627b` (fix)

**Plan metadata:** TBD (docs commit)

## Files Created/Modified

- `src/automil/runtime_helpers.py` — D-02: AUTOMIL_RESULTS_DIR target + T-09-06 path guard + D-05 termination_reason annotation
- `src/automil/backends/_orchestrator_daemon.py` — D-03 fold-first probe, D-04 main-PID-first timeout, D-05/D-06 status canonicalization in synthesis path

## Decisions Made

- Used existing `_read_fold_count_for_node` helper (L1091) for fold count in the D-03 probe — no new helper needed.
- Fold-aggregation result written with atomic `tempfile.mkstemp + os.replace` inside the `if fold_files:` branch to avoid a partial-write race on retry.
- `_handle_timeout` is LOCAL BACKEND ONLY: no backend-abstraction seam added per D-04 plan note. SLURM/Ray bypass the method entirely.
- `is_absolute()` guard on `AUTOMIL_RESULTS_DIR` value falls back to `Path.cwd()` for relative paths — mirrors T-09-06 threat mitigation without using the path as a shell argument.

## Deviations from Plan

None — plan executed exactly as written. The plan's action blocks were followed precisely; `_read_fold_count_for_node` was confirmed to already exist rather than needing creation.

## Issues Encountered

None. All 8 target tests turned GREEN on first attempt after applying the two targeted edits. Full suite: 991 passed, 14 pre-existing RED stubs (terminal_writer, migrate, reconcile-from-archive, mil-model cell identity, workstation-gated acceptance test) — no new failures introduced.

## Threat Surface Scan

No new network endpoints, auth paths, or file-access patterns introduced. Changes are confined to:
- `AUTOMIL_RESULTS_DIR` → filesystem write path in flush handler (T-09-06 — path validated as absolute before use; already in plan threat register)
- `os.kill(pid, SIGTERM)` / `os.killpg(pgid, SIGKILL)` — PID reuse guarded (T-09-05 — already in plan threat register)

No new threat flags beyond those already registered in the plan.

## Known Stubs

None — this plan produces complete implementations, not stubs. Partial-fold quarantine (exclusion from keep/discard and best_node) is Plan 06's scope (D-01); the correct `status=partial` payload is now produced by this plan.

## Next Phase Readiness

- Plan 05 (CLI + migrate, wave 3 parallel) can execute immediately — no dependency on Plan 04 outputs.
- Plan 06 (terminal_writer + quarantine, wave 4) depends on Plan 04: `_handle_timeout` now calls `_handle_completion` with correct `_timed_out` state and the synthesis path produces canonical statuses that `terminal_writer` will consume.
- Plans 02 and 03 (already complete) provide cell identity + schema foundations that Plan 05/06 build on.

---
*Phase: 09-state-recovery-integrity*
*Completed: 2026-06-11*
