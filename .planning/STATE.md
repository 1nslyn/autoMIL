---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Bug Fixing
status: executing
last_updated: "2026-06-12T10:48:00.000Z"
last_activity: 2026-06-12
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 16
  completed_plans: 16
  percent: 83
---

# State: autoMIL - v1.1 Bug Fixing

## Project Reference

**Project:** autoMIL — v1.1 Bug Fixing
**Core Value:** An agent can autonomously discover model improvements for any user's training code under a 6-hour-per-cell budget, with discovered variants reproducible, attributable to their parents, and portable across machines and LLM runtimes.

**Canonical documents** (load these before any phase work):

- `.planning/PROJECT.md` — milestone definition, validated/active/out-of-scope sets, constraints, key decisions
- `.planning/REQUIREMENTS.md` — 20 v1.1 REQ-IDs across 6 categories (REC/APL/CFG/SCH/OPS/DBT) with phase traceability
- `.planning/ROADMAP.md` — 6-phase bug-fix plan (Phases 9–14), success criteria, parallel-execution candidates
- `.planning/codebase/` — existing system map (ARCHITECTURE.md, CONCERNS.md, STACK.md). Do NOT re-document.
- `tasks/test-run-issues.md` — dated defect triage (2026-06-07), scope source for this milestone
- `CLAUDE.md` — project instructions and Leo's standing directives
- `~/.claude/projects/-home-jma-Documents-yinshuol-autoMIL/memory/MEMORY.md` — Leo's standing memory

**Current focus:** Phase 13 — CLI Lifecycle & Operability

## Current Position

Phase: 13 (CLI Lifecycle & Operability) — STARTING (discuss → plan → execute)
Plan: 0 of N (not yet planned)
Status: Phase 12 complete + gated; beginning Phase 13
Last activity: 2026-06-12

## Performance Metrics

| Metric | Value | Notes |
|---|---|---|
| Total v1.1 requirements | 20 | Mapped 100% to phases |
| Total phases (v1.1) | 6 | Phases 9–14 |
| Granularity | fine | Per `.planning/config.json` |
| Parallelization | enabled | Phases 11–14 are parallel-candidates after Phase 9 clears |
| Mode | yolo | Auto-approve gates within roadmap; Leo reviews artifacts |
| Phase 09 P02 | 16 | 2 tasks | 9 files |
| Phase 09 P03 | 8 | 2 tasks | 5 files |
| Phase 09 P04 | 8 | 2 tasks | 2 files |
| Phase 09 P05 | 15 | 2 tasks | 13 files |
| Phase 09 P06 | 35 | 2 tasks | 9 files |
| Phase 10 P02 | 15 | 2 tasks | 2 files |
| Phase 10 P03 | 10min | 2 tasks | 4 files |
| Phase 10 P04 | 10m | 1 tasks | 2 files |
| Phase 11-config-run-fidelity P01 | 12 | 2 tasks | 2 files |
| Phase 11-config-run-fidelity P02 | 4m | 1 tasks | 1 files |
| Phase 11-config-run-fidelity P03 | 8m | 2 tasks | 3 files |
| Phase 12-scheduling-overlay-isolation P02 | 8 | 2 tasks | 4 files |

## Parallel Execution Map

After Phase 9 (REC) completes, the following phases are independent and may execute in parallel:

- Phase 10 (APL) — depends on Phase 9 (REC-04 mil_model first-class); should start after Phase 9
- Phase 11 (CFG) — independent; parallel-candidate alongside 12, 13, 14
- Phase 12 (SCH) — independent; parallel-candidate alongside 11, 13, 14
- Phase 13 (OPS) — independent; parallel-candidate alongside 11, 12, 14
- Phase 14 (DBT) — independent; parallel-candidate alongside 11, 12, 13

Recommended execution order: Phase 9 → then Phase 10 + Phases 11/12/13/14 in parallel.

## Accumulated Context

### Decisions logged (from PROJECT.md -> ROADMAP.md)

- MockSLURMBackend: PENDING/RUNNING->CRASHED on restart (timer threads cannot resume) — Done (02-06, 2026-05-02)
- BCK-04 lint allowlist includes viz/server.py (viz daemon PID lifecycle, not job-control) — Done (02-07, 2026-05-02)
- Registry-first, not config-first, for cross-dataset isolation — Done (Phase 1)
- Skills only for autonomous setup; CLI for everything else — Done (Phase 7)
- Pluggable orchestrator backends with `local` as default — Done (Phase 2 ABC, Phase 6 SLURM/Ray)
- Multi-runtime agent support is in v1, not deferred — Done (Phase 3)
- 6h cap = per-cell-total, framework-enforced — Done (Phase 4)
- Search-scope mode flag (`architecture-preserving | free`) — Done (Phase 1, default `free`)
- autoMIL is generic; autobench is one consumer — Done (audited and verified in Phase 8)
- Tier 1 mechanical fixes before structural refactor — Done (5 commits, 2026-05-01)
- `port-variant` and `promote-variant` are CLI, not skills — Done (Phase 1)
- env.required is mandatory, env.passthrough is consumer-controlled — Done (Phase 8)
- Framework purity: zero autobench refs in src/automil/ — Done (Phase 8, D-206)
- D-02: SIGTERM flush writes to AUTOMIL_RESULTS_DIR (absolute-validated); falls back to cwd — Done (09-04)
- D-03: _collect_or_synthesize_result probes fold files before log-heuristic synthesis — Done (09-04)
- D-04: _handle_timeout: main-PID SIGTERM + timeout_grace_seconds grace + pgid SIGKILL; LOCAL BACKEND ONLY — Done (09-04)
- D-05/D-06 synthesis path: oom/timeout -> status=crash+termination_reason (not non-enum values) — Done (09-04)
- D-12: submit --mil-model resolution chain: flag → run.mil_model config → propose node metadata → ClickException — Done (09-05)
- D-15: automil cells migrate: mode-aware budget merge (agent_active sums consumed; wall_clock keeps min started_at) — Done (09-05)
- D-09: write_terminal_state writes graph→completed/<node>.json→archive result.json→results.tsv in fixed order — Done (09-06)
- D-10: terminal_writer sole archive result.json writer; reconcile_budget_kill write_text removed — Done (09-06)
- D-11: automil reconcile --from-archive NODE_OR_ALL refreshes existing nodes; skips running nodes — Done (09-06)
- D-01: graph.best_node() and _reevaluate_descendants() skip status=partial nodes (quarantine) — Done (09-06)

### Critical pitfalls defended (from research/PITFALLS.md -> ROADMAP.md anti-acceptance notes)

- Pitfall 1 (still uses old path) -> Phase 1 disable-old gate + protected-files validator (DONE)
- Pitfall 2 (leaky backend ABC) -> Phase 2 MockSLURM in parallel with LocalBackend (DONE)
- Pitfall 3 (multi-runtime untested-but-claimed) -> Phase 3 >=2 runtimes end-to-end smoke test (DONE)
- Pitfall 4 (mid-fold guillotine) -> Phase 4 per-fold checkpoint protocol ships WITH cap (DONE)
- Pitfall 5 (trajectory leak/bloat/fossilize) -> Phase 3 redaction-on-capture + bounded JSONL + schema-version metadata (DONE)
- Pitfall 6 (gate calibration) -> Phase 5 pre-registered held-out manifest + paired statistical test (DONE)
- Pitfall 7 (decoupling shipped wrong) -> Phase 8 sklearn-iris second consumer + end-to-end (DONE)
- Pitfall 8 (hardware mis-detect) -> Phase 7 detect-and-warn pattern + >=3 hardware shapes (DONE)
- Pitfall 9 (setup skill mis-scaffold) -> Phase 7 mandatory `automil check` + 1-min dry-run gate (DONE)

### milestone v1.0 complete

All 9 phases shipped. 92 plans executed. 69 v1 REQ-IDs delivered.
D-208 11-clause acceptance gate green (sub-gate B CI; sub-gates A+C workstation).
CHANGELOG 8.0.0 entry with BREAKING migration text published.
Framework purity: zero autobench refs in src/automil/ (D-206 grep gate).
Second consumer (sklearn-iris) runs end-to-end via documented contract (DEC-02/07).

### milestone v1.1 started

20 defect-remediation requirements defined across 6 categories (REC/APL/CFG/SCH/OPS/DBT).
Scope source: `tasks/test-run-issues.md` triage (2026-06-07) + 3 STATE.md deferred items + APL gap.
ISSUE-006 is won't-fix (documented design constraint). RTA cluster deferred (loop-opening).
Roadmap: Phases 9–14. Phase 9 (REC) first due to S0 priority of REC-01; Phases 11–14 parallel after Phase 9 clears.

## Session Continuity

**Last action:** Phase 12 (Scheduling & Overlay Isolation) COMPLETE (2026-06-12). All 3 plans executed, verified 2/2, code-reviewed (CR-01 `site.getsitepackages` AttributeError guard + 3 warnings + clause-07 stale-line fix, all applied). Definitive full-framework gate: **1039 passed, 1 failed, 53 skipped** — the single failure is the pre-existing `clause_11` stale v1.0 acceptance test (reads `.planning/REQUIREMENTS.md`, now the v1.1 file, expecting v1.0 DEC rows) — slated for Phase 14/DBT. Iris/sklearn tests PASS in the full run (the fixer's "sklearn not installed" was a subprocess-isolation artifact). Tracking artifacts committed: 2676515.

**Phase 12 deliverables:** SCH-01 `orchestrator.scheduling_policy` (best_fit default | least_loaded | round_robin) in `_find_best_gpu`; SCH-02 opt-in `editable_overlay_guard` + always-on `automil check` warning (D-199 honored — detect-and-warn always, inject only when opted in).

**Next:** Phase 13 (OPS — CLI Lifecycle & Operability, OPS-01..05): discuss → plan → execute → verify → code-review. Then Phase 14 (DBT — Housekeeping), then milestone audit → complete → cleanup. Continuing autonomously per Leo's directive (no pausing except true hard blockers).

---
*State initialised: 2026-05-01 after roadmap creation*
*milestone v1.0 complete: 2026-05-08*
*milestone v1.1 started: 2026-06-10 — roadmap Phases 9–14 defined*
*Mode: yolo, granularity: fine, parallelization: true*

## Deferred Items (from v1.0 close, 2026-05-08)

Items acknowledged and deferred at milestone v1.0 close on 2026-05-08:

| Category | Item | Status |
|----------|------|--------|
| verification_gap | 08-VERIFICATION.md sub-gate A (CCRCC reproduction, requires AUTOBENCH_CCRCC_ROOT) | human_needed |
| verification_gap | 08-VERIFICATION.md sub-gate C (heterogeneous consumers in same project) | human_needed |
| tech_debt | 3 pre-existing tick_cells failures (Phase 4-origin) | **addressed in Phase 14 (DBT-02)** |
| tech_debt | Phase 5 calibration pilot K-determination (Leo runs with CCRCC + CLWD cells) | deferred |
| tech_debt | Real SLURM/Ray cluster verification (BCK-05/06 success criterion 5) | deferred behind requires_slurm/requires_ray markers |
| tech_debt | External hardware shapes (CPU-only, ROCm laptop) per Phase 7 D-197 MEDIUM portability | deferred |

## Phase 8 follow-ups (deferred from v1.0, status updated)

1. **Sub-gate C (composability) workstation completion**: D-205 sub-gate C is a pytest.skip() in CI; the workstation-side body requires Leo's autobench + sklearn-iris co-registered project layout. Run manually on workstation; commit the active body when shape is stable.
2. **Schema version bump for graph.json**: **Addressed in v1.1 Phase 14 (DBT-01)** — `_load` detects legacy schema and dict-spreads on read.
3. **results.tsv schema generalization**: per OQ-8, the results.tsv writer keeps autobench-shaped 4-key columns (val_auc/val_bacc/test_auc/test_bacc). Sklearn-iris consumers write 0.0 for these columns (correct: no auc to record). Generalize when a third consumer surfaces that needs different display columns.
4. **viz dashboard generic-metric rendering**: per CONTEXT D-200 deferred, the viz metricFields array stays autobench-shaped. Auto-detect available keys from node.metrics and render dynamic sparklines for any consumer's metric set. Defer to post-v1.
5. **Allowlist anchor neighbor cleanup**: **Addressed in v1.1 Phase 14 (DBT-03)** — em-dashes neighboring the allowlist anchor removed.

## Phase 6 follow-ups (deferred from v1.0, status updated)

1. **Pre-existing tick_cells failures** (3 tests in `tests/test_tick_cells.py`): **Addressed in v1.1 Phase 14 (DBT-02)**.

2. **Real-cluster verification (BCK-05/06 success criterion 5)**: D-180/D-181 deferred. CCRCC `node_0176`-equivalent end-to-end on a real SLURM cluster + multi-node Ray cluster. Behind `@pytest.mark.requires_slurm`/`requires_ray` markers in `test_contract_real_slurm.py` / `test_contract_real_ray.py` — runs nightly only, not CI.

## Phase 5 Leo Follow-up (deferred, not a v1.1 item)

The calibration pilot (D-151, Plan 05-12) framework-side scaffold is committed at `90011e8`. The actual empirical K-threshold determination requires Leo to:

1. Choose a known-good change (recommended: CCRCC `node_0176` config applied to fresh cells).
2. Pick 3-5 fresh cells (3 CCRCC + 2 CLWD per recommendation).
3. Register a calibration manifest, submit, run `automil promote --calibrate <candidate_id>`.
4. Inspect the delta matrix in `archive/<candidate_id>/gate_evaluation.jsonl` and pick K such that the change passes consistently.
5. Update `.planning/phase-05-calibration.md` with chosen K and rationale; commit.
