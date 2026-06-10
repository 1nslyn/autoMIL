---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: Bug Fixing
status: planning
last_updated: "2026-06-10T17:47:42.632Z"
last_activity: 2026-06-10
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# State: autoMIL - F2-readiness framework refactor

## Project Reference

**Project:** autoMIL -- F2-readiness framework refactor
**Core Value:** An agent can autonomously discover model improvements for any user's training code under a 6-hour-per-cell budget, with discovered variants reproducible, attributable to their parents, and portable across machines and LLM runtimes.

**Canonical documents** (load these before any phase work):

- `.planning/PROJECT.md` -- milestone definition, validated/active/out-of-scope sets, constraints, key decisions
- `.planning/REQUIREMENTS.md` -- 69 v1 REQ-IDs across 10 categories (CLN, REG, BCK, TRJ, MRT, CAP, GTE, CLI, STP, DEC) with traceability table
- `.planning/ROADMAP.md` -- 9 phases, success criteria, dependency graph, parallel-execution candidates
- `.planning/codebase/` -- existing system map (ARCHITECTURE.md, CONCERNS.md, STACK.md). Do NOT re-document.
- `.planning/research/{SUMMARY,STACK,FEATURES,ARCHITECTURE,PITFALLS}.md` -- research synthesis informing phase design
- `tasks/automil_qa.md`, `tasks/automil_proposal.md` -- design-gap diagnostic + F2 proposal that motivated this milestone
- `CLAUDE.md` -- project instructions and Leo's standing directives
- `~/.claude/projects/-home-jma-Documents-yinshuol-autoMIL/memory/MEMORY.md` -- Leo's standing memory (saturate GPUs, research before submit, never blind-checkout, architectural-not-hyperparam, never ask continue autonomously)

**Current focus:** Phase 08 -- Decoupling completion + acceptance (COMPLETE; milestone v1.0 complete)

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-06-10 — Milestone v1.1 started

## Phase 8 follow-ups (deferred , not Phase 8 blockers)

1. **Sub-gate C (composability) workstation completion**: D-205 sub-gate C is a pytest.skip() in CI; the workstation-side body requires Leo's autobench + sklearn-iris co-registered project layout. Run manually on workstation; commit the active body when shape is stable.
2. **Schema version bump for graph.json**: D-200 changes storage shape but did NOT bump schema_version. Optional one-shot migration helper in graph.py:_load detecting v1 and dict-spreading existing nodes' val_auc/etc on read. Cost: ~30 lines; benefit: pre-D-200 graph.json files round-trip cleanly. Defer to v1.1 unless Leo encounters silent KeyError on legacy graph.json files.
3. **results.tsv schema generalization**: per OQ-8, the results.tsv writer keeps autobench-shaped 4-key columns (val_auc/val_bacc/test_auc/test_bacc). Sklearn-iris consumers write 0.0 for these columns (correct: no auc to record). Generalize when a third consumer surfaces that needs different display columns.
4. **viz dashboard generic-metric rendering**: per CONTEXT D-200 deferred, the viz metricFields array stays autobench-shaped. Auto-detect available keys from node.metrics and render dynamic sparklines for any consumer's metric set. Defer to post-v1.
5. **Allowlist anchor neighbor cleanup**: per F-13 LOW finding, `_orchestrator_daemon.py` lines 45-55 contain em-dashes in neighboring comments to the allowlist anchor at line 54. Future re-flow could break the anchor; low-priority cleanup for v1.1.

## Phase 6 follow-ups (deferred , not Phase 6 blockers)

1. **Pre-existing tick_cells failures** (3 tests in `tests/test_tick_cells.py`): `test_tick_cells_active_to_refusing_new`, `test_tick_cells_terminating_fires_cancel_with_cap_reason`, `test_tick_cells_finalized_when_running_empty`. Origin: Phase 4 `_orchestrator_daemon.py:_tick_cells` -- the test mocks expect `cells_dir = tmp_path / "automil" / "cells"` but the daemon's actual cells directory resolution may have shifted. Pre-existing at start of Phase 6 session (verified via `git checkout cca0bc0 -- src/automil/backends/_orchestrator_daemon.py` bisection). Not caused by namespace migration. Recommended fix: either (a) Phase 8 cleanup audits Phase 4 wiring, or (b) Leo runs targeted `/gsd-debug` on these 3 specific tests.

2. **Real-cluster verification (BCK-05/06 success criterion 5)**: D-180/D-181 deferred. CCRCC `node_0176`-equivalent end-to-end on a real SLURM cluster + multi-node Ray cluster. Behind `@pytest.mark.requires_slurm`/`requires_ray` markers in `test_contract_real_slurm.py` / `test_contract_real_ray.py` -- runs nightly only, not CI.

## Phase 5 Leo Follow-up (deferred , not a blocker for Phase 6)

The calibration pilot (D-151, Plan 05-12) framework-side scaffold is committed at `90011e8`. The actual empirical K-threshold determination requires Leo to:

1. Choose a known-good change (recommended: CCRCC `node_0176` config applied to fresh cells).
2. Pick 3-5 fresh cells (3 CCRCC + 2 CLWD per recommendation).
3. Register a calibration manifest, submit, run `automil promote --calibrate <candidate_id>`.
4. Inspect the delta matrix in `archive/<candidate_id>/gate_evaluation.jsonl` and pick K such that the change passes consistently.
5. Update `.planning/phase-05-calibration.md` with chosen K and rationale; commit.

## Performance Metrics

| Metric | Value | Notes |
|---|---|---|
| Total v1 requirements | 69 | Mapped 100% to phases |
| Total phases | 9 | Phase 0 + 8 substantive |
| Estimated wall-clock work | ~36-44 days | Per research/SUMMARY.md sub-totals |
| Granularity | fine | Per `.planning/config.json` |
| Parallelization | enabled | Phase 6 and Phase 7 are the strongest parallel pair |
| Mode | yolo | Auto-approve gates within roadmap; Leo reviews artifacts |
| Phase 02 P02-01 | 6m | 3 tasks | 5 files |
| Phase 02 P02-06 | 8m | 3 tasks | 1 file |
| Phase 02 P02-07 | 8m | 5 tasks | 5 files |
| Phase 02 P02-08 | 25 | 6 tasks | 5 files |
| Phase 06 P01 | 360 | 3 tasks | 8 files |
| Phase 06 P07 | 900 | 1 tasks | 1 files |
| Phase 07 P05 | 55 | 3 tasks | 4 files |
| Phase 08 P01 | 600 | 2 tasks | 6 files |
| Phase 08-decoupling-completion-acceptance P06 | 8m | 3 tasks | 7 files |
| Phase 08 P09 | 1800 | 3 tasks | 5 files |
| Phase 08 P10 | 17m | 4 tasks | 5 files |

## Accumulated Context

### Decisions logged (from PROJECT.md -> ROADMAP.md)

- MockSLURMBackend: PENDING/RUNNING->CRASHED on restart (timer threads cannot resume) -- Done (02-06, 2026-05-02)
- BCK-04 lint allowlist includes viz/server.py (viz daemon PID lifecycle, not job-control) -- Done (02-07, 2026-05-02)
- Registry-first, not config-first, for cross-dataset isolation -- Done (Phase 1)
- Skills only for autonomous setup; CLI for everything else -- Done (Phase 7)
- Pluggable orchestrator backends with `local` as default -- Done (Phase 2 ABC, Phase 6 SLURM/Ray)
- Multi-runtime agent support is in v1, not deferred -- Done (Phase 3)
- 6h cap = per-cell-total, framework-enforced -- Done (Phase 4)
- Search-scope mode flag (`architecture-preserving | free`) -- Done (Phase 1, default `free`)
- autoMIL is generic; autobench is one consumer -- Done (audited and verified in Phase 8)
- Tier 1 mechanical fixes before structural refactor -- Done (5 commits, 2026-05-01)
- `port-variant` and `promote-variant` are CLI, not skills -- Done (Phase 1)
- env.required is mandatory, env.passthrough is consumer-controlled -- Done (Phase 8)
- Framework purity: zero autobench refs in src/automil/ -- Done (Phase 8, D-206)

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

## Session Continuity

**Last action:** Phase 08 plan 10 (08-10) executed 2026-05-08. D-208 acceptance gate, CHANGELOG 8.0.0, STATE/ROADMAP/REQUIREMENTS updates. milestone v1.0 complete.

**Phase 08 prior session log:** `/gsd-discuss-phase 8` (CONTEXT bootstrapped, 37 engineering decisions D-199..D-208), `/gsd-plan-phase 8` (10 plans across 4 waves), plan-checker iter-1 + iter-2 PASS. Wave 1 (08-01/02/03 parallel): result schema, dict-spread, sklearn-iris consumer. Wave 2 (08-04/05 parallel): env.required check + passthrough, autobench purge. Wave 3 (08-06/07/08 parallel): contract doc, docs-exist test, framework purity test. Wave 4 sequential: 08-09 (final acceptance gate) -> 08-10 (D-208 aggregator + CHANGELOG + milestone close).

**Resume file:** None

**To resume in a fresh session:** milestone v1.0 is complete. No active phase work. Run `/gsd-verify-work 8` for final verification, then tag v1.0.0.

---
*State initialised: 2026-05-01 after roadmap creation*
*Phase 0 context gathered: 2026-05-01*
*Mode: yolo, granularity: fine, parallelization: true*
*milestone v1.0 complete: 2026-05-08*

## Deferred Items

Items acknowledged and deferred at milestone v1.0 close on 2026-05-08:

| Category | Item | Status |
|----------|------|--------|
| verification_gap | 08-VERIFICATION.md sub-gate A (CCRCC reproduction, requires AUTOBENCH_CCRCC_ROOT) | human_needed |
| verification_gap | 08-VERIFICATION.md sub-gate C (heterogeneous consumers in same project) | human_needed |
| tech_debt | 3 pre-existing tick_cells failures (Phase 4-origin, documented as Phase 6 follow-up) | deferred |
| tech_debt | Phase 5 calibration pilot K-determination (Leo runs with CCRCC + CLWD cells) | deferred |
| tech_debt | Real SLURM/Ray cluster verification (BCK-05/06 success criterion 5) | deferred behind requires_slurm/requires_ray markers |
| tech_debt | External hardware shapes (CPU-only, ROCm laptop) per Phase 7 D-197 MEDIUM portability | deferred |
