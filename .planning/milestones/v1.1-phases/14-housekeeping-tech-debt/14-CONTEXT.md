# Phase 14: Housekeeping & Tech Debt - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning
**Mode:** Auto-decided (`/gsd-autonomous` → discuss `--auto`). The FINAL v1.1 phase. Grey areas resolved via best-practice defaults; grounded in DBT requirements + verified code anchors (read during scout). Research likely skippable — all items are contained fixes with exact anchors (DBT-01 is the only one with non-trivial logic).

<domain>
## Phase Boundary

Pre-D-200 `graph.json` files round-trip cleanly (legacy flat-key node schema migrates on
read), the (already-passing) tick_cells tests are guarded against regression, the orchestrator
daemon's acceptance/allowlist anchor is protected from re-flow breakage, and the milestone-close
test suite is fully green. Covers DBT-01, DBT-02, DBT-03 — PLUS a folded-in housekeeping fix for
the pre-existing `clause_11` D-208 acceptance test so the suite passes 100% at v1.1 close.
Independent of Phases 9–13. This is the last phase before milestone audit → complete → cleanup.

</domain>

<decisions>
## Implementation Decisions

### DBT-01 — pre-D-200 graph.json legacy-schema round-trip (Phase 8 follow-up #2)
- **D-01:** `ExperimentGraph.__init__` (`graph.py:73-124`) currently fills only TOP-LEVEL and
  META defaults; it does NOT migrate per-NODE schema. Pre-D-200 nodes stored flat metric keys
  (`val_auc`/`test_auc`/`val_bacc`/`test_bacc`) directly on the node; D-200/DEC-04 moved consumer
  metrics into an opaque `node["metrics"]` dict (`add_node` `:220`, `promote` `:304`). A pre-D-200
  graph.json therefore has nodes WITHOUT a `metrics` dict → any reader doing `node["metrics"][...]`
  KeyErrors. **Fix:** add a per-node migrate-on-read step in `__init__` (after the top-level/meta
  defaults block): for each node lacking a `metrics` dict, dict-spread the legacy flat keys into a
  freshly-built `node["metrics"]` (preserve the flat keys too for back-compat, or drop them — see
  D-02). Bump `schema_version` default to `2` and gate the migration on `schema_version < 2` (an
  absent `schema_version` reads as legacy/1). Loading a pre-D-200 file must succeed with a
  fully-populated node tree and no KeyError.
- **D-02 (Claude's discretion / planner's call):** exact legacy key set to spread (at minimum
  `val_auc`/`val_bacc`/`test_auc`/`test_bacc`; also consider `composite`, `global_delta`,
  `elapsed_min`, `vram_gb`, `gpu` if they were flat pre-D-200) and whether to keep or strip the
  flat keys after spreading. Research/plan confirms against the actual pre-D-200 node shape (check
  git history or `.planning/milestones/v1.0-*` artifacts for a real legacy graph.json sample).

### DBT-02 — three pre-existing tick_cells failures (Phase 6 follow-up #1)
- **D-03:** **VERIFY-AND-GUARD, do NOT re-implement.** The three named tests
  (`test_tick_cells_active_to_refusing_new`, `test_tick_cells_terminating_fires_cancel_with_cap_reason`,
  `test_tick_cells_finalized_when_running_empty`) ALREADY PASS — confirmed during scout: the full
  `tests/test_tick_cells.py` runs 13 passed, and the authoritative full-framework gate showed these
  passing. The Phase-4-origin `cells_dir` resolution mismatch was incidentally fixed (most likely by
  Phase 9's tick_cells / `self.graph` wiring + cells_dir work, REC-01/REC-02). **Action:** the plan
  must (a) confirm all three pass in isolation AND in the full suite, (b) identify the commit/change
  that resolved them (git blame / Phase 9 SUMMARYs) and record it, (c) ensure a regression guard
  exists (the tests themselves serve as the guard — confirm they assert the real cells_dir behavior,
  not a masked path), and (d) mark DBT-02 satisfied. If — and only if — a re-run reveals they fail in
  some context (e.g. a specific test-ordering), THEN apply the cells_dir fix. Do NOT manufacture a
  fix for an already-green requirement.

### DBT-03 — daemon allowlist/acceptance anchor em-dash cleanup (Phase 8 follow-up #5 / F-13)
- **D-04:** Replace the em-dash characters (`—`, U+2014) with ASCII (`-` or `--`) in the daemon
  comments NEIGHBORING the acceptance/allowlist anchor, so a future comment re-flow or a tool that
  mangles multi-byte typography cannot shift/break the anchor line that the D-208 acceptance test
  (clause_07) and the framework-purity test hardcode. Concrete sites found during scout:
  `_orchestrator_daemon.py:34-36` (the explicit "Left here until Phase 14 / DBT-03 anchor cleanup"
  marker comment) and the env-whitelist block em-dashes at `:58` and `:63`. **Fix:** convert those
  em-dashes to ASCII; then update/confirm the marker comment (drop the "Left here until Phase 14"
  deferral note once done). Add a guard: a CI grep / `automil check` assertion that no em-dash
  characters remain adjacent to the anchor comment. NOTE: the acceptance test (clause_07) currently
  hardcodes a daemon line number (`_orchestrator_daemon.py:62` per Phase 12) — if this cleanup shifts
  that line, update the hardcoded assertion in the SAME plan (this is exactly the brittleness DBT-03
  exists to end; consider making the anchor match by stable comment text rather than line number).

### clause_11 fold-in — D-208 acceptance test reads the wrong requirements file (milestone-close hygiene)
- **D-05:** Fold in a fix for `tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete`
  — the persistent suite failure carried since v1.1 began. Root cause (verified): the test resolves
  `req_path` as `.planning/REQUIREMENTS.md` FIRST and only falls back to
  `.planning/milestones/v1.0-REQUIREMENTS.md` if the former is ABSENT (`test_phase8_acceptance.py:302-309`).
  Since v1.1 created a fresh `.planning/REQUIREMENTS.md` (v1.1 content, ZERO DEC-01..07 rows), the test
  reads it and fails. The archived `.planning/milestones/v1.0-REQUIREMENTS.md` DOES exist and HAS all 7
  `| DEC-NN | Phase 8 | Complete |` rows (verified `:247-253`). **Fix:** this is the *v1.0* D-208
  acceptance gate — it must validate the v1.0 record. Flip the precedence: when
  `.planning/milestones/v1.0-REQUIREMENTS.md` exists, read it (the authoritative archived v1.0
  requirements); fall back to `.planning/REQUIREMENTS.md` only when the archive is absent (mid-v1.0
  state). This is a test-logic fix (the test's own comment already says "Honor either location" — the
  precedence is just backwards for the post-archive, new-milestone case). Verified: this makes clause_11
  pass and brings the full suite to 100% green at v1.1 close.

### Claude's Discretion
- DBT-01 exact legacy key set + keep-vs-strip flat keys (D-02).
- DBT-03 whether to additionally make clause_07's anchor match by stable comment text vs line number
  (reduces future brittleness — recommended if low-risk).
- Whether DBT-02's "regression guard" needs any new test or the existing 3 tests suffice (likely suffice).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope source
- `.planning/REQUIREMENTS.md` §"Housekeeping & tech debt (DBT)" — DBT-01/02/03 text.
- `.planning/ROADMAP.md` §"Phase 14: Housekeeping & Tech Debt" — 3 success criteria.
- `.planning/STATE.md` §"Phase 8 follow-ups" (#2 DBT-01, #5 DBT-03) + §"Phase 6 follow-ups" (#1 DBT-02)
  + §"Deferred Items" — the origin records for each tech-debt item.

### Code anchors (verified 2026-06-12 during scout)
- `src/automil/graph.py:73-124` (`ExperimentGraph.__init__` / `_load` — top-level+meta defaults only;
  DBT-01 adds per-node migration here), `:90` (`schema_version: 1` default → bump to 2), `:220`/`:304`
  (D-200 `metrics` opaque-dict storage — the post-migration shape).
- `tests/test_tick_cells.py` — the 3 DBT-02 tests (all currently PASS); 13 tests total green.
- `src/automil/backends/_orchestrator_daemon.py:34-36` (DBT-03 anchor marker comment), `:58`,`:63`
  (em-dashes to convert), `:62` (the env-whitelist line clause_07 hardcodes per Phase 12).
- `tests/acceptance/test_phase8_acceptance.py:302-320` (clause_11 `req_path` precedence — the D-05 fix
  site), and the clause_07 daemon-line assertion in the same file (update if DBT-03 shifts the line).
- `.planning/milestones/v1.0-REQUIREMENTS.md:247-253` (the v1.0 DEC-01..07 `Complete` rows clause_11
  must validate against).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ExperimentGraph.__init__`'s existing missing-key fill + `logger.warning` paper-trail pattern
  (`graph.py:106-124`) is the natural home for the DBT-01 per-node migration (extend, don't replace).
- The D-200 `metrics` opaque-dict contract (`add_node`/`promote`) defines the post-migration target shape.
- The clause_11 test already has the dual-location logic — DBT-05 only flips the precedence.

### Established Patterns
- Framework purity D-206: graph.py + _orchestrator_daemon.py are in `src/automil/` — zero
  autobench/AUTOBENCH_/benchmarks refs. Tests may touch `tests/` freely.
- Acceptance-anchor brittleness (hardcoded daemon line numbers in test_phase8_acceptance.py) is the
  recurring pain DBT-03 targets — prefer stable-text anchoring where feasible.
- Run `tests/` and `benchmarks/tests/` SEPARATELY (rootdir collision — standing lesson).

### Integration Points
- DBT-01 touches the read path every graph consumer hits (`rank`, viz, reconcile) — migration must be
  idempotent and not corrupt already-D-200 graphs (gate strictly on `schema_version < 2` / absent metrics).
- clause_11/clause_07 both live in the D-208 acceptance gate (`test_phase8_acceptance.py`) — keep both
  green; a DBT-03 line shift must be reflected in clause_07 in the same plan.

</code_context>

<specifics>
## Specific Ideas
- DBT-02 through-line: don't fix what isn't broken — verify the already-green tests, attribute the
  resolving change, guard against regression, close the requirement honestly.
- clause_11 through-line: a v1.0 acceptance gate must read the v1.0 record; the precedence bug only
  surfaced once v1.1 created a new live REQUIREMENTS.md. One-line precedence flip.
- Milestone-close goal: the FULL framework suite (`tests/`) should be 100% green (zero failures) when
  Phase 14 closes — DBT-01/03 + clause_11 are the remaining gaps; DBT-02 is already green.
</specifics>

<deferred>
## Deferred Ideas
- results.tsv schema generalization + viz generic-metric rendering (Phase 8 follow-ups #3/#4) remain
  deferred to post-v1 — NOT in DBT scope (they activate "when a third consumer surfaces").
- Real SLURM/Ray cluster verification + external hardware shapes remain deferred behind markers.

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 14-Housekeeping & Tech Debt (final v1.1 phase)*
*Context gathered: 2026-06-12 (auto-decided via /gsd-autonomous)*
