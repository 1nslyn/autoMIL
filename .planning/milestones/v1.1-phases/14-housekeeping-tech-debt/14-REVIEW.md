---
phase: 14-housekeeping-tech-debt
reviewed: 2026-06-12T16:25:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - src/automil/graph.py
  - src/automil/backends/_orchestrator_daemon.py
  - tests/test_graph.py
  - tests/test_framework_purity.py
  - tests/acceptance/test_phase8_acceptance.py
  - tests/test_tick_cells.py
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-12T16:25:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found (1 Info only — no Blockers, no Warnings)

## Summary

Phase 14 (Housekeeping & Tech Debt, final v1.1 phase) contains one real-logic
change (DBT-01 legacy-schema migration in `graph.py`), one comment-only edit to
framework code (DBT-03 in `_orchestrator_daemon.py`), two test-anchor updates
(`test_framework_purity.py`, `test_phase8_acceptance.py`), three genuine new
regression tests (`test_graph.py`), and one comment-only attribution block
(`test_tick_cells.py`).

I reviewed adversarially against the four named risk areas. **The DBT-01
migration is correct and airtight; the comment-only edits introduced zero
behavioral change; the new graph tests are genuine (real-constructor
round-trips, not hand-built results); clause_11 precedence is correct.** The only
finding is an Info-level note on a tolerated brittle line-number anchor that the
phase brief explicitly permitted.

This is an honest low-finding result, not a soft pass. Each risk area was traced
to ground truth and (where relevant) re-executed against the live tree.

## Narrative Findings (AI reviewer)

### DBT-01 migration (graph.py:84-151) — verified CORRECT

- **Ordering fix is real.** `_on_disk_schema_version` is captured at L87 *before*
  the `setdefault` defaults block (L93-115) writes `schema_version=2`. An
  absent-key legacy graph therefore reads `1` and migrates. Re-executed
  `test_legacy_schema_absent_version` — passes.
- **Idempotency is airtight.** The gate `_on_disk_schema_version < 2` is False for
  a post-D-200 graph (`schema_version=2`), so the migration loop never runs. The
  inner `"metrics" not in _node` guard is a second layer: any node written by
  `add_executed`/`promote` already carries a `metrics` dict (always populated via
  `dict(metrics)` containing at least `composite`), so even a malformed
  `schema_version=1` graph that already has metrics is left untouched.
  `test_post_d200_graph_not_remigrated` asserts no spurious keys — passes.
- **No crash on partial/absent flat keys.** `{k: _node.get(k, 0.0) for k in ...}`
  defaults each missing legacy key to `0.0`. Handles "some but not all 4" and
  "neither metrics nor flat keys" without KeyError.
- **Read-only in memory.** No `save()` / `locked_update()` in `__init__`; the
  warning explicitly says "Re-save to persist." Confirmed.
- **No reader interaction breakage.** `best_node`/`promote`/`recompute_best` and
  production consumers read `node["metrics"]` defensively; the migrated dict
  satisfies them.

### Test genuineness (test_graph.py:497-629) — GENUINE, not theater

All three new tests write a real fixture via `graph_path.write_text(json.dumps(...))`
and load through the real `ExperimentGraph(graph_path)` constructor. Assertions
target migrated in-memory state (`node["metrics"]["val_auc"]`,
`g._data["schema_version"] == 2`, exact key-set equality). No hand-constructed
result shortcuts the migration path.

### DBT-03 (_orchestrator_daemon.py) — clean, comment-only

`git diff --stat` shows 3 insertions / 5 deletions, all on `#`-prefixed lines
(deferral comment trimmed, em-dash → `--`). No code line changed. The anchor
`Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)` lands exactly on L60 (verified
via `grep -n` and `sed -n '60p'`), matching the updated `_ALLOWLIST` key. The
`AUTOBENCH_*_ROOT` mention is the pre-existing allowlisted comment (not a new
D-206 ref). clause_07 now anchors on stable comment text. All framework-purity
and clause tests re-executed — 5 passed.

### clause_11 (test_phase8_acceptance.py:297-316) — correct

Both `.planning/milestones/v1.0-REQUIREMENTS.md` and `.planning/REQUIREMENTS.md`
exist on disk; the flipped precedence correctly selects the archived v1.0 record
for the v1.0 acceptance gate. `req_path` is always assigned before `.read_text()`
(the `else` branch assigns `live_path` purely to trip the assert message). Test
re-executed — passes. No other clause depends on the old precedence.

### DBT-02 (test_tick_cells.py:82-89) — clean

An 8-line attribution comment was added above the existing test structure. No
assertion, import, fixture, or dataclass was altered. The regression guard
remains valid.

## Info

### IN-01: Allowlist still uses a brittle file:line key for the daemon anchor

**File:** `tests/test_framework_purity.py:47`
**Issue:** The `_ALLOWLIST` entry is keyed on
`"src/automil/backends/_orchestrator_daemon.py:60"` — a hardcoded line number
that must be hand-updated on every import shift above it (it has drifted
:62 → :60 this phase, and :56 → :62 in prior phases). DBT-03 moved the
acceptance-side anchor (clause_07) to stable comment text, but the source-of-truth
allowlist key here remains line-number-coupled.
**Fix:** This is a tolerated design, explicitly permitted by the phase brief
(the line-number fallback is acceptable). The companion test
`test_allowlist_anchors_still_present` plus the content-anchor substring check in
`_is_allowlisted` make drift fail loudly rather than silently bypass, so there is
no correctness risk. If a future phase wants to eliminate the recurring manual
update, migrate the allowlist key to a `(rel_path, anchor_substring)` lookup that
greps for the anchor line at test time instead of pinning a line number. Not
required for this phase.

---

_Reviewed: 2026-06-12T16:25:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
