# Phase 9: State & Recovery Integrity - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

The framework's recorded truth — terminal state, status vocabulary, and budget-cell
identity — must be **correct, single-sourced, and survive a mid-run kill without losing
completed work**. Four requirements:

- **REC-01 (S0):** SIGTERM/timeout-killed runs aggregate completed folds into `result.json`
  instead of reporting `composite=0.0`.
- **REC-02 (S1):** A single terminal-state writer produces `graph.json`, `completed/<node>.json`,
  archive `result.json`, and `results.tsv` — so rank/dashboard/TSV/completion JSON never disagree.
- **REC-03 (S1):** One canonical status vocabulary; all framework recovery payloads validate
  against `result.schema.json`.
- **REC-04 (S1):** Budget cells keyed by `(dataset, encoder, mil_model)` via a required
  `--mil-model` field; re-parenting no longer opens a fresh 6h budget for the same model.

**Not in this phase:** opening any closed MIL training loop (RTA / ISSUE-007 — deferred);
variant application (Phase 10); config/argparse fidelity (Phase 11). This phase only fixes
how the framework *records and recovers* terminal state and budget identity.

</domain>

<decisions>
## Implementation Decisions

### A. Partial-fold recovery semantics (REC-01)
- **D-01:** Partial results are **quarantined**. A mid-run kill with N of M folds aggregates
  the **mean over completed folds** (`sum(composites)/n`, the existing `aggregate_folds`
  computation) and records `status=partial`. Partials are **excluded from keep/discard and
  `best_node` selection** but remain visible in `automil rank` and the dashboard. Rationale:
  a 2-fold mean must never spuriously beat a full-fold parent by variance; preserves
  honest-claims-at-budget discipline.
- **D-02:** `runtime_helpers.register_sigterm_flush()` aggregates and writes to
  `AUTOMIL_RESULTS_DIR` (the archive) when set, **not** `Path.cwd()` (the current bug).
- **D-03:** `_collect_or_synthesize_result()` MUST try archive fold-aggregation
  (`aggregate_folds(archive, expected)`) **before** synthesizing a timeout/crash result.
- **D-04:** Timeout handling becomes **main-PID-first**: SIGTERM the main PID first so its
  flush handler can write the partial cleanly, then SIGKILL the whole process group after a
  **configurable `orchestrator.timeout_grace_seconds` (default 10s)**. Replaces today's
  "SIGTERM the process group, sleep 5s, SIGKILL group."

### B. Status vocabulary (REC-03)
- **D-05:** Split the model: `status` stays a **tight enum** =
  `[completed, crash, budget_killed, cancelled, partial]`; add a separate **free-form
  `termination_reason`** field (`timeout`, `oom`, `sigterm`, etc.) for granularity.
  Schema stays small and validatable; detail preserved out-of-band.
- **D-06:** Canonicalize the `crashed` → `crash` drift (today `cells/reconcile.py` emits
  `crashed` for zero folds, which is not in the enum). All framework payloads canonicalize
  to the enum before writing `result.json`.
- **D-07:** `result.schema.json` is updated to allow the new enum value (`partial`) and the
  optional `termination_reason` field. (Whether to bump a schema version is a planner
  decision — see Open Implementation Notes.)
- **D-08:** Partial rows are **written to `results.tsv`** carrying their status, so the
  operator sees every terminal outcome in one ledger. Visibility ≠ comparability — D-01's
  quarantine still governs whether a partial affects `best_node`. (Today the TSV excludes
  `partial`.)

### C. Single terminal-state writer (REC-02)
- **D-09:** Extract a **standalone `terminal_writer` module/function** (not a daemon-private
  method) that writes all four artifacts in a **fixed order**: graph node (via the locked
  graph API / atomic `save()`) → `completed/<node>.json` → archive `result.json` →
  `results.tsv`. Both `_handle_completion` and `_handle_cap_killed_completion` call it.
  Rationale: unit-testable without a live daemon; daemon stays orchestration-only.
- **D-10:** The terminal_writer is the **sole `results.tsv` writer** (preserves the existing
  invariant that the orchestrator — never `train.py` — writes the TSV) and updates the graph
  **through the locked API**, never by direct dict mutation, for lock-safety/atomicity.
- **D-11:** Reconcile gains an **opt-in `automil reconcile --from-archive [<node>|all]`** that
  treats archive `result.json` as authoritative and refreshes **existing** nodes. Default
  `reconcile` stays missing-node recovery only (no surprise clobbers of live graph state).
  Matches the triage's "authoritative when explicitly requested."

### D. Budget-cell identity (REC-04)
- **D-12:** `--mil-model` is **required-with-inference**: resolve as explicit `--mil-model`
  flag → config (`run.mil_model`) → **error if neither**. Honors the "required" intent (a
  model is always pinned to the cell) without breaking existing scripted submits that declare
  the model in config. Applies to both `propose` and `submit`.
- **D-13:** Cell key changes from `sha256(dataset|encoder|parent_id)` to
  `sha256(dataset|encoder|mil_model)`. **Graph parent lineage stays separate from budget
  identity** (re-parenting must not fork the budget). `make_cell_id` signature changes
  `parent_id` → `mil_model`; all callers updated.
- **D-14:** `mil_model` is **free-form, normalized** — strip + lowercase + collapse internal
  whitespace before hashing into the key. **No registry validation** (autoMIL is generic and
  cannot enumerate a consumer's models — PROJECT.md "framework is domain-agnostic"). Document
  the canonical form so e.g. `CLAM_SB` and `clam_sb` collapse to one cell.
- **D-15:** Existing parent-keyed cells migrate via a **back-fill helper**
  (`automil cells migrate`, or a reconcile step): re-derive `mil_model` for existing executed
  nodes and **merge their elapsed budget** into the new `(dataset, encoder, mil_model)` cell
  so Leo's live TCGA-LUAD / CCRCC budget totals carry across the cutover. New submits use the
  new key immediately.

### Claude's Discretion
- Grace-window default (10s) is a starting value; the planner may tune.
- Exact `termination_reason` value set beyond `timeout`/`oom`/`sigterm` (e.g. `sigkill`,
  `unknown`) — pick a small documented set.
- Internal module/function naming (`terminal_writer`), and whether the back-fill is a new
  `cells migrate` subcommand vs. a `reconcile` extension — planner's call, as long as the
  budget-merge semantics in D-15 hold.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope source & verified defect behavior (read first)
- `tasks/test-run-issues.md` — the dated triage (2026-06-07) with per-issue **verified
  behavior + proposed fix**. Directly relevant entries:
  - **ISSUE-009 / 018 / 019** (S0) → REC-01 (partial-fold recovery)
  - **ISSUE-013 / 014** (S1) → REC-02 (split terminal-state writers)
  - **ISSUE-002** (S1) → REC-03 (status/schema drift)
  - **ISSUE-024** (S1) + **CONSTRAINT-01** → REC-04 (cell key by MIL model, not parent)

### Requirements & roadmap (the locked "what")
- `.planning/REQUIREMENTS.md` §"State & recovery integrity (REC)" — REC-01..04 text.
- `.planning/ROADMAP.md` §"Phase 9: State & Recovery Integrity" — goal + 4 success criteria.

### System map & constraints
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/CONCERNS.md` — existing
  orchestrator/graph/cells architecture (do NOT re-document).
- `.planning/PROJECT.md` §"Key Decisions" / §"Constraints" — "autoMIL is generic; autobench
  is one consumer" (grounds D-14, no registry validation) and "6h cap = per-cell-total"
  (grounds REC-04 budget semantics).
- `CLAUDE.md` §"Result Contract" + §"Key design decisions" — `results.tsv` is
  orchestrator-only; keep/discard is single-axis composite; `_recover_orphans` only in the
  daemon loop (grounds D-09/D-10).

</canonical_refs>

<code_context>
## Existing Code Insights

### Integration Points (file:line from codebase scout, 2026-06-10)
- **REC-01:**
  - `src/automil/runtime_helpers.py` ~L32-59 — `register_sigterm_flush()` + `aggregate_folds()`
    caller; **writes to `Path.cwd()` today — must use `AUTOMIL_RESULTS_DIR`** (D-02).
  - `src/automil/cells/reconcile.py` ~L72-85 — `aggregate_folds` returns
    `partial`/`crashed`/`completed`, composite = `sum(composites)/n`; `_crashed_payload`
    emits `crashed` (the drift to canonicalize, D-06). `reconcile_budget_kill()` writes
    archive `result.json` ~L138.
  - `src/automil/backends/_orchestrator_daemon.py` ~L1322-1386 —
    `_collect_or_synthesize_result()` synthesizes `oom`/`timeout`/`crash`/`completed`;
    **does NOT try archive aggregation first** (D-03 fixes). `_handle_timeout()` ~L1434-1456:
    `os.killpg(...SIGTERM)`, `sleep(5)`, SIGKILL group, sets `self._timed_out[exp_id]`
    (D-04 rewrites to main-PID-first + configurable grace).
- **REC-02:**
  - `_orchestrator_daemon.py` `_handle_completion()` ~L1151-1234 — writes
    `completed/<node>.json` (L1207-1209) + `results.tsv` via `_append_results_tsv` (L1212);
    **does NOT write `graph.json`**.
  - `_handle_cap_killed_completion()` ~L1258-1320 — writes `graph.json` (L1296/1307) +
    archive `result.json`; **does NOT write completed JSON or TSV**. → both converge on the
    new `terminal_writer` (D-09).
  - `src/automil/graph.py` — `save()` atomic via tempfile→`os.rename` ~L977-989;
    `locked_update()` ~L58; `reconcile()` recovers only **missing** nodes ~L735-786, skips
    existing executed ~L611 (D-11 adds opt-in existing-node refresh).
  - `src/automil/cli/reconcile.py` ~L29-79 — recompute-best + default missing-only recovery;
    add `--from-archive` here.
- **REC-03:**
  - `src/automil/schemas/result.schema.json` ~L14-17 — enum
    `["completed","crash","budget_killed","cancelled"]`; add `partial` + optional
    `termination_reason` (D-05/D-07).
- **REC-04:**
  - `src/automil/cells/state.py` ~L97-106 — `make_cell_id(dataset, encoder, parent_id)`
    → `(dataset, encoder, mil_model)` (D-13); `write_cell()` atomic ~L141-151.
  - `src/automil/cells/registry.py` ~L33-96 — `get_or_create_cell(...)`; cells persisted at
    `automil/cells/<cell_id>.json` (back-fill target for D-15).
  - `src/automil/cli/submit.py` ~L23-37 (options) / L343-371 (cell creation) — add
    `--mil-model`; resolve flag → config `run.mil_model` (D-12).
  - `src/automil/cli/propose.py` ~L80-87 (options) — add `--mil-model`.

### Established Patterns (must follow)
- **Atomic file writes** via `tempfile` + `os.replace`/`os.rename` (`graph.save`,
  `write_cell`) — the new `terminal_writer` must follow this for all four artifacts.
- **`results.tsv` is orchestrator-only**, never written by `train.py` — terminal_writer is
  the single TSV writer (CLAUDE.md).
- **`_recover_orphans()` only runs in the daemon loop**, never on construction — don't let
  the new writer/reconcile path trigger orphan recovery from `status`/`stop`.
- **Cell IDs are deterministic `sha256(...)[:16]`** — re-keying must stay deterministic so
  re-submits join the existing cell.

</code_context>

<specifics>
## Specific Ideas

- The decisions deliberately keep **visibility and comparability orthogonal**: a partial (or
  any non-`completed`) outcome should be *seen* everywhere (rank, dashboard, TSV) but only
  affect `best_node`/keep-discard when it is a fair, full comparison. This principle (D-01 +
  D-08) is the through-line for Phase 9 — apply it wherever a status influences search.
- The single-writer fixed order is **graph node first** (the authoritative state), then the
  derived artifacts, so a crash mid-write leaves the graph correct rather than the TSV.

</specifics>

<deferred>
## Deferred Ideas

- **`termination_reason` → viz dashboard rendering** — surfacing the new reason field in the
  3D dashboard is **deferred to post-v1** (STATE.md Phase 8 follow-up #4: viz `metricFields`
  stays autobench-shaped until generic-metric rendering lands). Phase 9 only persists the
  field; viz keeps rendering `status`.
- **Opening the closed CLAM training loop** (ISSUE-007 / RTA-01/02) — explicitly out of
  scope for v1.1; future registry-adoption milestone.
- **graph.json legacy schema round-trip (DBT-01)** — separate concern, owned by Phase 14.
  Phase 9 touches `result.schema.json`, not the graph schema migration.

### Open Implementation Notes (for the researcher/planner, not blocking)
- **`result.schema.json` version bump** — adding `partial` + `termination_reason` is a schema
  change; planner decides whether it warrants an explicit schema version field/migration
  (distinct from DBT-01's graph.json versioning).
- **Per-fold checkpoint atomicity** — REC-01 relies on `fold_<i>_result.json` files written
  after each fold; researcher should confirm those writes are atomic so a kill mid-fold-write
  can't leave a half-written fold result that corrupts aggregation.

</deferred>

---

*Phase: 9-State & Recovery Integrity*
*Context gathered: 2026-06-10*
