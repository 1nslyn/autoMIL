# Phase 9: State & Recovery Integrity - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-10
**Phase:** 9-State & Recovery Integrity
**Areas discussed:** Partial-fold recovery, Status vocabulary, Single terminal-state writer, Budget-cell identity
**Mode:** `--all` (all gray areas auto-selected; discussed interactively)

---

## A. Partial-fold recovery (REC-01)

### A1 — Partial result participation in search

| Option | Description | Selected |
|--------|-------------|----------|
| Quarantine partials | Record mean-over-completed with `status=partial`, exclude from keep/discard & best_node; visible in rank/dashboard | ✓ |
| Discount partials | Fold-count-scaled composite (mean × completed/expected), still comparable | |
| Treat as normal | Mean-over-completed competes like any full node | |

**User's choice:** Quarantine partials (Recommended)
**Notes:** Prevents a noisy 2-fold mean from spuriously beating a full-fold parent; preserves honest-claims-at-budget discipline.

### A2 — Timeout kill signal handling

| Option | Description | Selected |
|--------|-------------|----------|
| Main-PID-first + config grace | SIGTERM main PID first (flush handler writes partial), then SIGKILL group after configurable `orchestrator.timeout_grace_seconds` (default 10s) | ✓ |
| Main-PID-first, fixed grace | Signal main PID first, keep fixed 5s, no new config knob | |
| Keep group-SIGTERM, longer grace | Leave process-group SIGTERM, lengthen grace window | |

**User's choice:** Main-PID-first + config grace (Recommended)
**Notes:** Robust against PyTorch DataLoader workers; grace window tunable per environment.

---

## B. Status vocabulary (REC-03)

### B1 — Reconciling 8 runtime statuses with the 4-value schema enum

| Option | Description | Selected |
|--------|-------------|----------|
| status + termination_reason | Tight `status` enum (+`partial`) plus free-form `termination_reason` for detail; `crashed`→`crash` | ✓ |
| Expand the enum | Add partial/timeout/oom/crashed to the enum with normalization map | |
| Canonicalize to 4 | Map everything to the existing 4 before write; lose granularity | |

**User's choice:** status + termination_reason (Recommended)
**Notes:** Schema stays small and validatable; granularity preserved out-of-band.

### B2 — results.tsv inclusion of partial rows

| Option | Description | Selected |
|--------|-------------|----------|
| Include with status column | Write partial rows to results.tsv carrying status/reason | ✓ |
| Completed-only TSV | Keep TSV to fully-completed rows; partials in graph.json/dashboard only | |

**User's choice:** Include with status column (Recommended)
**Notes:** Visibility independent of comparability — A1 quarantine still governs best_node impact.

---

## C. Single terminal-state writer (REC-02)

### C1 — Where the single writer lives

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone module | `terminal_writer` writes all 4 artifacts in fixed order via locked graph API; both completion paths call it; unit-testable | ✓ |
| Daemon-internal method | Private `_finalize_node()` on the daemon; less indirection, daemon-only tests | |

**User's choice:** Standalone module (Recommended)
**Notes:** Daemon stays orchestration-only; writer testable without a live daemon.

### C2 — reconcile refreshing existing nodes from archive

| Option | Description | Selected |
|--------|-------------|----------|
| Opt-in `--from-archive` flag | Default stays missing-only recovery; explicit flag treats archive result.json as authoritative for existing nodes | ✓ |
| Automatic refresh | reconcile always refreshes existing nodes when archive differs | |
| Leave reconcile unchanged | Single writer alone closes the split | |

**User's choice:** Opt-in `--from-archive` flag (Recommended)
**Notes:** Matches triage's "authoritative when explicitly requested"; no surprise clobbers of live graph state.

---

## D. Budget-cell identity (REC-04)

### D1 — `--mil-model` contract strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Required-with-inference | flag → config (`run.mil_model`) → error if neither | ✓ |
| Hard-required flag | Error unless `--mil-model` passed explicitly; no fallback | |
| Optional with sentinel | Fall back to parent's mil_model / "default" | |

**User's choice:** Required-with-inference (Recommended)
**Notes:** Honors "required" intent without breaking existing scripted submits that declare the model in config.

### D2 — Migration of existing parent-keyed cells

| Option | Description | Selected |
|--------|-------------|----------|
| Back-fill helper | `automil cells migrate` re-derives mil_model for existing nodes, merges elapsed budget into the new key | ✓ |
| Clean break | Old cells go dormant; budget restarts under new keying | |
| Dual-read transition | Read both keys for a window, summing budget | |

**User's choice:** Back-fill helper (Recommended)
**Notes:** Preserves live TCGA-LUAD/CCRCC budget totals across the cutover.

### D3 — mil_model handling in the cell-key hash

| Option | Description | Selected |
|--------|-------------|----------|
| Free-form, normalized | Strip + lowercase + collapse whitespace before hashing; no registry validation | ✓ |
| Validate against registry | Reject models not in the registry | |

**User's choice:** Free-form, normalized (Recommended)
**Notes:** Consistent with the generic-framework constraint (autoMIL can't enumerate a consumer's models).

---

## Claude's Discretion

- Grace-window default (10s) — starting value, planner may tune.
- Exact `termination_reason` value set beyond `timeout`/`oom`/`sigterm`.
- Internal naming (`terminal_writer`); back-fill as new `cells migrate` subcommand vs. `reconcile` extension (budget-merge semantics in D-15 must hold).

## Deferred Ideas

- `termination_reason` → viz dashboard rendering — deferred to post-v1 (STATE.md Phase 8 follow-up #4).
- Opening the closed CLAM training loop (ISSUE-007 / RTA) — out of scope for v1.1.
- graph.json legacy schema round-trip (DBT-01) — owned by Phase 14.
- Open implementation notes (not blocking): `result.schema.json` version bump; per-fold checkpoint write atomicity.
