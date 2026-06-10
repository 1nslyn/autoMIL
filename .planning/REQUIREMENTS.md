# Requirements: autoMIL — v1.1 Bug Fixing

**Defined:** 2026-06-10
**Core Value:** An agent can autonomously discover model improvements for any user's training code under a 6-hour-per-cell budget, with discovered variants reproducible, attributable to their parents, and portable across machines and LLM runtimes.

**Scope source:** the dated defect triage `tasks/test-run-issues.md` (reviewed 2026-06-07, verified line-by-line against the current source tree) plus three v1.1-tagged housekeeping items from `STATE.md`, plus the registry-application integrity gap surfaced during scoping. This milestone **restores intended behavior** — including making the variant registry genuinely *apply* to the live model (no inert variants), verified by real experiments. It does **not open any closed MIL training loop**; that, and the variant kinds whose only application route requires it, stay deferred (see Out of Scope).

**"User" = operator/agent** driving autoMIL (the framework has no end-user UI; requirements are operator-centric).

## v1 Requirements

Committed scope of milestone v1.1. REQ-IDs grouped by defect theme. New category prefixes (REC/CFG/SCH/OPS/DBT) — no collision with v1.0's CLN/REG/BCK/TRJ/MRT/CAP/GTE/CLI/STP/DEC. Each maps to exactly one roadmap phase (filled in during roadmap creation).

### State & recovery integrity (REC)

<!-- The framework's recorded truth (terminal state, status, budget identity) must be correct and single-sourced. Highest-severity cluster. -->

- [ ] **REC-01**: When an experiment is SIGTERM/timeout-killed mid-run, completed folds already written to the archive are aggregated into `result.json` instead of being reported as a `composite=0.0` crash/timeout — `runtime_helpers` aggregates to `AUTOMIL_RESULTS_DIR`, and `_collect_or_synthesize_result()` tries archive fold-aggregation before synthesizing a timeout/crash. (ISSUE-009/018/019, S0)
- [ ] **REC-02**: Normal completion and cap-kill completion write `graph.json`, `completed/<node>.json`, archive `result.json`, and `results.tsv` through a single terminal-state writer, so dashboard / rank / TSV / completion JSON never disagree until manual reconciliation. (ISSUE-013/014, S1)
- [ ] **REC-03**: One canonical terminal/recovery status vocabulary is used everywhere; framework-generated recovery payloads (`partial`, `timeout`, `oom`, `crashed`) validate against `result.schema.json` (via schema aliases+normalization or canonicalization before write). (ISSUE-002, S1)
- [ ] **REC-04**: Budget cells are keyed by `(dataset, encoder, mil_model)`, fed by a required `--mil-model` metadata field on `propose`/`submit`; re-parenting a node no longer opens a fresh 6h budget for the same MIL model. (ISSUE-024, S1 — also satisfies CONSTRAINT-01)

### Variant application integrity (APL)

<!-- A registered variant MUST actually take effect on the model when selected — never silently inert. Apply through existing OPEN seams; do NOT open closed training loops (deferred → RTA / ISSUE-007). Verification by real experiment is in-bounds. -->

- [ ] **APL-01**: Selecting a registered variant via config (`automil apply <node>`) causes the live experiment to run with that variant applied to the actual model — a registered variant is never silently inert. Demonstrated end-to-end on the sklearn-iris reference, whose `classifier_v0` variant is currently never imported by its `train.py`.
- [ ] **APL-02**: Registered model / config / hyperparameter variants for the autobench CLAM consumer apply to the actual model through the existing `clam_train` args seam (`model_type`, `model_size`, `B`, `bag_weight`, `dropout`, optimizer/`lr`, `bag_loss`/`inst_loss` selectors) — **without editing `lib/` and without opening the closed training loop**. Verified by a real experiment whose composite differs from the un-applied baseline.
- [ ] **APL-03**: A variant whose only application route requires opening a closed training loop (e.g. a custom `LossVariant` callable injected into CLAM's `train_loop_clam`) is **detected and reported loudly** by the apply/validate path as "requires loop opening — deferred (ISSUE-007 / RTA)", never silently no-op'd. (Closes the inert-variant trap at its general root.)

### Config & run fidelity (CFG)

<!-- Config/snapshot values must drive runs; CLI/argparse defaults must not silently mask them. -->

- [ ] **CFG-01**: `run_experiment.py` argparse defaults for training overrides (`--seed`, `--lr`, `--max_epochs`, `--patience`, `--stop_epoch`, `--n_folds`) default to `None` and are only passed into `TrainConfig` when explicitly supplied, so snapshot/config dataclass defaults are honored. (ISSUE-015, S1)
- [ ] **CFG-02**: `submit --timeout` defaults to `None` and omits `timeout_min` from the queue spec when unset, so the orchestrator's configured `orchestrator.default_timeout_min` controls per-job timeout. (ISSUE-022, S1)
- [ ] **CFG-03**: A queue spec can carry a per-node run-command override (e.g. `--seed`, `--encoder`, `--lr`, `--n_folds`) layered on a config `run.command` base, without editing snapshotted code. (ISSUE-008, S1)

### Scheduling & overlay isolation (SCH)

<!-- Orchestrator scheduling correctness + worktree-overlay import integrity. -->

- [ ] **SCH-01**: An `orchestrator.scheduling_policy` knob (`best_fit | round_robin | least_loaded`) selects GPU placement; best-fit is reserved for memory-bound workloads so low-VRAM compute-bound jobs no longer over-stack one GPU while others idle. (ISSUE-005, S1)
- [ ] **SCH-02**: A generic daemon-side guard/injection ensures editable-installed consumer packages import from the per-experiment worktree overlay (not the main checkout), and `automil check` warns when editable package paths are snapshotted without a worktree import guard. (ISSUE-010, S1)

### CLI lifecycle & operability (OPS)

<!-- Operators can drive the full node lifecycle from the CLI without manual file surgery. -->

- [ ] **OPS-01**: `automil cancel` can cancel daemon-launched local jobs — it resolves a local running spec via top-level `opaque_id` or falls back to `metadata.pid` / `metadata.pgid`. (ISSUE-011, S1)
- [ ] **OPS-02**: An operator can cleanly dequeue a queued node (`automil dequeue <node>` or a `cancel` extension) that removes the queue spec and marks the graph node `cancelled`, leaving no orphaned pending proposal. (ISSUE-016, S2)
- [ ] **OPS-03**: Submitting against an existing `type=proposed,status=pending` node transitions it to `running` (`graph.mark_running`) after the queue spec is written, so cancellation and portfolio accounting stay consistent. (ISSUE-023, S2)
- [ ] **OPS-04**: A CLI `--project PATH` option routes project discovery so commands target an overlay from outside its project root (monorepo / sibling-overlay layouts). (ISSUE-012, S2)
- [ ] **OPS-05**: `automil viz start` resolves the port as explicit `--port` → `automil/config.yaml: viz.port` → default `8420`, matching the existing `viz.host` fallback. (ISSUE-004, S2)

### Housekeeping & tech debt (DBT)

<!-- v1.1-tagged debt from STATE.md deferred items. -->

- [ ] **DBT-01**: Pre-D-200 `graph.json` files load without a silent `KeyError` — `_load` detects the legacy schema and dict-spreads existing nodes' `val_auc`/`test_auc`/etc. on read (with `schema_version` handling). (STATE.md Phase 8 follow-up #2)
- [ ] **DBT-02**: The 3 pre-existing `tick_cells` failures pass — `test_tick_cells_active_to_refusing_new`, `test_tick_cells_terminating_fires_cancel_with_cap_reason`, `test_tick_cells_finalized_when_running_empty` (Phase-4-origin `cells_dir` resolution mismatch). (STATE.md Phase 6 follow-up #1)
- [ ] **DBT-03**: The em-dashes neighboring the daemon allowlist anchor (`_orchestrator_daemon.py` ~L45-55) are removed so future re-flow cannot break the anchor at the allowlist comment. (STATE.md Phase 8 follow-up #5 / F-13)

## v2 Requirements

Acknowledged but deferred from v1.1. Promotion to a future milestone requires a roadmap update.

### Registry variants requiring an open training loop (RTA)

<!-- Deferred ONLY because they require OPENING a closed MIL training loop. Variant *application through existing seams* is NOT here — it is v1 scope (see APL above). -->

- **RTA-01**: A closed MIL training loop is opened **in the consumer** (autobench), never by editing `lib/`, so loss/attention variants can be injected mid-loop (ISSUE-007 — CLAM is the canonical blocked case).
- **RTA-02**: A concrete loss/attention `LossVariant` (e.g. focal) **executes inside a CLAM cell's training loop** (requires RTA-01).
- **RTA-03**: `automil init` / `/automil-setup` auto-scaffolds the consumer-side dispatch glue for arbitrary new repos (beyond the documented contract + iris reference delivered under APL).

## Out of Scope

Explicitly excluded from v1.1.

| Item | Reason |
|------|--------|
| Opening any closed MIL training loop in the consumer (ISSUE-007 / RTA-01) + loss/attention variants that require it (RTA-02) | Opens the search loop + reproduction blast-radius. **Variant application through existing open seams IS in v1 (APL-01/02); only the loop-opening route is deferred.** Future registry-adoption milestone. |
| Auto-scaffolding registry dispatch for arbitrary new consumers (RTA-03) | Beyond v1.1; the documented dispatch contract + fixed sklearn-iris reference (APL-01) suffice for this milestone. |
| ISSUE-006 (`submit --parent` refuses unfinished parents) | **Won't-fix.** The triage's own remedy is "keep the guard." Documented design constraint, not a defect. |
| Full F2 experimental grid, paper writing, Mac/MPS, containerized execution | Carried from v1.0 Out of Scope — unchanged. |

## Traceability

Which phase covers which requirement. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REC-01 | TBD | Pending |
| REC-02 | TBD | Pending |
| REC-03 | TBD | Pending |
| REC-04 | TBD | Pending |
| APL-01 | TBD | Pending |
| APL-02 | TBD | Pending |
| APL-03 | TBD | Pending |
| CFG-01 | TBD | Pending |
| CFG-02 | TBD | Pending |
| CFG-03 | TBD | Pending |
| SCH-01 | TBD | Pending |
| SCH-02 | TBD | Pending |
| OPS-01 | TBD | Pending |
| OPS-02 | TBD | Pending |
| OPS-03 | TBD | Pending |
| OPS-04 | TBD | Pending |
| OPS-05 | TBD | Pending |
| DBT-01 | TBD | Pending |
| DBT-02 | TBD | Pending |
| DBT-03 | TBD | Pending |

**Coverage:**
- v1.1 requirements: 20 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 20 ⚠️ (resolved by roadmapper)

---
*Requirements defined: 2026-06-10*
*Last updated: 2026-06-10 after milestone v1.1 definition*
