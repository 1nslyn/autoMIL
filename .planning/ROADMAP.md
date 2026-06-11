# Roadmap: autoMIL

## Milestones

- ✅ **v1.0 F2-readiness framework refactor** — Phases 0-8 (shipped 2026-05-08), see [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- 🔄 **v1.1 Bug Fixing** — Phases 9-14 (started 2026-06-10)

## Phases

<details>
<summary>✅ v1.0 F2-readiness framework refactor (Phases 0-8) — SHIPPED 2026-05-08</summary>

- [x] Phase 0: Tier 2 cleanup + CLI split + compat shim (7/7 plans, completed 2026-05-01)
- [x] Phase 1: Variant registry + config-driven train + CCRCC reproduction sanity (12/12 plans, completed 2026-05-02)
- [x] Phase 2: Backend ABC + LocalBackend re-export + MockSLURM fixture (8/8 plans, completed 2026-05-03)
- [x] Phase 3: Trajectory recorder + multi-runtime asset reorg (11/11 plans, completed 2026-05-04)
- [x] Phase 4: 6h per-cell hard cap + cell-concept formalization (10/10 plans, completed 2026-05-05)
- [x] Phase 5: Generalization gate (12/12 plans, completed 2026-05-06)
- [x] Phase 6: SLURM backend (submitit) + Ray backend (raw ray.remote) (10/10 plans, completed 2026-05-06)
- [x] Phase 7: Hardware autodetect + /automil-setup skill (12/12 plans, completed 2026-05-07)
- [x] Phase 8: Decoupling completion + acceptance (10/10 plans, completed 2026-05-08)

</details>

### v1.1 Bug Fixing (Phases 9-14)

- [x] **Phase 9: State & Recovery Integrity** — Correct terminal-state recording, partial-fold aggregation, status vocabulary, and budget-cell identity (completed 2026-06-11)
- [x] **Phase 10: Variant Application Integrity** — Registered variants actually apply to the live model through existing open seams; loud failure for loop-opening variants (completed 2026-06-11)
- [x] **Phase 11: Config & Run Fidelity** — Config/snapshot values drive runs; argparse defaults stop masking them; per-node run-command overrides *(completed 2026-06-11)*
- [ ] **Phase 12: Scheduling & Overlay Isolation** — GPU scheduling-policy knob; generic editable-install worktree overlay guard *(parallel-candidate: independent of Phases 10–11)*
- [ ] **Phase 13: CLI Lifecycle & Operability** — Cancel daemon-launched jobs, dequeue queued nodes, resubmit pending→running, cross-project targeting, viz.port fallback *(parallel-candidate: independent of Phases 10–12)*
- [ ] **Phase 14: Housekeeping & Tech Debt** — Legacy graph.json round-trip, tick_cells test fixes, allowlist anchor cleanup *(parallel-candidate: independent of Phases 10–13)*

## Phase Details

### Phase 9: State & Recovery Integrity
**Goal**: The framework's recorded truth (terminal state, status, budget identity) is correct, single-sourced, and survives mid-run interruption without data loss.
**Depends on**: Nothing (first v1.1 phase; highest-severity cluster, S0 item REC-01 drives ordering)
**Requirements**: REC-01, REC-02, REC-03, REC-04
**Success Criteria** (what must be TRUE):
  1. An experiment killed mid-run by SIGTERM or timeout with completed folds on disk reports those folds' aggregated composite in `result.json` — not `composite=0.0`.
  2. Normal completion and cap-kill completion both write `graph.json`, `completed/<node>.json`, archive `result.json`, and `results.tsv` through a single writer; `automil rank` and the dashboard never disagree about a node's terminal state without manual reconciliation.
  3. `result.json` payloads written by framework recovery paths (`partial`, `timeout`, `oom`, `crashed`) all validate against `result.schema.json` without normalization errors.
  4. `automil propose` and `automil submit` require a `--mil-model` field; the budget cell key is `(dataset, encoder, mil_model)` and re-parenting a node does not open a fresh 6h budget for the same MIL model.
**Plans**: 6 plans
Plans:
**Wave 1**
- [x] 09-01-PLAN.md — Wave-0 test stubs (all 4 REC requirements, Nyquist compliance)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 09-02-PLAN.md — Cell identity foundation: make_cell_id + normalize_mil_model + read_cell shim (REC-04)
- [x] 09-03-PLAN.md — Schema update + _crashed_payload canonicalization (REC-03)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 09-04-PLAN.md — Partial-fold recovery: SIGTERM flush + fold-first synthesis + main-PID-first timeout (REC-01)
- [x] 09-05-PLAN.md — CLI mil_model wiring + automil cells migrate command (REC-04)

**Wave 4** *(blocked on Wave 3 completion)*
- [x] 09-06-PLAN.md — terminal_writer module + daemon refactor + reconcile --from-archive + partial quarantine (REC-01, REC-02)

**Cross-cutting constraints:**
- No regressions in existing 48-test suite
**Parallel note**: Must complete before Phase 10 (APL depends on mil_model being first-class per REC-04). Phases 11–14 may start in parallel once Phase 9 is in progress — they do not depend on Phase 9 outputs.

---

### Phase 10: Variant Application Integrity
**Goal**: A registered variant is never silently inert — it applies to the actual live model through existing open seams, and any variant that would require opening a closed training loop fails loudly instead of no-op'ing.
**Depends on**: Phase 9 (REC-04 establishes mil_model as a first-class field, which APL-02 depends on for correct cell-keyed variant dispatch)
**Requirements**: APL-01, APL-02, APL-03
**Success Criteria** (what must be TRUE):
  1. `automil apply <node>` on the sklearn-iris reference causes the experiment to run with the registered `classifier_v0` variant imported and applied — the variant's `train.py` import path is no longer inert; observable by a different model object being instantiated.
  2. A registered model/config/hyperparameter variant for the autobench CLAM consumer applies through the existing `clam_train` args seam (`model_type`, `model_size`, `B`, `bag_weight`, `dropout`, optimizer/`lr`, `bag_loss`/`inst_loss`) — verified by a real experiment whose composite differs from the un-applied baseline by more than noise.
  3. Attempting to apply a variant whose only route requires injecting a callable into CLAM's `train_loop_clam` produces a loud error message citing "requires loop opening — deferred (ISSUE-007 / RTA)" — it never silently no-op's.
**Plans**: 4 plans

Plans:
**Wave 1**
- [x] 10-01-PLAN.md — Wave-0 test stubs (APL-01/02/03 RED tests + variant_dispatch.py stub)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 10-02-PLAN.md — APL-03 loud-fail classifier + A1 fix: applied_variant.json overlay write (APL-01, APL-03)
- [x] 10-03-PLAN.md — APL-02 variant_dispatch.py implementation + run_experiment.py wiring (APL-02)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 10-04-PLAN.md — APL-01 iris train.py dispatch + end-to-end human verify (APL-01, APL-02)

---

### Phase 11: Config & Run Fidelity
**Goal**: Config file and snapshot values drive experiment runs without being silently overridden by argparse or CLI defaults; per-node run-command overrides are expressible without editing snapshotted code.
**Depends on**: Nothing (independent; may execute in parallel with Phases 12, 13, 14 after Phase 9 is underway)
**Requirements**: CFG-01, CFG-02, CFG-03
**Success Criteria** (what must be TRUE):
  1. A `run_experiment.py` invocation that does not pass `--seed`, `--lr`, `--max_epochs`, `--patience`, `--stop_epoch`, or `--n_folds` on the command line uses the values from the snapshot/config dataclass — not argparse's hard-coded defaults.
  2. `automil submit` called without `--timeout` omits `timeout_min` from the queue spec entirely, so the orchestrator's `orchestrator.default_timeout_min` setting takes effect unmasked.
  3. A queue spec can carry per-node run-command overrides (e.g. `--seed 42 --lr 1e-4`) layered on top of `config.yaml run.command`, and `automil submit --override "..."` writes them into the spec without modifying any snapshotted file.
**Plans**: 3 plans
Plans:
**Wave 1**
- [x] 11-01-PLAN.md — Wave-0 test stubs (CFG-01/02/03 RED tests, Nyquist compliance)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 11-02-PLAN.md — CFG-01: run_experiment.py None-default training overrides + conditional TrainConfig construction (CFG-01)
- [x] 11-03-PLAN.md — CFG-02/03: --timeout None-default + D-03 sentinel fix + --override CLI + daemon suffix-append (CFG-02, CFG-03)

**Cross-cutting constraints:**
- No regressions in existing test suites (run tests/ and benchmarks/tests/ SEPARATELY — combined run triggers rootdir collision)
**Parallel note**: Independent of Phases 10, 12, 13, 14. Can run in parallel with any of them after Phase 9 clears.

---

### Phase 12: Scheduling & Overlay Isolation
**Goal**: The orchestrator offers a configurable GPU placement policy so compute-bound jobs do not over-stack one GPU, and the daemon guards editable-installed consumer packages so experiments import from their worktree overlay.
**Depends on**: Nothing (independent; may execute in parallel with Phases 11, 13, 14)
**Requirements**: SCH-01, SCH-02
**Success Criteria** (what must be TRUE):
  1. Setting `orchestrator.scheduling_policy: round_robin` (or `least_loaded`) in `config.yaml` causes the daemon to distribute new jobs across GPUs in round-robin order rather than packing them onto the least-used GPU; observable by job placement logs showing different GPU IDs for successive submits.
  2. `automil check` warns when an editable-installed consumer package path is snapshotted in a worktree without a daemon-side import guard in place; a real experiment run with the guard active imports from the worktree overlay rather than the main checkout.
**Plans**: TBD
**Parallel note**: Independent of Phases 11, 13, 14. Can run in parallel with any of them after Phase 9 clears.

---

### Phase 13: CLI Lifecycle & Operability
**Goal**: Operators can drive the full node lifecycle — cancel running jobs, dequeue queued nodes, resubmit pending proposals, target a project from outside its root, and reliably reach the viz dashboard — without manual file surgery.
**Depends on**: Nothing (independent; may execute in parallel with Phases 11, 12, 14)
**Requirements**: OPS-01, OPS-02, OPS-03, OPS-04, OPS-05
**Success Criteria** (what must be TRUE):
  1. `automil cancel <node>` terminates a daemon-launched local job by resolving its `opaque_id`, `metadata.pid`, or `metadata.pgid` — the process is killed and the node transitions to `cancelled` in `graph.json`.
  2. `automil dequeue <node>` (or `automil cancel` extension) removes the queue spec for a queued node and marks it `cancelled` in `graph.json` — no orphaned pending proposal remains.
  3. `automil submit` against an existing `type=proposed, status=pending` node calls `graph.mark_running` after writing the queue spec, so the node transitions to `running` immediately.
  4. Every automil CLI command accepts a `--project PATH` option that routes project discovery to that path, enabling targeting from outside the project root (monorepo / sibling-overlay layouts).
  5. `automil viz start` without `--port` reads `viz.port` from `automil/config.yaml` and falls back to `8420`; the port-resolution order matches the existing `viz.host` fallback pattern.
**Plans**: TBD
**Parallel note**: Independent of Phases 11, 12, 14. Can run in parallel with any of them after Phase 9 clears.

---

### Phase 14: Housekeeping & Tech Debt
**Goal**: Pre-D-200 graph files round-trip cleanly, three pre-existing tick_cells test failures are resolved, and the orchestrator daemon allowlist anchor is protected from future re-flow breakage.
**Depends on**: Nothing (independent; may execute in parallel with Phases 11, 12, 13)
**Requirements**: DBT-01, DBT-02, DBT-03
**Success Criteria** (what must be TRUE):
  1. Loading a pre-D-200 `graph.json` (one that uses the old `val_auc`/`test_auc`/etc. flat-key schema) succeeds without `KeyError` and produces a fully-populated node tree; `schema_version` detection triggers automatic dict-spread migration on read.
  2. All three pre-existing `tick_cells` test failures pass: `test_tick_cells_active_to_refusing_new`, `test_tick_cells_terminating_fires_cancel_with_cap_reason`, `test_tick_cells_finalized_when_running_empty` — the Phase-4-origin `cells_dir` resolution mismatch is corrected.
  3. The em-dashes neighboring the daemon allowlist anchor in `_orchestrator_daemon.py` (lines ~45-55) are removed; `automil check` or a CI grep confirms no em-dash characters remain adjacent to the anchor comment.
**Plans**: TBD
**Parallel note**: Independent of Phases 11, 12, 13. Can run in parallel with any of them after Phase 9 clears.

---

## Progress

| Milestone | Phases | Plans | Status | Shipped |
|-----------|--------|-------|--------|---------|
| v1.0 F2-readiness framework refactor | 9 | 92 | Complete | 2026-05-08 |
| v1.1 Bug Fixing | 6 | TBD | In progress | — |

---

## Workstation UAT items deferred to /gsd-verify-work 8

These are workstation-data-gated tests that require Leo's environment with `AUTOBENCH_CCRCC_ROOT` set:

- **Sub-gate A**: CCRCC `node_0176` ±0.005 reproduction (D-205 / DEC-07)
- **Sub-gate C**: heterogeneous consumers (sklearn-iris + CCRCC side-by-side in same project)
- Real SLURM cluster verification (`@pytest.mark.requires_slurm` marker) (BCK-05 success criterion 5)
- Real Ray multi-node cluster verification (`@pytest.mark.requires_ray` marker) (BCK-06)
- External hardware shapes (CPU-only laptop, ROCm system) per Phase 7 D-197 MEDIUM portability

## Pre-existing tech debt for v1.1+

- 3 pre-existing tick_cells failures (Phase 4-origin, documented as Phase 6 follow-up #1) — addressed in Phase 14 (DBT-02)
- Phase 5 calibration pilot K-determination (Leo runs with CCRCC + CLWD cells)
