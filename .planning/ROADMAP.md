# Roadmap: autoMIL

## Milestones

- ✅ **v1.0 F2-readiness framework refactor** — Phases 0-8 (shipped 2026-05-08), see [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)
- ✅ **v1.1 Bug Fixing** — Phases 9-14 (shipped 2026-06-12), see [milestones/v1.1-ROADMAP.md](milestones/v1.1-ROADMAP.md)

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

<details>
<summary>✅ v1.1 Bug Fixing (Phases 9-14) — SHIPPED 2026-06-12 (1058 tests green, 20/20 reqs)</summary>

- [x] **Phase 9: State & Recovery Integrity** — Correct terminal-state recording, partial-fold aggregation, status vocabulary, and budget-cell identity (completed 2026-06-11)
- [x] **Phase 10: Variant Application Integrity** — Registered variants actually apply to the live model through existing open seams; loud failure for loop-opening variants (completed 2026-06-11)
- [x] **Phase 11: Config & Run Fidelity** — Config/snapshot values drive runs; argparse defaults stop masking them; per-node run-command overrides *(completed 2026-06-11)*
- [x] **Phase 12: Scheduling & Overlay Isolation** — GPU scheduling-policy knob; generic editable-install worktree overlay guard *(parallel-candidate: independent of Phases 10–11)* (completed 2026-06-12)
- [x] **Phase 13: CLI Lifecycle & Operability** — Cancel daemon-launched jobs, dequeue queued nodes, resubmit pending→running, cross-project targeting, viz.port fallback *(parallel-candidate: independent of Phases 10–12)* (completed 2026-06-12)
- [x] **Phase 14: Housekeeping & Tech Debt** — Legacy graph.json round-trip, tick_cells test fixes, allowlist anchor cleanup *(parallel-candidate: independent of Phases 10–13)* (completed 2026-06-12)

</details>

<!-- Full v1.1 phase details archived in milestones/v1.1-ROADMAP.md -->
