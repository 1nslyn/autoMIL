# Curated Source Inventory

The required mechanical inventory was run over the repository and found 80,865
files, largely because it traversed cached bytecode and historical worktrees.
Those duplicates are not useful evidence. This curated inventory retains the
current, authoritative materials that may support the manuscript.

| ID | Path | Type | Authority / role | Permitted use | Known limitation |
|---|---|---|---|---|---|
| S01 | `/Users/leoyin/Development/autoMIL/paper/preprint/CONTRIBUTIONS.md` | claim register | Authoritative contribution hierarchy | C1/C2 wording and promotion gates | Must stay synchronized with implementation semantics |
| S02 | `/Users/leoyin/Development/autoMIL/paper/preprint/PLAN.md` | protocol plan | Intended benchmark and paper protocol | cohorts, arms, task definitions, analysis plan | Intent is not execution evidence |
| S03 | `/Users/leoyin/Development/autoMIL/paper/preprint/EXPERIMENT_GRID.md` | campaign ledger | Latest dated roster and cluster-status blocks | roster dimensions and provisional execution status | Contains older contradictory sections; no local result artifacts |
| S04 | `/Users/leoyin/Development/autoMIL/paper/preprint/CODE-AUDIT-FIXES.md` | audit ledger | Latest defect, decision, and verification record | claim boundaries, open blockers, interim screen | Some progress-log numbers are stale relative to canonical-roster notes |
| S05 | `/Users/leoyin/Development/autoMIL/paper/preprint/PRELAUNCH_REVIEW.md` | data audit | Patient/task integrity and attrition review | cohort counts, split cautions, survival attrition | Predates some later code fixes |
| S06 | `/Users/leoyin/Development/autoMIL/paper/preprint/RELATED_WORK.md` | literature note | Closest-prior positioning and draft BibTeX | autonomous ML and pathology-MIL comparison | Final bibliography metadata still needs normalization |
| S07 | `/Users/leoyin/Development/autoMIL/.planning/PROJECT.md` | framework status | Shipped scope and deferred validation | system inventory and generality limits | Test counts are historical, not a current green-suite claim |
| S08 | `/Users/leoyin/Development/autoMIL/src/automil/runner.py` | source code | Worktree creation, overlay validation, result collection | C1 worktree, manifest, traversal, sealing mechanisms | Legacy result path remains compatibility behavior |
| S09 | `/Users/leoyin/Development/autoMIL/src/automil/graph.py` | source code | Experiment tree, selection, reconciliation | node lineage and validation-only search mechanics | Large module; claims require targeted line/test anchors |
| S10 | `/Users/leoyin/Development/autoMIL/src/automil/backends/_orchestrator_daemon.py` | source code | Scheduling, launch, accounting, completion, recovery | launched-attempt semantics and orchestration | Real distributed-backend equivalence is unverified |
| S11 | `/Users/leoyin/Development/autoMIL/src/automil/cells/state.py` | source code | Budget-cell state and counters | exact definition of attempt and usable-result counts | Equal attempts do not imply equal compute or difficulty |
| S12 | `/Users/leoyin/Development/autoMIL/src/automil/cells/cap.py` | source code | Time/evaluation cap state machine | cap behavior and failure-inclusive accounting | Production roster accounting still requires artifact audit |
| S13 | `/Users/leoyin/Development/autoMIL/src/automil/registry/config.py` | source code | Registration mode and protected surfaces | source-permission contract | File-level protection does not by itself enforce the settled per-arm method-identity contract or provide a runnable programmatic-recipe seam |
| S14 | `/Users/leoyin/Development/autoMIL/src/automil/cli/submit.py` | source code | Submission validation and protected-file rejection | enforcement path for source permissions | Depends on consumer declarations |
| S15 | `/Users/leoyin/Development/autoMIL/src/automil/runtime_helpers.py` | source code | Atomic and split result writing | validation-visible versus sealed-result storage | Framework-mediated boundary, not OS access control |
| S16 | `/Users/leoyin/Development/autoMIL/src/automil/cli/certify.py` | source code | Deliberate held-out reveal | certification workflow | User can misuse revealed results after certification |
| S17 | `/Users/leoyin/Development/autoMIL/src/automil/trajectory/` | source package | Trajectory schema, recording, redaction, export | provenance and potential C4 artifact | No complete released campaign corpus yet |
| S18 | `/Users/leoyin/Development/autoMIL/src/automil/backends/` | source package | Local, SLURM, and Ray interfaces | backend architecture | Strong portability claim blocked without real comparable runs |
| S19 | `/Users/leoyin/Development/autoMIL/src/automil/gate/` | source package | nomination and transfer gate | C5 mechanism and future-work design | Real preregistered transfer evidence absent |
| S20 | `/Users/leoyin/Development/autoMIL/benchmarks/src/autobench/pipeline/provenance.py` | source code | Per-arm recipe provenance | what native/upstream components each baseline uses | Documentation defects ES-1 and OPT-1 remain open |
| S21 | `/Users/leoyin/Development/autoMIL/benchmarks/src/autobench/pipeline/search_space.py` | source code | Declared per-arm searchable fields | search-space disclosure | Must be checked against actual runner consumption |
| S22 | `/Users/leoyin/Development/autoMIL/benchmarks/scripts/run_experiment.py` | source code | Experiment execution and result writing | end-to-end consumer contract | Campaign results are not stored in this local checkout |
| S23 | `/Users/leoyin/Development/autoMIL/benchmarks/datasets/` | configuration | Cohort/task/feature definitions | dataset table and reproducibility appendix | Environment variables resolve only on configured machines |
| S24 | `/Users/leoyin/Development/autoMIL/benchmarks/experiments/` | configuration | Per-cohort autoMIL overlays | consumer adaptation and permission surfaces | No archived graph/result artifacts in this checkout |
| S25 | `/Users/leoyin/Development/autoMIL/tests/test_runner.py` | tests | Worktree/overlay/result-collection invariants | C1 verification | Unit/integration evidence, not a production campaign |
| S26 | `/Users/leoyin/Development/autoMIL/tests/test_born_sealed_firewall.py` | tests | Born-sealed held-out isolation | C1 evidence-layer verification | Threat model excludes adversarial same-user processes |
| S27 | `/Users/leoyin/Development/autoMIL/tests/test_launch_cap_enforcement.py` | tests | Launch-path budget enforcement | attempt-cap verification | Does not establish equal search-space difficulty |
| S28 | `/Users/leoyin/Development/autoMIL/tests/test_daemon_eval_accounting.py` | tests | Attempt and completion counter semantics | accounting verification | Production ledger still must be reported |
| S29 | `/Users/leoyin/Development/autoMIL/tests/test_submit_protected_files.py` | tests | Protected-file hard rejection | permission enforcement | Consumer policy quality remains external |
| S30 | `/Users/leoyin/Development/autoMIL/tests/test_framework_purity.py` | tests | Framework/consumer knowledge separation | plug-in mechanism evidence | Additional non-toy consumers still needed for broad generality |
| S31 | `/Users/leoyin/Development/autoMIL/tests/test_synthetic_consumer_roundtrip.py` | tests | End-to-end synthetic consumer | cross-project contract smoke evidence | Synthetic consumer is not a non-toy scientific project |
| S32 | selected invariant-test run, 2026-07-30 | test result | 51 targeted tests passed in 4.80 s | current C1 regression evidence | Full collection produced 114 dependency/import errors and is not claimed green |
| S33 | `/Users/leoyin/Development/autoMIL/paper/preprint/figures/mock/` | mock images | Historical layout prototypes only | Never use as evidence; may inspire layout after rebuilding | Numerically fabricated by design |
| S34 | `/Users/leoyin/Development/autoMIL/paper/preprint/figures/make_figures.py` | plotting code | Intended audited-result readers | future quantitative figure generation | No local result artifacts currently feed final plots |
| S35 | `/Users/leoyin/Development/autoMIL/paper/preprint/NeurIPS26_autoMIL/neurips_2026.sty` | template | Official-format local style | final public preprint formatting | Must not be modified |
| S36 | `/Users/leoyin/Development/autoMIL/paper/preprint/NeurIPS26_autoMIL/checklist.tex` | template | Mandatory NeurIPS checklist | final PDF checklist | Answers must reflect the final manuscript and evidence |
