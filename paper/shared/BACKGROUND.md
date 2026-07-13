# Background — shared across both paper phases

_Compiled 2026-07-01. This is the context common to both
[`../preprint/PLAN.md`](../preprint/PLAN.md) (Phase 1, active) and
[`../journal/PLAN.md`](../journal/PLAN.md) (Phase 2, provisional) — read this
first, then the phase-specific doc for what's actually happening now._

## Target

- **Venue:** Nature Biomedical Engineering (long-horizon target).
- **Strategy:** ship a **preprint fast** on a deliberately reduced grid
  (Phase 1), then scale to the full grid for the formal journal submission
  (Phase 2). The decision: drop the largest datasets to improve experiment
  speed for the preprint; the formal journal covers all of them.

## Two layers of this project

1. **Framework layer — `autoMIL`.** The agent-driven experiment-orchestration
   tool itself (`src/automil/`). This is engineering, not a paper claim on its
   own; it's the infrastructure the empirical work runs on.
   - **Core value** (`.planning/PROJECT.md`): an agent can autonomously discover
     model improvements — architecture and training-recipe — for any user's
     training code under a 6-hour-per-cell budget, with variants that are
     reproducible, attributable to their parents, and portable across machines
     and LLM runtimes.
   - **Status:** v1.0 "F2-readiness" shipped 2026-05-08 (69 requirements, 9
     phases); v1.1 "Bug Fixing" shipped 2026-06-12 (20 requirements, 6 phases).
     Both merged to `main`. Full test suite green (1058 passed).
   - **The auto pipeline is the main contribution** — this framework is still
     the paper's core claim in both phases. What changes between phases is
     scope and evaluation, not the core claim.

2. **Empirical layer — `autobench` + cluster experiments.** MIL benchmarking
   across TCGA/CPTAC cohorts — the evidence base. Moved through Phases A–C
   below as shared groundwork; Phase 1 (preprint) and Phase 2 (journal) each
   build on top of this differently.

## Empirical trajectory (how the plan got here)

### Phase A — Broad baseline sweep (~May 2026)

Ran CLAM (`clam_mb`) and nnMIL (`simple_mil`) across **15 TCGA cancer types, 35
(dataset, task) mutation-prediction pairs**, 3 foundation-model encoders
(H‑Optimus‑1, UNI v2, Virchow2), 5-fold patient-stratified CV. Write-up:
[`../../tasks/baseline_summary/REPORT.md`](../../tasks/baseline_summary/REPORT.md).

Headline: 14/35 pairs reach test AUC ≥ 0.70 (best: THCA‑BRAF 0.925); weak
signal concentrated in low-prevalence single-gene tasks. Model gap CLAM vs
nnMIL ~3 AUC points; encoder choice is a smaller effect (~2 points) than
per-dataset/task variance.

Comparing against published numbers (GOLDMARK, arXiv:2603.20848; nnMIL,
arXiv:2511.14907 — [`../../tasks/baseline_summary/COMPARISON.md`](../../tasks/baseline_summary/COMPARISON.md))
showed our numbers reading ~2–4 AUC points **below** theirs on overlapping
tasks. That gap is what triggered Phase B.

### Phase B — GOLDMARK protocol-parity investigation (2026-06-14 to 06-19)

Branch `feat/goldmark-parity` — **cluster-local only, never pushed to GitHub**
(`/home/yinshuol/scratch/autoMIL/wt-goldmark-parity`, worktree off `main@340ae85`).
Full write-up: `docs/goldmark_parity.md` on that branch.

Question: is the gap a real pipeline deficiency, or a comparison/protocol
artifact? Method: read GOLDMARK's actual training code
(github.com/chadvanderbilt/GOLDMARK) stage by stage against ours.

**Finding:** almost entirely protocol artifact, not a pipeline defect:
- GOLDMARK reports AUC on the **same fold used for checkpoint selection**
  (`val == test`); we report a disjoint held-out test fold — strictly more
  conservative. Their own ablation: best-epoch selection on the reported fold
  is worth **+0.039 AUC** by itself.
- Their headline numbers (LUAD‑EGFR 0.896, THCA‑BRAF 0.937, LGG‑IDH1 0.827) are
  the **fine-tuned EAGLE (Prov-GigaPath)** encoder, not a frozen-encoder
  baseline — not a fair comparison point.
- Against GOLDMARK's *actual published frozen-encoder* per-task numbers (from
  their results portal, not the arXiv PDF), **our honest disjoint-test numbers
  already meet or beat every one of their encoders on LUAD EGFR/KRAS,
  including their fine-tuned EAGLE**, e.g. EGFR hoptimus1 0.853 vs GOLDMARK
  EAGLE 0.831.
- One open confound flagged but not yet closed: our comparison uses `clam_mb`
  (gated attention + instance-clustering loss), which is a stronger head than
  GOLDMARK's plain-GMA aggregator — the literal model-matched control would be
  our `ab_mil`. **This is now addressed in Phase 1** (ab_mil is a confirmed
  addition — see `../preprint/PLAN.md`).

Follow-on: **exact-protocol reproduction** (`goldmark_exact/`, same branch,
2026-06-17) — reproduced GOLDMARK's literal recipe (AdamW, 120 epochs, no
early-stop, best-val-AUC checkpoint, GMA-style `clam_sb`/no-inst-loss) on our
newer encoders across LUAD/LGG/COAD. Result
(`goldmark_exact/COMPARISON.csv`): under GOLDMARK's own recipe, our (newer)
encoders generally **exceed** their published per-split numbers (e.g. LGG‑IDH1
uni_v2 +0.078, COAD‑BRAF virchow2 +0.044); our existing default recipe is more
mixed, occasionally trailing on LUAD (egfr uni_v2 −0.09).

**Net effect of Phase B on the plan:** the pipeline and encoder choices are
validated as competitive-to-SOTA — this is the evidence behind the
"protocol-parity validation" claim in Phase 1's comparison table. Nothing here
is merged to `main` yet — nothing is written to the shared tree pending
review (per the branch's own merge-plan note).

### Phase C — Survival-analysis expansion (2026-06 ongoing, unmerged)

Branch `feat/nnmil-survival` (pushed to GitHub, not yet merged). Adds a second
task family beyond mutation/subtype classification: **overall-survival
(time-to-event) prediction**.

- Losses: Cox proportional hazards (`cox`, `clam_sb`/nnMIL-attention only —
  degenerate for `clam_mb`) and discrete-time NLL (`nllsurv`, works for both).
- Evaluation: patient-level concordance index via `scikit-survival`, **5-fold**
  CV (not 10 — with few events, 10-fold starves per-fold event counts),
  model selection on **validation loss** (not val c-index, which is a coin
  flip with few events).
- Labels: joined from GDC clinical exports (`OS_event`/`OS_time`) per cohort —
  tooling in `benchmarks/scripts/manifests/add_os_to_manifest.py`.
- Cohorts enabled so far: TCGA‑LGG (first), TCGA‑BLCA/BRCA/GBM/HNSC and
  CPTAC‑CCRCC/GBM configs added — 4 of these (BLCA, BRCA, HNSC, CCRCC) overlap
  directly with the Frontiers embedding-choice paper's 6-cohort survival
  benchmark (see Phase 1 plan), which is likely not a coincidence.
- Methodology doc: `docs/tutorials/survival_benchmark_experiments_tutorial.md` (on this
  branch — the survival counterpart of
  [`../../docs/tutorials/benchmark_experiments_tutorial.md`](../../docs/tutorials/benchmark_experiments_tutorial.md)).

This is a genuine axis expansion for the paper: classification-only →
classification + survival (+ regression, planned in Phase 1) to cover all
three standard MIL task types.

## Data scale available (shared cluster storage)

`/home/yinshuol/projects/rrg-jma/shared/Pathology/`:

- **TCGA — 16 cohorts extracted:** BLCA, BRCA, CESC, COAD, GBM, HNSC, LGG,
  LUAD, PAAD, PCPG, SKCM, STAD, TGCT, THCA, UCEC, UCS.
- **CPTAC — 10 cohorts extracted:** BRCA, CCRCC, COAD, GBM, HNSC, LSCC, LUAD,
  OV, PDAC, UCEC.

Only a subset of these (ovarian, CLWD, CCRCC, HANCOCK, TCGA‑COAD/LGG/LUAD, plus
the in-progress survival cohorts) have `autobench/datasets/*.yaml` configs on
GitHub today. **Phase 1 (preprint) uses 5 of these; Phase 2 (journal) is
scoped to the full 26-cohort inventory** — see the respective phase plans.

## Prior framing — superseded, kept for reference

An older Notion proposal ("AutoMIL Proposal," `wanglab`, updated 2026-06-26,
archived at [`../references/automil-proposal-2026-04-29.md`](../references/automil-proposal-2026-04-29.md))
framed the paper around **auditing training-recipe bias**: give every MIL
architecture an equal-effort agent-driven recipe search under a frozen,
architecture-preserving protocol, then report how much published leaderboards
change once that bias is controlled for. It included a 90-cell full grid (6
tasks × 3 encoders × 5 models), a pilot study, a full statistical-analysis
plan, and reviewer-attack rebuttals.

This doc is **not exactly right** relative to the confirmed
pivot — it predates that pivot and doesn't mention slide-level PFMs,
regression, or the PathBench-MIL/Frontiers comparison at all. It's too heavy
for the fast preprint (Phase 1), but its ambition level and rigor (frozen
recipe protocol, mixed-effects variance decomposition, the "recipe planner"
follow-on idea) look like a closer match for **Phase 2's** larger scope — see
[`../journal/PLAN.md`](../journal/PLAN.md) for how it's being carried forward,
provisionally.

## Sources consulted (general)

**Local repo** (`/Users/leoyin/Development/autoMIL`): `CLAUDE.md`,
`.planning/PROJECT.md`, `.planning/STATE.md`, `README.md`,
`benchmarks/README.md`, `benchmarks/datasets/*.yaml`,
`docs/tutorials/benchmark_experiments_tutorial.md`, `tasks/baseline_summary/{README,
REPORT,COMPARISON}.md`, git log on `main` and diff against
`origin/feat/nnmil-survival`.

**Cluster** (`fir.alliancecan.ca`, via existing SSH multiplex socket):
- `/home/yinshuol/scratch/autoMIL/wt-goldmark-parity/` — `docs/goldmark_parity.md`,
  `tasks/todo.md`, `tasks/lessons.md` (branch `feat/goldmark-parity`)
- `/home/yinshuol/scratch/autoMIL/goldmark_exact/COMPARISON.csv`
- `/home/yinshuol/projects/rrg-jma/shared/Pathology/{TCGA,CPTAC}/` — cohort inventory

Pivot-specific sources (PathBench-MIL/Patho-Bench/EVA/TITAN
research) are listed in [`../preprint/PLAN.md`](../preprint/PLAN.md).
