# Phase 1 — Preprint plan (active)

_Background/shared context: [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Full-scope journal plan: [`../journal/PLAN.md`](../journal/PLAN.md). Raw source
material: [`../references/`](../references/). Compiled 2026-07-01, updated
same day from the confirmed pivot._

## Status: confirmed pivot

### 1. Dataset scope: 5 datasets, drop by runtime

**Confirmed: 5 datasets, and "largest" means wall-clock runtime, not slide
count** — drop whichever cohort(s) take the longest to run, to cut
compute time for the preprint. Full-grid coverage (16 TCGA + 10 CPTAC) is
Phase 2 (journal) scope only.

**Still open:** which 5 datasets. Needs actual per-cohort wall-clock numbers,
not slide count — `tasks/baseline_summary/REPORT.md`'s 53–1000 slide range
(UCS→BRCA) is *not* the right proxy now that the metric is confirmed as
runtime. SLURM `sacct` history on `fir.alliancecan.ca` would have real
wall-clock per cohort if useful — say the word and I'll pull it.

### 2. Model roster: expand from 2 to 4 MIL aggregators

Baseline reporting today filters to one canonical head per framework
(`clam_mb`, `simple_mil` — [`../../tasks/baseline_summary/README.md`](../../tasks/baseline_summary/README.md)
§Scope). A citation-ranking literature study (PDF not yet in this
repo) put the top candidates at **DTFD-MIL, DSMIL, AB-MIL, TransMIL**.

**Confirmed: 4 models total** — `clam_mb, simple_mil, ab_mil, dtfd_mil`.
AB-MIL and DTFD-MIL are the two additions. **DSMIL is out** — the roster is
held at 4, not expanded to 5.

**Partly cheap, but not uniformly free:** both additions are already wired
into every dataset's `nnmil_models` list (e.g.
[`../../benchmarks/datasets/ovarian.yaml`](../../benchmarks/datasets/ovarian.yaml):
`ab_mil, trans_mil, ds_mil, dtfd_mil, ...`), so neither needs new model
integration. But "wired in" is not the same as "already run":
- **`ab_mil`** results already exist on disk — the baseline README lists it
  among heads "found on disk" that were deliberately *excluded* from the
  cross-dataset report for apples-to-apples cleanliness. Adding it is just a
  **re-aggregation/re-report** job (re-run `scripts/00_aggregate.py` +
  `01_tables.py` with a wider `KEEP_AND_RENAME` map).
- **`dtfd_mil`** is *not* in that on-disk list, so it likely still needs
  actual training runs per dataset before it can be aggregated.

So expanding the roster is a mix of free re-reporting (`ab_mil`) and real
compute (`dtfd_mil`) — needs a per-dataset coverage check before assuming
it's all free.

### 3. Competitive positioning: a feature/coverage table vs. two named papers

The decision: build a similar feature/coverage table showing our advantages
vs. the two named benchmarks, with the evaluation section as the clear
differentiator. The two benchmarks named:

- **PathBench-MIL** (arXiv:2512.17517, github.com/Sbrussee/PathBench-MIL) —
  the closer competitor. AutoML-flavored MIL benchmarking framework:
  **20+ MIL aggregators**, **40+ encoders**, supports **slide-level
  foundation models** and **classification + regression + survival**. No
  built-in dataset/cohort corpus (bring-your-own WSIs), and — as far as the
  README shows — no agent-driven, code-level recipe search; its "AutoML" is
  pipeline/hyperparameter configuration, not autonomous code modification.
- **"Practical guidelines for MIL... embedding choice impacts overall
  survival prediction"** (Frontiers, `fbinf.2026.1809049`) — 4 patch encoders
  (ResNet50, ProvGigaPath, UNI, CONCH) × 3 aggregators (ABMIL, DSMIL,
  TransMIL) + 2 **slide-level encoders** (TITAN, ProvGigaPath slide encoder,
  paired with Cox PH), across **6 cohorts** (TCGA-BLCA/BRCA/COAD/HNSC/STAD +
  CPTAC-CCRCC, 8,224 WSIs), **survival-only**. Headline finding — **encoder
  choice matters far more than aggregator choice** (TITAN c-index 0.62 best
  overall) — this is very likely the paper that motivated the survival-phase
  cohort picks (4 of its 6 cohorts overlap with `feat/nnmil-survival`'s
  target list, see shared background Phase C) and is also the direct
  precedent for the encoder>aggregator claim surfaced in the
  citation-ranking review.

**The reference table format.** It's the PathBench-MIL paper's own **Table 1**
(arXiv:2512.17517), comparing itself against **Patho-Bench**
(mahmoodlab/patho-bench) and **EVA** (kaiko-ai/eva) — cite it as such if
reused. We need our own row in this exact format. Filled in below from
checking each framework directly (Patho-Bench and EVA columns transcribed
from the source image; PathBench-MIL cross-checked against its repo):

| Feature | PathBench-MIL | Patho-Bench | EVA | **autobench (ours)** |
|---|:---:|:---:|:---:|:---:|
| Covers entire MIL pipeline | ✓ | ✗ | ✗ | ✓ *(prep→train→eval, plus agentic recipe search on top — see below)* |
| Patch foundation models | ✓ | ✓ | ✓ | ✓ *(H-Optimus-1, UNI v2, Virchow2)* |
| Slide-level foundation models | ✓ | ✓ *(via Trident)* | ✗ | ✓ *(TITAN — confirmed, §4)* |
| Classification | ✓ | ✓ | ✓ | ✓ |
| Regression | ✓ | ✗ | ✗ | ✗ → **planned (§4)** |
| Deep continuous survival | ✓ | ✗ *(Cox linear probe, not deep)* | ✗ | ✓ *(Cox, shared background Phase C)* |
| Deep discrete survival | ✓ | ✓ | ✗ | ✓ *(discrete-time NLL, shared background Phase C)* |
| AutoML capabilities | ✓ | ✗ | ✗ | ✓ *(but ours is agentic code-level recipe search, not config/hyperparameter menu search — a different, stronger category; worth its own row in the real table)* |
| Interactive visualization | ✓ | ✗ | ✗ | ✓ *(live 3D SSE dashboard)* |
| CPU multiprocessing | ✓ | ✓ | ✓ | ✓ *(likely — not explicitly verified, check before claiming)* |
| GPU parallelization | ✓ | ✓ | ✗ | ✓ *(multi-GPU best-fit bin-packing orchestrator)* |
| Built-in datasets & tasks | ✗ | ✓ *(95 tasks / 33 datasets)* | ✓ | ✓✓ *(16 TCGA + 10 CPTAC already extracted — see shared background; likely our single strongest row)* |
| Semantic segmentation | ✗ | ✗ | ✓ | ✗ *(out of scope — MIL is bag-level by design; recommend not chasing this)* |
| Patch-level prediction tasks | ✗ | ✗ | ✓ | ✗ *(same reasoning)* |
| Retrieval | ✗ | ✓ | ✗ | ✗ *(not currently supported)* |

Two rows worth calling out as the actual differentiators, since the checkbox
format hides them:
- **AutoML capabilities** — all four tools check this box, but PathBench-MIL's
  is Optuna-style search over a configured pipeline/menu. Ours is an LLM
  agent modifying training-recipe **code** directly, in isolated git
  worktrees, with every candidate's diff, logs, and result archived. This is
  the paper's actual headline claim and doesn't fit in a single checkmark —
  it needs its own comparison row or a footnote in the real table.
- **Published-SOTA protocol-parity validation** — none of these three papers
  claim this. The GOLDMARK exact-protocol reproduction (shared background
  Phase B) is evidence our pipeline reproduces published SOTA numbers under
  identical training logic, not just "runs the models." Worth a row.

### 4. Two capability gaps — status

Two gaps identified: slide-level PFM coverage and regression. **Both
confirmed to be added**, and expected to be quick given the reduced 5-dataset
preprint scope.

- **Slide-level pathology foundation model — confirmed: TITAN**
  (`MahmoodLab/TITAN` on Hugging Face). Emits one embedding per whole slide
  directly — **no patch tiling via TRIDENT, and no separate MIL aggregator**
  for this arm. New code path in `autobench`, not just config: `TRIDENT →
  patch features → MIL aggregator` becomes `TITAN → embedding → head` for
  this arm. Why TITAN:
  1. Built directly on **CONCH v1.5 patch features** — already one of our
     configured encoders (`MahmoodLab/conchv1_5` → `conch_v15` in every
     dataset YAML). Same lab, same feature family we already pull.
  2. **Won the Frontiers embedding-choice paper's own slide-level
     comparison** (c-index 0.62, beating the ProvGigaPath slide encoder) —
     citing our own TITAN numbers engages that paper's specific result
     directly.
  3. License is **CC-BY-NC-ND 4.0** — fine for academic/preprint use, but
     disclose it: no redistribution of derivative model weights, commercial
     use needs Mahmood Lab's approval.
- **Regression.** A third task-type family alongside classification and
  survival — e.g. continuous biomarker/score prediction. **Target label(s)
  still undecided** — needs a regression head + loss (MSE/Huber) + metric
  (Pearson/Spearman/R²) added to `autobench/src/autobench/pipeline/`
  alongside the existing classification and survival paths.

## Open / pending — to confirm

1. **Final 5-dataset list.** Metric confirmed as **runtime** (not slide
   count). Need actual per-cohort wall-clock numbers to pick which 5 — the
   `sacct` history on `fir.alliancecan.ca` can supply this if useful.
2. **Regression target(s).** Still undecided — needed before the code path
   can be scoped. Note this is inherently per-dataset (see §4): there is no
   continuous label in any config today, so it means choosing which
   continuous variable to predict, for which cohort(s).
3. **Unmerged branches block a single source of truth.** Two pieces of
   finished work currently live outside `main`:
   - `feat/goldmark-parity` — the protocol-validation work (shared background
     Phase B). Lives only on the cluster (`wt-goldmark-parity` worktree on
     `fir.alliancecan.ca`), **never pushed to GitHub at all**.
   - `feat/nnmil-survival` — the survival pipeline (shared background Phase
     C). Pushed to GitHub, not merged into `main`.
   Not urgent to merge today, but the preprint will eventually need one
   commit/tag as "the" pipeline version, and neither is in `main` yet.

## Sources (pivot-specific, in addition to shared background)

- arXiv:2512.17517 (PathBench-MIL) — abstract + github.com/Sbrussee/PathBench-MIL README
- Frontiers `fbinf.2026.1809049` (embedding choice for OS prediction) — full text summary
- github.com/mahmoodlab/patho-bench README
- github.com/kaiko-ai/eva README
- huggingface.co/MahmoodLab/TITAN model card
- Table 1 screenshot (PathBench-MIL vs Patho-Bench vs EVA)
