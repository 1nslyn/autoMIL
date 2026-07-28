# Phase 1 — Preprint plan (active)

_Background/shared context: [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Full-scope journal plan: [`../journal/PLAN.md`](../journal/PLAN.md). Raw source
material: [`../references/`](../references/). Compiled 2026-07-01; §5 resolved
2026-07-15, §1 roster resolved 2026-07-17, last reconciled against `main`
2026-07-21. **Counts are owned by [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md)** —
where this file disagrees on a number, that one wins._

> **Open item gating the campaign — see [`PRELAUNCH_REVIEW.md`](PRELAUNCH_REVIEW.md).**
> One decision in this document is contradicted by our own evidence and is
> **not yet resolved**: §3/§4's *encoder ≫ aggregator* headline — our 210-config
> baseline measured the reverse on classification (encoder spread 2.0 pts vs
> aggregator 3.0 pts). See PRELAUNCH_REVIEW §3, item O1. That review also flagged §2's
> `ab_mil` "free re-aggregation" claim; **that one is now fixed** — §2 below
> budgets both additions as from-scratch training.

## Status: confirmed pivot

### 1. Dataset scope: 5 datasets, diverse tasks across TCGA + CPTAC

**Confirmed: 5 cohorts (3 TCGA + 2 CPTAC), chosen for classification-task
diversity, not survival power.** Full-grid coverage (16 TCGA + 10 CPTAC) remains
Phase 2 (journal) scope only.

**Resolved (2026-07-17): the final 5 are TCGA-LUAD, TCGA-LGG, CPTAC-GBM,
CPTAC-PDAC, TCGA-HNSC** — each pinned to one classification task + an OS
survival task. The roster deliberately spans three classification *task types*
across two data sources:

| Dataset | Source | Classification task | Task type | n | OS deaths |
|---|---|---|---|--:|--:|
| TCGA-LUAD | TCGA | KRAS mutation | binary | 465 | 167 |
| TCGA-LGG | TCGA | IDH1 mutation | binary | 491 | 115 |
| CPTAC-GBM | CPTAC | TP53 mutation | binary | 99 | 72 |
| CPTAC-PDAC | CPTAC | immune subtype (low/med/high) | 3-class | 105 | 81 |
| TCGA-HNSC | TCGA | tumor grade (G1/G2/G3) | 3-class | 431 (414 gradeable) | 205 |

Selection now prioritizes **classification-task diversity** (binary mutation +
3-class immune subtype + 3-class tumor grade) and **cross-source coverage**
(TCGA GOLDMARK + CPTAC Patho-Bench), with OS survival retained as a secondary
axis on all five. This **replaces** the earlier survival-power rule: the old
"≥100 OS deaths hard gate" is dropped — CPTAC-GBM (72 deaths) and CPTAC-PDAC (81
deaths) fall below it — because the roster intentionally trades survival power
for task/source diversity and a genuine small-sample regime (GBM n=99, PDAC
n=105). Tasks are all distinct. The two 3-class tasks required **no pipeline
changes**: the metric, model-head, and split path are already multi-class-safe
(per-class one-vs-rest AUC, `n_classes`-sized heads, stratified folds — see
[`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §2). This also **supersedes**
EXECUTION_PLAN.md §2's earlier {THCA, LGG, LUAD, HNSC, COAD} recommendation.
Full grid (33 exps/dataset, 165 total, 825 fold-trainings) + compute estimate +
figure plan: [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md).

### 2. Model roster: expand from 2 to 4 MIL aggregators

Baseline reporting today filters to one canonical head per framework
(`clam_mb`, `simple_mil` — `tasks/baseline_summary/README.md` §Scope; note
`tasks/` is **gitignored**, so that path resolves only in a local checkout, not
on GitHub). A citation-ranking literature study (PDF not yet in this
repo) put the top candidates at **DTFD-MIL, DSMIL, AB-MIL, TransMIL**.

**Confirmed: 4 models total** — `clam_mb, simple_mil, abmil, dtfd_mil`.
AB-MIL and DTFD-MIL are the two additions. **DSMIL is out** — the roster is
held at 4, not expanded to 5.

**Naming and wiring (corrected 2026-07-21).** The model key is `abmil`, not
`ab_mil` — and ABMIL and DTFD are each their **own framework**
(`Framework.ABMIL` / `Framework.DTFD`), configured via their own YAML keys, not
via `nnmil_models`. All five roster YAMLs pin one model per framework:

```yaml
clam_models:  [clam_mb]
nnmil_models: [simple_mil]
abmil_models: [abmil]
dtfd_models:  [dtfd_mil]
```

An earlier draft of this section claimed both additions were "already wired into
every dataset's `nnmil_models` list" and cited `ovarian.yaml`. That was wrong:
`ovarian.yaml`'s `nnmil_models` is `[trans_mil, ds_mil, dtfd_mil, ilra_mil,
wikg_mil, simple_mil, vision_transformer, rrt]`, with `abmil` in a separate
`abmil_models` key. The string `ab_mil` appears nowhere in the codebase.

**Neither addition is free — both need training runs.** An earlier draft called
`abmil` a "free re-aggregation/re-report" job. The 2026-07-03 cluster audit
refutes that: `abmil` has **essentially zero usable coverage** on disk (one
partial TGCT fragment; every other cohort's fold-metric tree holds only
`clam_mb` + `simple_mil`), so it needs real compute exactly like `dtfd_mil`. See
[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) §0 and §0b-A. Budget both as
from-scratch training.

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
  cohort picks (2 of its 6 cohorts — TCGA-HNSC and CPTAC-CCRCC — overlap with
  the survival configs now on `main`; the earlier "4 of 6" count assumed
  TCGA-BLCA/BRCA configs that were never merged — see shared background
  Phase C) and is also the direct
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
| Regression | ✓ | ✗ | ✗ | ✗ → **Phase 2 (§4)** |
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

**Positioning note — encoder vs. MIL benchmark.** autobench is a *full-pipeline*
benchmark: it varies both the **encoder** axis (patch + slide foundation models)
and the **MIL aggregator** axis. That two-axis design isn't optional — the
headline *encoder > aggregator* finding is a statement about the relative
variance of those two axes, so a single-axis benchmark couldn't make it. But
covering both is **table-stakes, not the edge**: PathBench-MIL already spans
both axes (Patho-Bench and EVA are encoder-centric — the "covers entire MIL
pipeline" row above), and the Frontiers study does it at small scale. The wedge
is therefore *not* "first to cover both" — it's **recipe-bias control**:
PathBench-MIL varies both axes with fixed/menu-configured recipes, whereas we
give every cell an equal-effort agentic recipe search, so our encoder-vs-
aggregator comparison is de-biased in a way theirs isn't. Frame the paper on
that, not on coverage.

### 4. Two capability gaps — status

Two gaps identified: slide-level PFM coverage and regression. **Priority split:
TITAN is in the preprint; regression is deferred to Phase 2.**

The reasoning: TITAN is the higher-value, lower-risk move for a fast preprint —
it directly serves the headline *encoder > aggregator* claim (it's the encoder
that won the Frontiers slide-level comparison), it's table-stakes vs. the two
closest competitors (PathBench-MIL and Patho-Bench both have slide-level PFMs),
and it reuses the existing task types, metrics, and `result.json` contract.
Regression is the opposite: it's the actual *differentiator* vs. Patho-Bench /
EVA (both lack it), but it's a new task type with **no continuous target wired
into any dataset today**, needs new loss + metrics + a re-thought composite
contract, and sits off to the side of the core claim. It doesn't earn its
engineering + validation cost against a ship-fast phase, so it moves to Phase 2
as a proper task-type axis. (An exception was once held open that could have
pulled it back in; the 2026-07-03 audit closed it — see the note below the two
items.)

- **Slide-level pathology foundation model — confirmed for preprint: TITAN**
  (`MahmoodLab/TITAN` on Hugging Face). Emits one **768-d** embedding per whole
  slide directly, so TITAN *is* both the encoder and the aggregator for this arm
  — there is no tile-encoder sweep and no separate MIL aggregator axis. It
  needed a new code path in `autobench`, not just config (`TRIDENT → patch
  features → MIL aggregator` becomes `TITAN → embedding → head`); **that code is
  now merged on `main`** (`Framework.TITAN` + `benchmarks/src/autobench/pipeline/titan/`),
  configured per dataset as `titan: {head: linear}`. Why TITAN:
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
- **Regression — deferred to Phase 2.** A third task-type family alongside
  classification and survival (continuous biomarker/score prediction). Needs a
  regression head + loss (MSE/Huber) + metric (Pearson/Spearman/R²) added to
  `autobench/src/autobench/pipeline/`, plus a re-thought composite score
  (AUC/BACC don't apply). **No continuous target exists in any dataset config
  today** — every task is currently classification — so it also needs a target
  chosen per cohort. Too much new surface for the fast preprint; lands in
  Phase 2.

  *The ovarian exception is closed; a second candidate is open.* The original
  candidate was **ovarian HRD** — natively a continuous genomic-scar score. The
  2026-07-03 audit killed it twice over: `HRD_label` is a **pre-binarized 0/1
  manifest column** with no threshold or regression logic anywhere in the code,
  and the ovarian data root is **not on fir** at all. Ovarian is also not one of
  the 5 roster cohorts. See [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) §0.

  **But the premise "no continuous target exists in any dataset" was only ever
  tested against ovarian, and it does not survive contact with the roster.**
  CPTAC-PDAC's `immune_class` is an exactly-balanced 35/35/35 tertile split of a
  *continuous* tumour-immune infiltration score (PRELAUNCH_REVIEW §5 item 5) — i.e. a
  binned continuous target sitting inside the preprint roster. Whether the
  underlying score is recoverable from the CPTAC source has **not been checked**.
  Two consequences: (a) the tertile derivation must be disclosed in Table 1
  either way, and (b) if the score is recoverable, a single genuine regression
  arm becomes cheap and the deferral should be revisited rather than asserted.
  **Action: check the CPTAC-PDAC source for the continuous score before treating
  the regression deferral as final.**

### 5. Agent search-space scope — the frozen data substrate

**Resolved (2026-07-15): the agent's equal-effort recipe search runs on a frozen
data substrate — it may not change data preparation.** This is what makes the
headline *encoder ≫ aggregator* decomposition interpretable: a two-axis variance
claim is only valid if the third axis (data prep) is held constant across every
cell. That is exactly what the neighbors do — Patho-Bench, EVA, and the Frontiers
embedding paper (the very paper this claim argues with) all freeze mag/patch/
splits and let no search touch preprocessing; the Frontiers paper even disables
stain-norm on purpose "to avoid preprocessing-induced biases across cohorts."
PathBench-MIL is the lone exception, and it exposes tiling/normalization as a
*declared, compared* menu axis — not a hidden per-cell optimization run under an
"isolate the encoder" claim. Freezing the substrate is also on-brand for the
"MIL-benchmark rigor backbone" positioning: the freeze *is* the rigor claim.

**The line — what the agent may vs. may not change:**

| Layer | Agent may change? | Rationale |
|---|---|---|
| Splits / folds / test-set identity / labels | **No — frozen** | The evaluation protocol; mutable → cross-cell incomparable + test-leak surface |
| Feature extraction (mag, patch size, tiling, segmentation, stain-norm policy) | **No — shared substrate** | This *is* the encoder axis; per-cell change confounds the headline |
| Encoder-spec-dictated prep (native mpp / patch size / channel-norm) | Standardized per encoder, not searched | Field norm; standardization, not tuning |
| Train-only feature handling (norm fit-on-train, bag sampling, dropout, aug, class weighting) | **Optional — declare it** | Downstream of fixed features; doesn't touch test identity |
| Architecture / aggregator internals / training hyperparams / loss | **Yes — the search target** | The de-biasing objective |

"Model-only" is too narrow — the recipe legitimately includes the training
procedure (lr, schedule, optimizer, regularization, loss). "Anything including
prep" is too broad — it collapses the headline. The frozen substrate is the line.

**Enforcement — the protected list is necessary but not sufficient.** Today
`registry.protected` ships empty ([`registry/config.py:37`](../../src/automil/registry/config.py)),
`files.readonly` is warning-only and bypassed by explicit `--files`, and
`run_experiment.py` (split entry point + composite writer) is `files.editable`.
Defense-in-depth, cheapest first:

1. **`registry.protected` — config-only, do this now.** Add to each dataset's
   `automil/config.yaml` (schema per `registry/config.py`; globs are project-root
   relative; a submit that overlays any match is hard-rejected at
   [`submit.py:263-274`](../../src/automil/cli/submit.py), independent of the soft
   `readonly` list):

   ```yaml
   registry:
     mode: "architecture-preserving"   # optional; complements the list below
     protected:
       # --- evaluation protocol: splits + metrics ---
       - "benchmarks/src/autobench/pipeline/splits.py"
       - "benchmarks/src/autobench/pipeline/prepare.py"
       - "benchmarks/src/autobench/pipeline/evaluate.py"
       - "benchmarks/src/autobench/pipeline/*/dataset.py"   # clam/nnmil/abmil/dtfd/titan/smmile split+feature consumers
       # --- composite writers (the val-selection guarantee) ---
       - "benchmarks/src/autobench/pipeline/*/runner.py"    # shared writer lives in clam/runner.py; nnmil/abmil/dtfd/titan/smmile all import it
       - "benchmarks/scripts/run_experiment.py"             # entry point + summary_to_result_json
       # --- feature extraction / encoder inputs ---
       - "benchmarks/scripts/run_feature_extraction.py"
       - "benchmarks/src/autobench/data.py"                 # WSI-list CSV feeding TRIDENT
   ```

   (Protecting `run_experiment.py` blocks the dispatch entry point — acceptable,
   since model variants belong in `automil/variants/*.py`. If a recipe ever needs
   the entry point, first extract `summary_to_result_json` into its own protected
   module and keep dispatch editable.)
2. **Operational — do before launching the loop.** (a) Pre-generate all split
   CSVs into the shared `benchmark_dir`; generation is idempotent, so the `seed`/
   `n_folds` that live in the editable `pipeline/config.py` + consumer `data:`
   block become **inert during search**. (b) Have the orchestrator **recompute
   `composite` from the val `metrics` block** rather than trusting the scalar
   ([`terminal_writer.py:205`](../../src/automil/terminal_writer.py)), so an edited
   writer cannot fold test into selection.
3. **Structural — airtight, larger refactor (Phase 2).** Withhold the test split
   from the worktree during search entirely; the orchestrator evaluates the frozen,
   selected model on test out-of-process. `train.py` currently receives
   `(train, val, test)` in-process ([`clam/runner.py:119`](../../benchmarks/src/autobench/pipeline/clam/runner.py)),
   so layers 1–2 rest on the agent not mislabeling test as val; layer 3 removes the
   leak at its root.

## Open / pending — to confirm

1. **Final 5-dataset list — RESOLVED (2026-07-17):** TCGA-LUAD, TCGA-LGG,
   CPTAC-GBM, CPTAC-PDAC, TCGA-HNSC (see §1). Chosen for classification-task
   diversity (binary mutation + 3-class immune subtype + 3-class grade) across
   TCGA + CPTAC; OS survival retained as a secondary axis on all five. Replaces
   the earlier survival-power roster (LUAD/LGG/SKCM/BLCA/COAD).
2. **Regression — deferred to Phase 2 (see §4), but the premise needs one
   check.** The ovarian-HRD exception is ruled out (HRD is pre-binarized; ovarian
   is off-cluster and not a roster cohort). However CPTAC-PDAC's `immune_class` is
   a tertile split of a continuous infiltration score — a binned continuous target
   already in the roster. Check whether that score is recoverable from the CPTAC
   source before calling the deferral final.
3. **One unmerged branch remains — narrower than this item once claimed.**
   - `feat/nnmil-survival` — **RESOLVED: merged.** The survival pipeline is on
     `main` (the branch no longer exists on `origin`), along with the roster
     configs and the TITAN arm.
   - `feat/goldmark-parity` — the protocol-validation work (shared background
     Phase B). **Pushed to GitHub** (`origin/feat/goldmark-parity`, `d42f0b4`) —
     it is *not* cluster-only and *not* at risk from a scratch purge — but still
     **not merged into `main`**.
   So the remaining task is a decision about goldmark-parity plus one
   commit/tag (`preprint-pipeline-v1`) as "the" pipeline version.
4. **Enforce the frozen substrate before the agentic loop (see §5).** Wire
   `registry.protected` into each dataset's `automil/config.yaml`, pre-generate
   splits into the shared `benchmark_dir`, and recompute `composite`
   orchestrator-side from the val block.

   _Status 2026-07-28._ The composite **is** now recomputed orchestrator-side from
   the val block (CR-1b, `731cea2`) — with the residual limit that
   `scoring.recompute_composite` averages whatever is in `metrics`, so a rewritten
   writer that relabels test as val still passes. The protected list drafted in §5
   has been applied to **`clwd`, `ovarian_hrd`, `placeholder`** (commit `0b2da55`) —
   three template projects — but **not to `ccrcc`, the only live one**, which has no
   `registry:` block at all and therefore no enforcement (the gate is written
   `if reg_cfg.protected and …`, so an empty tuple is a no-op). None of the five
   roster cohorts has an `automil/` overlay yet. See
   [`READINESS-2026-07-28.md`](READINESS-2026-07-28.md) §1.4–1.5.

## Sources (pivot-specific, in addition to shared background)

- arXiv:2512.17517 (PathBench-MIL) — abstract + github.com/Sbrussee/PathBench-MIL README
- Frontiers `fbinf.2026.1809049` (embedding choice for OS prediction) — full text summary
- github.com/mahmoodlab/patho-bench README
- github.com/kaiko-ai/eva README
- huggingface.co/MahmoodLab/TITAN model card
- Table 1 screenshot (PathBench-MIL vs Patho-Bench vs EVA)
