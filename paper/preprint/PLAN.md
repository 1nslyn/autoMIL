# Phase 1 — Preprint plan (active)

_Background/shared context: [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Full-scope journal plan: [`../journal/PLAN.md`](../journal/PLAN.md). Raw source
material: [`../references/`](../references/). Compiled 2026-07-01; §5 resolved
2026-07-15, §1 roster resolved 2026-07-17, last reconciled against `main`
2026-07-21. **Counts are owned by [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md)** —
where this file disagrees on a number, that one wins._

> **Contribution authority.** See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md):
> C1 = the autoMIL auditable research-operations framework, including its
> matched-evaluation and sealed-certification contract; C2 = the result-neutral
> pathology-MIL ranking audit. Fig. 3 is the planned main empirical result, not
> the whole paper. C3 is deliberately unassigned; C4/C5 remain candidates until
> their promotion gates are met. The older *encoder ≫ aggregator* headline is
> dropped by the settled 2026-07-28 scope decision.

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
  Phase C) and was also the direct precedent for the subsequently dropped
  encoder>aggregator claim surfaced in the citation-ranking review.

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
| AutoML capabilities | ✓ | ✗ | ✗ | ✓ *(ours is agentic code-level recipe search rather than config/hyperparameter menu search; it needs its own row or footnote in the real table)* |
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
  the mechanism behind C1 and doesn't fit in a single checkmark — it needs its
  own comparison row or a footnote in the real table.
- **Published-SOTA protocol-parity validation** — none of these three papers
  claim this. The GOLDMARK exact-protocol reproduction (shared background
  Phase B) is evidence our pipeline reproduces published SOTA numbers under
  identical training logic, not just "runs the models." Worth a row.

**Separate system-novelty comparison required.** The MIL capability table above
cannot establish C1's novelty because PathBench-MIL, Patho-Bench, and EVA are
not the closest autonomous-research systems. The paper must separately compare
autoMIL with AIDE, MLAgentBench, AI Scientist-v2, AIRA/AIRA2, RE-Bench,
MLRC-Bench, and AMID on repository scope, candidate representation,
parent-linked reconstruction, budget accounting, protected source surfaces,
held-out non-interference, and backend semantics. Source-level code editing,
tree search, memory, budgets, hidden evaluation, and isolation are prior art;
[`RELATED_WORK.md`](RELATED_WORK.md) records the resulting boundary. C1 rests on
the auditable lineage-comparison substrate, not on claiming those ingredients
individually.

**Positioning note — encoder vs. MIL benchmark.** autobench is a *full-pipeline*
benchmark: it varies both the **encoder** axis (patch + slide foundation models)
and the **MIL aggregator** axis. Covering both is **table-stakes, not the edge**:
PathBench-MIL already spans both axes (Patho-Bench and EVA are encoder-centric
— the "covers entire MIL pipeline" row above), and the Frontiers study does it
at small scale. The wedge is therefore *not* "first to cover both" or a
predeclared directional variance claim — it is **recipe-bias control**.
PathBench-MIL varies both axes with fixed/menu-configured recipes, whereas we
give every cell an equal-effort agentic search and measure whether the
cross-method ranking changes or remains stable. Frame empirical C2 on that
result-neutral comparison.

### 4. Two capability gaps — status

Two gaps identified: slide-level PFM coverage and regression. **Priority split:
TITAN is in the preprint; regression is deferred to Phase 2.**

The reasoning: TITAN is the higher-value, lower-risk move for a fast preprint —
it adds the distinct slide-level regime needed for credible full-pipeline
coverage, it's table-stakes vs. the two closest competitors (PathBench-MIL and
Patho-Bench both have slide-level PFMs), and it reuses the existing task types,
metrics, and `result.json` contract. Because TITAN does not receive the same
search regime as the tile-level arms, C2 must label it separately rather than
treat it as a fully matched cross-method comparison.
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

### 5. Agent search-space scope — frozen measurement, method identity, programmatic recipe

**Resolved (2026-07-15; boundary completed 2026-07-30): the agent's equal-effort
search runs on a frozen data/evaluation substrate and inside a predeclared
method-identity contract.** This is what makes C2's cross-method comparison
interpretable: recipe effects cannot be separated from data-preparation effects
if the latter changes independently in each cell, and a searched result cannot
be attributed to a published method if its defining mechanism was removed or
replaced. Patho-Bench, EVA, and the Frontiers embedding paper all freeze
mag/patch/splits and let no search touch preprocessing; the Frontiers paper even
disables stain-norm on purpose "to avoid preprocessing-induced biases across
cohorts."
PathBench-MIL is the lone exception, and it exposes tiling/normalization as a
*declared, compared* menu axis — not a hidden per-cell optimization run under an
"isolate the encoder" claim. Freezing the substrate is also on-brand for the
"MIL-benchmark rigor backbone" positioning: the freeze *is* the rigor claim.

**The line — what the agent may vs. may not change:**

| Layer | Agent may change? | Rationale |
|---|---|---|
| Splits / folds / test-set identity / labels | **No — frozen** | The evaluation protocol; mutable → cross-cell incomparable + test-leak surface |
| Feature extraction (mag, patch size, tiling, segmentation, stain-norm policy) | **No — shared substrate** | This defines the encoder input; per-cell change confounds C2's cross-method comparison |
| Encoder-spec-dictated prep (native mpp / patch size / channel-norm) | Standardized per encoder, not searched | Field norm; standardization, not tuning |
| Method-defining inference operator, forward branch, and core training mechanism | **No — frozen per arm** | Removing or replacing these changes which published method is being measured |
| Declared scalar configuration (lr, wd, epochs, dropout, width, etc.) | **Yes** | Valid configuration of the same method class |
| Programmatic train-only recipe (sampling, adaptive schedule, optimizer/gradient policy, additive regularization, stopping) | **Yes — the primary code-level surface** | Changes the training program without replacing the method's defining mechanism |
| Identity-breaking architecture or objective change | **No for C2** | Archive separately as an evolved lineage; never attribute it to the published method |

The boundary is therefore semantic, not "HP versus code." For arm \(a\), the
consumer freezes the inference method \(f_a\) and every defining mechanism in
\(L_a^{core}\). The searched recipe may be an executable program: it can define
train-only sampling \(S_r(D_{train}, history)\), parameter updates
\(U_r(\theta,\nabla L,t,stats)\), schedules, gradient transforms, and additive
regularization. It may not add or replace inference-time operators, introduce a
new learnable forward branch, or delete/bypass/zero a defining core mechanism.
A correctness bug is not an agentic gain: discovery pauses the campaign, fixes
and versions the common protocol, and regenerates the affected baseline.

**Decision — M-13 resolved.** The C2 searched leaderboard is
**method-identity-preserving but programmatic-recipe-searchable**. It is neither
best-evolved-head nor recipe-only in the sense of a finite hyperparameter menu.
Candidates are classified as `config-only`, `programmatic-recipe`,
`identity-breaking`, or `invalid`; only the first two enter C2. The selected
winner need not contain a source edit, but the campaign must genuinely execute
and archive the programmatic-recipe surface rather than merely advertise it.

**Enforcement status — the scientific decision is made; the code does not yet
implement it.** The 18-entry `registry.protected` substrate list landed across
all projects in `be01096`, and orchestrator-side composite recomputation landed
in `731cea2`. Those controls freeze measurement, not method identity:

1. `registry.mode: architecture-preserving` currently requires only a non-empty
   protected list; it does not encode per-arm defining mechanisms.
2. The roster configs still signpost CLAM model/core files as editable, while
   no other arm receives an equivalent model-file surface.
3. `search_space.py` declares the scalar override channel, but arbitrary source
   edits bypass that declaration.
4. The structured `PolicyVariant` and `LossVariant` paths hard-fail, so there is
   no runnable bounded programmatic-recipe channel between scalar HPO and
   unrestricted source editing.

Before the confirmatory campaign:

1. Protect every arm's model/forward implementation and defining loss/training
   stages symmetrically; lock degeneracies such as CLAM
   `no_inst_cluster=true` and `bag_weight=1`.
2. Add a consumer-side `RecipePolicy` module called from protected trainers,
   with programmable hooks for optimizer, scheduler, train-only sampling,
   additive regularization, gradient transforms, and epoch/stopping policy.
3. Enforce machine-readable per-arm method invariants on every submission and
   persist the four-way candidate classification in the archive.
4. Keep the data/evaluation firewall already described above; Phase 2 may still
   strengthen it by withholding the test split from the worktree entirely.

### 6. Framework vs. consumer — what "plug-and-play" may claim

**The question (2026-07-29).** §5 fixes *where* the line is for this benchmark.
It does not answer the structural one: every arm exposes a different set of
tunable knobs (CLAM 15, DTFD 13, nnMIL 10, ABMIL 8, TITAN 5 —
[`search_space.py:64-145`](../../benchmarks/src/autobench/pipeline/search_space.py)),
so if autoMIL needs per-arm adaptation to know what an agent may tune, in what
sense is it a plug-and-play optimization framework? Verified by reading the
permission chain end to end.

**Resolved: the framework's permission is FILE-level, not parameter-level.**
The generic framework never enumerates hyperparameters. The consumer currently
declares three channels into a run, but only the first two are broadly usable:

| Channel | How the agent uses it | What gates it |
|---|---|---|
| **1. Source edit** (primary) | edit project files, then `automil submit --files …` | `files.editable` supplies the auto-detect scope ([`submit.py:208`](../../src/automil/cli/submit.py)); `registry.protected` glob-matches every submitted path and **hard-rejects** ([`submit.py:263`](../../src/automil/cli/submit.py)) |
| **2. Arg append** (secondary) | `automil submit --override "--lr 1e-4"`, suffix-appended to `run.command` in the worktree ([`submit.py:43`](../../src/automil/cli/submit.py)) | reaches `run_experiment.py` → `apply_overrides(…, arm=…)`, which raises on any knob not in that arm's declared space |
| **3. Structured variant** (declared, incomplete) | register a model/loss/policy module | model variants currently collapse to a CLAM-shaped argument menu; loss and policy variants hard-fail (`VAR-1`) |

**The declared search space is enforced on channel 2 only.** Channel 1's boundary
is entirely the file white/black list, and channel 3 does not yet supply the
programmatic-recipe seam settled in §5. This matters for how the freeze is
described: the substrate is protected because `splits.py` / `prepare.py` /
`evaluate.py` / `*/runner.py` / `run_experiment.py` are *unwritable*, while
method identity and the allowed training-program surface remain a consumer-side
contract that still needs implementation.

**The framework/consumer boundary is real, and it is tested.**
`search_space.py` and `hparams.py` live under
`benchmarks/src/autobench/pipeline/` — **consumer side**. `src/automil/` contains
no hyperparameter or MIL vocabulary in any code path (only docstring examples),
and [`tests/test_framework_purity.py`](../../tests/test_framework_purity.py)
greps the package on every run so a future commit cannot quietly reintroduce one.
Per-arm adaptation is therefore autobench's cost, not the framework's:

| Task | Cost | Where |
|---|--:|---|
| Add a 6th MIL aggregator for config-only search | ~15 lines: one `SEARCH_SPACE` entry, one `apply_overrides(…, arm=…)` in its runner ([`abmil/runner.py:72`](../../benchmarks/src/autobench/pipeline/abmil/runner.py), [`dtfd/runner.py:78`](../../benchmarks/src/autobench/pipeline/dtfd/runner.py), [`titan/train.py:84`](../../benchmarks/src/autobench/pipeline/titan/train.py), [`nnmil/runner.py:74`](../../benchmarks/src/autobench/pipeline/nnmil/runner.py)), one scope declaration | consumer |
| Add that arm to the C2 programmatic-recipe audit | declare method invariants and connect the protected trainer to the shared `RecipePolicy` hooks; exact adapter cost must be reported, not hidden inside the framework claim | consumer |
| Attach autoMIL to a new project | **0 lines of code** — `config.yaml` only (`run.command`, `files.editable`, `registry.protected`, `scoring`) | consumer config |

**The layering already matches the fairness rule.** The two independent advisory
passes on the baseline question (2026-07-29) converged on the same three-tier
partition, and it maps one-to-one onto config keys that already exist:

| Tier | Content | Declared where | Enforced by |
|---|---|---|---|
| **0 — measurement apparatus**, identical across arms | splits, folds, features, labels, composite, val/test firewall | `registry.protected` | **framework**, at submit |
| **1 — method identity**, fixed per arm | inference operator family, defining forward branches, core loss/training mechanisms | per-arm invariant contract + protected core files | **consumer declares; framework must enforce** |
| **2 — budget**, equalized by quantity not value | eval count, wall-clock, keep-margin δ and `se_multiplier` | `cap`, `scoring` | **framework** |
| **3 — recipe program**, free per arm within Tier 1 | scalar configuration plus train-only sampling, optimization, gradient, regularization, and stopping programs | declared knobs + `RecipePolicy` hook | **consumer declares** |

The classification test is semantic: *would varying it change the measurement
apparatus, the published method being measured, or only how that method is
trained?* Those map to Tiers 0, 1, and 3 respectively. Per-arm recipe
differences are intended; silent differences in enforcement are not.

**What the paper may claim, and what it may not.**

- ❌ Do **not** claim autoMIL determines an arbitrary model's tunable parameters,
  or ships a declared search space out of the box. It does neither. The framework
  supplies the *lock*; the consumer supplies the *key list*.
- ✅ Claim plug-and-play at the **mechanism** layer — worktree overlay with
  manifest hash verification, file-level freeze enforced at submit, val-firewall
  with born-sealed test, noise-calibrated keep/discard, budget cells. None of it
  knows what a learning rate is, and attaching it to a new repo is config-only.
- ✅ Claim rigor at the **instance** layer only after this benchmark publishes
  each arm's method invariants and programmatic-recipe contract, records a reason
  for every lock, and rejects both undeclared knobs and identity-breaking source
  edits rather than silently accepting or dropping them.

Both halves are independently checkable (0-lines-of-code attachment; the
published per-arm table), which is why they should be stated separately rather
than merged into one sentence that overclaims.

**The honest weakness, stated before a reviewer states it.** Channel 1 is
source-level editing, so *by default* the search space is "whatever the editable
files allow" — unbounded and unauditable. `search_space.py` makes only the scalar
override channel finite; it does not govern source edits. The §5
method-invariant contract and `RecipePolicy` seam are therefore prerequisites
for the strongest C2 claim. Plug-and-play buys the harness and the
measurement freeze; it does not buy the consumer's scientific declaration.

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
