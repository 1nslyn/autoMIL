# Preprint — Experiment Grid & Estimation

> ## Cluster state — verified 2026-07-30 on `fir`
>
> Full report, per-cell tables and raw census:
> [`GRID-CENSUS-2026-07-30.md`](GRID-CENSUS-2026-07-30.md) ·
> [`data/grid_census_2026-07-30.tsv`](data/grid_census_2026-07-30.tsv).
> Every `summary.json` under
> `.../Pathology/autoMIL/phase2/<cohort>/benchmark_5fold/results/` was parsed
> (237 files, 0 unreadable) and cross-checked against 975 fold-level
> `metrics.json`. This supersedes the 2026-07-29 block, whose survival column
> double-counted the loss axis.
>
> **The grid is complete. Every roster cell is present.**
>
> | Cohort | roster task | classification | survival (`nllsurv`) | `cox` extras | owner |
> |---|---|--:|--:|--:|---|
> | tcga_luad | `kras` | **13/13** | **13/13** | +7 | yinshuol |
> | tcga_lgg | `idh1` | **13/13** | **13/13** | +7 | yws0322 |
> | cptac_gbm | `tp53` | **13/13** | **13/13** | +7 | kcuoft |
> | cptac_pdac | `immune_class` | **13/13** | **13/13** | +7 | atatc |
> | tcga_hnsc | `grade` | **13/13** | **13/13** | +7 | ryanwk |
>
> **Survival is `nllsurv`-only from here (2026-07-30 decision).** The earlier
> `20/20` counted both survival losses: 20 = `13 nllsurv + 7 cox`. Dropping `cox`
> makes the roster **26 experiments/cohort → 130 total / 650 fold-trainings**
> (was 33 → 165/825). §2.1 and §2.2 are restated accordingly; the YAMLs still
> declare `survival_losses: [cox, nllsurv]` and must be narrowed to `[nllsurv]`.
>
> **`cox` was never reportable anyway.** `clam_mb` and `dtfd_mil` have **zero**
> `cox` runs on any cohort (7/13 present, identical pattern all five) — closing
> that arm would take 30 more experiments. Where both losses do exist (35 paired
> cells) `nllsurv` wins **27/35**, mean Δ c-index **+0.0147**. So `nllsurv`-only
> is defensible on the evidence, not just on scope.
>
> **Validation of the 130 roster cells (recomputed 2026-07-30):**
>
> - **Fold integrity: perfect.** 975 fold directories = 195 × 5 exactly; on-disk
>   `fold_*/` count matches `n_folds` for **every** experiment, zero mismatches.
> - **Zero non-finite values in any primary metric** (`auc_roc` / `c_index` /
>   `balanced_accuracy`) across all 975 fold files and all 237 summaries. The 880
>   NaNs that do exist are all `sensitivity` / `specificity` — undefined for the
>   two 3-class tasks by construction, plus an nnMIL-only asymmetry on binary
>   tasks (the L-10 family). No composite is affected.
> - **No cache collision.** 237 experiments produced **237 distinct** per-fold
>   signatures. The CR-5 / CR-5b shared-cache hazard did **not** materialise here.
> - **Survival postdates the censoring fix.** Oldest `os` result is 2026-07-22
>   16:55; the fix landed 2026-07-21 ~21:20. All survival is 5-fold, and the
>   10-fold tree contains none, so the 10-fold cache-reuse hazard did not fire.
> - **Single seed.** All 237 are `seed=42`, `strategy=standard`. Every mean±std in
>   the results is **fold variance at one seed, not seed variance** — say so
>   wherever it appears.
> - **⚠ `kappa` is emitted by nnMIL alone.** 120/120 of nnMIL's classification
>   folds carry it; `clam` (0/150), `abmil` (0/90), `dtfd` (0/90) and `titan`
>   (0/25) never do — it exists only in
>   [`nnmil/evaluate.py:55`](../../benchmarks/src/autobench/pipeline/nnmil/evaluate.py)'s
>   metric map. **Kappa is unusable cross-arm**, which costs most on
>   `immune_class` and `grade`. For those two the comparable set is `auc_roc`,
>   `accuracy`, `balanced_accuracy`, `f1`. (Multi-class AUC *is* consistent —
>   both evaluators use `multi_class="ovr", average="macro"`.)
> - **⚠ `config.json` misreports the hyperparameters for 3 of 5 arms.** All 195
>   configs record `lr=2e-4, weight_decay=1e-5, max_epochs=200` — the shared
>   `TrainConfig`. Only CLAM and ABMIL actually used those. DTFD trained at its
>   own 1e-4/1e-4, nnMIL at its plan's 3e-4 (cls) / 1e-4 (surv) with 100 epochs,
>   TITAN at 1e-3/1e-4. That is finding H-3 made concrete: **102 of 195 configs
>   describe a recipe that did not run.** Do not build a methods table from
>   `config.json` for these results; use `pipeline/provenance.py`.
>
> **⚠ Off-roster material must be excluded by filter, not by directory sweep.**
> TCGA-LUAD carries **72** experiments beyond its 26 roster cells: an `egfr` task
> (21), extra aggregators `clam_sb`/`mil`/`trans_mil` on `kras` (9), and a
> separate `benchmark_10fold` tree (42 exps, 340 GPU-h, no survival). A naive
> `find` over LUAD returns 105 experiments where the roster is 26. Pin the filter:
> 5-fold trees only · roster task per cohort · `surv_loss == "nllsurv"` ·
> `model ∈ {clam_mb, simple_mil, abmil, dtfd_mil, titan}`.
>
> **Re-run cost is real, not zero.** `fix/audit-2026-07-23` returns CLAM to
> upstream `lr=1e-4` and ABMIL to upstream `5e-4 / 1e-4 / 20 epochs`, so **93 of
> the 195** on disk would produce different numbers — **50 of the 130 roster
> cells** (25 CLAM + 25 ABMIL). DTFD, nnMIL and TITAN are untouched by those
> changes and should reproduce.
>
> Oldest result 2026-07-10, newest **2026-07-27 09:30**. §3.2's "conch_v15 not
> confirmed extracted on any cohort" is stale — TITAN ran on all five.

## 0. TL;DR

| Quantity | Value |
|---|---|
| Datasets | **5** — 3 TCGA + 2 CPTAC |
| Task axes | **2** — classification (binary mutation · 3-class subtype · 3-class grade) + survival (OS) |
| MIL aggregators | **4** — `clam_mb`, `simple_mil`, `abmil`, `dtfd_mil` |
| Patch encoders | **3** — `uni_v2`, `virchow2`, `hoptimus1` |
| Slide-encoder arm | **TITAN** (its own encoder + aggregator, linear probe) |
| Survival loss | **`nllsurv` only** (2026-07-30 decision — `cox` dropped, §2.1) |
| **Experiments / dataset** | **26** (24 tile-encoder + 2 TITAN) — 13 classification + 13 survival |
| **Total experiments** | **130** |
| **Total fold-trainings** (×5-fold) | **650** |
| Static-grid compute | **~300–350 GPU-hours — measured, not estimated** (≥261 GPU-h over 115 of 130 cells; 15 untimed — §3.1) |
| Real long pole | **~~`conch_v15` feature extraction for TITAN~~ — done, TITAN ran on all five cohorts** |
| Status | **complete on `fir`** — all 130 roster cells present and validated (cluster-state block above) |
| **Not yet budgeted** | **the agentic recipe-search layer** — required C1 validation and C2 empirical campaign; ~15–20× the static grid at full scope (§3.3) |

The static grid is **not** as cheap as this doc long assumed — ~300–350 measured
GPU-hours against a ≈40 GPU-h estimate (§3.1). The primary contribution is the autoMIL framework
(C1), including its matched-evaluation contract; the expensive equal-effort
**agentic recipe search on top of each cell** validates C1 and produces planned
empirical C2. It is not yet fully
scoped in this compute plan and changes the compute picture by an order of
magnitude. Contribution authority: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md).

---

## 1. Final dataset roster

Confirmed 2026-07-17. This **supersedes** the earlier survival-power roster
(LUAD/LGG/SKCM/BLCA/COAD) and EXECUTION_PLAN.md §2's {THCA, LGG, LUAD, HNSC,
COAD} recommendation. The roster now spans **three classification task types
across two data sources**; each cohort is pinned to a single classification task
+ an OS survival task.

| Dataset | Source | Classification task | Task type | OS deaths | YAML |
|---|---|---|---|--:|---|
| TCGA-LUAD | TCGA | KRAS mutation | binary | 167 | [`tcga_luad.yaml`](../../benchmarks/datasets/tcga/tcga_luad.yaml) |
| TCGA-LGG | TCGA | IDH1 mutation | binary | 115 | [`tcga_lgg.yaml`](../../benchmarks/datasets/tcga/tcga_lgg.yaml) |
| CPTAC-GBM | CPTAC | TP53 mutation | binary | 72 | [`cptac_gbm.yaml`](../../benchmarks/datasets/cptac/cptac_gbm.yaml) |
| CPTAC-PDAC | CPTAC | immune subtype | 3-class | 81 | [`cptac_pdac.yaml`](../../benchmarks/datasets/cptac/cptac_pdac.yaml) |
| TCGA-HNSC | TCGA | tumor grade | 3-class | 205 | [`tcga_hnsc.yaml`](../../benchmarks/datasets/tcga/tcga_hnsc.yaml) |

Selection prioritizes **classification-task diversity** (binary mutation +
3-class immune subtype + 3-class tumor grade) and **cross-source coverage** (TCGA
GOLDMARK + CPTAC Patho-Bench); OS survival is retained as a secondary axis on all
five. The earlier **≥100 OS-deaths hard gate is dropped** — CPTAC-GBM (72) and
CPTAC-PDAC (81) fall below it — a deliberate trade of survival power for task/
source diversity and a small-sample regime (GBM n=99, PDAC n=105). Tasks are all
distinct. Full rationale is mirrored in `paper/preprint/PLAN.md` §1.

### 1.1 Dataset statistics (Table-1 candidate)

Rendered version: [`figures/mock/table1_dataset_stats.png`](figures/mock/table1_dataset_stats.png)
(`figures/make_dataset_table.py`). The roster mixes task types, so the Table-1
figure stacks three sub-tables — binary mutation, 3-class immune subtype, 3-class
tumor grade — matching each cohort's class structure.

**Binary mutation (mutant vs. wildtype):**

| Dataset | Cls task | Patients (n) | Mutant (+) | Wildtype (−) | Prevalence | OS deaths | OS prevalence |
|---|---|--:|--:|--:|--:|--:|--:|
| TCGA-LUAD | KRAS | 465 | 171 | 294 | 36.8% | 167 | 35.9% |
| TCGA-LGG | IDH1 | 491 | 382 | 109 ⚑ | 77.8% | 115 | 23.4% |
| CPTAC-GBM | TP53 | 99 | 32 | 67 | 32.3% | 72 | 72.7% |

**3-class immune subtype (low / medium / high):**

| Dataset | Cls task | Patients (n) | Low | Medium | High | OS deaths | OS prevalence |
|---|---|--:|--:|--:|--:|--:|--:|
| CPTAC-PDAC | immune_class | 105 | 35 | 35 | 35 | 81 | 77.1% |

**3-class tumor grade (G1 / G2 / G3):**

| Dataset | Cls task | Patients (n) | G1 | G2 | G3 | OS deaths | OS prevalence |
|---|---|--:|--:|--:|--:|--:|--:|
| TCGA-HNSC | tumor grade | 431 (414 gradeable) | 54 | 260 | 100 | 205 | 47.6% |

- **⚑ TCGA-LGG polarity:** IDH1-mutant is the **majority** class (382/491, 78%) —
  gliomas are commonly IDH-mutant, so the power-limiting minority is the 109
  **wildtype** cases. This also explains LGG's high baseline AUC.
- **Small-sample cohorts:** CPTAC-GBM (n=99) and CPTAC-PDAC (n=105) are the
  smallest, included on purpose to probe the small-sample noise-vs-memorization
  regime. **CPTAC-PDAC's immune subtype is balanced by construction (35/35/35)
  because it is a tertile split of a continuous tumour-immune infiltration
  score** — disclose that derivation in the paper's Table 1. An exactly equal
  3-way split presented without explanation reads as concealment; it also
  means this task is a *binned continuous target*, which bears on the
  regression deferral (see `PLAN.md` §4 and PRELAUNCH_REVIEW §5 item 5).
- **TCGA-HNSC grade:** 54 + 260 + 100 = 414 gradeable of 431 patients; the 17
  GX / not-reported cases carry no grade label and drop from the grade task
  (survival still uses all 431).
- **Units:** `n` is patient/case-level; TCGA is ~1 slide/case (slides ≈ cases),
  CPTAC carries ~2.4 slides/case. Each slide = one MIL bag. OS deaths are
  case-level events; OS prevalence = deaths / n.
- **⚠ The survival cohort is smaller than `n`.** Slides with a missing or
  non-positive `OS_time` are dropped from the survival task (non-positive
  follow-up is undefined for Cox's partial likelihood). The **usable** survival
  cohorts are:

  | Dataset | Usable patients | Usable deaths | (vs cohort `n` / raw deaths) |
  |---|--:|--:|---|
  | TCGA-LUAD | 453 | 162 | 465 / 167 |
  | TCGA-LGG | 488 | 114 | 491 / 115 |
  | CPTAC-GBM | 98 | 72 | 99 / 72 |
  | CPTAC-PDAC | 102 | 81 | 105 / 81 |
  | TCGA-HNSC | 429 | 204 | 431 / 205 |

  **Quote the usable figures in any survival results table** — the OS-deaths
  column above is the cohort-level count, which overstates what actually trains
  by 5 (LUAD), 1 (LGG) and 1 (HNSC) deaths. Attrition is now small (0.5–2.6%).

  > **Provenance note.** Until 2026-07-21 these cohorts were materially smaller
  > (LUAD 399, LGG 445, HNSC 397) because `add_os_to_manifest.py` reduced the GDC
  > clinical export with `drop_duplicates` (first *row* per case) rather than
  > first *non-null*. `days_to_last_follow_up` is a diagnoses-level field that is
  > often blank on a case's first row, so living patients received `OS_event=0`
  > with `OS_time=NaN` and were dropped — **129 patients across the three TCGA
  > cohorts, every one of them censored, and zero deaths.** That is informative
  > censoring: it inflated the event rate (LUAD 0.406 → corrected 0.358) and
  > would have biased every Cox/nllsurv fit and c-index on the TCGA arms. Fixed
  > by reducing per column over non-null values. Any survival number produced
  > before that date should be regenerated, not reused.
- **Provenance:** TCGA-LUAD/LGG mutation counts match the May `task_sizes.csv`
  baseline; CPTAC-GBM/PDAC and TCGA-HNSC counts + all OS-death counts are from
  the roster figure (2026-07-17). `task_sizes.csv` is mutation-only (15-cohort
  baseline) and does **not** cover the 3-class immune/grade tasks — verify the
  3-class and OS counts against the cluster manifests when the campaign runs.

---

## 2. The static grid (verified)

Each of the five YAMLs is structurally identical:

- **tasks:** one classification task (`n_classes: 2` for the binary-mutation cohorts LUAD/LGG/GBM; `n_classes: 3` for CPTAC-PDAC immune subtype and TCGA-HNSC grade) **+** `os` (survival, `survival_losses: [cox, nllsurv]`)
- **aggregators:** `clam_models: [clam_mb]`, `nnmil_models: [simple_mil]`, `abmil_models: [abmil]`, `dtfd_models: [dtfd_mil]` — one model per framework
- **encoders:** `uni_v2`, `virchow2`, `hoptimus1`
- **titan:** `head: linear` (slide-level arm, no tile-encoder sweep)

**Multi-class is transparent to the grid.** The two 3-class tasks need no code
changes: `compute_extended_metrics` computes per-class one-vs-rest AUC via
`label_binarize` + `nanmean` ([`pipeline/evaluate.py`](../../benchmarks/src/autobench/pipeline/evaluate.py)),
every model head (CLAM/nnMIL/DTFD/ABMIL/TITAN) is built to `n_classes`, and
splits use `StratifiedKFold` — all class-count-agnostic. Verified by running
`generate_all_experiments` over all five YAMLs: **the same count each, regardless
of `n_classes`** (33 as currently configured, 26 once `survival_losses` is narrowed
to `[nllsurv]` — §2.1), so the totals below are unchanged from the earlier
binary-only roster.

> ⚠ **One metric is not class-count-transparent after all.** Cohen's `kappa` is
> emitted only by the nnMIL evaluator
> ([`nnmil/evaluate.py:55`](../../benchmarks/src/autobench/pipeline/nnmil/evaluate.py)) —
> 120/120 of its classification folds carry it, and `clam`/`abmil`/`dtfd`/`titan`
> never do. So on the two 3-class tasks, where kappa is the natural agreement
> statistic, it **cannot be used cross-arm**. Same caveat as
> `sensitivity`/`specificity`, which are undefined for 3-class by construction. The
> cross-arm comparable set on `immune_class` and `grade` is `auc_roc`, `accuracy`,
> `balanced_accuracy`, `f1`. Multi-class AUC itself *is* consistent — both
> evaluators use `multi_class="ovr", average="macro"`.

### 2.1 How the grid expands — per dataset

The expansion is **not** a clean 4×3×2; survival loss-eligibility differs per
framework (`generate_all_experiments`, [`pipeline/config.py:296`](../../benchmarks/src/autobench/pipeline/config.py)).
Rendered version: [`figures/mock/table2_grid_breakdown.png`](figures/mock/table2_grid_breakdown.png).

**Restated 2026-07-30: survival is `nllsurv`-only, so the expansion is now a clean
4×3 + 1 on both axes.**

| Framework · model | Classification (1 gene × 3 enc) | Survival OS — `nllsurv` (× 3 enc) | Row total |
|---|--:|--:|--:|
| CLAM · `clam_mb` | 3 | 3 | 6 |
| nnMIL · `simple_mil` | 3 | 3 | 6 |
| ABMIL · `abmil` | 3 | 3 | 6 |
| DTFD · `dtfd_mil` | 3 | 3 | 6 |
| **TITAN** (1 pseudo-encoder) | 1 | 1 | 2 |
| **Per-dataset total** | **13** | **13** | **26** |

**Why `cox` is dropped.** The YAMLs declare `survival_losses: [cox, nllsurv]`, and
loss-eligibility is filtered per framework in code, which made the survival column
uneven — `clam_mb` and `dtfd_mil` are cox-ineligible, so the configured grid was
`13 nllsurv + 7 cox = 20` per dataset:

- **`clam_mb` — nllsurv only.** `cox` needs a single-risk output that only
  `clam_sb` exposes; `clam_mb` is multi-branch → cox is skipped (`config.py:374`).
- **`dtfd_mil` — nllsurv only.** Cox's partial-likelihood needs a cross-patient
  risk set that doesn't exist within one slide's pseudo-bags (`config.py:397`).
- **`simple_mil` / `abmil` / TITAN — both losses.** Attention/linear heads take
  either loss (arbitrary output width), so cox + nllsurv both ran.

That asymmetry is exactly why `cox` is not reportable: **a complete cox arm is
7/13 by construction**, and closing it would take 30 extra experiments to give
`clam_mb`/`dtfd_mil` a cox path they cannot have. On the 35 cells where both
losses ran, `nllsurv` wins 27/35 (mean Δ c-index +0.0147). `nllsurv`-only is
therefore both the cheaper and the better-supported choice.

> **Action:** narrow `survival_losses: [cox, nllsurv]` → `[nllsurv]` in all five
> roster YAMLs ([`tcga_luad`](../../benchmarks/datasets/tcga/tcga_luad.yaml:23) ·
> [`tcga_lgg`](../../benchmarks/datasets/tcga/tcga_lgg.yaml:23) ·
> [`cptac_gbm`](../../benchmarks/datasets/cptac/cptac_gbm.yaml:23) ·
> [`cptac_pdac`](../../benchmarks/datasets/cptac/cptac_pdac.yaml:30) ·
> [`tcga_hnsc`](../../benchmarks/datasets/tcga/tcga_hnsc.yaml:29)) so
> `generate_all_experiments` emits 26/dataset and the configured grid matches the
> reported one. The 35 `cox` runs already on disk stay as a supplementary
> loss-ablation note; they are not deleted.

### 2.2 Totals

| Scope | Experiments | Fold-trainings (×5) | Measured GPU-h |
|---|--:|--:|--:|
| Per dataset | 26 | 130 | ~52 (≥) |
| Classification only (5 ds) | 65 | 325 | 137.5 |
| Survival only (5 ds, `nllsurv`) | 65 | 325 | ≥123.5 ⚠ |
| **Campaign (5 datasets)** | **130** | **650** | **≥261.0 → ~300–350** |

> **Superseded numbers.** This table previously read 33/165/825 — that counted the
> `cox` axis now dropped (§2.1). The `cox`-inclusive configured grid was 165
> experiments / 825 fold-trainings; **35 of those 165 were cox** and remain on
> disk as a supplementary ablation.

> ⚠ **GPU-hours are a lower bound.** Measured from `elapsed_seconds_total`, but the
> **`clam_mb` survival arm (15 cells / 75 fold-trainings) records no timing at
> all** — not in `summary.json`, not in any of its 75 fold `metrics.json`. It is
> the only affected group; elsewhere the summary field and the fold-level sum agree
> exactly. Metrics are complete and valid; only compute accounting is missing.
> Pricing those 75 fold-trainings at `clam_mb`'s own classification rate
> (68.6 min/fold) adds ≈86 GPU-h → **≈347 GPU-h** all-in. Fix the writer before any
> re-run: equal-effort budgeting is a C1 claim and depends on trustworthy timings.

> **Everything is already run.** All 130 roster cells exist and are validated on
> `fir` (cluster-state block). The old "runnable today = 150 experiments, TITAN
> blocked on `conch_v15`" caveat is **obsolete** — TITAN ran on all five cohorts,
> so the encoder axis is not degenerate and PRELAUNCH_REVIEW §3 items O1/O3 are
> resolved on the facts.

> Reproduce: the `validate config` block inside
> [`submit_benchmark.sh`](../../benchmarks/scripts/slurm/submit_benchmark.sh:78)
> prints `experiments=…  fold-trainings=…` for any dataset before launching — it
> calls the same `generate_all_experiments`. (Set `AUTOBENCH_TCGA_*_ROOT` to any
> stub path first; grid generation reads only the config, not the data.)

### 2.3 Launch surface

- **Tile-encoder grid** (30 exps/ds): `sbatch submit_benchmark.sh <dataset>` — 4×H100, `FRAMEWORKS="clam nnmil dtfd abmil"`, idempotent, self-resubmits before the 24 h wall.
- **TITAN arm** (3 exps/ds): `sbatch submit_titan_extract.sh <dataset>` (once, fold-independent) → `sbatch submit_titan.sh <dataset>` (1×H100).
- **Survival** is already part of the tile grid because `os` is a configured task; `submit_survival_benchmark.sh` exists for a survival-only re-run.

---

## 3. Estimation

### 3.1 Static-grid compute

**Measured, 2026-07-30, from the completed campaign** — this replaces the earlier
estimate, which anchored to EXECUTION_PLAN §4 (120 CLAM-heavy fold-trainings =
10.5 h single-GPU) and assumed nnMIL/ABMIL/DTFD/TITAN were all lighter than CLAM.
Per-head cost over the **130 roster cells only** (off-roster LUAD and `cox`
excluded), from `summary.json:elapsed_seconds_total`:

| Head | Roster exps | Fold-trainings | min / fold-training | GPU-hours |
|---|--:|--:|--:|--:|
| `clam_mb` — classification | 15 | 75 | 68.6 | 85.7 |
| `clam_mb` — survival | 15 | 75 | **untimed** | **—** |
| `simple_mil` | 30 | 150 | 38.3 | 95.7 |
| `abmil` | 30 | 150 | 14.6 | 36.5 |
| `dtfd_mil` (two-tier pseudo-bag) | 30 | 150 | 17.0 | 42.6 |
| TITAN (linear probe, 1 vec/slide) | 10 | 50 | 0.5 | 0.4 |
| **Total instrumented** | **115** | **575** | **27.2** | **261.0** |
| **All-in estimate** | **130** | **650** | ~32 | **≈ 347** |

> ⚠ **The old estimate was low by ~7.5–8.75×, not the 1.4× its own caveat
> anticipated.** It predicted 2.5–5 min/fold-training for the tile-encoder heads;
> measured is **14.6–68.6**. Only TITAN's linear probe was estimated correctly.
> The ranking was also wrong: `clam_mb` is indeed the most expensive on
> classification (68.6 min/fold), but `simple_mil` — assumed half CLAM's cost — is
> the single largest line item at 95.7 GPU-h, and `abmil` is the cheapest real
> aggregator, not equal to `simple_mil`.

> ⚠ **15 of 130 roster cells are untimed.** The `clam_mb` × `nllsurv` arm records
> no `elapsed_seconds` in `summary.json` or in any of its 75 fold `metrics.json`;
> every other group is internally consistent (summary field = fold sum, to the
> decimal). Metrics are valid — only timing is absent. The **≈347 GPU-h** all-in
> figure prices those 75 fold-trainings at `clam_mb`'s own measured classification
> rate; the cheapest-survival-arm anchor (DTFD, 29.8 min/fold) gives ≈298 GPU-h, so
> the defensible band is **~300–350 GPU-h**. Fix the writer before any re-run —
> equal-effort budgeting is a C1 claim and depends on trustworthy timings.

At 261+ GPU-h the static grid is **~2½–3 days on 4×H100**, not the "≈10–13 h" this
section previously claimed, and not EXECUTION_PLAN's "~1 day in one
self-resubmitting job." The campaign in fact ran 2026-07-10 → 07-27 across five
lab members' allocations, which is consistent with the measured total.

### 3.2 Feature extraction — the real long pole

> **✅ Resolved — this section is historical.** Both dependencies below were
> satisfied during the campaign: all 130 roster cells ran, which requires
> `conch_v15` for the 10 TITAN cells and all three patch encoders on all five
> cohorts. Training was **not** cheap relative to extraction (§3.1: ~300–350 GPU-h),
> so the "features gate it" framing no longer holds for this grid. Retained for the
> record and for any cohort added later.

Training is cheap; **features gate it.**

- **TITAN dependency — `conch_v15` @ 20×/512px**, all 5 cohorts. ~~Not confirmed
  extracted on any TCGA cohort as of the 07-03 audit.~~ **Extracted and used —
  TITAN ran on all five cohorts** (10 roster cells, 2026-07-25 → 07-26).
  ~24 GPU-h/cohort, parallelizable → ~1 day wall across 5 GPUs.
- **Patch features** (`uni_v2`/`virchow2`/`hoptimus1`): **all present on all five
  cohorts** — every one of the 120 tile-encoder roster cells produced results, which
  is only possible with the features in place. TCGA-HNSC's GDC grade join also
  landed (its `grade` task has 13/13 cells).

**Preflight (do first):** on the cluster, verify per-cohort feature presence
before launching training, or the grid stalls on missing inputs.

### 3.3 The agentic recipe-search layer — C1/C2 campaign, and NOT yet budgeted

The static grid above is the **"default recipe" leaderboard** — the *before*
numbers. autoMIL itself is C1; the campaign asks what happens when **every
matched tile-level cell gets an equal-evaluation agentic search**, producing the
searched leaderboard required for the result-neutral C2 comparison. That layer
multiplies compute massively and **appears in no preprint doc's budget yet.**

Using the proposal's protocol ([`references/…proposal…`](../references/automil-proposal-2026-04-29.md) §6.3)
as the cost model — a *cell* = (dataset, task, encoder, aggregator):

| Stage | Per cell | Fold-trainings |
|---|---|--:|
| Discovery | ~60 candidates × (1 seed × 3 inner folds) | ~180 |
| Promotion | top-10 × 3 seeds | ~30 |
| Final (frozen) | 5 seeds × 5 folds | ~25 |
| **Per cell** | | **≈ 235** |

> ⚠ **This exceeds autoMIL's own per-cell budget cap — needs reconciling before
> the loop launches.** At ~700 GPU-h / 60 cells this is **~11.7 GPU-h per cell**,
> against the framework-enforced **6-hour hard wall-clock cap per cell**
> (`.planning/PROJECT.md`). The two also define "cell" differently: the proposal's
> is `(dataset, task, encoder, aggregator)`; the framework's is
> `(dataset, encoder, parent)`. Either the search protocol shrinks (fewer
> discovery candidates or fewer inner folds) or the cap is raised for the
> campaign — as written the plan is not executable under the enforced budget.

- **One dataset, classification only** (3 enc × 4 agg = 12 cells): ~2,800 fold-trainings — already **~4.3× the entire static 5-dataset grid** (650 fold-trainings).
- **All tile-encoder classification cells** (5 ds × 3 enc × 4 agg = 60 cells): ~14,000 fold-trainings.

> ⚠ **Re-priced 2026-07-30 — the wall-clock here was built on the old ≈40 GPU-h
> base and is far too low.** At the campaign's **measured** classification rates
> (§3.1), 14,000 fold-trainings is not ~700 GPU-h. Weighting the four aggregators
> by their measured classification cost (`clam_mb` 68.6 · `simple_mil` 35.0 ·
> `abmil` 1.9 · `dtfd_mil` 4.3 min/fold — mean ≈27.5 min/fold) gives
> **≈6,400 GPU-h ≈ 67 days on 4×H100.** Even the cheapest-aggregator-only variant
> is ~450 GPU-h. Sanity-check against the ratio rule: the full audit is 15–20× the
> static grid, and the static grid is now ~325 GPU-h, so **≈4,900–6,500 GPU-h** —
> the two derivations agree. **The 60-cell full audit is out of reach**, not merely
> "journal-scale"; only a pilot is feasible.

> Two exclusions to state plainly, since "all classification cells" reads broader
> than it is: this 60-cell count **omits the 5 TITAN classification arms** (1 per
> dataset — TITAN has no encoder×aggregator fan-out, so classification is 65
> experiments but only 60 tile-encoder cells), and it **omits all 100 survival
> experiments** entirely. A search over survival cells too would be a further
> multiple on top.

**Implication for the preprint:** a paper about an agentic framework needs *some*
agentic result, but the full 60-cell search is out of scope for "ship fast."
Static-only plus off-roster feasibility anchors is no longer an acceptable C2.
The remaining scope decision is a balanced, matched pilot (12–18 roster cells,
~1–2 days on 4×H100) versus the full 60-cell classification audit. The pilot
must predeclare coverage across cohorts and lineages and can support only a
narrower C2 than the full audit. This is the biggest open compute decision.

### 3.4 Rough end-to-end (preprint = static grid + pilot agentic)

**Restated 2026-07-30 — the first two rows are done, and the third is the only
remaining pole.**

| Component | Status | Wall-clock (4×H100) |
|---|---|---|
| `conch_v15` extraction (5 cohorts, parallel) | ✅ **done** | — |
| Static grid (650 fold-trainings, 130 cells) | ✅ **done** — ~300–350 GPU-h | ~2½–3 days *(spent)* |
| Pilot agentic search (12–18 cells) | ⬜ not started | **~7–10 days** (re-priced) |
| Aggregate + figures | ⬜ in progress | hours |
| **Remaining to preprint** | | **~7–10 days** |

> ⚠ **The pilot's "~1–2 days" was derived from the old ≈40 GPU-h base.** At ~235
> fold-trainings/cell (§3.3) and the measured ~27.5 min/fold classification mean,
> **12–18 cells is ≈1,290–1,940 GPU-h ≈ 13–20 days on 4×H100** — not 1–2 days. To
> land a pilot in ~7–10 days either the cell count drops to ~6–9, the per-cell
> search protocol shrinks (fewer discovery candidates or inner folds — which §3.3's
> callout already flags as necessary to fit the 6-hour cap), or the pilot is
> restricted to the cheap aggregators (`abmil` at 1.9 min/fold makes a 12-cell pilot
> ~90 GPU-h, which *is* ~1 day — but an `abmil`-only pilot cannot support a
> cross-aggregator C2). **This is now the binding scope decision for the preprint.**

Full-audit (60-cell) variant → ≈4,900–6,500 GPU-h, i.e. **months** on 4×H100, not
"+5–7 days". TITAN code is already merged (`pipeline/titan/` on `main`), so it is
**not** a dev-time pole anymore.

---

## 4. Figure plan

Eight figures map onto the paper's claims. Five are drafted as **clearly-labelled
mock-data examples** (regenerate with
[`figures/make_mock_figures.py`](figures/make_mock_figures.py)); three are
table/diagram deliverables that don't need mocking.

Separately, **Tables 1 and 2 (§1.1 and §2.1) are rendered from REAL data** by
[`figures/make_dataset_table.py`](figures/make_dataset_table.py). Both scripts
currently write into [`figures/mock/`](figures/mock/), so that directory holds
seven PNGs: five mocks **plus two real tables**. Mind the distinction — the
"MOCK DATA" warning at the end of this section applies **only** to the five
`fig*.png` files, not to `table1_dataset_stats.png` / `table2_grid_breakdown.png`.

| # | Figure | Claim it serves | Source data | Status |
|---|---|---|---|---|
| **1** | Classification leaderboard heatmap (5 ds × 4 agg×3 enc + TITAN) | Published/native-default starting comparison; final status awaits the corrected baseline | static grid `results.tsv` | **mock drafted** |
| **2** | Encoder-vs-aggregator variance (decomposition + per-dataset spread) | Descriptive variance decomposition; no directional conclusion is frozen before the completed analysis | mixed-effects on grid | **mock drafted** |
| **3** | autoMIL recipe-search effect (cross-lineage ranks + per-cell lift) | **Main C2 empirical result; validates C1:** measure ranking change or stability under equal effort (RQ1–2) | corrected agentic campaign | **mock drafted** |
| **4** | Survival OS c-index (5 ds × 5 arms) | Measure the second task axis; label TITAN as a distinct slide-level regime | survival grid | **mock drafted** |
| **5** | autoMIL search trajectory (one cell, val-only) | *How* the agent explores; test-quarantine discipline | `graph.json` of a real run | **mock drafted** |
| 6 | Competitive/coverage table (vs PathBench-MIL / Patho-Bench / EVA) | Positioning | PLAN.md §3 (already filled) | table — reuse PLAN.md |
| 7 | Protocol-parity panel (our honest test vs GOLDMARK) | "reproduces published SOTA" | `goldmark_exact/COMPARISON.csv` | needs cluster pull |
| 8 | Pipeline schematic (TRIDENT→features→{MIL, TITAN}→result.json + worktree search loop) | System overview | architecture | diagram — **intended to be Figure 1 of the paper** |

> **Numbering.** The `#` column above is a **planning ID, not the paper's figure
> number** — they will be renumbered at write-up (the schematic in row 8 is meant
> to open the paper). Likewise "Table 1" is used for two different things in this
> doc: §1.1's dataset-statistics table (the intended paper Table 1) and the row-1
> leaderboard heatmap, described below as "Table-1-as-a-figure" because it renders
> the results matrix. They are separate artifacts.

### The five drafted mocks

- **`fig1_leaderboard_heatmap.png`** — the core results table as a heatmap; rows =
  dataset/gene, columns grouped by aggregator then encoder, TITAN as its own
  column. This is Table-1-as-a-figure.
- **`fig2_encoder_vs_aggregator_variance.png`** — a historical mock for the
  dropped *encoder ≫ aggregator* claim. It may survive only as a descriptive
  variance decomposition with no directional headline.
- **`fig3_recipe_search_effect.png`** — the planned main C2 empirical figure:
  (A) compare default and equal-effort searched cross-lineage ranks without
  presupposing a flip; (B) per-cell lift. Replace the off-roster, pre-fix
  CCRCC/ovarian anchors with corrected roster evidence.
- **`fig4_survival_cindex.png`** — grouped bars, OS c-index per dataset × arm,
  death counts annotated, random-0.5 reference line; TITAN visually separated
  as a slide-level regime, with no winner presupposed.
- **`fig5_search_trajectory.png`** — candidate composites + running-best staircase
  over the UCB experiment tree for one cell, validation-only, with the frozen→test
  hand-off annotated (the anti-test-leakage story).

> ✅ **Fig 1 and Fig 4 are now built from real data** (2026-07-30) — see
> `figures/real/`, produced by `make_figures.py` from the 130 baseline roster
> cells. Reproduce with the two commands in §4.1. The versions in `figures/mock/`
> are superseded for these two.
>
> ⚠ **Figs 2, 3 and 5 remain fabricated for layout only**, loosely anchored to the
> May-baseline AUC ranges; each carries a red "MOCK DATA" tag and a "(MOCK)"
> title. Fig 3 and Fig 5 need the agentic-search layer, which has not run. Fig 2
> is a historical mock for the dropped *encoder ≫ aggregator* claim — and §5.3 of
> [`GRID-CENSUS-2026-07-30.md`](GRID-CENSUS-2026-07-30.md) finds the real data does
> **not** establish that ordering, so it must not be revived as a directional
> claim. **This warning does not cover `table1_dataset_stats.png` and
> `table2_grid_breakdown.png`** — those are real cohort counts and the verified
> grid, produced by `make_dataset_table.py`, and carry no MOCK tag.

### 4.1 Reproducing the real figures

The results trees live on `fir` under
`/home/yinshuol/projects/rrg-jma/shared/Pathology/autoMIL/phase2/<cohort>/benchmark_5fold`.
Collect, then plot. **Pass `LABEL=PATH` roots** — real `summary.json` files carry
no `dataset` field, so without labels every cohort collapses into one blank row:

```bash
python benchmarks/scripts/collect_results.py \
  --roots tcga_luad=<phase2>/tcga_luad/benchmark_5fold \
          tcga_lgg=<phase2>/tcga_lgg/benchmark_5fold \
          cptac_gbm=<phase2>/cptac_gbm/benchmark_5fold \
          cptac_pdac=<phase2>/cptac_pdac/benchmark_5fold \
          tcga_hnsc=<phase2>/tcga_hnsc/benchmark_5fold \
  --out /tmp/results.csv --per-fold-out /tmp/per_fold.csv
```

```bash
python paper/preprint/figures/make_figures.py \
  --results /tmp/results.csv --per-fold /tmp/per_fold.csv \
  --out-dir paper/preprint/figures/real
```

`make_figures.py` applies the baseline roster filter
([`figures/roster.py`](figures/roster.py)) by default: 130 of the 195 collected
experiments are kept, and it **fails loudly** rather than plotting if any cohort is
short of its 26 cells. It prints exactly what it dropped (35 `cox` + 30 off-roster
LUAD). `--no-roster-filter` and `--allow-incomplete-roster` exist as deliberate
escape hatches; neither should be used for paper figures.

---

## 5. Open decisions / gaps (that affect the grid)

1. **Agentic-search scope (§3.3).** Balanced matched pilot (12–18 roster cells)
   vs full classification audit (60 cells). Static-only is no longer in scope.
   Dominates compute; needed to validate C1 and establish C2. **Decide first.**
2. **Val-based selection — DONE; enforcement is the remaining gap.** The
   keep/discard composite is now **validation-based** as of `bf9a2d6`
   ([`clam/runner.py:58,68`](../../benchmarks/src/autobench/pipeline/clam/runner.py):
   `val_c_index` and `(val_auc+val_bacc)/2`; test is sealed into the `held_out`
   block at `:57,64-66`), computed by the single shared writer that the nnmil/abmil/titan/smmile/
   dtfd runners import. The orchestrator quarantines that test block **in code**
   (born-seal → strip → `certify.json`). What is *not* yet enforced: it **trusts the
   `composite` scalar verbatim** ([`terminal_writer.py:205`](../../src/automil/terminal_writer.py)),
   and the substrate freeze rests only on the **soft** `files.readonly` list —
   `registry.protected` ships empty ([`registry/config.py:37`](../../src/automil/registry/config.py)),
   so an agent could still overlay `splits.py` / `run_experiment.py` or fold test
   into `composite`. Before the agentic loop: (a) populate `registry.protected`
   (frozen-substrate list — [`PLAN.md`](PLAN.md) §5), and (b) recompute the composite
   orchestrator-side from the val `metrics` block. Static grid is unaffected; the
   agentic layer is blocked on (a)+(b).
3. **Feature preflight (§3.2).** Confirm `conch_v15` (TITAN) + the 3 patch
   encoders exist for all 5 cohorts, especially the three new members
   (CPTAC-GBM, CPTAC-PDAC, TCGA-HNSC).
4. **Pipeline single-source-of-truth.** Tag one commit (`preprint-pipeline-v1`) so
   every submit runs the same code; survival + roster + TITAN are on `main` now,
   but goldmark-parity is on `origin` (`d42f0b4`) and still unmerged.
5. **Runtime instrumentation.** Record per-fold elapsed during the campaign — the
   paper wants an honest runtime-per-cohort figure and history can't supply it.
6. **Per-cell budget conflict (§3.3).** The proposal's search protocol costs
   ~11.7 GPU-h/cell against autoMIL's framework-enforced **6 h/cell cap**, on two
   different definitions of "cell." Shrink the protocol or raise the cap —
   blocking for the agentic layer.
7. **Static-grid estimate reconciliation (§3.1).** ≈40 GPU-h (per-head model) vs
   ~55 GPU-h (per-head model scaled to the measured EXECUTION_PLAN §4 anchor).
   Cheap to settle once the first cohort's instrumented timings land.

---

## 6. Provenance note — why "33", not "44" or "21"

Two internal memories disagreed on grid size; both were stale, so this file was
built by executing the generator. **44/dataset** was the old multi-model roster
(CLAM ran `clam_sb`+`clam_mb`+`mil`, nnMIL ran `trans_mil`+`simple_mil`).
**21/dataset** ("7 arms × 3 enc + TITAN") counted classification-style and missed
the survival loss fan-out (cox+nllsurv for nnMIL/ABMIL/TITAN). The committed
config produces **33** — verified by
`generate_all_experiments` over all five YAMLs. Minor hygiene bug spotted in
passing: the header comment in `submit_benchmark.sh` still lists the old
`clam_sb,mil,trans_mil` roster — misleading, though the code reads the pinned
single-model YAML lists correctly, so the actual grid is unaffected.
