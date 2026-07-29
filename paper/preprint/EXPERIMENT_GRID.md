# Preprint — Experiment Grid & Estimation

> ## Cluster state — verified 2026-07-29 on `fir`
>
> _This block replaces an earlier version that was **wrong**. It was written from
> `<cohort>/benchmark/results`, which holds the May–June 15-cohort mutation
> sweep. The preprint campaign writes to
> `.../Pathology/autoMIL/phase2/<dataset>/benchmark_5fold/`, one directory per
> lab member. Scanning the wrong tree produced a "campaign never launched"
> conclusion that is the opposite of the truth._
>
> **The 5-fold grid is complete.** 195 experiments on disk; all 165 roster cells
> present, verified cell-by-cell against §2.1:
>
> | Cohort | roster task | classification | survival | owner |
> |---|---|--:|--:|---|
> | tcga_luad | `kras` | 13/13 | 20/20 | yinshuol |
> | tcga_lgg | `idh1` | 13/13 | 20/20 | yws0322 |
> | cptac_gbm | `tp53` | 13/13 | 20/20 | kcuoft |
> | cptac_pdac | `immune_class` | 13/13 | 20/20 | atatc |
> | tcga_hnsc | `grade` | 13/13 | 20/20 | ryanwk |
>
> All five arms ran (clam · nnmil · abmil · dtfd · titan); all 100 survival
> experiments exist. TCGA-LUAD carries 30 extra experiments beyond the roster
> (an `egfr` task, plus `clam_sb`/`mil`/`trans_mil`) and a separate 10-fold tree.
> Newest result 2026-07-26. §3.2's "conch_v15 not confirmed extracted on any
> cohort" is also stale — TITAN ran on all five.
>
> **Validation of those 195 (2026-07-29):**
>
> - **Fold integrity: clean.** No fold-count anomaly; **zero non-finite values in
>   any primary metric** (`auc_roc` / `c_index` / `balanced_accuracy`) across
>   8,440 per-fold records. The 880 NaNs that do exist are all `sensitivity` /
>   `specificity` — undefined for the two 3-class tasks by construction, plus an
>   nnMIL-only asymmetry on binary tasks (the L-10 family). No composite is
>   affected.
> - **No cache collision.** 195 experiments produced 195 distinct per-fold
>   signatures. The CR-5 / CR-5b shared-cache hazard did **not** materialise here.
> - **Survival postdates the censoring fix.** Every `os` result is ≥ 2026-07-22
>   01:55; the fix landed 2026-07-21 ~21:20. Splits are 5-fold everywhere, so the
>   10-fold cache-reuse hazard did not fire either.
> - **⚠ `config.json` misreports the hyperparameters for 3 of 5 arms.** Every
>   config records `lr=2e-4, weight_decay=1e-5, max_epochs=200` — the shared
>   `TrainConfig`. Only CLAM and ABMIL actually used those. DTFD trained at its
>   own 1e-4/1e-4, nnMIL at its plan's 3e-4 (cls) / 1e-4 (surv) with 100 epochs,
>   TITAN at 1e-3/1e-4. That is finding H-3 made concrete: **102 of 195 configs
>   describe a recipe that did not run.** Do not build a methods table from
>   `config.json` for these results; use `pipeline/provenance.py`.
>
> **Re-run cost is real, not zero.** `fix/audit-2026-07-23` returns CLAM to
> upstream `lr=1e-4` and ABMIL to upstream `5e-4 / 1e-4 / 20 epochs`, so **93 of
> the 195** (45 CLAM + 48 ABMIL) would produce different numbers. DTFD (33),
> nnMIL (54) and TITAN (15) are untouched by those changes and should reproduce.

## 0. TL;DR

| Quantity | Value |
|---|---|
| Datasets | **5** — 3 TCGA + 2 CPTAC |
| Task axes | **2** — classification (binary mutation · 3-class subtype · 3-class grade) + survival (OS) |
| MIL aggregators | **4** — `clam_mb`, `simple_mil`, `abmil`, `dtfd_mil` |
| Patch encoders | **3** — `uni_v2`, `virchow2`, `hoptimus1` |
| Slide-encoder arm | **TITAN** (its own encoder + aggregator, linear probe) |
| **Experiments / dataset** | **33** (30 tile-encoder + 3 TITAN) — but **30 today**: the 3 TITAN arms cannot run until `conch_v15` is extracted (§3.2) |
| **Total experiments** | **165** |
| **Total fold-trainings** (×5-fold) | **825** |
| Static-grid compute | **≈ 40 GPU-hours ≈ ½–1 day on 4×H100** (possibly ~55 GPU-h — see the §3.1 caveat) |
| Real long pole | **`conch_v15` feature extraction for TITAN** (~1 day wall) |
| **Not yet budgeted** | **the agentic recipe-search layer** — the paper's headline; ~15–20× the static grid at full scope (§3.3) |

The static grid is cheap. The paper's actual contribution — the equal-effort
**agentic recipe search on top of each cell** — is the expensive part and is not
yet scoped in any preprint doc. That is the one decision that changes the compute
picture by an order of magnitude.

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
`generate_all_experiments` over all five YAMLs: **33 experiments each,
regardless of `n_classes`**, so the totals below are unchanged from the earlier
binary-only roster.

### 2.1 How the grid expands — per dataset

The expansion is **not** a clean 4×3×2; survival loss-eligibility differs per
framework (`generate_all_experiments`, [`pipeline/config.py:296`](../../benchmarks/src/autobench/pipeline/config.py)).
Rendered version: [`figures/mock/table2_grid_breakdown.png`](figures/mock/table2_grid_breakdown.png).

| Framework · model | Classification (1 gene × 3 enc) | Survival OS (× 3 enc) | Row total |
|---|--:|--:|--:|
| CLAM · `clam_mb` | 3 | **nllsurv only** → 3 | 6 |
| nnMIL · `simple_mil` | 3 | **cox + nllsurv** → 6 | 9 |
| ABMIL · `abmil` | 3 | **cox + nllsurv** → 6 | 9 |
| DTFD · `dtfd_mil` | 3 | **nllsurv only** → 3 | 6 |
| **TITAN** (1 pseudo-encoder) | 1 | **cox + nllsurv** → 2 | 3 |
| **Per-dataset total** | **13** | **20** | **33** |

Why the survival column is uneven (all enforced in code, not by hand):
- **`clam_mb` — nllsurv only.** `cox` needs a single-risk output that only
  `clam_sb` exposes; `clam_mb` is multi-branch → cox is skipped (`config.py:374`).
- **`dtfd_mil` — nllsurv only.** Cox's partial-likelihood needs a cross-patient
  risk set that doesn't exist within one slide's pseudo-bags (`config.py:397`).
- **`simple_mil` / `abmil` / TITAN — both losses.** Attention/linear heads take
  either loss (arbitrary output width), so cox + nllsurv both run.

### 2.2 Totals

| Scope | Experiments | Fold-trainings (×5) |
|---|--:|--:|
| Per dataset | 33 | 165 |
| Classification only (5 ds) | 65 | 325 |
| Survival only (5 ds) | 100 | 500 |
| **Campaign (5 datasets)** | **165** | **825** |

> **Configured vs. runnable today.** The 165/825 figures are the *configured*
> grid — what `generate_all_experiments` emits from the committed YAMLs. Until
> `conch_v15` features exist (§3.2), the TITAN arm has no inputs, so a launch
> today yields **30 exps/dataset = 150 experiments / 750 fold-trainings** with an
> empty TITAN row. TITAN is therefore a **launch gate for the encoder axis**, not
> an optional extra: without it the encoder axis is three near-identical modern
> ViT foundation models (see PRELAUNCH_REVIEW §3, items O1 and O3).

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

Anchored to EXECUTION_PLAN §4 (120 CLAM-heavy fold-trainings = 10.5 h single-GPU)
and the fact that nnMIL/ABMIL/DTFD/TITAN heads are all lighter than CLAM.
Order-of-magnitude per-head single-GPU cost (to be replaced by the campaign's own
instrumented timings):

| Head | Fold-trainings (5 ds × 5-fold) | ~min / fold-training | GPU-hours |
|---|--:|--:|--:|
| `clam_mb` (instance loss, heaviest) | 150 | ~5 | ~12.5 |
| `simple_mil` | 225 | ~2.5 | ~9.4 |
| `abmil` | 225 | ~2.5 | ~9.4 |
| `dtfd_mil` (two-tier pseudo-bag) | 150 | ~3 | ~7.5 |
| TITAN (linear probe, 1 vec/slide) | 75 | ~0.5 | ~0.6 |
| **Total** | **825** | — | **≈ 40 GPU-h** |

On 4×H100 with the best-fit orchestrator → **≈ 10–13 h wall-clock** for the whole
5-dataset static grid (or ~2–3 h each if the five datasets run as parallel jobs).
Consistent with EXECUTION_PLAN's "~1 day in one self-resubmitting job."

> ⚠ **This may be ~1.4× optimistic against its own anchor — unreconciled.** The
> EXECUTION_PLAN §4 reference point (120 runs / 10.5 h) is an **equal-count
> `clam_mb`/`simple_mil`** mix, i.e. **~5.25 min/fold-training measured**. The
> per-head rates above predict **3.75 min** for that same equal-count mix — so
> they run ~1.4× fast. Scaling every head by that factor gives **~55 GPU-h ≈
> ~14 h on 4×H100**, not ≈40 GPU-h ≈ 10–13 h. (Pricing all 825 fold-trainings at
> the flat 5.25 min gives ~72 GPU-h, but that overstates — it charges TITAN's
> linear probe at CLAM's rate.) Both are placeholders until the campaign's
> instrumented timings land (§5, item 5) — but plan against the pessimistic end.
> This propagates to the §0 TL;DR and §3.4.

### 3.2 Feature extraction — the real long pole

Training is cheap; **features gate it.**

- **TITAN dependency — `conch_v15` @ 20×/512px**, all 5 cohorts. Not confirmed
  extracted on any TCGA cohort as of the 07-03 audit. ~24 GPU-h/cohort,
  parallelizable → **~1 day wall** across 5 GPUs. **This is the bottleneck.**
- **Patch features** (`uni_v2`/`virchow2`/`hoptimus1`): likely present for
  TCGA-LUAD/LGG; **CPTAC-GBM, CPTAC-PDAC, and TCGA-HNSC are new to the roster** —
  confirm/extract per cohort (~24 GPU-h each). CPTAC cohorts extract via the
  Patho-Bench/TRIDENT path; TCGA-HNSC additionally needs the GDC grade column
  joined into its manifest (same clinical-join pattern as `add_os_to_manifest.py`).

**Preflight (do first):** on the cluster, verify per-cohort feature presence
before launching training, or the grid stalls on missing inputs.

### 3.3 The agentic recipe-search layer — headline, and NOT yet budgeted

The static grid above is the **"default recipe" leaderboard** — the *before*
numbers. autoMIL's actual claim (BACKGROUND.md: "the auto pipeline is the main
contribution") is that **every cell gets an equal-effort agentic recipe search**,
producing the *corrected* leaderboard. That layer multiplies compute massively and
**appears in no preprint doc's budget yet.**

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

- **One dataset, classification only** (3 enc × 4 agg = 12 cells): ~2,800 fold-trainings — already **~3.4× the entire static 5-dataset grid**.
- **All tile-encoder classification cells** (5 ds × 3 enc × 4 agg = 60 cells): ~14,000 fold-trainings ≈ **~700 GPU-h ≈ ~7 days on 4×H100** — journal-scale.

> Two exclusions to state plainly, since "all classification cells" reads broader
> than it is: this 60-cell count **omits the 5 TITAN classification arms** (1 per
> dataset — TITAN has no encoder×aggregator fan-out, so classification is 65
> experiments but only 60 tile-encoder cells), and it **omits all 100 survival
> experiments** entirely. A search over survival cells too would be a further
> multiple on top.

**Implication for the preprint:** a paper about an agentic framework needs *some*
agentic result, but the full 60-cell search is out of scope for "ship fast." The
realistic preprint options are **(a)** a pilot-scale agentic demo (12–18 cells, ~1–2
days on 4×H100) plus the existing CCRCC/ovarian-HRD feasibility anchors, or **(b)**
static grid + feasibility anchors only, deferring the full audit to Phase 2. **This
is the single biggest open scoping decision** and it dominates the compute plan.

### 3.4 Rough end-to-end (preprint = static grid + pilot agentic)

| Component | Wall-clock (4×H100) |
|---|---|
| `conch_v15` extraction (5 cohorts, parallel) | ~1 day |
| Static grid (825 fold-trainings) | ~½–1 day |
| Pilot agentic search (12–18 cells) | ~1–2 days |
| Aggregate + figures | hours |
| **Total** | **~3–4 days** (extraction + pilot dominate) |

Full-audit (60-cell) variant → add ~5–7 days. TITAN code is already merged
(`pipeline/titan/` on `main`), so it is **not** a dev-time pole anymore.

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
| **1** | Classification leaderboard heatmap (5 ds × 4 agg×3 enc + TITAN) | The corrected benchmark exists & is complete | static grid `results.tsv` | **mock drafted** |
| **2** | Encoder-vs-aggregator variance (decomposition + per-dataset spread) | **Headline:** encoder ≫ aggregator | mixed-effects on grid | **mock drafted** |
| **3** | autoMIL recipe-search effect (ranking bump + per-cell lift) | **Framework contribution:** equal-effort search flips rankings / lifts composite (RQ1–2) | agentic layer + CCRCC/HRD anchors | **mock drafted** |
| **4** | Survival OS c-index (5 ds × 5 arms) | Second task axis works; TITAN wins slide-level | survival grid | **mock drafted** |
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
- **`fig2_encoder_vs_aggregator_variance.png`** — the paper's headline in two
  panels: (A) variance-component bar showing encoder % ≫ aggregator %, (B)
  per-dataset AUC spread when you swap encoder vs swap aggregator. **Note the
  de-biasing caveat:** this claim is only fair *after* the agentic recipe search
  equalises recipe effort — otherwise a reviewer says the aggregator gap is just
  under-tuning. Fig 2 + Fig 3 must be read together.
- **`fig3_recipe_search_effect.png`** — the framework's money figure: (A) bump
  chart, aggregator ranking flips from default → equal-effort searched recipe; (B)
  per-cell dumbbell lift, with **CCRCC (0.744→0.807)** and **ovarian HRD
  (0.814→0.851)** as the *real* feasibility anchors already on record.
- **`fig4_survival_cindex.png`** — grouped bars, OS c-index per dataset × arm,
  death counts annotated, random-0.5 reference line; TITAN drawn as the
  slide-level winner (matches the Frontiers precedent).
- **`fig5_search_trajectory.png`** — candidate composites + running-best staircase
  over the UCB experiment tree for one cell, validation-only, with the frozen→test
  hand-off annotated (the anti-test-leakage story).

> ⚠ Every number in **the five `fig*.png` mocks** is fabricated for layout only,
> loosely anchored to the May-baseline AUC ranges. Each carries a red "MOCK DATA"
> tag and a "(MOCK)" title. Swap in `results.tsv` / `graph.json` once the campaign
> runs. **This warning does not cover `table1_dataset_stats.png` and
> `table2_grid_breakdown.png`** — those are real cohort counts and the verified
> grid, produced by `make_dataset_table.py`, and carry no MOCK tag.

---

## 5. Open decisions / gaps (that affect the grid)

1. **Agentic-search scope (§3.3).** Pilot (12–18 cells) vs full audit (60 cells)
   vs none. Dominates compute; needed for the paper's headline. **Decide first.**
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
