# Phase-2 Grid Census — verified 2026-07-30 on `fir`

**Source of truth:** `/home/yinshuol/projects/rrg-jma/shared/Pathology/autoMIL/phase2/<cohort>/benchmark_5fold/results/`
**Method:** every `summary.json` under every cohort tree was parsed (237 files, 0
unreadable), cross-checked against on-disk `fold_*/` directories and 975
fold-level `metrics.json`. Raw census: [`data/grid_census_2026-07-30.tsv`](data/grid_census_2026-07-30.tsv).

**Directory grammar:** `results/<framework>/<strategy>/<task>/<encoder>/<model>/[<surv_loss>/]fold_<k>/`.
The `<surv_loss>` level exists **only** for `task=os`.

---

## 0. The headline correction

The `20/20` survival column in EXPERIMENT_GRID.md's cluster-state block is
**counting two survival losses**. Survival splits on a `cox` / `nllsurv` axis,
and the 20 per cohort is `13 nllsurv + 7 cox`.

**If only `nllsurv` is required — which is the stated plan — the survival roster
is 13 cells per cohort, and it is complete.**

| Cohort | roster task | classification | survival (`nllsurv`) | `cox` extras | owner |
|---|---|--:|--:|--:|---|
| tcga_luad | `kras` | **13/13** | **13/13** | +7 | yinshuol |
| tcga_lgg | `idh1` | **13/13** | **13/13** | +7 | yws0322 |
| cptac_gbm | `tp53` | **13/13** | **13/13** | +7 | kcuoft |
| cptac_pdac | `immune_class` | **13/13** | **13/13** | +7 | atatc |
| tcga_hnsc | `grade` | **13/13** | **13/13** | +7 | ryanwk |

13 cells = 4 aggregators (`clam_mb`, `simple_mil`, `abmil`, `dtfd_mil`) × 3 tile
encoders (`uni_v2`, `virchow2`, `hoptimus1`) + 1 TITAN slide-encoder arm.

### Restated grid totals

| Quantity | EXPERIMENT_GRID.md §0 | Verified (nllsurv-only roster) |
|---|--:|--:|
| Experiments / cohort | 33 | **26** (13 cls + 13 surv) |
| Total roster experiments | 165 | **130** |
| Total fold-trainings | 825 | **650** |

The old 33/cohort decomposes exactly as `13 cls + 13 nllsurv + 7 cox`, and its
"30 tile-encoder + 3 TITAN" split is arithmetically correct — it just silently
included the partial `cox` axis. **Every roster cell is present. Nothing is
missing. The grid is complete and over-complete.**

---

## 1. What is actually on disk — full accounting

**237 experiments** across all trees. All 237 reconcile:

| Bucket | n | GPU-h | Keep? |
|---|--:|--:|---|
| **Roster classification** (5 cohorts × 13) | **65** | 137.5 | ✅ core |
| **Roster survival, `nllsurv`** (5 × 13) | **65** | 123.5 ⚠ *15 cells untimed* | ✅ core |
| `cox` survival extras (5 × 7) | 35 | 96.0 | ❌ drop (§2) |
| LUAD off-roster classification | 30 | 123.9 | ❌ drop (§3) |
| LUAD `benchmark_10fold` tree | 42 | 339.9 | ❌ drop (§3) |
| **Total** | **237** | **820.7** ⚠ | |

`65 + 65 + 35 + 30 = 195` = the five `benchmark_5fold` trees; `+42` 10-fold = 237.

> ### ⚠ The `clam_mb` survival arm has no timing at all
> All **15** `clam_mb` × `nllsurv` experiments (**75 fold-trainings**) record
> neither `elapsed_seconds_total` in `summary.json` nor `elapsed_seconds` in any of
> their 75 fold `metrics.json`. They are the **only** affected group — for all
> other 180 experiments the summary field and the sum of fold fields agree to the
> decimal (480.9 GPU-h both ways), so the instrumentation is otherwise sound.
> Their **metrics are complete and valid**; only the timing is missing.
>
> Every GPU-hour figure above is therefore a **lower bound**. Pricing the 75
> missing fold-trainings at `clam_mb`'s own measured classification rate
> (68.6 min/fold) adds **≈ 86 GPU-h**; at the cheapest observed survival rate
> (DTFD, 29.8 min/fold) it adds ≈ 37 GPU-h.

**Roster compute: ≥ 261.0 GPU-h measured over 115 of 130 cells → ~300–350 GPU-h
all-in** (best single estimate **≈ 347 GPU-h**, using `clam_mb`'s own rate).

> ### ⚠ The §0 compute estimate is low by ~7.5–8.75×
> EXPERIMENT_GRID.md §0 budgets the static grid at **"≈ 40 GPU-hours"** (with a
> "possibly ~55" caveat). The roster as actually run cost **~300–350 GPU-h**. This
> matters downstream: §3.3 scopes the agentic recipe-search layer at **15–20× the
> static grid**. Against a ~325 GPU-h base that is **≈ 4,900–6,500 GPU-h**, not
> 600–1,100. Re-scope §3.1/§3.3 before committing to the agentic campaign.

### 1.1 Per-head measured cost (roster cells only)

| Head | Roster exps | Fold-trainings | GPU-h | min / fold-training |
|---|--:|--:|--:|--:|
| `clam_mb` — classification | 15 | 75 | 85.7 | 68.6 |
| `clam_mb` — survival | 15 | 75 | **untimed** | — |
| `simple_mil` | 30 | 150 | 95.7 | 38.3 |
| `abmil` | 30 | 150 | 36.5 | 14.6 |
| `dtfd_mil` | 30 | 150 | 42.6 | 17.0 |
| `titan` (linear probe) | 10 | 50 | 0.4 | 0.5 |
| **Total instrumented** | **115** | **575** | **261.0** | **27.2** |

The §3.1 estimate table predicted 2.5–5 min/fold-training for the tile-encoder
heads. Measured is **14.6–68.6** — the per-head rates in that table are wrong by
roughly an order of magnitude, not by the 1.4× its own caveat anticipated. `abmil`
is the cheapest real aggregator and `clam_mb` classification the most expensive at
68.6 min/fold. TITAN's linear probe is the one head the estimate got right.

---

## 2. The `cox` axis — drop it, and it was never usable anyway

Dropping `cox` is not merely a scope choice; **the `cox` arm is structurally
incomplete and could not have been reported as-is.**

| Framework | `nllsurv` / cohort | `cox` / cohort |
|---|--:|--:|
| `clam` (`clam_mb`) | 3 | **0** |
| `nnmil` (`simple_mil`) | 3 | 3 |
| `abmil` | 3 | 3 |
| `dtfd` (`dtfd_mil`) | 3 | **0** |
| `titan` | 1 | 1 |
| **total** | **13** | **7** |

`clam_mb` and `dtfd_mil` have **zero** `cox` runs on any cohort. A `cox`-vs-`nllsurv`
comparison would need **30 further experiments** (2 models × 3 encoders × 5
cohorts) to close. The pattern is identical across all five cohorts, so this is a
launch-configuration decision, not sporadic failure.

**Where the two losses do overlap (35 paired cells), `nllsurv` is the better
choice on the evidence:**

- `nllsurv` wins **27 / 35** paired cells
- mean Δ c-index = **+0.0147** (`nllsurv` − `cox`), range −0.041 … +0.130
- largest gaps are all small-sample CPTAC-GBM cells where `cox` degrades badly
  (`simple_mil`/`hoptimus1`: 0.388 `cox` vs 0.518 `nllsurv`)

So `nllsurv`-only is defensible on its own merits and needs no apology in the
paper. If a reviewer asks why not Cox, the 35-cell paired comparison is a real
answer — worth keeping as a supplementary note rather than deleting.

---

## 3. Off-roster material (LUAD only)

TCGA-LUAD carries 72 experiments beyond its 26 roster cells, all from earlier
exploratory work by the same owner:

- **`egfr` task, 5-fold (21):** a second mutation task — 7 models × 3 encoders.
  Not in the §1 roster (LUAD is pinned to `kras`).
- **Extra aggregators on `kras`, 5-fold (9):** `clam_sb`, `mil`, `trans_mil` × 3
  encoders. Beyond the 4-model roster.
- **`benchmark_10fold` tree (42):** `kras` 21 + `egfr` 21, `n_folds=10`, 7 models ×
  3 encoders. **No survival at all** in this tree.

These are harmless but must be **excluded by explicit filter**, not by directory
sweep — a naive `find` over LUAD returns 105 experiments where the roster is 26.
The 10-fold tree is the biggest trap: it is 340 GPU-h and 42 experiments that
would silently inflate any LUAD aggregate.

---

## 4. Validation of the 130 roster experiments

Everything below was recomputed from disk, not carried over from the 2026-07-29 block.

✅ **Fold integrity: perfect.** 975 fold directories across the five 5-fold trees
= 195 × 5 exactly. `summary.json`'s `n_folds` matches the on-disk `fold_*/` count
for **every** experiment; zero mismatches. Every 5-fold-tree experiment has
`n_folds=5`, every 10-fold-tree experiment `n_folds=10`.

✅ **Zero non-finite values in any primary metric.** `auc_roc`, `c_index` and
`balanced_accuracy` are finite in all 975 fold files and all 237 summaries. No
missing or NaN aggregate means.

✅ **No cache collision.** 237 experiments produced **237 distinct** per-fold
metric signatures. The CR-5 / CR-5b shared-cache hazard did not materialise.

✅ **Survival postdates the censoring fix.** Oldest `os` result is
**2026-07-22 16:55**; the fix landed 2026-07-21 ~21:20. All survival is 5-fold, so
the 10-fold cache-reuse hazard did not fire (the 10-fold tree contains no survival).

✅ **Single seed, single strategy, as designed.** All 237: `seed=42`, `strategy=standard`.
Multi-seed is therefore **not** available — every mean±std in §5 is fold variance
at one seed, not seed variance. Any "variance" claim in the paper must say so.

✅ **Timestamps.** Oldest 2026-07-10 05:18, newest **2026-07-27 09:30** (the
2026-07-29 block's "newest 07-26" was marginally stale).

⚠ **Timing provenance gap:** the `clam_mb` survival arm (15 cells / 75
fold-trainings) carries no `elapsed_seconds` anywhere — see the §1 callout. Metrics
are unaffected; only compute accounting is. Worth fixing in the writer before any
re-run, since equal-effort budgeting (the C1 claim) depends on trustworthy timings.

### 4.1 Two metrics are not cross-arm comparable

⚠ **`sensitivity` / `specificity`: 880 non-finite values** (220 each × {test,val} ×
{sens,spec}). Undefined by construction for the two 3-class tasks, plus an
`nnmil`-only asymmetry on binary tasks (the L-10 family). **No composite is affected.**

⚠ **`kappa` is emitted by `nnmil` alone — newly confirmed here.** Cohen's kappa
appears in **120/120** of `nnmil`'s classification folds and **0** of
`clam` (0/150), `abmil` (0/90), `dtfd` (0/90), `titan` (0/25). It lives only in
`benchmarks/src/autobench/pipeline/nnmil/evaluate.py:55`'s metric map; the shared
`pipeline/evaluate.py` never computes it. **Consequence:** kappa cannot be used as
a cross-arm metric anywhere, which is most costly on `immune_class` and `grade`
where it is the natural 3-class agreement statistic. For those two tasks the
cross-arm comparable set is **`auc_roc`, `accuracy`, `balanced_accuracy`, `f1`** only.

✅ **Multi-class AUC is consistently defined.** Both evaluators document and use
`roc_auc_score(multi_class="ovr", average="macro")`, so the 3-class AUCs in §5 are
comparable to the binary ones and to each other across arms.

### 4.2 `config.json` still misreports hyperparameters (H-3)

Confirmed and now exact: **all 195** `config.json` files in the 5-fold trees record
the identical shared `TrainConfig` — `lr=2e-4`, `weight_decay=1e-5`, `max_epochs=200`
(`abmil` 48, `clam` 45, `dtfd` 33, `nnmil` 54, `titan` 15). Only CLAM and ABMIL
actually trained at those values. DTFD ran 1e-4/1e-4, nnMIL 3e-4 (cls) / 1e-4 (surv)
at 100 epochs, TITAN 1e-3/1e-4.

**102 of 195 configs describe a recipe that did not run.** Do **not** build the
methods table from `config.json`; use `pipeline/provenance.py`.

### 4.3 Re-run cost is real

`fix/audit-2026-07-23` returns CLAM to upstream `lr=1e-4` and ABMIL to upstream
`5e-4 / 1e-4 / 20 epochs`. **93 of the 195** (45 CLAM + 48 ABMIL) would produce
different numbers. DTFD (33), nnMIL (54) and TITAN (15) are untouched and should
reproduce bit-for-bit. Within the **roster** specifically, that is 25 CLAM + 25
ABMIL = **50 of 130** cells exposed.

---

## 5. Results — the 130 roster cells

Test metrics, mean ± std over 5 folds, seed 42. Val means are in the census TSV.
Per the val-firewall convention these test numbers are for reporting only and did
not drive any selection.

### 5.1 Classification — test AUC (macro-OvR for 3-class)

| Cohort (task) | model | `uni_v2` | `virchow2` | `hoptimus1` |
|---|---|--:|--:|--:|
| **tcga_luad** (`kras`) | `clam_mb` | 0.642±0.068 | 0.634±0.075 | 0.675±0.088 |
| | `simple_mil` | 0.633±0.076 | 0.624±0.085 | 0.656±0.092 |
| | `abmil` | 0.624±0.032 | 0.663±0.077 | 0.688±0.085 |
| | `dtfd_mil` | 0.641±0.081 | 0.629±0.048 | **0.701±0.064** |
| | `titan` | — | — | 0.614±0.069 |
| **tcga_lgg** (`idh1`) | `clam_mb` | 0.850±0.038 | 0.791±0.054 | 0.866±0.023 |
| | `simple_mil` | 0.822±0.070 | 0.814±0.039 | 0.792±0.089 |
| | `abmil` | 0.855±0.028 | 0.780±0.047 | **0.882±0.036** |
| | `dtfd_mil` | 0.856±0.039 | 0.789±0.043 | 0.850±0.012 |
| | `titan` | — | — | 0.852±0.036 |
| **cptac_gbm** (`tp53`) | `clam_mb` | 0.707±0.091 | 0.743±0.041 | 0.747±0.044 |
| | `simple_mil` | 0.635±0.127 | 0.570±0.141 | 0.606±0.144 |
| | `abmil` | 0.708±0.041 | 0.714±0.060 | 0.768±0.053 |
| | `dtfd_mil` | 0.707±0.043 | **0.799±0.029** | 0.704±0.148 |
| | `titan` | — | — | 0.790±0.076 |
| **cptac_pdac** (`immune_class`) | `clam_mb` | 0.552±0.060 | **0.581±0.064** | 0.540±0.070 |
| | `simple_mil` | 0.525±0.065 | 0.554±0.034 | 0.559±0.083 |
| | `abmil` | 0.540±0.073 | 0.561±0.052 | 0.541±0.056 |
| | `dtfd_mil` | 0.510±0.060 | 0.579±0.063 | 0.518±0.069 |
| | `titan` | — | — | 0.536±0.064 |
| **tcga_hnsc** (`grade`) | `clam_mb` | 0.695±0.090 | **0.714±0.080** | 0.674±0.082 |
| | `simple_mil` | 0.664±0.099 | 0.713±0.075 | 0.641±0.128 |
| | `abmil` | 0.696±0.096 | 0.660±0.059 | 0.673±0.075 |
| | `dtfd_mil` | 0.641±0.089 | 0.627±0.107 | 0.647±0.088 |
| | `titan` | — | — | 0.665±0.052 |

### 5.2 Survival (`nllsurv`) — test c-index

| Cohort | model | `uni_v2` | `virchow2` | `hoptimus1` |
|---|---|--:|--:|--:|
| **tcga_luad** | `clam_mb` | 0.595±0.091 | 0.609±0.086 | 0.589±0.087 |
| | `simple_mil` | 0.590±0.089 | 0.594±0.046 | 0.559±0.085 |
| | `abmil` | 0.594±0.091 | 0.603±0.089 | 0.590±0.079 |
| | `dtfd_mil` | 0.595±0.077 | **0.617±0.038** | 0.599±0.036 |
| | `titan` | — | — | 0.550±0.050 |
| **tcga_lgg** | `clam_mb` | 0.758±0.047 | 0.708±0.039 | 0.752±0.038 |
| | `simple_mil` | 0.758±0.057 | 0.737±0.044 | 0.772±0.048 |
| | `abmil` | 0.754±0.050 | 0.726±0.034 | 0.760±0.047 |
| | `dtfd_mil` | **0.769±0.030** | 0.703±0.041 | 0.763±0.038 |
| | `titan` | — | — | **0.769±0.011** |
| **cptac_gbm** | `clam_mb` | 0.548±0.100 | 0.549±0.097 | 0.561±0.085 |
| | `simple_mil` | 0.537±0.098 | 0.520±0.134 | 0.518±0.103 |
| | `abmil` | 0.558±0.108 | 0.529±0.116 | 0.556±0.092 |
| | `dtfd_mil` | 0.524±0.075 | **0.574±0.088** | 0.548±0.091 |
| | `titan` | — | — | 0.519±0.131 |
| **cptac_pdac** | `clam_mb` | 0.558±0.030 | 0.569±0.070 | 0.536±0.109 |
| | `simple_mil` | 0.553±0.054 | 0.528±0.067 | 0.528±0.106 |
| | `abmil` | 0.556±0.030 | 0.550±0.072 | 0.541±0.098 |
| | `dtfd_mil` | 0.551±0.078 | 0.557±0.095 | 0.537±0.078 |
| | `titan` | — | — | **0.577±0.100** |
| **tcga_hnsc** | `clam_mb` | 0.585±0.036 | 0.584±0.041 | 0.582±0.070 |
| | `simple_mil` | 0.548±0.065 | 0.568±0.040 | 0.594±0.068 |
| | `abmil` | 0.565±0.037 | 0.582±0.037 | 0.580±0.060 |
| | `dtfd_mil` | **0.600±0.038** | 0.557±0.061 | 0.581±0.054 |
| | `titan` | — | — | 0.592±0.084 |

### 5.3 Reading the grid

- **Task difficulty spans the intended range.** LGG `idh1` is easy (0.78–0.88 AUC,
  consistent with the 78% IDH1-mutant majority noted in §1.1), LUAD `kras` and
  HNSC `grade` are mid (0.62–0.71), and **PDAC `immune_class` is at chance**
  (0.510–0.581 AUC, best cell 0.581). A balanced-by-construction tertile split of a
  continuous infiltration score landing at chance is the expected outcome for a
  binned continuous target at n=105 — it is a real result, but it will not support
  any encoder-vs-aggregator claim, and it should be presented as the
  small-sample/noise-regime probe it was chosen to be.
- **Survival is uniformly weak except LGG** (0.70–0.77 c-index). The other four
  cohorts sit at 0.52–0.62 with std 0.03–0.13 — the ±std bands overlap chance on
  GBM and PDAC. Survival cannot carry a headline claim at one seed.
- **Encoder effects do not dominate aggregator effects in this data.** Within a
  cohort, spread across the three encoders for a fixed aggregator is comparable to
  spread across the four aggregators for a fixed encoder, and the best encoder is
  not consistent across cohorts (`hoptimus1` on LUAD/LGG, `virchow2` on GBM/HNSC/PDAC).
  **This is in tension with the "encoder ≫ aggregator" framing** that the frozen-data-
  substrate constraint was adopted to support. At one seed, with these std bands,
  the honest statement is that no ordering is established. Resolve this against
  Fig-2's intended message before drafting.
- **TITAN is competitive but not dominant** — best survival cell on LGG (0.769) and
  PDAC (0.577), best-ish classification on GBM (0.790), but worst on LUAD `kras`
  (0.614) and mid on HNSC. Its linear probe uses one encoder by construction, so it
  occupies a single column.

---

## 5.4 Baseline figures (built 2026-07-30)

Fig 1 and Fig 4 are now produced from these 130 cells — no mock data — into
`figures/real/`:

- **[`fig1_leaderboard_heatmap.png`](figures/real/fig1_leaderboard_heatmap.png)** —
  5 cohorts × 13 (aggregator, encoder) classification cells. Plots a
  **within-dataset centred ΔAUC**, not raw AUC, and prints no cross-dataset
  summary row: binary AUC and macro-OvR AUC are different quantities and must not
  share a colour scale (PRELAUNCH_REVIEW O3). Each row's own mean is in its y-tick
  label (0.55 PDAC → 0.83 LGG).
- **[`fig4_survival_cindex.png`](figures/real/fig4_survival_cindex.png)** — OS
  c-index per cohort × arm, 5 arms (all `nllsurv`), error bars from the pooled
  per-fold test values, random-0.5 reference line. Arms are drawn in fixed
  alphabetical order with one neutral colour cycle, so nothing pre-draws TITAN as
  the slide-level winner (also O3). The ±sd bands visibly cross 0.5 on CPTAC-GBM
  and CPTAC-PDAC — consistent with §5.3.

Figs 2, 3 and 5 stay mock: 3 and 5 need the agentic-search layer, and 2 is the
dropped *encoder ≫ aggregator* claim that §5.3 finds unsupported.

### Two fixes were needed to build them

1. **`collect_summaries` could not tell cohorts apart.** No framework runner writes
   a `dataset` key into `summary.json` — 0 of 195 real summaries have one — and the
   collector discarded which root each summary came from, so `aggregate_results`'
   `s.get("dataset", "")` gave all 195 the same blank cohort. Every per-cohort
   figure would have silently pooled all five. Roots may now be passed as
   `LABEL=PATH`, and the label is stamped only when the summary has no non-empty
   `dataset` of its own. The existing test suite could not catch this: its fixture
   writes `dataset` into the JSON, which the real runners never do.
2. **Nothing enforced the roster.** Added [`figures/roster.py`](figures/roster.py) —
   `make_figures.py` now filters to the 130 cells by default, reports what it
   dropped, and fails loudly rather than plotting a partial grid.

---

## 6. Actions

1. **Replace the cluster-state block** in
   [EXPERIMENT_GRID.md](paper/preprint/EXPERIMENT_GRID.md) — `20/20` → `13/13
   nllsurv (+7 cox extras)`; totals `165/825` → `130/650`.
2. **Correct §0's compute estimate** from ≈40 GPU-h to **~300–350 GPU-h**, and
   re-derive §3.3's agentic budget from the corrected base (≈4,900–6,500 GPU-h).
   Fix the missing-timing bug on the `clam_mb` survival path so the figure can be
   stated exactly rather than as a band.
3. **Pin the roster filter** in whatever loads results for figures: 5-fold trees
   only, roster task per cohort, `surv_loss == "nllsurv"`, `model ∈ {clam_mb,
   simple_mil, abmil, dtfd_mil, titan}`. Without it LUAD contributes 105 experiments
   instead of 26.
4. **Drop `kappa` from any cross-arm 3-class table** (nnmil-only), and keep
   `sensitivity`/`specificity` out of cross-arm tables for the same reason.
5. **State the single-seed limitation** wherever mean±std appears — it is fold
   variance at seed 42, not seed variance.
6. **Decide the H-3 methods-table path** — `provenance.py`, not `config.json`.
7. **Decide on the 50 exposed roster cells** (25 CLAM + 25 ABMIL) before merging
   `fix/audit-2026-07-23`: re-run them or freeze the current numbers and document
   the recipe actually used.
