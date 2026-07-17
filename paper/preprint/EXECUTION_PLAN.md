# Preprint — Execution Plan (compute campaign)

_Companion to [`PLAN.md`](PLAN.md) (strategy) and [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Compiled 2026-07-03 from a live audit of the codebase + the cluster
(`fir.alliancecan.ca`) results tree. This document is grounded in **what is
actually on disk today**, which differs from PLAN.md's cost assumptions in two
load-bearing ways — see §0._

> **Status (2026-07-17): roster resolved — see §2.** The dataset decision is
> **closed**: the roster is TCGA-LUAD/LGG/HNSC + CPTAC-GBM/PDAC. The §0 audit,
> §0b gap checklist, and §1 on-disk coverage tables are the **2026-07-03
> planning snapshot** and still name the earlier candidate set
> (THCA/COAD/SKCM/BLCA) — kept for provenance. Re-run the coverage/preflight
> checks against the resolved roster before launching. Canonical roster:
> [`PLAN.md`](PLAN.md) §1 + [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §1.

---

## 0. TL;DR — how the audit revises PLAN.md

| PLAN.md assumption | Audit finding | Impact |
|---|---|---|
| §2: `ab_mil` results already on disk → adding it is "free re-aggregation" | **`ab_mil` has essentially zero usable coverage** — one partial cohort (TGCT, 1 task, 5 folds). `_completed.json` and the fold-metric tree everywhere else contain **only `clam_mb` + `simple_mil`**. | ab_mil is **not free** — it needs training runs like dtfd_mil. The "free re-aggregation" path does not exist. |
| §2: `dtfd_mil` needs training | Confirmed — `dtfd_mil` is on **zero** cohorts, and isn't even in the 3 TCGA YAMLs' `nnmil_models`. | Correct as written. Pass it via `--nnmil_models` (CLI overrides YAML; no YAML edit strictly required). |
| §1: pick 5 datasets "by wall-clock runtime"; `sacct` will supply it | **Clean per-cohort wall-clock is not recoverable.** `metrics.json`/`_completed.json` carry no timing; `sacct` history is date-capped and dominated by goldmark LUAD jobs; log mtime spans are calendar envelopes (BRCA 239h, LUAD 554h), not run times. | Moot — since every cohort needs re-running for the 2 new heads, the **new campaign itself yields clean timing**. Select on signal × task-count now; measure real runtime from the campaign. |
| §4: ovarian HRD might be a ready continuous target for a regression arm | **No continuous HRD score exists** — `HRD_label` is a pre-binarized 0/1 manifest column; no threshold/regression logic in the code. Also, **the ovarian root `/mnt/pool/ovariancancer/...` is not on fir** (off-cluster). | Regression stays in Phase 2 (the "exception" does not hold). Ovarian is not runnable on the cluster as-is. |
| §4: TITAN is "new code path, not just config" | Confirmed, and **cheaper than feared**: TRIDENT already ships a `TitanSlideEncoder` (loads `MahmoodLab/TITAN`, 4096-d). Missing piece is ~500–650 lines of *autobench* wiring, reusing the `metrics.json`/`summary.json` contract unchanged. | TITAN is tractable. The real cost is **`conch_v15` patch-feature extraction for the TCGA cohorts** (not yet extracted), which TITAN depends on. |

**Net:** the 4-model roster is a **from-scratch training campaign for `ab_mil` + `dtfd_mil` on the chosen datasets** (not a re-report), plus a scoped TITAN build. The good news: nnMIL heads are cheap — the whole ab_mil+dtfd campaign fits in roughly **one 24-h 4×H100 job** (see §4). The longer pole is TITAN's `conch_v15` feature extraction.

---

## 0b. Gap checklist — what's missing for the preprint

`☐` = gap to close · `✅` = already done (shown for contrast) · effort/owner in parens.
Detailed "how" for each is in §3.

**A. Experiments — MIL model coverage** _(the bulk; ~1 day compute — §4)_
- ☐ **`ab_mil`** — train on all 5 chosen datasets. **0 folds on disk today** (only a partial TGCT fragment); PLAN.md's "free re-aggregation" does not hold. _(compute)_
- ☐ **`dtfd_mil`** — train on all 5. 0 folds on disk; not in the TCGA YAMLs' `nnmil_models` (pass via `--nnmil_models`, no YAML edit needed). _(compute)_
- ☐ **Re-run `clam_mb` + `simple_mil`** for any *roster* cohort lacking on-disk folds. Per the 2026-07-03 audit, LGG/LUAD/GBM/HNSC had them but **CPTAC-PDAC did not** — re-verify current cluster state for the resolved roster. _(compute)_
- ✅ `clam_mb` + `simple_mil` already on disk for **LGG, LUAD** (and BLCA/BRCA/CESC/GBM/HNSC/PAAD/PCPG/UCS).

**B. Datasets & configs**
- ✅ **Final 5 decided** (§2): TCGA-LUAD/LGG/HNSC + CPTAC-GBM/PDAC. _(closed 2026-07-17)_
- ✅ **Roster YAML configs built**: `tcga_hnsc.yaml` (grade), `cptac_gbm.yaml` (tp53), `cptac_pdac.yaml` (immune_class) created; `tcga_luad`/`tcga_lgg` already present. _(done 2026-07-17)_
- ☐ **Preflight: confirm patch features exist** for the roster, especially the 3 new cohorts CPTAC-GBM/PDAC + TCGA-HNSC (`benchmark/features/{uni_v2,virchow2,hoptimus1}/` or `trident_output/`) — extract if missing. _(check)_

**C. TITAN slide-encoder arm** _(longer pole)_
- ☐ **autobench code path** — `Framework.TITAN` + dispatch + new `pipeline/titan/` package (~500–650 lines, greenfield; reuses `metrics.json`/`summary.json`). _(1–2 dev-days)_
- ☐ **`conch_v15` patch features for the 5 TCGA cohorts** — TITAN's dependency, **not extracted** on any TCGA cohort (only on the custom datasets). ~24 h/GPU, parallelizable. _(compute)_
- ☐ **TITAN config keys** (`titan:` encoder + `4096` dim) in each chosen YAML + the training runs (cheap, slide-level). _(small + compute)_

**D. Aggregation & reporting**
- ☐ **Extend `KEEP_AND_RENAME`** in `tasks/baseline_summary/scripts/00_aggregate.py:41` to add `ab_mil`, `dtfd_mil` (+ titan once tagged); re-run `00→01→02→04`. _(small)_
- ☐ **Competitive/coverage table** vs PathBench-MIL / Patho-Bench / EVA (PLAN.md §3) + the encoder-vs-aggregator de-biasing narrative. _(writing)_

**E. Pipeline reproducibility**
- ☐ **One canonical pipeline commit + tag** (`preprint-pipeline-v1`). Work is split across `main`, `origin/feat/goldmark-parity` (orchestrator free-VRAM fix), `origin/feat/nnmil-survival` (+6 configs). Merge/cherry-pick the needed pieces. _(decision + small)_
- ☐ **Runtime instrumentation** — record per-fold elapsed so the campaign yields the honest "runtime per cohort" the paper wants (history can't supply it). _(small)_

**F. Hygiene / durability**
- ☐ **Rotate `.env` secrets** (`HF_TOKEN`, `WANDB_API_KEY`) — plaintext in `benchmarks/.env` (gitignored, not in git; verify history). _(quick)_
- ☐ Note scratch-purge exposure of custom-dataset features (not blocking a TCGA-only campaign). _(awareness)_

**Deferred — NOT preprint gaps:** regression arm → Phase 2 (no continuous target exists); ovarian dataset (root not on fir, HRD pre-binarized).

---

## 1. Ground truth — on-disk coverage (2026-07-03)

**Source-of-truth results tree:** `…/shared/Pathology/TCGA/TCGA-<X>/benchmark/` on fir
(`/home/yinshuol/projects/rrg-jma/shared/Pathology`, durable `/projects` storage).
Patch features live in `trident_output/` + per-encoder `benchmark/features/{uni_v2,virchow2,hoptimus1}/`.

**Model coverage — `metrics.json` fold count per (cohort × head), all locations incl. archives:**

| Cohort | clam_mb | simple_mil | ab_mil | dtfd_mil | trans_mil | Local YAML in `main`? |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| BLCA, BRCA, CESC, GBM, HNSC, LGG, LUAD, PAAD, PCPG, UCS | ✓ | ✓ | ✗ | ✗ | ✗ (stray logs only) | LGG/LUAD only |
| **COAD, SKCM, STAD, THCA, UCEC** | **✗ (no `results/` dir at all)** | **✗** | ✗ | ✗ | ✗ | COAD only |
| TGCT | ✓ | ✓ | partial (1 task, 5 folds) | ✗ | ✗ | ✗ |

Two consequences that aren't in PLAN.md:
1. **5 of the 15 baseline cohorts currently have no on-disk fold metrics** — including the two headline results (THCA-BRAF **0.925**, COAD-MSI **0.871**). Their May-11 report numbers came from data since cleaned/purged (the `_unfair_archive_2026-05-29` cleanup + scratch purge). If a chosen dataset is in this set, its **canonical heads must be re-run too**, not just the new ones.
2. **`ab_mil`/`dtfd_mil`/`trans_mil` are effectively greenfield everywhere** (custom datasets CCRCC/CLWD/HANCOCK on `/scratch` also carry only clam_mb+simple_mil).

**Encoders:** TCGA cohorts have 3 patch encoders (uni_v2, virchow2, hoptimus1). `conch_v15` (TITAN's dependency) is configured only on the 4 custom datasets, **not** on any TCGA cohort.

---

## 2. Decision 1 — the 5 datasets (the primary fork)

PLAN.md §3's competitive table leans on **"built-in datasets & tasks (16 TCGA + 10 CPTAC)"** as "our single strongest row," so the 5-dataset showcase should be **TCGA + CPTAC cohorts** (Pool A), not the fully custom datasets (Pool B: ovarian is off-cluster; ccrcc/clwd/hancock are on purgeable scratch and no more ready than TCGA/CPTAC). This plan assumes Pool A. _If you'd rather showcase the custom datasets, say so — it changes §3–§4._

**Final roster (5 cohorts — 3 TCGA + 2 CPTAC): each pinned to one classification task plus an OS survival task.**

| Dataset | Source | Cls task | Task type | n | cls prevalence | OS deaths | OS % |
|---|---|---|---|:--:|:--:|:--:|:--:|
| **TCGA-LUAD** | TCGA / GOLDMARK | KRAS | binary (mut/wt) | 465 | 36.8% | 167 | 35.9% |
| **TCGA-LGG** | TCGA / GOLDMARK | IDH1 | binary (mut/wt) | 491 | 77.8% | 115 | 23.4% |
| **CPTAC-GBM** | CPTAC / Patho-Bench | TP53 | binary (mut/wt) | 99 | 32.3% | 72 | 72.7% |
| **CPTAC-PDAC** | CPTAC / Patho-Bench | immune_class | 3-class (low/med/high) | 105 | balanced | 81 | 77.1% |
| **TCGA-HNSC** | TCGA / GDC clinical | tumor grade | 3-class (G1/G2/G3) | 431 | — | 204 | 47.3% |

**Rationale:** the roster prioritizes **classification-task diversity** (binary
mutation + 3-class immune subtype + 3-class tumor grade) across **two data
sources** (TCGA + CPTAC), while retaining OS survival as a secondary axis on
all five cohorts. The old "≥100 OS deaths hard gate" is **dropped** —
CPTAC-GBM (72 deaths) and CPTAC-PDAC (81 deaths) fall below it; the roster
deliberately trades survival-power for task/source diversity and a
small-sample regime (GBM n=99, PDAC n=105). Genes/tasks are all distinct
(KRAS/IDH1/TP53/immune-subtype/grade).

**Decision closed:** the 5 datasets above are final. TCGA-SKCM (NRAS),
TCGA-BLCA (PIK3CA), and TCGA-COAD (BRAF) — considered in an earlier draft of
this roster — are dropped in favor of the CPTAC-GBM/CPTAC-PDAC/TCGA-HNSC set
above. Grid math is unchanged: 33 experiments/dataset (30 tile-encoder + 3
TITAN), 165 total, 825 fold-trainings — a 3-class task generates the same 33
experiments as a binary one.

_Note: this document predates the final roster pivot and is superseded by
[`PLAN.md`](PLAN.md); commands below are illustrative of the campaign
mechanics, not a live task list._

---

## 3. The campaign — phased

### Phase 0 — preflight (½ day, do first)
- **Verify patch features exist** for the chosen 5, per encoder. LUAD/LGG have them; **THCA/COAD had their `results/` purged — confirm their `benchmark/features/{uni_v2,virchow2,hoptimus1}/` (or `trident_output/`) survived.** If features are gone, add a TRIDENT extraction step (24 h/cohort/GPU) before training.
- **Lock a reproducible pipeline commit** (PLAN.md open item #3). The pipeline is currently split across `main`, `origin/feat/goldmark-parity` (orchestrator free-VRAM budgeting fix + parity mode — now backed up), and `origin/feat/nnmil-survival` (+6 dataset configs incl. HNSC). Decide the canonical set, merge/cherry-pick to `main`, and **tag it** (e.g. `preprint-pipeline-v1`). Every submit script should run that tag.
- **Build the 2 missing configs** (THCA, and HNSC if not pulled from the survival branch) from `benchmarks/datasets/templates/tcga_template.yaml`, mirroring `tcga/tcga_luad.yaml` (3 encoders; `nnmil_models: [ab_mil, trans_mil, simple_mil]` — dtfd added via CLI).

### Phase A — train `ab_mil` + `dtfd_mil` on the 5 (the bulk; ~1 day compute)
One idempotent multi-GPU job per cohort (skips completed automatically). CLAM (`clam_mb`) is already on disk for LUAD/LGG — only re-run canonical heads where results are missing (CPTAC-GBM, CPTAC-PDAC, HNSC).

```bash
# from benchmarks/scripts/ on the cluster, against the tagged pipeline
DATASET=tcga_luad FRAMEWORKS=nnmil NNMIL_MODELS="ab_mil dtfd_mil" \
  ENCODERS="uni_v2 virchow2 hoptimus1" N_FOLDS=5 SEED=42 \
  sbatch submit_benchmark.sh          # 4×H100, 24h, mem=0, self-resubmits on timeout
# repeat for tcga_lgg
# TCGA-HNSC (tumor grade, 3-class):
DATASET=tcga_hnsc FRAMEWORKS=nnmil NNMIL_MODELS="ab_mil dtfd_mil" TASKS="grade" \
  ENCODERS="uni_v2 virchow2 hoptimus1" N_FOLDS=5 SEED=42 sbatch submit_benchmark.sh
# CPTAC-PDAC (immune subtype, 3-class):
DATASET=cptac_pdac FRAMEWORKS=nnmil NNMIL_MODELS="ab_mil dtfd_mil" TASKS="immune_class" \
  ENCODERS="uni_v2 virchow2 hoptimus1" N_FOLDS=5 SEED=42 sbatch submit_benchmark.sh
# repeat for cptac_gbm (TP53, binary)
```
Idempotency: reruns skip experiments already in `results/_completed.json`, so a timeout-resubmit is safe.

### Phase B — TITAN arm (longer pole: code + `conch_v15` features)
1. **Code (~500–650 lines, greenfield — subagent-scoped):**
   - `Framework.TITAN = "titan"` in `benchmarks/src/autobench/pipeline/config.py:27`
   - dispatch branch in `orchestrator.py:_run_single_experiment_dispatch` (~353)
   - new `benchmarks/src/autobench/pipeline/titan/` package (model `nnmil/`'s shape, ~400 lines): `TitanDataset` returns one `[1,4096]` embedding/slide (no bag/patch loop) → linear/MLP head → **same `metrics.json`/`summary.json` contract**.
   - route TITAN prep to `run_slide_feature_extraction_job` (skip H5→PT); add `titan: <HF repo>` / `titan: 4096` to each chosen YAML.
2. **`conch_v15` patch features for the 5 TCGA cohorts** (TITAN's dependency; not yet extracted) — `submit_feature_extraction.sh` per cohort, ~24 h/GPU, parallelizable across 5 GPUs → ~1 day wall-clock.
3. **TITAN runs** — slide-level, 1 vector/slide, very cheap once features + code land.

### Phase C — aggregate + report
- Extend `tasks/baseline_summary/scripts/00_aggregate.py` `KEEP_AND_RENAME` (`:41`) to add `("nnmil","ab_mil"):"ab_mil"`, `("nnmil","dtfd_mil"):"dtfd_mil"` (and a titan row once framework-tagged), re-run `00→01→02→04`. Point `ROOT` at the chosen 5.
- Build PLAN.md §3's competitive/coverage table (autobench row) and the encoder-vs-aggregator de-biasing narrative.

---

## 4. Compute budget (order-of-magnitude)

Reference point: `autobench_luad_sgpu` ran clam+simple × 2 tasks × 3 enc × 10 folds = 120 runs in **10.5 h single-GPU** (CLAM-heavy). nnMIL heads (ab_mil, dtfd_mil) are lighter than CLAM.

| Work | Runs | Est. wall-clock |
|---|---|---|
| ab_mil+dtfd, one 2-task cohort (3 enc ×10 folds ×2 heads) | 120 | ~2–3 h on 4×H100 |
| ab_mil+dtfd, all 5 (each cohort ≈ 2 task-units: 1 cls + 1 OS) | ~600 | **~1 day in one self-resubmitting 4×H100 job** |
| CPTAC-GBM/PDAC + HNSC canonical-head rerun (clam_mb) | 60 | folded into the above |
| `conch_v15` extraction ×5 cohorts | — | ~24 h/GPU, ~1 day across 5 GPUs |
| TITAN training runs ×5 | ~50 | hours (slide-level) |
| TITAN code | — | ~1–2 dev-days |

**Bottleneck is TITAN's feature extraction + code, not the MIL training.** If TITAN slips, the 4-model × 5-dataset table can ship on its own.

---

## 5. Getting the real wall-clock (for the paper's runtime claims)
Since the campaign re-runs everything, instrument it: the orchestrator already writes per-experiment logs (`benchmark/logs/{fw}/{strategy}/…​.log`). Add an elapsed line per fold (or parse `sacct -j <jobid> --format=JobID,Elapsed,ElapsedRaw` right after each cohort's job), and record per-cohort totals. That produces the honest "runtime per cohort" figure PLAN.md wants — as a *result of* the campaign, not a prerequisite to it.

---

## 6. Open decisions for you
1. **The 5 datasets** — final roster is {TCGA-LUAD, TCGA-LGG, TCGA-HNSC, CPTAC-GBM, CPTAC-PDAC} per §2. _(closed)_
2. **Dataset pool** — TCGA + CPTAC (assumed) vs. custom (ovarian/ccrcc/clwd/hancock). _(§2)_
3. **Pipeline single-source-of-truth** — what merges into `main` + gets tagged before the campaign (goldmark orchestrator fix? survival configs?). _(Phase 0)_
4. **TITAN in scope for the preprint, or fast-follow?** — decouples cleanly if timeline is tight. _(Phase B)_

## 7. Risks & durability
- **Scratch purge.** The custom datasets + the goldmark worktree live on `/scratch` (Alliance purges on inactivity). goldmark is now on GitHub; the custom-dataset *features* are not backed up (regenerable from WSIs, but expensive). Not blocking for a TCGA-only campaign (TCGA is on durable `/projects`).
- **Secrets.** `benchmarks/.env` holds a plaintext `HF_TOKEN` + `WANDB_API_KEY`. It's gitignored (not in git), but recommend rotating both and confirming they never entered git history.
- **Config sprawl.** Dataset YAMLs are split across `main` + `origin/feat/nnmil-survival`; the 15-cohort mutation baseline used configs not all in `main`. Consolidating (Phase 0) prevents pulling from ≥2 branches mid-campaign.
