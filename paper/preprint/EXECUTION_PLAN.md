# Preprint — Execution Plan (compute campaign)

_Companion to [`PLAN.md`](PLAN.md) (strategy) and [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Compiled 2026-07-03 from a live audit of the codebase + the cluster
(`fir.alliancecan.ca`) results tree. This document is grounded in **what is
actually on disk today**, which differs from PLAN.md's cost assumptions in two
load-bearing ways — see §0._

> **Status (updated 2026-07-21).** Two things have moved since this doc was
> compiled:
>
> 1. **Roster resolved (2026-07-17), decision closed** — TCGA-LUAD/LGG/HNSC +
>    CPTAC-GBM/PDAC (§2). The §0 audit, §0b checklist, and §1 on-disk coverage
>    tables are the **2026-07-03 planning snapshot** and still name the earlier
>    candidate set (THCA/COAD/SKCM/BLCA) — kept for provenance. Re-run the
>    coverage/preflight checks against the resolved roster before launching.
> 2. **The TITAN build and the survival pipeline are now merged on `main`** —
>    `Framework.TITAN` + `benchmarks/src/autobench/pipeline/titan/` ship today,
>    so §0b-C's code item and §6's "TITAN in scope?" question are **closed**.
>    §4's compute table is also built on a **10-fold** reference point; the
>    campaign is **5-fold** — corrected in place below.
>
> Canonical roster: [`PLAN.md`](PLAN.md) §1. **Canonical counts:**
> [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) — where this doc disagrees on a
> number, that one wins.

---

## 0. TL;DR — how the audit revises PLAN.md

| PLAN.md assumption | Audit finding | Impact |
|---|---|---|
| §2: `abmil` results already on disk → adding it is "free re-aggregation" | **`abmil` has essentially zero usable coverage** — one partial cohort (TGCT, 1 task, 5 folds). `_completed.json` and the fold-metric tree everywhere else contain **only `clam_mb` + `simple_mil`**. | `abmil` is **not free** — it needs training runs like `dtfd_mil`. The "free re-aggregation" path does not exist. |
| §2: `dtfd_mil` needs training | Confirmed — `dtfd_mil` is on **zero** cohorts. | Correct as written. **Since resolved in config:** ABMIL and DTFD are each their own framework, and all 5 roster YAMLs now pin `abmil_models: [abmil]` / `dtfd_models: [dtfd_mil]`. No CLI override needed — just include them in `FRAMEWORKS`. |
| §1: pick 5 datasets "by wall-clock runtime"; `sacct` will supply it | **Clean per-cohort wall-clock is not recoverable.** `metrics.json`/`_completed.json` carry no timing; `sacct` history is date-capped and dominated by goldmark LUAD jobs; log mtime spans are calendar envelopes (BRCA 239h, LUAD 554h), not run times. | Moot — since every cohort needs re-running for the 2 new heads, the **new campaign itself yields clean timing**. Select on signal × task-count now; measure real runtime from the campaign. |
| §4: ovarian HRD might be a ready continuous target for a regression arm | **No continuous HRD score exists** — `HRD_label` is a pre-binarized 0/1 manifest column; no threshold/regression logic in the code. Also, **the ovarian root `/mnt/pool/ovariancancer/...` is not on fir** (off-cluster). | Regression stays in Phase 2 (the "exception" does not hold). Ovarian is not runnable on the cluster as-is. |
| §4: TITAN is "new code path, not just config" | Confirmed, and **cheaper than feared**: TRIDENT already ships a `TitanSlideEncoder` (loads `MahmoodLab/TITAN`, **768-d**). Missing piece was ~500–650 lines of *autobench* wiring, reusing the `metrics.json`/`summary.json` contract unchanged. | **Now built and merged on `main`** (`Framework.TITAN` + `pipeline/titan/`). The remaining cost is **`conch_v15` patch-feature extraction for the roster cohorts** (not yet extracted), which TITAN depends on. |

**Net:** the 4-model roster is a **from-scratch training campaign for `abmil` + `dtfd_mil` on the chosen datasets** (not a re-report). The TITAN build that this doc scoped as pending is **done** — it is merged on `main`, so it is no longer a dev-time pole. The `abmil`+`dtfd_mil` heads are cheap and fit in roughly **one 24-h 4×H100 job** (see §4). The remaining long pole is TITAN's `conch_v15` feature extraction.

---

## 0b. Gap checklist — what's missing for the preprint

`☐` = gap to close · `✅` = already done (shown for contrast) · effort/owner in parens.
Detailed "how" for each is in §3.

**A. Experiments — MIL model coverage** _(the bulk; ~1 day compute — §4)_
- ☐ **`abmil`** — train on all 5 chosen datasets. **0 folds on disk today** (only a partial TGCT fragment); the "free re-aggregation" path does not hold. _(compute)_
- ☐ **`dtfd_mil`** — train on all 5. 0 folds on disk. Now pinned in each roster YAML's `dtfd_models`, so no CLI override is needed. _(compute)_
- ☐ **Re-run `clam_mb` + `simple_mil`** for any *roster* cohort lacking on-disk folds. **The 2026-07-03 audit covers the `…/Pathology/TCGA/` tree only**, so its GBM/HNSC rows are **TCGA**-GBM/HNSC — they say nothing about the roster's **CPTAC**-GBM. On the audit's evidence, TCGA-LGG/LUAD/HNSC had folds and **CPTAC-GBM and CPTAC-PDAC are unverified**. Re-verify all five against the current cluster state. _(compute)_
- ✅ `clam_mb` + `simple_mil` already on disk for **TCGA-LGG, TCGA-LUAD, TCGA-HNSC** (and TCGA BLCA/BRCA/CESC/GBM/PAAD/PCPG/UCS).

**B. Datasets & configs**
- ✅ **Final 5 decided** (§2): TCGA-LUAD/LGG/HNSC + CPTAC-GBM/PDAC. _(closed 2026-07-17)_
- ✅ **Roster YAML configs built**: `tcga_hnsc.yaml` (grade), `cptac_gbm.yaml` (tp53), `cptac_pdac.yaml` (immune_class) created; `tcga_luad`/`tcga_lgg` already present. _(done 2026-07-17)_
- ☐ **Preflight: confirm patch features exist** for the roster, especially the 3 new cohorts CPTAC-GBM/PDAC + TCGA-HNSC (`benchmark/features/{uni_v2,virchow2,hoptimus1}/` or `trident_output/`) — extract if missing. _(check)_

**C. TITAN slide-encoder arm** _(the feature extraction is the remaining pole)_
- ✅ **autobench code path** — `Framework.TITAN` + dispatch + `pipeline/titan/` package. **Built and merged on `main`.** _(done)_
- ✅ **TITAN config keys** — `titan: {head: linear}` is present in all 5 roster YAMLs. (The embedding is **768-d**, not 4096; the dim is a code default, `pipeline/config.py: titan_embed_dim = 768`, not a YAML field.) _(done)_
- ☐ **`conch_v15` patch features for all 5 roster cohorts** — TITAN's dependency, **not extracted** on any TCGA cohort as of the audit (only on the custom datasets); the two CPTAC cohorts go via the Patho-Bench/TRIDENT path. ~24 h/GPU, parallelizable. _(compute)_
- ☐ **TITAN training runs** — cheap, slide-level, once features land. _(compute)_

**D. Aggregation & reporting**
- ☐ **Extend `KEEP_AND_RENAME`** in `tasks/baseline_summary/scripts/00_aggregate.py` to add `abmil`, `dtfd_mil` (+ titan); re-run `00→01→02→04`. Key on the right framework — `("abmil","abmil")` and `("dtfd","dtfd_mil")`, **not** `("nnmil", …)`. (`tasks/` is gitignored, so this script exists only in a local checkout.) _(small)_
- ☐ **Competitive/coverage table** vs PathBench-MIL / Patho-Bench / EVA (PLAN.md §3) + the encoder-vs-aggregator de-biasing narrative. _(writing)_

**E. Pipeline reproducibility**
- ☐ **One canonical pipeline commit + tag** (`preprint-pipeline-v1`). **Narrower than originally written:** survival, the roster configs, and TITAN are all **merged on `main`** now (`feat/nnmil-survival` no longer exists on `origin`). The only work still outside `main` is `origin/feat/goldmark-parity` (orchestrator free-VRAM fix + parity mode). Decide whether to merge/cherry-pick it, then tag. _(decision + small)_
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

| Cohort | clam_mb | simple_mil | abmil | dtfd_mil | trans_mil | Local YAML in `main`? |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| BLCA, BRCA, CESC, GBM, HNSC, LGG, LUAD, PAAD, PCPG, UCS | ✓ | ✓ | ✗ | ✗ | ✗ (stray logs only) | LGG/LUAD only |
| **COAD, SKCM, STAD, THCA, UCEC** | **✗ (no `results/` dir at all)** | **✗** | ✗ | ✗ | ✗ | COAD only |
| TGCT | ✓ | ✓ | partial (1 task, 5 folds) | ✗ | ✗ | ✗ |

Two consequences that aren't in PLAN.md:
1. **5 of the 15 baseline cohorts currently have no on-disk fold metrics** — including the two headline results (THCA-BRAF **0.925**, COAD-MSI **0.871**). Their May-11 report numbers came from data since cleaned/purged (the `_unfair_archive_2026-05-29` cleanup + scratch purge). If a chosen dataset is in this set, its **canonical heads must be re-run too**, not just the new ones.
2. **`abmil`/`dtfd_mil`/`trans_mil` are effectively greenfield everywhere** (custom datasets CCRCC/CLWD/HANCOCK on `/scratch` also carry only clam_mb+simple_mil).

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
| **TCGA-HNSC** | TCGA / GDC clinical | tumor grade | 3-class (G1/G2/G3) | 431 (414 gradeable) | — | 205 | 47.6% |

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
- **Lock a reproducible pipeline commit** (PLAN.md open item #3). Survival, the roster configs, and TITAN are **already on `main`**; the only outstanding piece is `origin/feat/goldmark-parity` (orchestrator free-VRAM budgeting fix + parity mode). Decide whether it's in, merge/cherry-pick, and **tag it** (e.g. `preprint-pipeline-v1`). Every submit script should run that tag.
- ~~**Build the 2 missing configs** (THCA, HNSC)~~ — **done, and THCA is no longer in the roster.** All five roster YAMLs exist on `main` with one model pinned per framework (`clam_models: [clam_mb]`, `nnmil_models: [simple_mil]`, `abmil_models: [abmil]`, `dtfd_models: [dtfd_mil]`). Do **not** widen `nnmil_models` — a multi-model list would break the verified 33-experiments/dataset grid.

### Phase A — train `abmil` + `dtfd_mil` on the 5 (the bulk; ~1 day compute)
One idempotent multi-GPU job per cohort (skips completed automatically). CLAM (`clam_mb`) is already on disk for TCGA-LUAD/LGG/HNSC — re-run canonical heads only where results are missing (verify CPTAC-GBM and CPTAC-PDAC; the 2026-07-03 audit did not cover the CPTAC tree).

**`submit_benchmark.sh` reads exactly three inputs** — `<dataset>` (arg 1 or `DATASET`), `<n_folds>` (arg 2 or `N_FOLDS`, default **5**), and `FRAMEWORKS` (default `clam nnmil dtfd abmil`). Everything else — tasks, encoders, per-framework model lists, seed — comes from the dataset YAML. An earlier draft of this block passed `NNMIL_MODELS`/`ENCODERS`/`TASKS`/`SEED`; **the script ignores all four**, so those commands would have silently run the full default grid instead of the intended subset.

```bash
# from the repo root on the cluster, against the tagged pipeline
# just the two new heads (abmil + dtfd are their own frameworks):
FRAMEWORKS="abmil dtfd" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad
FRAMEWORKS="abmil dtfd" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_lgg
FRAMEWORKS="abmil dtfd" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_hnsc

# CPTAC pair — canonical heads unverified, so run the full 4-framework default:
sbatch benchmarks/scripts/slurm/submit_benchmark.sh cptac_gbm
sbatch benchmarks/scripts/slurm/submit_benchmark.sh cptac_pdac
```
Each job is 4×H100, 24 h, `mem=0`, and self-resubmits before the wall. Idempotency: reruns skip experiments already in `results/_completed.json`, so a timeout-resubmit is safe. The `validate config` block prints `experiments=… fold-trainings=…` before launching — sanity-check it against [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §2.

### Phase B — TITAN arm (**code done**; `conch_v15` features are the remaining pole)
1. ~~**Code (~500–650 lines, greenfield)**~~ — **shipped on `main`.** For reference, what landed:
   - `Framework.TITAN = "titan"` in `benchmarks/src/autobench/pipeline/config.py:29`
   - dispatch branch in `pipeline/orchestrator.py`
   - `benchmarks/src/autobench/pipeline/titan/` package: `TitanDataset` returns one **`[1,768]`** embedding/slide (no bag/patch loop) → linear head → **same `metrics.json`/`summary.json` contract**
   - TITAN prep routed to `run_slide_feature_extraction_job` (skips H5→PT); the YAML key is `titan: {head: linear}` — the 768-d dim is a code default (`titan_embed_dim`), **not** a YAML field
2. **`conch_v15` patch features for all 5 roster cohorts** (TITAN's dependency; not yet extracted) — `submit_titan_extract.sh` per cohort, ~24 h/GPU, parallelizable across 5 GPUs → ~1 day wall-clock. Note the CPTAC pair extracts via the Patho-Bench/TRIDENT path, not the TCGA one.
3. **TITAN runs** — `submit_titan.sh` (1×H100), slide-level, 1 vector/slide, very cheap once features land.

### Phase C — aggregate + report
- Extend `tasks/baseline_summary/scripts/00_aggregate.py`'s `KEEP_AND_RENAME` to add `("abmil","abmil"):"abmil"`, `("dtfd","dtfd_mil"):"dtfd_mil"`, and `("titan","titan"):"titan"` — key on each model's **own** framework, not `nnmil`. Re-run `00→01→02→04`; point `ROOT` at the chosen 5.
- Build PLAN.md §3's competitive/coverage table (autobench row) and the encoder-vs-aggregator de-biasing narrative.

---

## 4. Compute budget (order-of-magnitude)

Reference point: `autobench_luad_sgpu` ran clam+simple × 2 tasks × 3 enc × **10 folds** = 120 runs in **10.5 h single-GPU**, i.e. **~5.25 min per fold-training averaged over a CLAM-heavy mix**. `abmil` and `dtfd_mil` are lighter than CLAM.

> **The campaign is 5-fold, not 10.** An earlier version of this table carried the 10-fold reference straight into its run counts and roughly doubled every row. Counts below are the **verified 5-fold grid** from [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §2.1–2.2 (33 exps/dataset; per dataset: `clam_mb` 6, `simple_mil` 9, `abmil` 9, `dtfd_mil` 6, TITAN 3).

| Work | Fold-trainings | Est. wall-clock |
|---|--:|---|
| `abmil`+`dtfd_mil`, one cohort (15 exps × 5 folds) | 75 | ~1–1.5 h on 4×H100 |
| `abmil`+`dtfd_mil`, all 5 cohorts | **375** | **well under one 24-h 4×H100 job** |
| CPTAC-GBM/PDAC canonical-head rerun (`clam_mb`+`simple_mil`, 15 exps × 5 folds × 2 cohorts) | 150 | folded into the above |
| `conch_v15` extraction ×5 cohorts | — | ~24 h/GPU, ~1 day across 5 GPUs |
| TITAN training runs (3 exps × 5 folds × 5 cohorts) | 75 | hours (slide-level) |
| ~~TITAN code~~ | — | **done — merged on `main`** |

**Bottleneck is TITAN's `conch_v15` feature extraction, not the MIL training** (the code pole is gone). If extraction slips, the 4-model × 5-dataset table can ship on its own.

⚠ **Open discrepancy — see [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §3.1.** That doc's per-head model (`clam_mb` ~5 min, `simple_mil` ~2.5 min) predicts **3.75 min** for the same equal-count clam+simple mix this anchor measured at **5.25 min** — so its rates run ~1.4× fast, and the headline "≈ 40 GPU-h" becomes **~55 GPU-h** once every head is scaled by that factor. Both are order-of-magnitude placeholders pending the campaign's own instrumented timings; **not yet reconciled.**

---

## 5. Getting the real wall-clock (for the paper's runtime claims)
Since the campaign re-runs everything, instrument it: the orchestrator already writes per-experiment logs (`benchmark/logs/{fw}/{strategy}/…​.log`). Add an elapsed line per fold (or parse `sacct -j <jobid> --format=JobID,Elapsed,ElapsedRaw` right after each cohort's job), and record per-cohort totals. That produces the honest "runtime per cohort" figure PLAN.md wants — as a *result of* the campaign, not a prerequisite to it.

---

## 6. Open decisions for you
1. **The 5 datasets** — final roster is {TCGA-LUAD, TCGA-LGG, TCGA-HNSC, CPTAC-GBM, CPTAC-PDAC} per §2. _(closed)_
2. **Dataset pool** — TCGA + CPTAC (assumed) vs. custom (ovarian/ccrcc/clwd/hancock). _(§2)_
3. **Pipeline single-source-of-truth** — whether `origin/feat/goldmark-parity`'s orchestrator free-VRAM fix merges into `main` before the tag. Survival + roster + TITAN are already in. _(Phase 0)_
4. ~~**TITAN in scope for the preprint, or fast-follow?**~~ — **closed: in scope, and the code is merged.** _(Phase B)_
5. **Compute estimate reconciliation** — §4's 5.25 min/fold-training anchor vs `EXPERIMENT_GRID.md` §3.1's 3.75 min for the same mix. Decides whether the static grid is ~40 or ~55 GPU-h. _(§4)_

## 7. Risks & durability
- **Scratch purge.** The custom datasets + the goldmark worktree live on `/scratch` (Alliance purges on inactivity). goldmark is now on GitHub; the custom-dataset *features* are not backed up (regenerable from WSIs, but expensive). Not blocking for a TCGA-only campaign (TCGA is on durable `/projects`).
- **Secrets.** `benchmarks/.env` holds a plaintext `HF_TOKEN` + `WANDB_API_KEY`. It's gitignored (not in git), but recommend rotating both and confirming they never entered git history.
- **Config sprawl — largely resolved.** All five roster YAMLs are now on `main` (the `feat/nnmil-survival` split is gone). The residual risk is historical: the 15-cohort mutation baseline used configs that were never all in `main`, so its numbers aren't reproducible from `main` alone.
