# Preprint — Execution Plan (compute campaign)

_Companion to [`PLAN.md`](PLAN.md) (strategy) and [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Compiled 2026-07-03 from a live audit of the codebase + the cluster
(`fir.alliancecan.ca`) results tree. This document is grounded in **what is
actually on disk today**, which differs from PLAN.md's cost assumptions in two
load-bearing ways — see §0._

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
- ☐ **Re-run `clam_mb` + `simple_mil`** for any chosen cohort whose results were purged: **COAD, THCA** (also SKCM/STAD/UCEC if picked). _(compute)_
- ✅ `clam_mb` + `simple_mil` already on disk for **LGG, LUAD** (and BLCA/BRCA/CESC/GBM/HNSC/PAAD/PCPG/UCS).

**B. Datasets & configs**
- ☐ **Decide the final 5** (recommendation §2: THCA, LGG, LUAD, HNSC, COAD). _(your call)_
- ☐ **Build missing YAML configs**: THCA (from `tcga_template.yaml`); HNSC exists on `origin/feat/nnmil-survival`; UCEC/BLCA if swapped in. _(small)_
- ☐ **Preflight: confirm patch features survived** for the purged cohorts (THCA/COAD `benchmark/features/*` or `trident_output/`) — if gone, add TRIDENT extraction. _(check)_

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

PLAN.md §3's competitive table leans on **"built-in datasets & tasks (16 TCGA + 10 CPTAC)"** as "our single strongest row," so the 5-dataset showcase should be **TCGA mutation cohorts** (Pool A), not the custom datasets (Pool B: ovarian is off-cluster; ccrcc/clwd/hancock are on purgeable scratch and no more ready than TCGA). This plan assumes Pool A. _If you'd rather showcase the custom datasets, say so — it changes §3–§4._

**Ranking — signal (best test AUC, May baseline) vs. campaign cost (`tasks × n`, the driver of how many new runs `ab_mil`+`dtfd_mil` add):**

| Cohort | tasks | n | cost `t×n` | best AUC | headline task | ready? |
|---|:--:|:--:|:--:|:--:|---|---|
| **THCA** | 2 | 495 | 990 | **0.925** | braf, nras | needs config + rerun (no results) |
| **LGG** | 2 | 491 | 982 | **0.849** | idh1 | ✅ YAML+results |
| **LUAD** | 2 | 465 | 930 | 0.788 | egfr, kras | ✅ YAML+results (goldmark focus) |
| **HNSC** | 2 | 431 | 862 | 0.811 | hras | config on `origin/feat/nnmil-survival` |
| **COAD** | 5 | 550 | 2750 | **0.871** | msi, braf | YAML in `main`, **no results** (rerun) |
| UCEC | 5 | 499 | 2495 | 0.841 | pten | no config, no results |
| BLCA | 5 | 386 | 1930 | 0.838 | fgfr3 | no config |
| STAD | 1 | 374 | 374 | 0.734 | pik3ca | no config, no results |
| PAAD/PCPG | 1 | ~180 | ~180 | ~0.72 | kras/hras | no config |
| BRCA | 1 | 1000 | 1000 | 0.692 | pik3ca | ✅ results |
| CESC/GBM/SKCM/UCS | — | — | — | ≤0.65 | (near-chance) | — |

**Recommended 5: THCA, LGG, LUAD, HNSC, COAD.**
Rationale: the four 2-task cohorts give five strong, distinct mutation stories (BRAF/NRAS, IDH1, EGFR/KRAS, HRAS) at low cost; COAD adds the MSI headline (0.871). To trim COAD's 5-task cost, **subset it to `msi` + `braf`** (its only strong tasks) — that drops its cost from 2750 → ~955 and keeps the headline. Net campaign cost of the recommended 5 with COAD subset ≈ **10 task-units**, all comparable size.
- **Swap options:** UCEC (pten 0.841) or BLCA (fgfr3 0.838) instead of HNSC if you want a bigger cohort; both cost more (5 tasks) and need configs built.
- **Leanest possible 5** (all ready, minimal new config): LUAD, LGG, COAD, BRCA + one of THCA/HNSC — but BRCA (pik3ca 0.692) is a weaker story.

**Decision needed from you:** confirm the 5 (recommended set, or a swap), and whether to subset COAD's tasks.

---

## 3. The campaign — phased

### Phase 0 — preflight (½ day, do first)
- **Verify patch features exist** for the chosen 5, per encoder. LUAD/LGG have them; **THCA/COAD had their `results/` purged — confirm their `benchmark/features/{uni_v2,virchow2,hoptimus1}/` (or `trident_output/`) survived.** If features are gone, add a TRIDENT extraction step (24 h/cohort/GPU) before training.
- **Lock a reproducible pipeline commit** (PLAN.md open item #3). The pipeline is currently split across `main`, `origin/feat/goldmark-parity` (orchestrator free-VRAM budgeting fix + parity mode — now backed up), and `origin/feat/nnmil-survival` (+6 dataset configs incl. HNSC). Decide the canonical set, merge/cherry-pick to `main`, and **tag it** (e.g. `preprint-pipeline-v1`). Every submit script should run that tag.
- **Build the 2 missing configs** (THCA, and HNSC if not pulled from the survival branch) from `benchmarks/datasets/tcga_template.yaml`, mirroring `tcga_luad.yaml` (3 encoders; `nnmil_models: [ab_mil, trans_mil, simple_mil]` — dtfd added via CLI).

### Phase A — train `ab_mil` + `dtfd_mil` on the 5 (the bulk; ~1 day compute)
One idempotent multi-GPU job per cohort (skips completed automatically). CLAM (`clam_mb`) is already on disk for LUAD/LGG — only re-run canonical heads where results are missing (THCA, COAD).

```bash
# from benchmarks/scripts/ on the cluster, against the tagged pipeline
DATASET=tcga_luad FRAMEWORKS=nnmil NNMIL_MODELS="ab_mil dtfd_mil" \
  ENCODERS="uni_v2 virchow2 hoptimus1" N_FOLDS=10 SEED=42 \
  sbatch submit_benchmark.sh          # 4×H100, 24h, mem=0, self-resubmits on timeout
# repeat for tcga_lgg, tcga_thca, tcga_hnsc
# COAD (rerun ALL heads incl. clam_mb + subset tasks):
DATASET=tcga_coad FRAMEWORKS="clam nnmil" NNMIL_MODELS="ab_mil dtfd_mil simple_mil" \
  MODELS="clam_mb" TASKS="msi braf" ENCODERS="uni_v2 virchow2 hoptimus1" \
  N_FOLDS=10 SEED=42 sbatch submit_benchmark.sh
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
| ab_mil+dtfd, all 5 (COAD subset to 2 tasks ≈ 10 task-units) | ~600 | **~1 day in one self-resubmitting 4×H100 job** |
| COAD canonical-head rerun (clam_mb, 2 tasks) | 60 | folded into the above |
| `conch_v15` extraction ×5 cohorts | — | ~24 h/GPU, ~1 day across 5 GPUs |
| TITAN training runs ×5 | ~50 | hours (slide-level) |
| TITAN code | — | ~1–2 dev-days |

**Bottleneck is TITAN's feature extraction + code, not the MIL training.** If TITAN slips, the 4-model × 5-dataset table can ship on its own.

---

## 5. Getting the real wall-clock (for the paper's runtime claims)
Since the campaign re-runs everything, instrument it: the orchestrator already writes per-experiment logs (`benchmark/logs/{fw}/{strategy}/…​.log`). Add an elapsed line per fold (or parse `sacct -j <jobid> --format=JobID,Elapsed,ElapsedRaw` right after each cohort's job), and record per-cohort totals. That produces the honest "runtime per cohort" figure PLAN.md wants — as a *result of* the campaign, not a prerequisite to it.

---

## 6. Open decisions for you
1. **The 5 datasets** — confirm recommended {THCA, LGG, LUAD, HNSC, COAD} (COAD subset to msi+braf?), or a swap. _(§2)_
2. **Dataset pool** — TCGA (assumed) vs. custom (ovarian/ccrcc/clwd/hancock). _(§2)_
3. **Pipeline single-source-of-truth** — what merges into `main` + gets tagged before the campaign (goldmark orchestrator fix? survival configs?). _(Phase 0)_
4. **TITAN in scope for the preprint, or fast-follow?** — decouples cleanly if timeline is tight. _(Phase B)_

## 7. Risks & durability
- **Scratch purge.** The custom datasets + the goldmark worktree live on `/scratch` (Alliance purges on inactivity). goldmark is now on GitHub; the custom-dataset *features* are not backed up (regenerable from WSIs, but expensive). Not blocking for a TCGA-only campaign (TCGA is on durable `/projects`).
- **Secrets.** `benchmarks/.env` holds a plaintext `HF_TOKEN` + `WANDB_API_KEY`. It's gitignored (not in git), but recommend rotating both and confirming they never entered git history.
- **Config sprawl.** Dataset YAMLs are split across `main` + `origin/feat/nnmil-survival`; the 15-cohort mutation baseline used configs not all in `main`. Consolidating (Phase 0) prevents pulling from ≥2 branches mid-campaign.
