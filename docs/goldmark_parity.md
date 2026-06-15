# GOLDMARK pipeline diff + parity mode — findings & changes

**Branch:** `feat/goldmark-parity` (off `origin/main` @ 340ae85)
**Date:** 2026-06-14 · **Author:** Leo + Claude
**Motivating doc:** `…/TCGA/reports/baseline_summary/COMPARISON.md` — our matched-task
TCGA mutation AUCs read ~2–4 points below GOLDMARK & nnMIL.
**Verification job:** SLURM `44355372` (`gmparity_luad`), output in
`/scratch/yinshuol/autoMIL/goldmark_parity_luad/` (scratch only — not shared).

---

## 1. Question

Is our "lower" number a **pipeline deficiency** (data → training → eval) or a
**comparison/protocol artifact**? If there's a real difference, fix it on a
branch and verify on LUAD before touching shared results.

## 2. Method

A research workflow mapped **our** pipeline per stage (file:line) and extracted
**GOLDMARK's** real protocol from its **source code**
(`github.com/chadvanderbilt/GOLDMARK`) and the arXiv:2603.20848 paper, then
synthesized a classified per-stage diff. Key code/config sites were
independently re-read and the parity behavior unit-tested + dry-run on real LUAD
data before any GPU use.

## 3. The decisive finding (HIGH confidence, verified against GOLDMARK code)

GOLDMARK's internal TCGA-CV number sets, in `scripts/train_task_v2.py`:

```
train_split_value='train', val_split_value='test', test_split_value='test'
```

i.e. the **same** held-out fold of a 2-way ~70/30 patient `StratifiedShuffleSplit`
is used **both** for best-epoch checkpoint selection (val) **and** as the reported
metric (test). With `PATIENCE=999`/`VAL_INTERVAL=999`, validation runs only at a
sparse epoch grid and the **best-AUC epoch on that fold** is kept. GOLDMARK
**self-reports +0.039 mean AUROC** of optimism from this best-epoch selection alone.

**Ours** reports the held-out **TEST** AUC of a 3-way ~70/10/20 split — a fold the
model never trains on **and** never selects on (selection uses a disjoint 10% val).
So our number is strictly more conservative. **This protocol gap — not a pipeline
deficiency — is the dominant reason our matched-task AUCs read lower.**

Corroborating: our pipeline already records **both** `val_` and `test_` AUC per
run; on LUAD the `val_`(select-set) figure already sits ~0.02–0.05 above `test_`.

## 4. Per-stage diff (classified)

| Stage | Aspect | Ours | GOLDMARK | Class | Impact |
|---|---|---|---|---|---|
| eval | reported split | held-out **TEST** (3-way 70/10/20) | **val==test** holdout (2-way 70/30) | **protocol_artifact** | high |
| ckpt | selection | best val-loss (CLAM) / val-bacc (nnMIL), disjoint from report | best-AUC epoch **on the reported fold** (+0.039) | **protocol_artifact** | high |
| reporting | aggregation | report **all** encoders/models, mean±CI, no cherry-pick | **best-encoder** per task (Borda rank-sum); headline 0.896=EAGLE | **protocol_artifact** | high |
| feature_extraction | nnMIL patch cap | train cap `0.5×median` (random sub-bag) | keep **all** tiles | **real_deficiency** | med |
| feature_extraction | tissue/QC | per-patch tissue≥0; no slide-area QC | per-tile tissue≥0.5; drop slides <25 mm² | real_deficiency | med (needs re-extract → deferred) |
| feature_extraction | tile FOV | 112 µm (224px@0.5mpp) | code: **112 µm (= ours)**; paper prose: 128 µm | unknown | med (ambiguous; deferred) |
| splits | fractions / folds | 70/10/20, 10-fold (LUAD) | 70/30, 5-fold | protocol_artifact | low/med |
| training | aggregator | CLAM gated-attention / nnMIL heads | GMA (gated attention) | **equivalent** | low |
| training | epochs | 200+earlystop (CLAM); 100 (nnMIL) | 120 fixed, best-epoch | protocol_artifact | low |
| labels | definition | OncoKB `{GENE}_binary` (verbatim GOLDMARK) | OncoKB L1/2/3 `label_index` | **equivalent** | none |

### Corrections to prior notes
- **Tiling:** we are **224px @ 20×** (store `20x_224px_0px_overlap`), **matching**
  GOLDMARK's 224px. COMPARISON.md's "ours = 256px" is **wrong** (fix it post-merge).
- **Encoders:** ours are **newer** — H-Optimus-**1** (vs 0), UNI **v2** (vs v1),
  Virchow2 (exact). An advantage, not a deficit.
- **EAGLE out of scope:** GOLDMARK's headline LUAD-EGFR 0.896 / THCA-BRAF 0.937 /
  LGG-IDH1 0.827 are the **fine-tuned Prov-GigaPath (EAGLE)** encoder — not a
  frozen-encoder target. Compare only against frozen-encoder cells / the 0.831
  top-8 aggregate. (Verified: these numbers are NOT in the arXiv text; they're
  Download-Center/Figure-3 EAGLE artifacts.)

## 5. What changed (this branch)

Design principle: **keep our conservative held-out-TEST default 100% intact**;
add parity as an **opt-in** comparison instrument.

1. `feat: add opt-in GOLDMARK-parity split mode` (110dcf1→)
   - `splits.py::_splits_goldmark_parity` — 2-way `StratifiedShuffleSplit`
     (`holdout_frac`), patient-level, writes the **same** holdout slide_ids into
     **both** `val` and `test` columns (`val==test`). 2-way leakage assert.
   - Threaded `holdout_frac`: `config.BenchmarkConfig` → `run_benchmark.py`
     (`--goldmark_parity` / `--holdout_frac` 0.30) → `orchestrator._prepare_data`
     → `prepare_all` → `create_strategy_splits`.
   - Because `val==test`, the existing select-on-val / report-on-test machinery
     (CLAM early-stop ckpt + test summary; nnMIL `evaluate('test')`) reproduces
     GOLDMARK's select-and-report-on-same-fold with **no downstream change**.
   - `TestGoldmarkParity`: val==test, 70/30, patient-level, reproducibility,
     default-unchanged regression, feasibility guard.
2. `feat: make nnMIL train-time patch cap configurable` (the one real fix)
   - `--nnmil_max_seq_multiplier` (default **0.5** = upstream) and
     `--nnmil_use_original_length` (default **off**) threaded through
     `config` → `run_benchmark` → `orchestrator._prepare_nnmil_plans` →
     `nnmil/prepare.py` (`_analyze_features`, `_generate_training_config`).
     Defaults are bit-identical to before; raise to retain rare discriminative
     tiles (CLAM already keeps all). `TestPatchCapKnobs`.
3. `test: LUAD GOLDMARK-parity verification submit script`
   - `submit_goldmark_parity_luad.sh` (this job).

Plus a baseline commit carrying pre-existing in-flight infra (cuBLAS pin, nnMIL
`num_workers`, tcga_lgg/coad configs, approx submit scripts).

**Tests:** 78 relevant autobench tests pass (incl. 5 new parity + 4 new cap).
Default 3-way path proven unchanged.

## 6. Verification — what to look for

Job `44355372` runs CLAM `{clam_sb,clam_mb} × {egfr,kras} × {hoptimus1,uni_v2,
virchow2} × 5 folds` in **default** and **parity** modes (fold-matched).

Compare per (encoder, task): `parity_holdout_auc − default_test_auc`.
- **Expected:** parity ≈ +0.02..+0.06 above default (val-vs-test + best-epoch +
  select-on-report optimism; GOLDMARK self-reports +0.039 from selection alone).
- **Confirmation that "gap = protocol, not pipeline":** parity-mode **Virchow2**
  (exact-encoder control) lands within ~0.02–0.03 of GOLDMARK's frozen Virchow2
  cell, while our conservative default sits ~0.03–0.05 lower.
- Do **not** compare against 0.896 (EAGLE). Use frozen-encoder cells / 0.831 top-8.

Results land in `…/goldmark_parity_luad/{parity,default}/benchmark/aggregated/clam/standard.csv`.

## 6a. RESULTS (first-order; job 44355922)

Parity mode completed all 12 CLAM experiments (clam_sb + clam_mb × {egfr,kras} ×
3 encoders, 5-fold). The default-mode arm hit the 12h wall (full-bag CLAM on LUAD
is slow) — fold-matched default-5fold is being re-run; numbers below compare
parity-5fold against our **existing default-10fold** held-out-TEST baseline.

**Parity (val==test holdout AUC, 70/30) vs default-10fold (held-out TEST), clam_mb:**

| task | encoder | parity | default(test) | Δ |
|---|---|---|---|---|
| egfr | hoptimus1 | 0.836 | 0.809 | +0.027 |
| egfr | uni_v2 | 0.775 | 0.742 | +0.034 |
| egfr | virchow2 | 0.799 | 0.802 | −0.003 |
| kras | hoptimus1 | 0.653 | 0.607 | +0.046 |
| kras | uni_v2 | 0.615 | 0.592 | +0.023 |
| kras | virchow2 | 0.602 | 0.585 | +0.017 |

**Mean Δ ≈ +0.024** (5/6 positive). `val_auc == test_auc` in every parity row
(aliasing verified). This is in the predicted +0.02–0.06 band and is a
conservative *lower bound* on GOLDMARK's optimism (we still select the checkpoint
on val-loss, not best-AUC; GOLDMARK self-reports +0.039 from best-AUC selection
alone). clam_sb parity (egfr: 0.823/0.756/0.798; kras: 0.635/0.631/0.617) tracks
clam_mb closely.

**Conclusion: adopting GOLDMARK's reporting protocol raises our number by ~the
size of the reported gap — the deficit is a reporting/protocol artifact, not a
pipeline deficiency.** Fold-matched default-5fold will tighten the per-cell delta.

## 7. Open questions / caveats (carried, not blocking)

- GOLDMARK FOV: code says 112 µm (= ours); paper prose says 128 µm. If the
  manuscript truly used 128 µm a real FOV confounder exists, but it would need
  re-extraction (~17.5×) — deferred, documented.
- Frozen-encoder per-task LUAD cells live in the GOLDMARK Download-Center
  supplement, not the arXiv text; pull them for an exact comparison target.
- Label-tier exactness: confirm our `{GENE}_binary` were derived with the same
  OncoKB L1/2/3-actionable filter (prevalence check: EGFR≈11%, KRAS≈37% on our
  465-case cohort — consistent).

## 8. Merge plan (after concrete results)

Only after parity results are in and reviewed: merge `feat/goldmark-parity` →
`main` (PR), and move/curate experiment results from scratch into the shared
benchmark dir. Until then, **nothing is written to the shared tree**.
