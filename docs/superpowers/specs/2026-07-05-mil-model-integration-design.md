# Design Spec — Integrating ABMIL, DTFD-MIL, and TITAN into `autobench`

**Date:** 2026-07-05
**Status:** design approved (architecture + fidelity knobs), pending spec review → implementation plan
**Companion:** [`../../../paper/preprint/EXECUTION_PLAN.md`](../../../paper/preprint/EXECUTION_PLAN.md) (compute campaign),
[`../../../paper/preprint/PLAN.md`](../../../paper/preprint/PLAN.md) (preprint strategy)
**Provenance:** synthesized from two independent, blind architecture passes (deep-reasoner/Opus + Codex),
which converged on the same hybrid structure and independently flagged the two corrections below.

---

## 1. Goal

Extend the `autobench` MIL benchmark so the preprint's 4-model roster —
`clam_mb`, `simple_mil`, **ABMIL**, **DTFD-MIL** — plus a new **TITAN** slide-level
foundation-model arm all run through the existing multi-GPU orchestrator and emit the
**unchanged** `metrics.json` / `summary.json` contract, so aggregation and reporting need no changes.

This spec is **dataset-independent engineering only.** Dataset selection and the compute
campaign are deferred (previous per-dataset runs are incomplete).

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Overall structure | **Hybrid** (per-model home by training-loop shape) | Each model's structure should match its home; forcing uniformity creates giant `if` branches. |
| **ABMIL** | Faithful port of the **original gated attention** (`lib/AttentionDeepMIL` `GatedAttention`), adapted to precomputed features, run as a `model_type` under the **nnMIL** framework | Uses the citable original directly; ABMIL's loop is the standard `forward→loss→step`, so it needs no new framework. |
| **DTFD-MIL** | New **`Framework.DTFD`** + `pipeline/dtfd/` package driving `lib/DTFD-MIL` modules directly; **AFS** distillation | Two optimizers + two-tier loss + pseudo-bag split can't honestly run in the standard trainer. AFS is the paper default, robust to variable bag sizes. |
| **TITAN** | New **`Framework.TITAN`** + `pipeline/titan/` package; **linear probe** on the frozen slide embedding | One vector/slide — no bag, no aggregator. Linear probe is the standard frozen-encoder eval protocol. |
| TITAN embedding dim | **Read from the feature file at prepare time** (TRIDENT emits **768-d**, not 4096) | `lib/TRIDENT/.../load.py:419` sets 768; never hard-code. |

## 3. Two corrections baked in (both found independently by both experts)

1. **The vendored nnMIL `dtfd_mil` is NOT DTFD-MIL.** `lib/nnMIL/network_architecture/model_factory.py:83-116`
   wraps only the tier-2 `Attention_with_Classifier` (gated attention) — it omits the pseudo-bag
   split, tier-1, and instance distillation that *define* DTFD-MIL. Benchmarking it under that name
   would misrepresent the model in the preprint. The new `pipeline/dtfd/` drives `lib/DTFD-MIL` directly.
2. **TITAN is 768-d.** The head is sized from the extracted feature file, not a hard-coded constant.

## 4. Unifying contract (why the orchestrator/aggregation don't change)

Dispatch is purely on `exp_cfg.framework` (`orchestrator.py:353-358`, `_gpu_worker.py:113-118`);
the orchestrator reads back only `summary.json` (`orchestrator.py:488-524`). So **adding a framework =
one enum member + one dispatch branch + one `run_*_experiment()` that writes the shared summary.**
The multi-GPU scheduler, VRAM bin-packing, completion/failure tracking, split CSVs, and the
aggregation/reporting layer are all untouched. Precedent in-tree: `pipeline/smmile/` is already a
standalone package for a model that breaks the standard loop.

Every new arm reuses, unchanged:
- `evaluate.py::compute_extended_metrics` + `compute_confidence_intervals` (the metric contract),
- the fold split CSVs `splits/<strategy>/<task>/splits_<fold>.csv` (identical folds across all arms → fair comparison),
- the `summary.json` writer shape (`nnmil/runner.py:61-81`) and the autoMIL archive contract (`clam/runner.py:16-70`).

## 5. Component 1 — ABMIL (original gated attention, under nnMIL)

**Source of truth:** `lib/AttentionDeepMIL/model.py` `GatedAttention` (`:72-128`).

**Adaptation (only the two MNIST-specific ends change; the attention is verbatim):**
- Replace `feature_extractor_part1/2` (Conv2d for 28×28 images) with a linear projection
  `Linear(in_dim → M)` of the precomputed patch features. (In the original, that extractor merely
  produces instance embeddings, which we already have from the patch encoder.)
- Keep gated attention exactly: `attention_V` (Linear→Tanh), `attention_U` (Linear→Sigmoid),
  `attention_w`, softmax over instances, weighted sum.
- Replace the binary `Sigmoid` head with a `Linear(M → num_classes)` softmax head, so multi-class
  tasks (CLWD 6/7-class) work and the loss/metrics path matches the other models (CE + `compute_extended_metrics`).

**Home:** `model_type` under `Framework.NNMIL` — the standard `ClassificationTrainer`
(`classification_trainer.py:62-90`) already fits (single forward → CE → one optimizer step).
**Two distinct config keys (updated 2026-07-06, per Leo):** `ab_mil` = the original **non-gated**
attention (restored to nnMIL's historical meaning + the AttentionDeepMIL repo default), and
`ab_mil_gated` = the faithful **gated** port (M=500/L=128). No key is overloaded; which variant the
preprint reports is a roster decision (deferred). Both run on nnMIL's standard trainer.

**Methods-note fidelity:** "ABMIL (Ilse et al., 2018), gated-attention variant, from the authors'
reference implementation; instance feature extractor replaced by a linear projection of precomputed
patch-encoder features; classifier generalized to K-class softmax." Hidden dims: paper uses M=500/L=128 —
**decision to confirm at review:** match the paper (500/128) or keep 512/128 for uniformity with the nnMIL zoo.

**New files:** 1 small model module (~70 lines) + factory registration. No trainer changes.

## 6. Component 2 — DTFD-MIL (new `Framework.DTFD`)

**Source of truth:** `lib/DTFD-MIL/` — `Main_DTFD_MIL.py` (two-tier train/eval `:151-376`),
`Model/network.py` (`DimReduction`, `Classifier_1fc`, `residual_block`), `Model/Attention.py`
(`Attention_Gated`, `Attention_with_Classifier`), `utils.py:5-8` (`get_cam_1d`).

**Why its own framework:** four structural incompatibilities with the standard trainer, each verifiable:
two optimizers over disjoint module groups (`Main_DTFD_MIL.py:104-105`); two-tier loss with
`loss0.backward(retain_graph=True)` then `loss1.backward()` and per-module grad clipping between
(`:351-365`); random pseudo-bag split re-drawn every forward (`:311-314`); AFS/MaxS/MaxMinS instance
distillation from tier-1 CAM rankings (`:326-344`). No hook into the single-optimizer trainer can host this.

**Where each concern lives:**
- **Pseudo-bag split → trainer** (it's random per forward, not a fixed data artifact). The dataset stays
  trivial: return the full `[N, embed_dim]` H5 bag + int label, reusing the shared split CSVs.
- **Instance distillation (AFS) → trainer** (inseparable from the tier-1 forward).
- **Two optimizers + schedulers + two-tier loss → trainer**, ported from `train_attention_preFeature_DTFD`
  (`:272-376`); eval from `test_attention_DTFD_preFeat_MultipleMean` (`:151-269`).

**Metric contract (non-negotiable, top risk):** discard DTFD's native optimal-threshold `eval_metric`
(`utils.py:22-40`); feed tier-2 softmax probs + labels into the shared `compute_extended_metrics` so
`test_auc`/`test_bacc` are computed by identical code across all four models.

**Config (`DTFDConfig`, defaults from `Main_DTFD_MIL.py:39-50,104-108`):** `numGroup=4`,
`distill="AFS"` (locked), `mDim=512`, `grad_clip=5`, `lr=1e-4`, `wd=1e-4`, MultiStepLR decay 0.2.
Guards: `numGroup ≤ N_patches`; seed via `train.seed + fold` (matches nnMIL); restore pristine torch
state on return (respect `_isolated_torch_state()`, `orchestrator.py:300-343`).

**Multi-class distillation semantics (confirm at review):** reference ranks on
`patch_pred_softmax[:, -1]` (last class) — a binary assumption. AFS (attention-feature-sum) sidesteps
per-class ranking, so it's safe for CLWD; documented as the multi-class rule.

**VRAM:** keep `model_type == "dtfd_mil"` so `_MODEL_BASE_VRAM["dtfd_mil"]=4.0` (`orchestrator.py:158`) applies. No scheduler change.

**New files (`pipeline/dtfd/`):** `_imports.py` (~20, `sys.path` + re-export the 4 modules + `get_cam_1d`),
`config.py` (~40), `model.py` (~60, immutable 4-module bundle), `dataset.py` (~60), `train.py` (~220,
two-tier loop + eval; split `train.py`/`eval.py` if it exceeds the band), `runner.py` (~80, clone of `nnmil/runner.py`).

## 7. Component 3 — TITAN (new `Framework.TITAN`)

**Data path:** TITAN slide embeddings come from TRIDENT's slide encoder (`lib/TRIDENT/.../load.py:419-425`,
768-d), one vector per slide, in a separate `features_titan/<slide_id>.h5` dir (analogous to per-patch-encoder
`features_<key>/`). `titan/prepare.py` validates a slide feature exists for every slide in each task CSV,
**reads the true dim from the first file** (fail fast if missing, mirroring `nnmil/prepare.py:147-153`),
and reuses the **same split CSVs** so folds match every other arm.

**Encoder-key semantics:** TITAN *is* the encoder — there is no tile-encoder axis to sweep. Model it as a
single pseudo-encoder key `"titan"` with `embed_dim` = detected dim, so `results_subdir =
titan/standard/<task>/titan/<model>/` is well-formed and `generate_all_experiments` is unchanged. The grid
then yields exactly `tasks × folds` TITAN experiments (one head).

**Head + trainer:** `titan/model.py` = frozen-input **linear probe** `Linear(D, num_classes)` (locked).
`titan/train.py::train_titan_fold` = standard CE + Adam, early-stop on val AUC, eval →
`compute_extended_metrics`. Structurally the simplest arm; its *data path* (not its loop) is why it's a
separate package. VRAM key `_MODEL_BASE_VRAM["titan"]≈2.0`.

**Out of scope here:** actually running TRIDENT to extract `conch_v15` patch features + TITAN slide
features on the TCGA cohorts (that's compute, in the deferred campaign). This spec delivers the code path;
a tiny synthetic fixture verifies it end-to-end.

**New files (`pipeline/titan/`):** `config.py` (~30), `model.py` (~40), `prepare.py` (~90),
`dataset.py` (~70), `train.py` (~90), `runner.py` (~80), optional `_imports.py`.

## 8. Shared edits (register the two new frameworks)

- `pipeline/config.py` — add `DTFD`, `TITAN` to `Framework` (`:23-27`); extend `generate_all_experiments`
  (`:265-269`) so DTFD reads `dtfd_model_types` and TITAN yields the single `titan` arm; add
  `dtfd_models`/`titan_head` to `BenchmarkConfig` (`:138-153`). ~+40 lines.
- `config.py` (autobench) — add `dtfd_models` + optional `titan:` block to `DatasetConfig`. ~+15 lines.
- `orchestrator.py` — two dispatch branches (`:353-358`) + two in `_gpu_worker` (`:113-118`); prepare-gates
  next to `_prepare_nnmil_plans` (`:551-557`); VRAM key `titan`. ~+50 lines.
- `scripts/run_benchmark.py` — extend `_FRAMEWORK_MAP` (`:56-59`) with `dtfd`, `titan`; add `--dtfd_models`. ~+15 lines.

## 9. Also in scope (from the approved engineering bundle)

- **`dtfd_mil` end-to-end verification** — a smoke test that overfits one tiny bag and checks tier-2 loss
  decreases + a fold runs to a valid `metrics.json`. This is how we de-risk before any cluster compute.
- **Runtime instrumentation** — record per-fold `elapsed_seconds` into the fold result so the campaign
  yields honest per-cohort wall-clock. Small, shared across arms.
- **Aggregator/reporting** — extend `KEEP_AND_RENAME` (`00_aggregate.py:41`) for `ab_mil`, `dtfd_mil`, and
  make the tables titan-aware, so new rows flow in with no rework.
- **Pipeline consolidation + tag** — merge the goldmark orchestrator free-VRAM fix into `main` and tag
  `preprint-pipeline-v1` as the canonical version everything builds on. (Git decision; sequenced first.)

## 10. Test strategy (TDD)

Per-arm, fixture-driven (no cluster, no real WSIs):
- ABMIL: unit test the gated forward on a random `[N, in_dim]` bag → `[num_classes]` logits; gradient flows.
- DTFD: smoke test — one tiny bag, `numGroup=2`, 3 epochs, assert tier-2 loss ↓ and a `metrics.json` is written with the shared schema.
- TITAN: fixture `features_titan/*.h5` of shape `[1,768]`, assert prepare detects dim=768, a fold trains, `metrics.json` emitted.
- Cross-cutting: each new `Framework` appears in `generate_all_experiments` output for a fixture config; dispatch routes to the right runner; a completed run's `summary.json` validates against the existing schema.

## 11. Risks

1. **DTFD metric divergence** (highest) — must use `compute_extended_metrics`, not native eval. Baked in (§6).
2. **Faithful DTFD two-tier port** — `retain_graph` first backward, grad-clip order, per-slide optimization are
   easy to get subtly wrong. Port line-by-line from `lib/DTFD-MIL`; smoke test guards it.
3. **TITAN dim/feature existence** — detect dim from file; fail fast in prepare if slide features absent.
4. **Multi-class DTFD ranking** — AFS avoids per-class ranking; documented.
5. **Cross-framework process state** — heterogeneous trainers under `max_tasks_per_child>1`; DTFD must restore
   pristine torch state so it can't perturb a following experiment's metrics. Prefer the multi-GPU path (fresh
   process per experiment) for the campaign.

## 12. Out of scope / deferred

- Dataset selection + the compute campaign (per Leo — prior runs incomplete).
- `conch_v15` + TITAN feature extraction runs (compute).
- Regression arm (Phase 2; no continuous target exists).
- Expanding the roster beyond 4 (PLAN.md locks it).

## 13. Decisions resolved (autonomous, per Leo's delegation 2026-07-05)

Leo delegated these with two governing goals: **(1) a robust/rigorous pipeline; (2) a fair
environment — no model intentionally handicapped.** Resolved accordingly:

1. **ABMIL hidden dims → paper-exact `M=500, L=128`.** Fidelity to Ilse et al. + the fairness
   principle "each model runs at its authors' canonical config." The 500-vs-512 gap is negligible and
   not a handicap — it *is* ABMIL's designed capacity.
2. **ABMIL keys → separated (updated 2026-07-06, per Leo).** `ab_mil` = original **non-gated**
   attention (restored to its historical meaning + the repo default); `ab_mil_gated` = the gated port
   (M=500/L=128). No key is overloaded, so old non-gated `ab_mil` results (e.g. the TGCT partial) stay
   valid — the earlier contamination caveat is moot. Which variant the preprint reports is a roster
   decision, deferred with dataset selection; to report gated, list `ab_mil_gated` in `nnmil_models`.
3. **Commit** → all work on branch `feat/mil-model-integration` with granular conventional commits;
   **no push, no merge to `main`** (Leo reviews first). Design docs committed on the branch too.

## 14. Review-list items (surfaced for Leo on wake)
- Confirm the resolved decisions above (esp. gated ABMIL as `ab_mil_gated`; `ab_mil` = non-gated original).
- Roster: pick which ABMIL variant the preprint reports — `ab_mil` (non-gated) vs `ab_mil_gated` (gated) — and list it in the datasets' `nnmil_models` (deferred with dataset selection).
- Pipeline consolidation/tag (`preprint-pipeline-v1`, goldmark orchestrator fix into `main`) — a git
  decision left for Leo; the engineering branch is cut from current `main`.
- Any hyperparameters the fixtures can't validate against real data (DTFD `numGroup`/`total_instance`
  vs real WSI bag sizes; TITAN linear-probe lr/epochs) — tuned to reference defaults, listed for review.
