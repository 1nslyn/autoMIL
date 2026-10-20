# A training recipe for MIL on small cohorts: what 900 attempts found (v1, draft)

Status: draft of 2026-10-20. One number is still *pending*: the last
attempt of the 6-class CLAM cell (EMA + 1024 tiles). The held-out sanity
readings of §7 follow once that cell closes.
Everything here is validation-only; the held-out readings of the two
development cohorts are taken once at the end and appear in §7 only as a
sanity check, never as the measurement.

## 1. Summary

Across four MIL aggregators (ABMIL, CLAM-MB, DTFD-MIL, nnMIL simple_mil) on
frozen H-optimus-1 features, a training-only recipe (architecture, features,
seed and measurement untouched) gains:

| Piece | Arms | Weak-signal task (CCRCC PBRM1, AUC ~0.6) | Strong-signal task (6-class growth pattern, macro AUC 0.88) | Transfer check (CCRCC high_grade, AUC ~0.72; one attempt, no search) | Seed-robust |
|---|---|---|---|---|---|
| Weight EMA (bias-corrected, swapped in for validation) | ABMIL 0.998 | +0.008 ± 0.009 | +0.015 ± 0.004 | **+0.031 ± 0.013, kept** | yes (3 seeds, two tasks) |
| | CLAM 0.9995 | +0.007 ± 0.009 | +0.006 ± 0.001, guard fail (balanced accuracy −0.043) | +0.014 ± 0.019 | — |
| | nnMIL 0.98 | +0.005 ± 0.009 | −0.001 ± 0.002 | — | — |
| | DTFD 0.998 | +0.007 (guard fail) | +0.003 (guard fail) | +0.011 ± 0.007 with the native lr step at epoch 9 (guard pass) | sign yes, size no |
| nnMIL batch size 4 (native 24) | nnMIL | **+0.034 ± 0.020, kept** | +0.007 ± 0.004 (every fold up; under the keep margin) | −0.006 ± 0.015 | yes (+0.028 and +0.040 under seeds 43 and 44) |
| Any other piece or stack | all | within ±0.01 or worse | within ±0.003 of its parent | — | — |

Two findings carry the recipe. Weight EMA is universal: it removes the
epoch-to-epoch jitter that the smoothed selection metric pays for, worth
+0.007 to +0.031 depending on how noisy the native validation curve is
(ABMIL on high_grade, whose native folds jitter by up to 0.09, gains the
most), and nothing stacks on it beyond ±0.003. Its limit is the operating
point: on CLAM (6-class) and DTFD (PBRM1, 6-class) the averaged weights
rank better but lower balanced accuracy at the fixed threshold by 0.02 to
0.06, and the companion guard excludes them. nnMIL's native batch size is
mis-sized for cohorts under 400 training bags (9–12 optimizer steps per
epoch against an epoch-based patience of 10), and batch 4 alone recovers
+0.034 on CCRCC; the KRAS v5 nnMIL recipe (+0.046) carried the same change.
Batch 2, 4 and 8 are equal (+0.031 to +0.034) and scaling the lr down with
the batch gives the gain back, so the quantity that matters is optimizer
progress per epoch against the patience, not the batch size itself. The
size of the effect follows the under-training: on the 6-class task, where
every native fold already selects an epoch past 25, batch 4 is +0.007 ±
0.004 with every fold up, and on high_grade, where no fold stops early,
it is −0.006 ± 0.015. The fix applies where a native fold stops at the
patience with an early selected epoch, not to every small cohort.

The recipes the KRAS trials found (ABMIL warmup + cosine; CLAM label
smoothing, 512-tile bags, class balancing; DTFD label smoothing + dropout
0.7) did not transfer: on CCRCC they are neutral or lose 0.02–0.05, on the
6-class task they are neutral. They are KRAS-specific tuning of the kind a
30-attempt search finds by chance on 70 validation slides per fold.

So the honest ceiling of a training-only recipe, measured on tasks the
search never saw, is +0.01 to +0.03 for ABMIL (rising with the native
curve's jitter), about +0.01 for CLAM, nothing for DTFD under the
balanced-accuracy guard, plus the occasional correction of a mis-sized
default (nnMIL's batch where a fold stops early). A +0.05 average on the
KRAS arms will not come from training recipes alone.

## 2. Method

- **Development cells.** Plain autoMIL projects on two cohorts outside the
  paper roster, driven by the operator (no agent): CPTAC CCRCC PBRM1
  (142 mutant / 182 wild type, 340 slides) and a 6-class lung growth-pattern
  task (398 slides, cases with disagreeing slide labels dropped). Split
  design `devcv`: five-fold cross-validation inside standard fold 0's
  development slides, fixed test slides; seed 42; the same validation
  schema, keep margin (max(0.015, 1 × paired SE)), companion guard
  (`val_bacc`, margin from the class counts) and winner rule (max(0.01,
  2.93 × pooled paired SE), Sidak-adjusted for the pool size) as the
  campaign cells.
- **Attempts.** Each attempt is one generated policy file from a composite
  recipe template (switches: EMA decay, label smoothing, class weights,
  bag subsampling, warmup/cosine, lr step, minimum epochs, projection
  unfreezing, body-lr scale) plus an `--hparams` override, submitted through
  the same sealed policy seams and submit-time smoke the campaign agents
  use, so every recipe here is expressible by an agent. The root of every
  cell is the native recipe. An all-off policy reproduced the root bit for
  bit on CLAM and nnMIL (five identical prediction hashes each): the seam
  path is the identity.
- **Measurement.** Lift = paired five-fold mean difference against the root
  (or the parent for a stack) ± paired SE, read with `dev_cell.py readout`;
  seed replicates re-run the root and the best pieces under seeds 43 and 44
  on the same folds (the training seed changes, the splits do not). The bag
  cache used for ABMIL and DTFD readers was checked bit-identical against an
  uncached fold (407 s → 63 s).
- **Budget.** CCRCC: ABMIL 13 + 6 replicate runs, DTFD 12 + 6, CLAM 9 +
  identity, nnMIL 12 + identity + 4 replicate runs; 6-class: ABMIL 17 + 6,
  DTFD 8, nnMIL 8, CLAM 9; high_grade transfer check: one attempt per
  arm. All jobs ran at low priority beside the official
  trial, one at a time once the machine was shared.

## 3. Results by cell (validation only)

### 3.1 CCRCC PBRM1

| Arm | Root | Best attempt | Lift ± SE | Seeds 42/43/44 | Verdict |
|---|---|---|---|---|---|
| ABMIL | 0.612 | EMA 0.998 | +0.007 ± 0.009 | +0.007 / +0.015 / +0.002 | baseline kept; EMA real but small |
| CLAM-MB | 0.633 | EMA 0.9995 | +0.007 ± 0.009 | — | baseline kept |
| DTFD-MIL | 0.604 | dropout 0.7 | +0.005 ± 0.015 | +0.005 / −0.006 / +0.008 | baseline kept; nothing |
| nnMIL | 0.574 | batch 4 | +0.034 ± 0.020 | +0.034 / +0.028 / +0.040 | **kept**; seed-robust, balanced accuracy +0.02 to +0.05 |

What else was measured: ABMIL — warmup + cosine +0.002, without EMA
−0.010 (guard fail), 512 tiles −0.023, balanced weights −0.001, projection
unfreezing −0.031 (guard fail), HNSC recipe +0.001, EMA dose 0.99 / 0.995 /
0.999 = +0.000 / +0.005 / −0.001, EMA + LS 0.1 +0.005 (guard fail), late lr
step = EMA alone. CLAM — LS 0.2 −0.030, 512 tiles −0.025, EMA + LS −0.042,
EMA + 512 −0.039, EMA + balanced −0.029, three- and four-piece stacks −0.054
and −0.030, all guard failures. DTFD — every EMA variant fails the `val_bacc`
guard (−0.03 to −0.05) while the AUC holds; EMA + balanced weights −0.011;
dropout 0.5 / 0.8, LS, lr step: within noise. nnMIL — attention/body lr ×0.1
(the KRAS v5 piece) −0.010 to −0.020 with guard failures, EMA 0.98 +0.005,
LS 0.1 +0.002; batch 2 +0.032 ± 0.019 and batch 8 +0.031 ± 0.027 (both
kept), batch 4 + EMA 0.98 −0.001 ± 0.003 against batch 4, batch 4 with lr
1.5e-4 −0.029 ± 0.018 against batch 4 (fold 0 back to epoch 0).

Seed spread of the root's five-fold mean: ABMIL 0.026 (fixed 20 epochs; two
folds select epochs 0–4), DTFD 0.003, nnMIL 0.009.

### 3.2 Six-class growth pattern

| Arm | Root | Best attempt | Lift ± SE | Seeds 42/43/44 | Verdict |
|---|---|---|---|---|---|
| ABMIL | 0.883 | EMA + cosine + unfreezing (+ 512 tiles) | +0.017 ± 0.006 | EMA alone +0.011 / +0.015 / +0.019; the stack +0.016 / +0.012 / +0.017 | stack accepted by the rule; equal to EMA under seeds |
| DTFD-MIL | 0.886 | EMA + lr step at 9 | +0.004 ± 0.003 | — | baseline kept |
| nnMIL | 0.890 | batch 4 | +0.007 ± 0.004 | — | baseline kept; every fold up, balanced accuracy +0.046; EMA −0.001, KRAS v5 pieces −0.002 to −0.005, LS 0.1 +0.003 |
| CLAM-MB | 0.881 | EMA + LS 0.2 + 512 tiles | +0.009 ± 0.003 (guard fail) | — | baseline kept; every EMA variant fails the balanced-accuracy guard (−0.02 to −0.06); LS 0.2 alone +0.002, 512 tiles −0.003; EMA + 1024 tiles *pending* |

ABMIL batch 2 (eight variants on the EMA stack: unfreezing dose and
schedule-free form, EMA 0.995 / 0.999, + 512 tiles, + LS 0.1, + dropout 0.6)
all lie within +0.013 to +0.017 against the root; against their parent the
additions are +0.002, −0.002 and +0.001. Root seed spread 0.0008.

### 3.3 CCRCC high_grade transfer check

Recipe v1 applied with no search, one attempt per arm (219 training bags,
52 validation slides per fold):

| Arm | Root | Recipe v1 | Lift ± SE | Guard | Verdict |
|---|---|---|---|---|---|
| ABMIL | 0.725 | EMA 0.998 | **+0.031 ± 0.013** | pass | kept |
| DTFD-MIL | 0.730 | EMA 0.998 + lr ×0.2 at epoch 9 | +0.011 ± 0.007 | pass (balanced accuracy −0.002) | under the margin |
| CLAM-MB | 0.721 | EMA 0.9995 | +0.014 ± 0.019 | pass (+0.042) | under the margin |
| nnMIL | 0.711 | batch 4 | −0.006 ± 0.015 | pass | no effect: the native folds select epochs 5–21, none stops at epoch 0 |

The ABMIL and CLAM native curves on this task jitter by up to 0.09 and
0.12 per fold; EMA brings ABMIL's down to 0.003–0.05 and its lift is the
largest of any cell.

## 4. Mechanisms

- **EMA removes jitter, not error.** The selection score is the mean of the
  validation metric over five epochs around the restored epoch. ABMIL's
  native per-fold jitter on PBRM1 was 0.003–0.076; under EMA 0.002–0.026,
  and the lift equals the jitter removed. On high_grade the native jitter
  is 0.007–0.090 with hindsight gaps up to 0.052, and the lift is +0.031.
  The underlying curve (peak minus hindsight gap) did not rise in any cell
  by more than 0.005.
- **nnMIL's batch size is a step-count problem.** Steps per epoch = training
  bags / batch size; the plan's batch 24 gives 9 on CCRCC. With patience 10
  epochs, a fold that has not improved in 90 steps stops. Batch 4 gives 51
  steps per epoch and the same folds train for 500+ steps before the
  patience acts. Under three seeds the lift is +0.028 to +0.040, and fold 0,
  the under-trained fold, gains +0.09 to +0.14 every time. Scaling the lr
  down with the batch (sqrt rule, 1.5e-4 at batch 4) gives the whole gain
  back and returns fold 0 to epoch 0, while batch 2, 4 and 8 at the native
  lr are equal: the quantity that matters is optimizer progress per epoch
  (steps × lr) against the epoch-based patience. EMA on top of batch 4 adds
  nothing (−0.001): the jitter it removes is already gone.
- **EMA and the operating point.** On DTFD (PBRM1, 6-class) and on CLAM
  (6-class) the averaged weights rank validation slides as well or slightly
  better but shift the operating point at the fixed decision threshold,
  lowering balanced accuracy by 0.02–0.06. On the binary CLAM cells and on
  DTFD high_grade the guard passed. The companion guard is doing its job:
  the recipe would look better on AUC alone.
- **Why KRAS pieces did not transfer.** Each was selected as the best of
  ~30 attempts on five folds of about 70 validation slides. The CCRCC
  re-measurement shows them neutral or harmful with guard failures; the
  6-class task shows them neutral. A best-of-30 selection on that much
  noise has an expected optimism of about one pooled SE, which is the size
  of the "gains" they carried beyond EMA.

## 5. Recipe v1 (per arm, exact overrides)

- **ABMIL**: EMA decay 0.998 (0.995–0.999 equal), swapped in for
  validation; everything else native. Expected +0.008 to +0.031, rising
  with the native curve's jitter. Optional: 1-epoch warmup + cosine,
  neutral.
- **CLAM-MB**: EMA decay 0.9995; everything else native. Expected +0.007 to
  +0.014 in AUC on binary tasks; on the 6-class task the averaged weights
  lowered balanced accuracy by 0.04 at the fixed threshold, so re-tune the
  threshold or expect a companion guard to exclude it. Do not add label
  smoothing, bag subsampling or balancing by default: they lost on CCRCC
  and were guard-excluded on the 6-class task.
- **DTFD-MIL**: native recipe. EMA gains +0.003 to +0.011 in AUC but fails
  the balanced-accuracy guard on two of three tasks; if the deployment
  threshold is tuned rather than fixed, EMA 0.998 with the native lr step
  is worth that much.
- **nnMIL simple_mil**: batch size 4 (2 to 8 equal) on cohorts below ~400
  training bags, native lr 3e-4 and schedule; never scale the lr down with
  the batch (−0.029); EMA adds nothing on top (−0.001). Expected +0.03 on a
  cohort whose native run stops a fold at the patience with an early
  selected epoch (PBRM1, confirmed under three seeds), and nothing
  reliable where every fold already improves within the patience window
  (the 6-class task +0.007 ± 0.004, high_grade −0.006 ± 0.015). Read the
  baseline's selected epochs before applying it.

## 6. What this changes for the campaign

- The agent's instruction file gains a prior-experience section (§3c):
  open with EMA (and batch 4 for nnMIL), skip the settled axes, size every
  predicted effect against the paired SE, replicate before stacking. The
  expected effect on round 2 is fewer wasted attempts and nnMIL cells that
  find their batch-size gain in batch 1 instead of batch 3; the expected
  lift per arm is +0.01 to +0.03 for ABMIL depending on the task's jitter,
  about +0.01 for CLAM, nothing for DTFD under the balanced-accuracy
  guard, and +0.03 for nnMIL only where the native run under-trains.
- The +0.05 goal for the KRAS arms is not reachable by training recipes
  within the sealed seams. The two routes left are outside this report's
  scope: changing what the seams allow (architecture, features, ensembling)
  or changing the claim.

## 7. Held-out sanity readings (development cohorts only)

Taken once per node through `automil certify` after the cell closed
(2026-10-20). These cohorts are not in the paper, so the values may be
recorded here; the only clean test of a devcv-selected configuration is
split 0's own test slides (67 and 80 slides), so each reading is a sanity
check with an SE of about 0.04, not a measurement.

| Cell | Root (val → test AUC) | Recipe v1 node (val → test AUC) | Test delta |
|---|---|---|---|
| PBRM1 ABMIL | 0.612 → 0.519 | EMA 0.998: 0.619 → 0.523 | +0.004 |
| PBRM1 DTFD | 0.604 → 0.558 | EMA 0.998: 0.612 → 0.555 | −0.003 |
| PBRM1 CLAM | 0.633 → 0.523 | EMA 0.9995: 0.640 → 0.600 | +0.077 |
| PBRM1 nnMIL | 0.574 → 0.639 | batch 4: 0.607 → 0.685 | +0.046 |
| 6-class ABMIL | 0.883 → 0.880 | EMA 0.998: 0.894 → 0.883 | +0.002 |
| 6-class DTFD | 0.886 → 0.870 | EMA + step 9: 0.890 → 0.879 | +0.009 |
| 6-class nnMIL | 0.890 → 0.847 | batch 4: 0.897 → 0.863 | +0.016 |
| 6-class CLAM | *pending* | | |
| high_grade ABMIL | 0.725 → 0.762 | EMA 0.998: 0.756 → 0.748 | −0.014 |
| high_grade DTFD | 0.730 → 0.730 | EMA + step 9: 0.741 → 0.724 | −0.006 |
| high_grade CLAM | 0.721 → 0.728 | EMA 0.9995: 0.735 → 0.738 | +0.010 |
| high_grade nnMIL | 0.711 → 0.691 | batch 4: 0.705 → 0.682 | −0.010 |

What the readings can say: the recipe node is above its root on 7 of 11
single test splits, by a mean of +0.012, within one SE of zero everywhere
except where the validation lift was largest (PBRM1 nnMIL +0.046, CLAM
+0.077). The nnMIL batch fix carries to test on the two tasks where
validation said it worked and not on high_grade, matching §3.3. ABMIL's EMA
lifts (+0.008, +0.015, +0.031 on validation) show no test counterpart
(+0.004, +0.002, −0.014), consistent with §4: EMA removes jitter in the
smoothed selection metric and does not raise the underlying curve. The
PBRM1 roots sit 0.05–0.11 below their validation scores on this test
split, so that cohort's selection signal is itself optimistic.

## 8. Record

- Cells: `benchmarks/experiments/dev/<cohort>__<task>__hoptimus1__<arm>[__seedN]/automil/`
  (graph.json, learnings.md with every batch decision, batches/*.json with
  every attempt's constants and overrides, orchestrator archive on aihub).
- Tooling: `benchmarks/scripts/dev_cell.py` (materialize, readout, batch),
  `dev_loop.py`, `dev_recipe.py`, `dev_batch_file.py`; template
  `benchmarks/experiments/dev/_recipes/composite.py`.
- Earlier evidence: `docs/recipes/trial-synthesis-2026-10-15.md`,
  `docs/recipes/trial-attempts-2026-10-15.csv`, `docs/recipes/playbook-v0.md`.
