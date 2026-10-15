# MIL training-recipe search: prior-experience synthesis (Opus pass)

This is the full report. 
All numbers come from validation. Lifts are on `val_auc_smooth` for v5, v6, longA and longC, and
on `val_auc` (maximum, 3 folds) for v4. No held-out value was read or used.

## 0. Answer up front

1. **The ceiling is about +0.02 to +0.03.** That holds for recipe-only search on the devcv design
   (inner 5-fold CV, 74–75 validation slides per fold) for ABMIL, CLAM, DTFD and TITAN.
   - No devcv leader cleared its multiplicity-corrected winner bar (≈0.029–0.037).
   - The per-fold oracle across all attempts is +0.034 to +0.048 in every devcv cell. That
     oracle is biased upward.
   - Recipe-only search in the explored families cannot reach a +0.05 average across arms.
2. **Headroom: nnMIL first, CLAM second.**
   - nnMIL gained +0.046 ± 0.011 with all five folds up, on standard folds (47 validation
     slides). Its baseline jitter penalty is only ~0.010, so its gains raised the plateau. It
     has never run on devcv.
   - CLAM has three single additions on EMA worth +0.020 to +0.024 each. They gain on
     different folds and have not been stacked.
3. **About half of every devcv MIL gain is removed jitter.**
   - `val_auc_smooth` equals the peak minus a jitter penalty. The baseline penalty is
     0.028–0.029 for ABMIL and CLAM, and 0.017 for DTFD.
   - Weight EMA recovers most of that penalty. Beyond it, the plateau itself rose only +0.006
     to +0.013.
4. **Most attempts were wasted, and most of the waste was avoidable.**
   - 44% of the 712 completed attempts landed within 1 paired SE of their parent. 71% moved
     the score by less than 0.015.
   - 63 attempts (8.8%) went to families that never beat a parent by 1 SE in any arm.
   - In v6, 19 of 70 attempts re-tested axes that earlier cells had already shown to be null
     or harmful. None of the 19 beat its parent by 1 SE.
5. **Two selection-reuse risks for the next round.**
   - longC ran 96 attempts on exactly the v6 TITAN validation folds.
   - The v6 round-1 recipes were selected on the same folds that round 2 will use.
   - Pool those attempts into the winner bar. The Sidak z rises from 2.93 to 3.14 for 60
     comparisons, and to 3.35 for 126.

## 1. Corpus and evidence weights

| design | cells | completed attempts | val slides / fold | weight |
|---|---|---|---|---|
| devcv (longC TITAN, v6 ABMIL/CLAM/DTFD) | 4 (v6 nnMIL and TITAN have no attempts) | 166 | 74–75 (27–28 mutant) | strongest, for positives and negatives |
| HNSC grade (v5, 3-class) | 5 | 150 | 42 (G1 6 / G2 26 / G3 10) | positives real (orchestrator context) |
| standard KRAS (v4 3-fold max metric, v5, longA) | 11 | 396 | 47 (17 mutant) | negatives strong, positives weak |

Status counts:
- 712 attempts completed. This includes one partial run, v4 DTFD node_0019.
- 6 are running (v6 DTFD node_0026–0031).
- 50 rows are v4 promotion replications on validation folds 3–4.
- 22 rows are roots.

Per cell: n is completed attempts and Δp is the paired difference from the parent.

| cell | n | base | best node: lift ± SE | keeps | \|Δp\|<SE | \|Δp\|<0.015 | Δp<−SE | Δp>+SE |
|---|---|---|---|---|---|---|---|---|
| longA TITAN (std) | 96 | .6336 | node_0087 +.0749±.0401 | 2 | .54 | .86 | .27 | .15 |
| longC TITAN (devcv) | 96 | .6081 | node_0036 +.0322±.0067 | 43 | .31 | .47 | .03 | .66 |
| v4 ABMIL | 30 | .6458 | node_0018 +.0641±.0200 | 4 | .37 | .47 | .37 | .13 |
| v4 CLAM | 30 | .6843 | node_0017 +.0490±.0113 | 2 | .47 | .60 | .40 | .13 |
| v4 DTFD | 30 | .6719 | node_0028 +.0660±.0272 | 3 | .43 | .63 | .30 | .10 |
| v4 nnMIL | 30 | .6007 | node_0026 +.0327±.0047 | 3 | .30 | .80 | .13 | .53 |
| v4 TITAN | 30 | .6451 | node_0025 +.0399±.0225 | 3 | .47 | .83 | .03 | .37 |
| HNSC ABMIL | 30 | .6744 | node_0031 +.0354±.0137 | 2 | .47 | .90 | .07 | .47 |
| HNSC CLAM | 30 | .6922 | node_0031 +.0307±.0098 | 1 | .33 | .87 | .30 | .37 |
| HNSC DTFD | 30 | .6848 | node_0027 +.0258±.0093 | 9 | .53 | .70 | .03 | .43 |
| HNSC nnMIL | 30 | .7132 | node_0028 +.0282±.0085 | 1 | .40 | .87 | .23 | .37 |
| HNSC TITAN | 30 | .6943 | node_0029 +.0189±.0083 | 1 | .50 | .90 | .30 | .20 |
| v5 KRAS ABMIL | 30 | .6672 | node_0030 +.0451±.0186 | 4 | .57 | .70 | .20 | .20 |
| v5 KRAS CLAM | 30 | .6949 | node_0030 +.0238±.0110 | 2 | .43 | .60 | .47 | .10 |
| v5 KRAS DTFD | 30 | .7014 | node_0020 +.0134±.0056 | 0 | .30 | .60 | .63 | .07 |
| v5 KRAS nnMIL | 30 | .6341 | node_0031 +.0460±.0110 | 7 | .40 | .53 | .27 | .33 |
| v5 KRAS TITAN | 30 | .6336 | node_0024 +.0490±.0264 | 7 | .47 | .63 | .17 | .37 |
| v6 ABMIL | 30 | .7268 | node_0010 +.0250±.0123 | 1 | .50 | .83 | .33 | .17 |
| v6 CLAM | 16 | .7282 | node_0012 +.0244±.0130 | 0 | .50 | .94 | .06 | .44 |
| v6 DTFD | 24 | .7419 | node_0019 +.0209±.0103 | 1 | .54 | .79 | .21 | .25 |
| **total** | **712** | | | **96** | **.437** | **.711** | **.228** | **.309** |

v6 nnMIL (baseline .7199; folds .8074 .7440 .7128 .7135 .6216) and v6 TITAN (baseline .6081)
have no attempts yet.

**v4 promotion replication.** Discovery used folds 0–2; replication used the unused validation
folds 3–4.
- **ABMIL:** did not replicate. Every promoted candidate gained fold 3 and lost fold 4 by 0.06–0.13.
  For example, node_0018 scored .7882/.7902 against the baseline's .7333/.8824.
- **CLAM:** did not replicate. Every candidate lost fold 4; node_0004 (wd 1e-2) scored .8059/.7922
  against .8078/.8333.
- **nnMIL:** null.
- **DTFD:** the high-lr + wd 1e-2 + patience 40 family roughly held. node_0021 scored .8392/.8098
  against .8353/.7941. Plain wd 1e-2 at lr 1e-4 lost about 0.05.
- **TITAN:** the patience-30 candidates gained only on fold 4 (+0.13 to +0.14). This is the
  slow-climber artifact.

## 2. Facts that hold across arms (general MIL-on-small-cohort facts)

**F1. Score anatomy.** `val_auc_smooth` ≈ peak val AUC − a jitter penalty. Baseline penalties
(peak or selected-epoch AUC minus smoothed score):

| design | ABMIL | CLAM | DTFD | nnMIL | TITAN |
|---|---|---|---|---|---|
| v6 (devcv) | 0.028 (0.7545 vs 0.7268) | 0.029 | 0.017 (0.7593 vs 0.7419) | | |
| v5 KRAS | 0.044 (0.711 vs 0.667) | | ~0.028 | ~0.010 | |
| longA | | | | | 0.016 (0.6498 vs 0.6336) |

- The batch-1 attention arms carry about 2–4× the jitter penalty of batch-24 nnMIL.
- On ABMIL v6, the best 5-epoch window was 0.7402. Its 0.013 selection loss was fully
  recovered by EMA (0.7428 of a possible 0.7435).
- The anneal raised the window to 0.7528, a plateau gain of +0.0126.
- On CLAM v6, the AUC level over epochs 3–29 moved at most +0.006 under any change.
- On DTFD v6, the hump height did not move under any single change. The LS + dropout 0.5
  stack raised the peak by only +0.008 (0.7673).

**F2. Weight EMA is the universal cheap base on this metric.** Use a bias-corrected EMA and
swap it in at `before_validation`. Results on devcv:

| arm | decay per step | change | node |
|---|---|---|---|
| ABMIL | 0.998 | +0.016 ± 0.0085, kept | node_0003 |
| CLAM | 0.9995 | +0.0128 ± 0.0050 | node_0003 |
| CLAM | 0.998 | +0.008 | |
| DTFD | 0.998 | +0.0042 ± 0.0034, 4/5 folds | node_0002 |

- With about 300 steps per epoch, the horizon is roughly 1.7–7 epochs.
- Under v4's maximum metric, EMA and anneals lost. This is the metric's doing, not the method's.
- 7 of the 13 "coarse-dead family turned positive" cases in §5.4 are EMA or anneal schedules.

**F3. The 1536→512 projection must learn at full speed (ABMIL, CLAM, DTFD).** Constant-rate
changes to it failed in 13 of 13 attempts (none beat the parent by 1 SE; mean −0.047):

| arm | change | effect | design |
|---|---|---|---|
| DTFD | ×0.2 | −0.038 | devcv |
| DTFD | ×0.05 | −0.065 | devcv |
| DTFD | ×0.1 | −0.079 and −0.075 | std |
| DTFD | frozen | −0.133 | std |
| DTFD | ×2 | −0.027 | std |
| DTFD | ×4 | −0.051 | std |
| CLAM | 0.1× | −0.022 | v4 |
| CLAM | frozen | −0.055 | HNSC |
| CLAM | 1e-5 under 3e-5 | −0.042 | std |
| ABMIL | 0.25× | −0.017 | v4 |

The only positive schedule is gradual unfreezing: HNSC CLAM node_0028 (+0.0115) and node_0031
(+0.0119 over the parent). In DTFD the projection holds 786K of ~920K parameters, and that is
where the signal is learned.

**F4. Perturbing frozen features never helps.**
- Channel or input dropout and noise failed in 0 of 15 attempts across 5 arms. DTFD channel
  dropout 0.3 cost −0.0194 ± 0.0082 on devcv.
- Per-tile noise is averaged away by attention pooling. A shared channel mask removes weak
  signal.
- Changing which tiles get pooled acts on the slide embedding and sometimes helps:
  - CLAM: sub512 on devcv and 256-tile bags on HNSC.
  - DTFD: a 30% subsample on HNSC.
- The same idea was null for DTFD v6 (≤512 and ≤2048 tiles) and for nnMIL.

**F5. Cross-slide bag mixing never helps (0 of 7).**
- PseMix p0.8 produced the same fold pattern on CLAM and DTFD in v6:
  - CLAM: +.045 +.077 +.026 −.044 −.058
  - DTFD: +.042 +.016 −.010 −.070 −.043
- nnMIL same-class mixing (std) cost −0.0423 ± 0.0151. It makes training bags artificially
  separable.

**F6. Optimizer wrappers do not help.**
- SAM, MSAM and random weight perturbation: 0 of 10 beat the parent; mean −0.015. CLAM MSAM
  cost −0.050 and −0.064 on std.
- SAM combined with dropout is incoherent: the two passes see different masks (DTFD node_0024).
- Lookahead: 0 of 11; mean −0.012.
- Gradient clipping or AutoClip: 0 of 4; mean −0.037. DTFD clip 1 cost −0.100; CLAM AutoClip
  cost −0.074.

**F7. A larger effective batch does not help batch-1 arms.**

| arm | change | effect |
|---|---|---|
| DTFD v6 | accumulation 4 | −0.0045 |
| ABMIL v6 | 4-bag accumulation | −0.022 vs platform |
| CLAM v5 | B=32 | −0.012 |
| CLAM v5 | B=64 | −0.025 |

- CLAM v5 accumulation k4 (+0.0176) works the same way as a lower lr and does not add to it.
- nnMIL is the exception: smaller batches help once attention is slowed (§3.4).

**F8. The optimal lr is sharp at native for DTFD and nnMIL, flat for ABMIL.**
- DTFD 3e-5 cost −0.010 on devcv and −0.033 on every std fold.
- nnMIL 1.5e-4 cost −0.029.
- ABMIL: lr 1e-4 ≈ 5e-4 on devcv (−0.0003). 6e-4 cost −0.016.
- A lower lr delays memorization but does not raise the hump. On DTFD v6 node_0003 the peaks
  moved to epochs 12–24 and the plateau stayed level.

**F9. Coupled L2 at 1e-2 is dead.**
- On devcv: ABMIL −0.0012, DTFD on EMA −0.0128.
- On TITAN, coupled decay up to 1.0 is a no-op.
- v4's wd 1e-2 "wins" were jitter-assisted maxima that failed on folds 3–4.

**F10. One fold limits each design, and folds differ in optimal capacity and length.**
- On devcv, fold 3 was beaten by 1/30 ABMIL, 5/16 CLAM and 4/24 DTFD attempts. Fold 0 was
  beaten by 88–100% of attempts.
- On std KRAS, fold 4 was beaten by 17% of ABMIL and 10% of CLAM attempts.
- TITAN: fold 1's best model uses about 2 PCs, while fold 4 needs 18 or more.
- Principle: recipes that let capacity grow over epochs let per-fold early stopping pick the
  right point. Examples are warmup, the spectral curriculum, gradual unfreezing and strong head
  dropout.
- Recipes that compress learning cost the slow folds. Examples are a 20-epoch anneal to 0,
  `num_epochs 40` and a global slow-down.

**F11. Memorization inflates val loss, not val-AUC rank error.**
- CLAM v6: training loss reaches 0 by epochs 15–18, yet the AUC level over epochs 15–39 is at
  least as high as over epochs 3–14.
- DTFD v6: val loss triples by epochs 20–30.
- Do not use val loss as the overfitting signal for AUC.

**F12. The val_bacc guard breaks when logits are compressed.** The minority class (mutant, about
36%) falls under 0.5:

| arm | change | val_bacc change | outcome |
|---|---|---|---|
| DTFD | LS 0.2 + dropout 0.5 | −0.020 | guard failure |
| CLAM | lr 3e-5 | −0.026 | |
| CLAM | LS 0.2 | −0.013 | |
| TITAN | zero start without centring | −0.03 to −0.04 | |
| DTFD | instance subsampling | +0.014 to +0.018 | raises sensitivity |
| DTFD | dropout 0.7 | +0.024 to +0.034 | raises sensitivity |

**F13. Early-stopping traps.**
- Early stopping armed during warmup restores an untrained checkpoint (nnMIL v4; HNSC nnMIL
  fold 2 at epoch 0).
- Any change that slows early learning needs a minimum-epoch stop in `should_stop`.
- Patience × lr ≈ constant. On HNSC TITAN, lr/patience pairs of 1e-4/100, 1.4e-4/71 and
  7e-5/143 each gave +0.017 to +0.019.

**F14. Cadence coupling inflates slow-lr lifts.**
- The 5-epoch smoothing window covers less parameter travel at a lower lr, so slow-lr recipes
  get a measurement bonus.
- longC node_0036 (lr 1e-4) gained +0.032 raw but only about +0.010 to +0.012 when
  travel-matched.
- Its lr neighbours fall monotonically: 2e-4 → .6348, 5e-4 → .6277, native → .6243.

**F15. Some attempts are predictable no-ops.**
- 9 attempts (1.3%) reproduced an earlier node's fold vector exactly: longA node_0097 =
  node_0087; v4 ABMIL node_0013 = node_0002; v4 DTFD node_0018, node_0024 and node_0029;
  v4 nnMIL node_0024; v4 TITAN node_0009 and node_0031; HNSC DTFD node_0021.
- Typical cause: extending training when every fold had already stopped early.

## 3. Per-arm findings

### 3.1 ABMIL

**Native recipe:** Adam 5e-4, coupled L2 1e-4, dropout 0, batch 1, 20 epochs; M/L locked.
**Evidence:** v4 30, HNSC 30, v5 KRAS 30, v6 30 attempts.

**Dead or harmful.**
- On the v6 platform (EMA + cosine), every addition lowered the score:

| addition | change vs platform |
|---|---|
| SAM | −0.014 |
| dropout 0.4 | −0.012 |
| keep 25% | −0.019 |
| bag mixup | −0.024 |
| peak lr 1e-3 | −0.023 |
| LS 0.1 | −0.012 |
| class balance | −0.007 |
| LS + balance | −0.017 |
| within-bag interpolation | −0.015 |
| 4-bag accumulation | −0.022 |

- Platform ablations: removing the warmup cost −0.015; setting wd to 0 cost −0.010.
- At constant lr: 6e-4 cost −0.016; coupled L2 1e-2 −0.0012; lr 1e-4 −0.0003 (devcv).
- v5 KRAS (std):
  - L2-SP pull toward init is monotone harmful (−0.013 to −0.047).
  - Cap 128 and cap 1024 were worse than cap 512.
  - Attention lr ×0.1 cost −0.0117.
- HNSC nulls: bag caps at native lr, L2, EMA at 3e-5, Adam betas, the Breen recipe,
  feature-statistic augmentation.

**Positive.**
- EMA 0.998: halves the jitter (0.0215 → 0.0103) and recovers the selection loss.
- Warmup + per-step cosine to 0 on top of EMA: raises the plateau by stopping the late decline of
  the overfitting folds 1 and 4.
- On HNSC, slower lr (3e-5) plus noise on the slow trajectory helps:
  - dropout 0.6 with 2048-tile bags, mixing and STKIM
  - instance mixup of every instance

**Top 3**

| # | node | design | override | lift ± SE | per fold | val_bacc | flag |
|---|---|---|---|---|---|---|---|
| 1 | v6 node_0010 | devcv | `--policy-variant ema_cos` | +0.0250 ± 0.0123 | +.041 +.052 +.018 −.019 +.034 | +0.059 | Native 20 epochs. Not length-suspect. |
| 2 | v6 node_0003 | devcv | `--policy-variant ema_bc998` | +0.0160 ± 0.0085, kept | +.046 +.015 +.013 −.007 +.013 | | |
| 3 | HNSC node_0031 | HNSC | `--hparams '{"lr": 3e-5, "dropout": 0.6}' --policy-variant bag2048_mix_stkim` | +0.0354 ± 0.0137 | 5/5 folds up | +0.023 | Slower lr within the same 20 epochs: less travel, not more epochs. |

- node_0010 recipe: EMA 0.998 (bias-corrected), a 1-epoch linear warmup from 0.1× over 298
  steps, then a per-step cosine to 0 over 20 epochs. It is the center of a plateau whose
  siblings all scored 0.7460–0.7510: linear decay .7510, EMA 0.999 .7500, cosine-40 .7497,
  peak lr 2.5e-4 .7469, cosine-30 .7460.
- EMA 0.997 (node_0026) scored −0.0003 ± 0.0009 against node_0003.
- Std-fold lifts that came with longer training are suspect:
  - v5 node_0030 `--policy-variant ema9997_cap512 --hparams '{"max_epochs": 40}'`:
    +0.0451 ± 0.0186, fold 4 −0.012, val_bacc −0.014.
  - v5 node_0026 `--hparams '{"lr": 2.25e-5, "max_epochs": 40}'`: +0.0444 ± 0.0406,
    fold 4 −0.111.
  - v4 node_0018 (lr 1e-4, 40 epochs, AdamW wd 25): +0.064. It failed on folds 3–4.

**Interactions.**
- EMA and the anneal stack (+0.009 over EMA). Everything else on top of them subtracts, so the
  platform is not under-regularized.
- Fold 3 keeps rising under constant lr until epoch 19. Every anneal freezes it at 0.70–0.72,
  including cosines over 30 and 40 epochs.
- On v5 std, fold 4 needs high-lr noise: slow learners reach 0.68–0.75 there, against 0.83–0.87
  at native lr.

### 3.2 CLAM-MB

**Native recipe:** Adam 1e-4, coupled L2 1e-5, dropout 0.25 (attention dropout fixed at 0.25),
B=8, weighted sampler, patience 20, stop_epoch 50.
**Evidence:** v4 30, HNSC 30, v5 KRAS 30, v6 16 attempts.

**Broken or inert controls.**
- svm instance and bag losses crash because `topk` is not installed.
- `inst_loss=ce` is a no-op.
- The scheduler seam is inert.
- The clustering loss saturates in epoch 1 (cluster accuracy 1.0), so B is close to inert.

**Dead or harmful.**
- v6 (devcv):

| change | effect |
|---|---|
| SAM | −0.0189 ± 0.0111 |
| input dropout 0.2 | −0.0005 |
| sub30 | −0.0018 |
| dropout 0.5 on EMA | −0.0003 |
| intra-class composition on EMA | −0.0040 |
| lr 3e-5 | +0.0069, val_bacc −0.026 (guard) |

- v5 (std):

| change | effect |
|---|---|
| AutoClip p10 | −0.074 |
| MSAM ρ 0.5 / 1.5 | −0.050 / −0.064 |
| projection at lr/3 | −0.042 |
| decoupled decay to zero | −0.027 / −0.054 |
| decay to init | −0.018 |
| lr 1e-5 | −0.023 |
| B=64 / B=32 | −0.025 / −0.012 |

  STKIM, MIL-Dropout, channel dropout and cosine were null. Over 99 loss spikes, restarts
  averaged −0.005.
- HNSC: coupled L2 is monotone harmful; a frozen projection cost −0.055; a constant slow
  projection split the folds.

**Positive.**
- EMA 0.9995 is the most consistent single change (4/5 folds). The longer horizon beats 0.998.
- On top of EMA, each of these helps on its own:
  - LS 0.2
  - 512 random tiles per training bag
  - `weighted_sample false` with class-balanced loss weights
  - sub128
- HNSC: 256 random tiles plus EMA 0.999 lifts every fold. Adding gradual projection unfreezing
  helps further: 0.1× lr through epoch 11, then a linear ramp to 1× at epoch 20.

**Top 3**

| # | node | design | override | lift ± SE | per fold | val_bacc | flag |
|---|---|---|---|---|---|---|---|
| 1 | v6 node_0012 | devcv | `--policy-variant e95ls20` (EMA 0.9995 + LS 0.2) | +0.0244 ± 0.0130 | +.036 +.065 +.013 −.013 +.021 | −0.013 | Guard-borderline. |
| 2 | v6 node_0010 | devcv | `--policy-variant e95sub512` | +0.0207 ± 0.0072 | +.032 +.041 +.018 +.012 −.000 | +0.006 | Lowest SE of the leaders. |
| 3 | HNSC node_0031 | HNSC | `--policy-variant subemaunfreeze20` | +0.0307 ± 0.0098 | +.057 +.032 −.001 +.044 +.021 | +0.035 | |

- Near tie: v6 node_0016 `--hparams '{"weighted_sample": false}' --policy-variant e95bal`
  scored +0.0203 ± 0.0094.
- v5 std node_0004 `--hparams '{"lr": 3e-5}'` scored +0.0227 ± 0.0165. Fold 4 was −0.042 and
  part of the gain is a lucky late jump on fold 1. This is a slower-learning recipe on std
  folds, so treat it as weak.
- v4 node_0004 (wd 1e-2, +0.0425 on the maximum metric) failed on folds 3–4.

**Interactions.**
- LS 0.2 and sub512 gain on different folds: LS on folds 0/1/4 (fold 3 −0.013), sub512 on
  folds 0–3 (fold 4 flat). This is the same complementary pattern that stacked in DTFD.
- LS, sub512 and balance are untested in combination.
- Accumulation and a lower lr act through the same mechanism and do not add (v5).
- PseMix splits the folds.
- In v6 the epoch 3–29 level barely moves. The gains are jitter damping plus small level gains
  (EMA +0.009 over epochs 3–14).

### 3.3 DTFD-MIL

**Native recipe:** Adam 1e-4 on both tiers, L2 1e-4, numGroup 4, grad_clip 5, lr ×0.2 at
epoch 100, patience 20. Bags have a median of about 13k patches.
**Evidence:** v4 29 + 1 partial, HNSC 30, v5 KRAS 30, v6 24 (+6 running).

**Dead or harmful.**
- Projection lr in any direction (F3).

| change | effect | design |
|---|---|---|
| grad_clip 1 | −0.100 | std |
| lr 3e-5 | −0.033 on every fold | std |
| lr 3e-5 | −0.010 | devcv |
| 2e-4 / 4e-4 high-lr phases | −0.03 to −0.04 | std |
| channel dropout 0.3 | −0.019 | devcv |
| PseMix p0.8 | −0.013 ± 0.020 | devcv |
| wd 1e-2 on EMA | −0.013 | devcv |
| SAM + dropout | | incoherent (node_0024) |
| SAM + LS | fold 3 −0.043 | devcv |

- Null: numGroup 8 or 16 (8 attempts across 4 cells, all null), subsampling ≤512 or ≤2048,
  accumulation 4, SAM alone.

**Positive.**
- EMA 0.998 gives +0.004, tiny but free of trajectory noise.
- Label smoothing ε 0.2 on both tiers (soft target 0.9/0.1 through `train_step`) bounds the
  logits, which makes post-peak memorization milder.
- Dropout on the pooled features before the classifiers (droprate = droprate_2) unlocks long
  plateaus.
- On the LS 0.2 base:

| dropout | score | selected epochs, folds 0–4 |
|---|---|---|
| 0.3 | 0.7571 | 18 12 5 10 34 |
| 0.5 | 0.7591 | 22 11 4 10 34 |
| 0.7 | 0.7628 | 49 40 39 9 38 |

- LS 0.2 ≥ 0.3 ≥ 0.1 at every dropout dose.
- v5 std: one post-peak step helped: `lr_decay_step 9` (×0.2) gave +0.0134 ± 0.0056. A ratio
  of 0.1 cost −0.014 and 0.4 cost −0.007.
- HNSC: a 30% subsample + EMA 0.999 + wd 1e-3 + patience 40 helps, and slower variants lift
  every fold.

**Top 3**

| # | node | design | override | lift ± SE | per fold | val_bacc | flag |
|---|---|---|---|---|---|---|---|
| 1 | v6 node_0019 | devcv | `--policy-variant ema_ls20 --hparams '{"droprate": 0.7, "droprate_2": 0.7}'` | +0.0209 ± 0.0103 | +.031 +.016 +.009 −.006 +.054 | +0.034 | See note below. |
| 2 | v6 node_0022 | devcv | `--policy-variant ema_ls30 --hparams '{"droprate": 0.7, "droprate_2": 0.7}'` | +0.0164 ± 0.0068, kept | | +0.024 | |
| 3 | HNSC node_0030 | HNSC | `--policy-variant sub30_ema999 --hparams '{"wd": 0.001, "lr": 5e-5, "patience": 60}'` | +0.0241 ± 0.0039 | 5/5 folds up | +0.062 | Slower lr with longer patience. |

- node_0019 note: its selected epochs are later (49/40/39/9/38), so it trains longer through
  early stopping. On devcv this is not the std-fold artifact, but compare dropout doses at
  similar selected epochs.
- v4 node_0021 `--hparams '{"wd": 0.01, "lr": 0.0003, "patience": 40, "max_epochs": 100}'`
  scored +0.062 on the maximum metric. It is patience-coupled and roughly held on folds 3–4.

**Interactions.**
- LS and dropout gain on different folds (LS on 0/1/4, dropout on 3/4) and add up to +0.017 to
  +0.021.
- SAM cancels with dropout.
- LS + dropout 0.5 endangers the guard; dropout 0.7 repairs sensitivity.
- Pending runs: dropout 0.8 and 0.9, LS 0.25, a MultiStepLR step at epoch 30 (×0.2), and
  dropout 0.7 without LS.

### 3.4 nnMIL (SimpleMIL)

**Native recipe:** AdamW 3e-4 with decoupled wd 1e-4, cosine over 100 epochs with 10 warmup
epochs, patience 10, batch 24 balanced (9–10 steps per epoch), dropout 0.25.
**Evidence:** v4 30, HNSC 30, v5 KRAS 30. No devcv attempts.

**Arm facts.**
- Gated attention runs on a random 384-of-1536 channel subset. It has 1.18M attention
  parameters against a 3K head.
- Each slide always shows the same ≤3921-patch crop, because the crop uses
  `RandomState(seed+idx)`.
- 16% of slides are zero-padded, and the padding is pooled without a mask.
- `transform_bag` sees only the crop's real rows.
- The submit-time smoke test drives torch Adam, but the trainer uses AdamW (2 groups) +
  cosine_lr + GradScaler. Regroup through `opt.param_groups` / `add_param_group`.
  `cosine_lr` honours a per-group `lr_scale`.

**Dead or harmful.**
- v5 std:

| change | effect |
|---|---|
| lr 1.5e-4 | −0.029 |
| lr 6e-4 | −0.009 |
| Lookahead | −0.028, all folds |
| num_epochs 40 | −0.0125 (slow folds are under-trained) |
| same-class mixing | −0.042 |
| instance dropout on EMA | −0.016 / −0.020 |
| attention ×0.05 at b24 | −0.020 |

  DSU feature shift was an exact null.
- HNSC: frozen attention −0.066; attention ×0.1 at b24 −0.040; EMA 0.98 cuts off the late
  creep; channel dropout 0.5 −0.019.
- v4: early stopping armed during warmup produced untrained checkpoints.

**Positive.**
- Slowed attention × small balanced batches × EMA, with EMA decay min(0.98, (1+t)/(10+t)).
  Attention lr scale against batch size, all with EMA:

| attention lr | b24 | b12 | b8 | b6 | b4 |
|---|---|---|---|---|---|
| 1.0 | .6400 | .6489 | .6424 | .6337 | |
| 0.2 | .6490 | .6577 | .6558 | .6587 | .6695 |
| 0.1 | | | | .6697 | .6801 |

- The two changes are complements: smaller batches hurt at full attention lr and help once
  attention is slowed.

**Top 3**

| # | node | design | override | lift ± SE | per fold | val_bacc | flag |
|---|---|---|---|---|---|---|---|
| 1 | v5 KRAS node_0031 | std | `--policy-variant ema_attn10 --hparams '{"batch_size": 4}'` | +0.0460 ± 0.0110 | +.070 +.061 +.034 +.009 +.056 | +0.025 | See note below. |
| 2 | v5 KRAS node_0029 | std | `--policy-variant ema_attn10 --hparams '{"batch_size": 6}'` | +0.0356 ± 0.0117 | all folds up | | |
| 3 | HNSC node_0028 | HNSC | `--hparams '{"batch_sampler": "balanced", "batch_size": 9}' --policy-variant headwd30` | +0.0282 ± 0.0085 | +.019 +.042 +.048 +.000 +.031 | +0.179 | |

- node_0031 is the only attempt in the corpus that cleared its estimated winner bar by a clear
  margin: +0.046 against 0.027. Two HNSC leaders sit at their bars: ABMIL (+0.0354 vs ≈0.0355)
  and CLAM (+0.0307 vs 0.0314).
- node_0031 note: batch 4 means about 6× more optimizer steps per epoch. That is more travel at
  the same epoch count, on std folds, so it is partly a training-length change and needs
  devcv confirmation.
- node_0029 neighbour: node_0027 (`ema_attn20`, b4) scored +0.0354 ± 0.0093. Six configs with
  attention ×0.1–0.25 and batch 4–12 scored 0.656–0.680.
- node_0028 recipe: AdamW decoupled wd 30 on the head weight only.
- v4 node_0026 (attention slowing, dropout 0.5, wd 10) scored +0.033 on 3 folds and was null
  on folds 3–4.

**Interactions.**
- Attention slowing and small batches only help together.
- Heavy decay combined with a lower lr triggers the epoch-0 trap, so add a minimum-epoch guard.
- EMA 0.98 combined with patience 10 cuts off late creep on HNSC.

### 3.5 TITAN linear probe

**Native recipe:** Linear(768,2), Adam 1e-3, L2 1e-4, batch 32, patience 10, max 200 epochs.
**Evidence:** v4 30, HNSC 30, v5 KRAS 30, longA 96, longC 96 attempts. No v6 attempts.

**Embedding geometry.**
- |mean| ≈ 12.05 (squared 145), which exceeds the total variance (≈91).
- Top eigenvalues: 18.1, 10.3, 8.9, 5.7, 4.2.
- Cumulative explained variance: top-8 .64, top-16 .79, top-32 .91, top-64 .97. The effective
  dimension is about 13.

**Dead or harmful (devcv unless marked).**
- Neutral or null: coupled wd ≤1.0, EMA, input noise or dropout, supervised PC selection,
  winsorizing, soft spectral filters, per-PC Adam, a bagged projector, inner-CV k.

| change | effect | design |
|---|---|---|
| z-scoring | −0.013 | |
| removing PC1 | −0.031 | |
| dropping the top-4 PCs | −0.052 | |
| uncentred zero start | guard failure | |
| SGD on uncentred inputs | −0.021 | HNSC |
| span projection | −0.0044 | HNSC |
| PCA-64 | −0.006 | HNSC |
| SNR gate | −0.011 | HNSC |

**Positive.**
- Centring is the enabler.
- A centred top-k projection gives +0.015 to +0.019, with a k plateau of 18–24.
- The spectral curriculum works because folds need different capacities (F10).

**Top 3**

| # | node | design | override | lift ± SE | per fold | val_bacc | flag |
|---|---|---|---|---|---|---|---|
| 1 | longC node_0082 | devcv | `--policy-variant balcurr18s1` | +0.0295 ± 0.0102 | +.047 +.012 +.025 +.006 +.058 | +0.050 | Native cadence. |
| 2 | longC node_0036 | devcv | `--hparams '{"lr": 1e-4, "patience": 60, "max_epochs": 300}' --policy-variant pczc20` | +0.0322 ± 0.0067 | +.027 +.035 +.019 +.022 +.057 | +0.047 | Slow cadence; see note. |
| 3 | HNSC node_0029 | HNSC | `--hparams '{"lr": 1.4e-4, "patience": 71, "max_epochs": 357}'` | +0.0189 ± 0.0083 | | | Progress-matched slow Adam. |

- balcurr18s1 recipe:
  - Basis: class-balanced covariance 0.5Σmut + 0.5Σwt + 0.25ddᵀ.
  - Training: centred, zero start, curriculum k(e) = min(18, 1+4e) starting from PC1.
  - Validation: the projection is folded into the layer at `before_validation`, and
    `should_stop` projects W onto the current subspace.
  - Robustness: random init .6375, stagewise .6369, cap 22 .6323, +6 PCs per epoch .6328.
- node_0036 note: the raw lift is inflated by the 5-epoch window. It is about +0.012 when
  travel-matched (F14).
- Std-fold lifts that came with longer training are suspect:
  - longA node_0087 `--hparams '{"early_stopping": false, "max_epochs": 400}' --policy-variant pcskip1_hold30_b95_sd10_k136`:
    +0.0749 ± 0.0401; fold 0 +0.138, fold 4 +0.182, fold 3 −0.043. Every candidate with early
    stopping off gains +0.15 to +0.19 on fold 4 because of the patience-10 truncation.
  - v5 node_0024 `--policy-variant fb_nesterov_null_w30`: +0.049, fold 4 +0.149.
  - v4 node_0025 (patience 30, lr 3e-3, `l2warm15`): +0.040.

**Interactions.**
- Centring is required for a zero start, and zero start + centring + PC truncation + curriculum
  stack.
- Early stopping off interacts with the fold-4 truncation on std folds only.
- The curriculum was never tried at slow cadence. Do not combine them: the combination would
  stack the cadence bonus.

## 4. Interactions across arms

**Stacked:**

| arm | stack | result |
|---|---|---|
| ABMIL | EMA + warmup/cosine | +0.009 over EMA |
| DTFD | EMA + LS + pooled dropout | +0.017 to +0.021; complementary folds |
| CLAM HNSC | 256-tile bags + EMA + gradual unfreezing | +0.012 over subema256 |
| nnMIL | slowed attention × small batch × EMA | +0.046 |
| TITAN | centring × zero start × top-k × curriculum | +0.0295 |

**Cancelled or harmful together:**
- SAM × dropout (DTFD).
- Any regularizer on the ABMIL platform.
- Accumulation × lower lr (CLAM): same mechanism.
- Coupled wd × Lookahead (v4 CLAM): wd re-adds jitter and lowers Lookahead's plateau.
- An anneal × slow folds (ABMIL fold 3, nnMIL folds 3–4).
- Heavy decay × lower lr (nnMIL HNSC epoch-0 trap).
- LS × moderate dropout × guard (DTFD).
- Attention slowing at full batch, or small batches at full attention lr (nnMIL).

**Untested combinations with complementary fold profiles:**
- CLAM: LS 0.2 + sub512 + balance.
- DTFD: LS + dropout + the class-balanced weights the agent planned.
- nnMIL: the v5 recipe + headwd30 (HNSC).

## 5. Agent inefficiencies (from master.csv; 712 completed non-root attempts)

**5.1 Mostly null outcomes.**

| design | \|Δp\| < SE | \|Δp\| < 0.015 | worse by > 1 SE | better by > 1 SE |
|---|---|---|---|---|
| overall | 43.7% | 71.1% | 22.8% | 30.9% |
| devcv | 39.8% | 62.7% | 11.4% | 48.8% |
| HNSC | 44.7% | 84.7% | 18.7% | 36.7% |
| std | 44.9% | 69.4% | 29.0% | 21.2% |

- longC inflates the "better by > 1 SE" share because its nodes are parented on the root.
  Excluding longC it is 25.5%; v6 alone is 18 of 70 (25.7%).
- In v6, only 3 of 70 attempts reached Δp ≥ 0.015. Two were kept; the third, DTFD node_0018,
  failed the guard.
- longC reached its best score at attempt 35 of 96. Its top 15 configurations span
  0.6344–0.6403, and none of the last 61 attempts beat attempt 35. Those later attempts did find
  the native-cadence curriculum, which is the more trustworthy recipe.

**5.2 Families that never worked.** 63 attempts (8.8%) went to families with zero attempts
above +1 SE in any arm or cell:

| family | attempts | arms | mean Δp |
|---|---|---|---|
| channel or input dropout | 15 | 5 | −0.007 |
| pseudo-bag count / B | 12 | 2 (CLAM, DTFD) | −0.010 |
| Lookahead | 11 | 3 (CLAM, nnMIL, TITAN) | −0.012 |
| SAM-like | 10 | 3 (ABMIL, CLAM, DTFD) | −0.015 |
| cross-bag mixing | 7 | 4 (ABMIL, CLAM, DTFD, nnMIL) | −0.010 |
| gradient clipping | 4 | 3 (ABMIL, CLAM, DTFD) | −0.037 |
| decay lowered | 4 | 2 (CLAM, DTFD) | −0.006 |

A further 13 attempts were constant-rate projection-lr changes: 0 positive, mean −0.047.

**5.3 Re-tests of settled axes in v6: 19 of 70 attempts (27%).** 15 had same-arm evidence; 4
were DTFD SAM runs with cross-arm evidence. None of the 19 beat its parent by > 1 SE.

| arm | re-tests | axes |
|---|---|---|
| DTFD | 12 of 24 | lr 3e-5, channel dropout, accumulation, numGroup 8/16, projection ×0.2/×0.05, wd 1e-2, SAM ×4 |
| ABMIL | 5 of 30 | SAM ×3, wd 1e-2, peak lr 1e-3 |
| CLAM | 2 of 16 | input dropout, SAM |

**5.4 Coarse "dead" labels misfire.**
- Across all later cells, 55 of 562 attempts (9.8%) went to arm-and-family pairs with at least
  2 earlier tries and no positive.
- 13 of those 55 (24%) were positive in the new cell. Seven were EMA or anneal schedules, which
  had lost under v4's maximum metric.
- Priors must name the exact setting, the metric and the design.

**5.5 Persisting on a dead family within a cell.**
- 60 attempts (8.4%) went to a family already tried at least twice in the same cell with no
  positive. 10 of them turned positive.
- Worst families: module lr 10 (2 positive), lr up 9 (1), Lookahead 6 (0), instance
  subsampling 6 (1), coupled wd 6 (1).

**5.6 Exact no-ops: 9 (1.3%).** See F15.

**5.7 Over-optimistic predictions.**
- Agents predicted a gain for 89% of attempts (mean +0.0103). 52% were positive (mean −0.0003).
  21.5% beat their own prediction.
- Spearman correlation between predicted and observed: 0.44 overall.

| campaign | Spearman | share of ≥ +0.015 predictions that came true |
|---|---|---|
| v4 | 0.25 | 19% |
| v5 KRAS | 0.24 | 31% |
| HNSC | 0.48 | 27% |
| v6 | 0.33 | 23% |
| longA | 0.05 | 0% |
| longC | 0.81 | 67% (root-parented dose ladders) |

**5.8 Repeated research passes.**
- Every cell restarted the literature pass and re-found the same sources: 2311.17466, 2306.16180,
  2411.18704, 2405.09990, 2010.01412, 1907.08610.
- Each v6 cell re-screened 6–8 single axes from scratch.
- The v6 DTFD cell needed 16 attempts (two batches) to reach the LS × dropout region. Eight of
  those 16 repeated directions that earlier DTFD cells had settled: lr 3e-5, channel dropout,
  accumulation, numGroup 8 and 16, projection lr ×0.2 and ×0.05, and wd 1e-2.

**5.9 Longer or slower training.**
- 11 of 37 std keeps and 8 of 14 HNSC keeps involved longer or slower training.
- On std folds, such attempts were positive less often (16% vs 26%). The inflation shows up as
  single-fold wins, such as TITAN fold 4 +0.15 to +0.19, not in the share of wins.

## 6. Playbook draft (agent-facing; validation evidence only; 120-line cap)

```text
# Prior experience: training-recipe search for MIL on small cohorts
Source: ~710 completed attempts in 20 cells (ABMIL, CLAM-MB, DTFD-MIL, nnMIL, TITAN probe; LUAD KRAS, HNSC grade).
Tags: [devcv] inner 5-fold CV, 74-75 val slides/fold (strongest). [HNSC] 3-class grade, 42 val slides/fold.
[std] 47 val slides/fold: trust its negatives, not its positives. Lifts are val_auc_smooth vs the native baseline.
Fixed: architecture, features, seed, measurement. No ensembling. EMA of one run's weights is allowed.

## A. How the score behaves
1. val_auc_smooth = peak val AUC minus a jitter penalty (baseline: ABMIL 0.028, CLAM 0.029, DTFD 0.017 [devcv];
   nnMIL ~0.010, TITAN ~0.016 [std]). EMA recovers most of it; more must raise the plateau (devcv so far +0.006..+0.013).
2. A lower lr inflates the score: 5 epochs cover less parameter travel. TITAN at lr 1e-4: +0.032 raw,
   about +0.012 travel-matched [devcv]. Judge slow recipes by a higher curve, not a flatter one.
3. Paired SE for one change is 0.005-0.015 [devcv]; detectable effect ~0.02; winner bar ~0.029-0.037.
   Single additions usually give +0.005-0.012 and seldom clear the 0.015 keep-bar (v6: 3 of 70).
   The winner rule compares every candidate to the baseline, so a discard is not a loss.
4. Devcv fold 3 resists nearly every change (beaten by 1/30 ABMIL, 5/16 CLAM, 4/24 DTFD tries). Never tune for one fold.
5. Folds differ in best capacity and length (TITAN: fold 1 best with ~2 PCs, fold 4 needs >=18). Recipes that grow
   capacity over epochs (warmup, curriculum, gradual unfreezing, strong head dropout) let per-fold early stopping
   choose. Recipes that compress learning (short anneal to 0, num_epochs cuts, global slow-down) cost the slow folds.
6. Memorisation inflates val loss, not val-AUC rank error. Do not use val loss to diagnose AUC overfitting.
7. Guard (val_bacc): logit-shrinking changes push the mutant class (~36%) under 0.5.
   Examples: LS 0.2 + dropout 0.5 on DTFD (-0.020, fail); CLAM lr 3e-5 (-0.026); CLAM LS 0.2 (-0.013);
   TITAN zero start without centring (-0.03). Subsampling and dropout 0.7 raise sensitivity.
8. Early stopping armed during warmup restores an untrained checkpoint. Give every slowed recipe a
   minimum-epoch stop in should_stop. Keep patience x lr constant when changing lr.
9. Before extending training, check selected epochs against the new cap. 9 attempts were exact no-ops.

## B. Dead in every arm (0 of N attempts beat the parent by > 1 paired SE)
- Channel/input dropout or noise on frozen features: 0/15, 5 arms (DTFD channel drop 0.3: -0.019 [devcv]).
- Cross-slide bag mixing: 0/7. PseMix p0.8 on CLAM and DTFD [devcv]: folds 0-1 +0.02..+0.08,
  folds 3-4 -0.04..-0.07. Same-class mixing on nnMIL: -0.042 [std].
- SAM, MSAM, weight perturbation: 0/10. SAM with dropout is incoherent (different masks per pass).
- Lookahead 0/11. Gradient clipping or AutoClip 0/4 (DTFD clip 1: -0.100; CLAM AutoClip: -0.074 [std]).
- DTFD numGroup 8/16 and CLAM B: inert (0/12).
- Constant-rate changes to the 1536->512 projection lr (slower, faster, frozen) in ABMIL/CLAM/DTFD: 0/13, mean -0.047.
- Larger effective batch for batch-1 arms (accumulation 4, CLAM B 32/64): null to -0.025.
- Coupled L2 1e-2: ABMIL -0.001, DTFD -0.013 [devcv]. On TITAN, decay up to 1.0 is a no-op.

## C. Universal base
- Bias-corrected weight EMA, swapped in at before_validation and restored after.
  ABMIL 0.998/step: +0.016. CLAM 0.9995: +0.013 (beats 0.998). DTFD 0.998: +0.004 [devcv].
  This is mostly jitter removal. Start every MIL arm from EMA, then look for plateau-raisers.

## D. Arms
ABMIL (Adam 5e-4, L2 1e-4, dropout 0, batch 1, 20 epochs; M/L locked)
- Best [devcv]: EMA 0.998 + 1-epoch linear warmup from 0.1x + per-step cosine to 0 over 20 epochs
  (policy ema_cos): +0.025 +- 0.012, folds +.041 +.052 +.018 -.019 +.034, val_bacc +0.059.
- Load-bearing: warmup (-0.015 without), L2 1e-4 (-0.010 at 0). Neutral: linear decay, EMA 0.997-0.999,
  cosine over 30/40 epochs, peak lr 2.5e-4.
- Every addition to that platform lowered it by 0.007-0.024: SAM, dropout 0.4, keep 25%, bag mixup,
  LS 0.1, class balance, within-bag interpolation, 4-bag accumulation, peak lr 1e-3.
- At constant lr, 1e-4 ~ 5e-4 [devcv]; 6e-4 costs -0.016. Fold 3 still rises at epoch 19 under constant lr.
- [HNSC] lr 3e-5: +0.027; adding dropout 0.6 and 2048-tile bags with mixing: +0.035.
- [std] +0.044 gains with 40 epochs or lr <= 4e-5 traded fold 4 (down to -0.11). Treat them as length effects.
CLAM-MB (Adam 1e-4, L2 1e-5, dropout 0.25, B=8, weighted sampler, patience 20, stop_epoch 50)
- Never submit svm losses (topk missing: crash). inst_loss=ce is a no-op; the scheduler seam is inert.
- Best [devcv], each one addition on EMA 0.9995 (none stacked yet):
  LS 0.2 (e95ls20) +0.024 +- 0.013, val_bacc -0.013. 512 random tiles per training bag (e95sub512)
  +0.021 +- 0.007. weighted_sample false + class-balanced loss weights (e95bal) +0.020 +- 0.009.
  Fold profiles: LS helps folds 0/1/4 and loses fold 3; sub512 helps folds 0-3 and is flat on fold 4.
- Null on EMA [devcv]: dropout 0.5, sub30, input dropout 0.2, accumulation 4. SAM -0.019; lr 3e-5 +0.007, val_bacc -0.026.
- [HNSC] 256 random tiles + EMA 0.999: +0.019, all folds up. Adding gradual projection unfreezing
  (0.1x lr to epoch 11, linear to 1x at epoch 20): +0.031.
DTFD-MIL (Adam 1e-4 both tiers, L2 1e-4, numGroup 4, clip 5, lr x0.2 at epoch 100, patience 20)
- Best [devcv]: EMA 0.998 + LS 0.2 on both tiers (soft target via train_step) + droprate =
  droprate_2 = 0.7 (ema_ls20): +0.021 +- 0.010, val_bacc +0.034. LS 0.3 + 0.7: +0.016 (kept).
- Dropout 0.3/0.5/0.7 on LS 0.2 scored 0.757/0.759/0.763, with selected epochs moving later
  (to 49/40/39/9/38). LS 0.2 >= 0.3 >= 0.1 at every dose.
- LS and dropout lift different folds and add. Do not regularise or slow the projection.
- Sharp lr optimum at 1e-4: 3e-5 cost -0.010 [devcv] and -0.033 [std]. Subsampling, accumulation,
  numGroup and SAM were null [devcv].
- [std] A single post-peak step (lr_decay_step 9, x0.2): +0.013. [HNSC] 30% subsample + EMA 0.999
  + wd 1e-3 + patience 40: +0.024.
nnMIL (AdamW 3e-4, decoupled wd 1e-4, cosine 100 epochs with 10 warmup, patience 10, batch 24 balanced, dropout 0.25)
- The submit smoke test uses torch Adam; the trainer uses AdamW (2 groups) + cosine_lr + GradScaler. Test both.
  Regroup by editing opt.param_groups; cosine_lr honours a per-group lr_scale.
- Each slide always shows the same <=3921-patch crop, and 16% are zero-padded and pooled without a mask.
- Best [std, all 5 folds up]: EMA min(0.98,(1+t)/(10+t)) + attention/body lr x0.1, head x1 (ema_attn10)
  + batch_size 4: +0.046 +- 0.011. Six configs with attention x0.1-0.25 and batch 4-12 scored +0.022..+0.046.
- The two changes are complements: attention x0.05 at batch 24 cost -0.020, and batch 6 at full
  attention lr scored below batch 24. Note that batch 4 means ~6x more steps per epoch.
- Harmful [std]: lr 1.5e-4 -0.029; Lookahead -0.028; num_epochs 40 -0.013 (slow folds are
  under-trained); instance dropout -0.016. [HNSC] Frozen attention: -0.066.
- [HNSC] Balanced batch 9 + decoupled wd 30 on the head weight only (headwd30): +0.028, no fold down.
TITAN probe (Linear(768,2), Adam 1e-3, L2 1e-4, batch 32, patience 10, max 200 epochs)
- |mean| 12 exceeds the total spread (variance ~91, mean^2 145). Effective dimension ~13. Centre first.
- Best native-cadence recipe [devcv], balcurr18s1: +0.0295 +- 0.010, all folds up, val_bacc +0.050. Basis: class-
  balanced covariance 0.5S_mut + 0.5S_wt + 0.25dd'; centred, zero start, k(e) = min(18, 1+4e) from PC1, folded at validation.
- Centred top-k projection alone: k 18-24 plateau, +0.015..+0.019. PC1 carries signal (removing it: -0.031).
- Null or harmful [devcv]: decay <= 1.0, EMA, input noise/dropout, z-scoring (-0.013), supervised PC
  selection, spectral filters, per-PC Adam.
- These TITAN priors were selected on the same folds as v6 TITAN. Re-running them is not independent evidence.

## E. Untested training-only directions, highest prior first
1. CLAM: EMA 0.9995 + LS 0.2 + 512-tile bags, then add weighted_sample false + class-balanced weights.
   The components have complementary folds. Watch val_bacc (LS -0.013; subsampling and balance push back).
2. nnMIL [devcv]: ema_attn10 + batch 4 + a minimum-epoch stop. Then add headwd30. Strongest all-fold
   evidence in the corpus, but only on std folds.
3. Gradual projection unfreezing on KRAS: 0.1x to epoch ~10, linear to 1x by ~20, on the arm's EMA stack.
   Order: CLAM, then DTFD, then ABMIL. It is the only projection schedule that ever helped.
4. Label smoothing 0.1-0.2 for nnMIL. Never tried; it lengthened plateaus in DTFD and CLAM.
5. DTFD: LS 0.2 + dropout 0.7 + class-balanced per-bag weights via train_step. Aim: keep the
   sensitivity gain and lower the guard risk. Read the pending dropout 0.8/0.9 results first.
6. ABMIL: keep full lr longer inside the 20 epochs (constant to ~epoch 14, then decay to 0, or cosine
   to a 0.1x floor) on EMA. Low prior: 30/40-epoch cosines did not rescue fold 3.
7. TITAN: start the curriculum from a supervised direction (shrinkage class-mean difference in the
   centred top-18 basis) instead of zero. Low prior.

## F. Process
- Open each arm from its best [devcv] recipe. Skip axes listed in B: v6 spent 19 of 70 attempts
  re-testing them and none beat its parent.
- Write priors as exact setting x metric x design. Coarse "dead axis" labels were wrong 13 times in 55,
  mostly EMA/anneals that lost under a max-AUC metric.
- Predictions ran optimistic: 89% predicted a gain, 52% got one (mean -0.0003). Size every predicted
  effect against the paired SE.
- Prefer composite hypotheses tested against the baseline over dose ladders below the detectable size.
  Keep >= 2 attempts per batch for mechanisms not in this file.
- Restate every hparam and the policy file in each submit; children inherit nothing. Data loading dominates
  (~55-66 s/epoch): extra forward passes (SAM, mixup) are cheap, extra epochs are not (17 h timeout).
```

The playbook block is 118 lines inside the fence.

## 7. Reachable gain ceiling for recipe-only search

| arm | devcv best (lift ± SE) | devcv bar | per-fold oracle, devcv (upward-biased) | other designs | headroom |
|---|---|---|---|---|---|
| ABMIL | +0.025 ± 0.012 (node_0010) | ≈0.037 | +0.039 (fold 3: +0.000) | HNSC +0.035; std +0.045 (length-suspect) | Low. The plateau is reached and every addition subtracts. Realistic +0.025–0.03. |
| CLAM | +0.024 ± 0.013 (node_0012), 16 attempts | ≈0.029 (est.) | +0.039 | HNSC +0.031 | Moderate. Three complementary additions are unstacked and 14 attempts remain. Realistic +0.03. |
| DTFD | +0.021 ± 0.010 (node_0019) | ≈0.029 | +0.034 | HNSC +0.024–0.026 | Low to moderate. The dropout-dose trend is open (6 running). Realistic +0.025. |
| nnMIL | no devcv attempts | — | — | std +0.046 ± 0.011, above bar 0.027; HNSC +0.028 | Highest. Gains raise the plateau and jitter is small. Realistic +0.03–0.04 on devcv, not yet shown. |
| TITAN | +0.0295 ± 0.010 native; +0.032 slow (≈+0.012 travel-matched) | 0.0365 (96 attempts) | +0.048 | HNSC +0.019; std lifts are length artifacts | Low. longC's best came at attempt 35 of 96 and its top 15 span 0.634–0.640. v6 TITAN = longC folds. |

**Gain decomposition (devcv).**
- The removable jitter penalty is 0.017–0.029 per MIL arm. EMA collects most of it.
- Plateau gains seen so far are +0.006 to +0.013.
- A +0.05 lift would need plateau gains of about +0.035, roughly 3× anything observed on devcv.

**Noise floor.**
- At paired SE 0.010–0.012, the best of 30 null attempts is expected to reach about 2.04 × SE ≈
  +0.02 to +0.025.
- The devcv leaders sit at that level. What makes the EMA-family gains credible is that they are
  systematic with respect to the metric: they remove the penalty, replicate across 4–6 siblings,
  and lift 4 of 5 folds. Their size is capped by the penalty.

**Overall.**
- A realistic recipe-only average across the five arms on devcv is about +0.025 to +0.03.
- At most two arms (nnMIL and CLAM) have a plausible path past their winner bar.
- For comparison, HNSC grade leaders reached their bars (ABMIL +0.0354 vs ≈0.0355; CLAM +0.0307
  vs 0.0314; HNSC bars ≈0.030–0.036). On devcv KRAS no leader did.
- A +0.05 average is out of reach for recipe-only search on these cells.

## 8. Risks and open questions

1. **Leakage and selection reuse (TITAN).**
   - The longC baseline folds equal the v6 TITAN baseline folds (.6006 .6052 .5477 .6597 .6273,
     mean 0.6081). Every TITAN devcv prior was selected on exactly the v6 TITAN validation folds.
   - Pool longC's 96 attempts into v6 TITAN's winner bar: Sidak z 3.35 at 126.
   - The C0 held-out set is the v6 test set. The orchestrator-supplied beyond-validation
     confirmation of the C0 configs is test-derived. I kept it out of the report and the
     playbook, and it must never reach an agent-facing surface.
2. **Round-2 reuse on v6 ABMIL, CLAM and DTFD.** The round-1 recipes were selected on the same
   folds. Pool the attempts (z 3.14 at 60). Otherwise the round-2 bar is anti-conservative by
   about 0.2 × SE per arm.
3. **Transfer from std and HNSC.** All nnMIL positives, and the CLAM unfreezing and DTFD
   subsample-EMA positives, come from std folds or the 3-class HNSC grade task. Transfer to KRAS
   devcv is unverified.
4. **Pending data.** The 6 running v6 DTFD attempts may move DTFD's dose conclusion, and v6 CLAM
   still has 14 attempts to spend.
5. **Guard at freeze.** Two leaders lower val_bacc: CLAM e95ls20 (−0.013) and DTFD LS +
   dropout 0.5. A stack can pass on Δ and fail the guard.
6. **Metric dependence.** Every EMA and anneal prior holds only for the 5-epoch smoothed metric.
   Under a maximum metric they invert (v4).
7. **Homogenized search.** A playbook can push every agent to EMA stacks. Section F's reserved
   exploratory slots and the untested list are the counterweight.
8. **Classification error.** Family counts come from regex rules that I audited and refined
   (`analysis/audit1.txt`). Expect ±2–3 per count. The hand-labelled v6 re-test count (19 of 70)
   does not depend on the classifier.

## 9. Alternatives considered

- **A coarse family-level dead list** (what an automatic classifier produces). Rejected:
  - 24% of later re-tests of coarse-dead families were positive.
  - The main cause was v4's maximum metric reversing the sign of smoothing levers.
  - The playbook states exact setting, metric and design instead.
- **Withholding round-1 recipes from round-2 agents**, to keep round 2 independent. It loses:
  - the same cells would spend about 25% of their budget re-deriving them;
  - independence is better restored by pooling attempts in the winner bar.
- **Ranking recipes by raw lift across designs.** Rejected:
  - std-fold lifts include length artifacts (TITAN fold 4 +0.15 to +0.19; ABMIL fold 4 −0.11);
  - slow-lr lifts carry the cadence bonus.
  - I ranked by design, then by fold consistency, then by lift.

## Appendix: analysis artifacts (scratchpad, read-only inputs untouched)

Directory: `/private/tmp/claude-501/-Users-leoyin-Development-autoMIL/7ce1713c-e305-4b7d-9a21-55a231f39b2c/scratchpad/analysis/`

- `enrich.py` → `enriched.csv`: fold values, paired Δ against parent and root, val_bacc.
- `classify.py` → `classified.csv`.
- `refine.py` → `classified2.csv`: refined families.
- `waste.py` and `waste2.py` → `classified3.csv`: waste, no-op, calibration and length statistics.
- `promo.py`: v4 promotion validation folds. It filters held-out keys.
- `top_by_cell.txt`: the top 6 nodes per cell.
- `audit1.txt`: the classifier audit.
