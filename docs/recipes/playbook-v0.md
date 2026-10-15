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
