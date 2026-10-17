# Cell-local learnings: ccrcc__pbrm1__hoptimus1__simple_mil

## 2026-10-16 batch 1 (8 attempts + identity): batch size 4 kept, +0.034

Root 0.5739 (folds 0.47, 0.59, 0.53, 0.62, 0.66; marginal SE 0.032). The nnMIL plan picks batch 24 for this cohort (205 training bags), which is 9 optimizer steps per epoch; with patience 10 the native run on fold 0 stopped at epoch 11 having selected epoch 0, and the other folds selected epochs 26-37.

- Identity attempt: bit-identical to the root on all five folds (seam path verified on nnMIL).
- Batch 4 alone: +0.0335 ± 0.0196 paired, val_bacc +0.023, every fold but one up (fold 0 +0.09); kept by the framework; clears the best-of-6 Sidak bar (0.030), not the frozen 2.93-SE bar (0.037). Fold 0 now selects epoch 13 of 24. Batch 4 is 51 steps per epoch, six times the native count.
- EMA 0.98 alone: +0.0047 ± 0.0094. Label smoothing 0.1: +0.0022 ± 0.0041.
- Attention/body lr x0.1 (the KRAS v5 piece) reverses the sign: attn10_b4 -0.0095 (guard fail), ema_attn10_b4 -0.0184 (guard fail), ema_attn10_b8 -0.0125, ema_attn20_b6 -0.0067, the v5 recipe + LS 0.1 -0.0204 (guard fail). On this task the body-lr scaling costs about 0.05 against batch 4 alone.

Reading: the native batch size under-trains nnMIL on a small cohort (too few steps per epoch for the epoch-based patience); the first real recipe finding of the loop. Decision: replicate root + batch 4 under seeds 43 and 44 first, then batch 2: batch 4 + EMA 0.98, batch 8, batch 2, and batch 4 with lr 1.5e-4 (sqrt rule) to separate the step count from the effective learning rate.

## 2026-10-17 seed replicates (cells `__seed43`, `__seed44`; same folds, training seed only): batch 4 holds

Root five-fold mean by seed: 0.5739 / 0.5734 / 0.5897 (SD 0.0093). Batch 4 paired lift by seed: +0.0335 / +0.0276 / +0.0404 (mean +0.0338); val_bacc +0.023 / +0.033 / +0.045. Fold 0, the fold the native batch size left at epoch 0, gains +0.09 / +0.14 / +0.10; one fold per seed moves down by 0.004 to 0.020.

Reading: the lift is three to four times the root's seed spread and has the same sign on every seed, so batch size 4 is the nnMIL recipe for cohorts of this size (about 200 training bags), independent of the training seed. Batch 2 (submitted 10-17 01:57, one job at a time): batch 4 + EMA 0.98, batch 8, batch 2, batch 4 with lr 1.5e-4.
