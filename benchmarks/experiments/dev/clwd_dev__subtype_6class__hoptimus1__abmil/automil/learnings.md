# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__abmil

## 2026-10-16 batch 1 (8 attempts): winner ema_cos_unfreeze, +0.0155

Root 0.8828 macro AUC over 6 classes (strong signal; paired SEs 0.004-0.007, a quarter of CCRCC's).

- EMA 0.998 alone: +0.0112 ± 0.0039. Warmup + cosine alone: +0.0046 ± 0.0036. EMA + warmup + cosine (KRAS v6 leader): +0.0110 ± 0.0051; without warmup +0.0118 ± 0.0040; + 512 tiles +0.0118 ± 0.0044; + label smoothing 0.1 +0.0110 ± 0.0055. Every EMA variant sits at +0.011 to +0.012: EMA is the effect, the additions are neutral.
- EMA + warmup/cosine + projection unfreezing (0.1x to epoch 8, ramp to 1x by 16): +0.0155 ± 0.0068, kept; clears the frozen 2.93-SE bar (0.0143) and the best-of-7 Sidak bar (0.0119). The winner under the protocol's rule.
- HNSC recipe (lr 3e-5, dropout 0.6, 2048 tiles, EMA): +0.0164 ± 0.0046 but val_bacc -0.025, guard fail.

Reading: on a strong-signal morphology task the EMA gain is about +0.011 with a tight SE, and the only piece that adds to it is gradual projection unfreezing (+0.004 more). Decision: replicate root + ema998 + the winner under seeds 43 and 44; batch 2 stacks on the winner (unfreezing dose and schedule-free form, 512 tiles, LS 0.1, EMA dose, dropout 0.6).

## 2026-10-16 seed replicates and batch 2: EMA is the recipe, +0.015

Seeds 43 and 44 (same folds): root 0.8835 and 0.8819 (SD across the three seeds 0.0008). EMA 0.998 alone: +0.0112 (42), +0.0150 (43), +0.0185 (44), mean +0.015. The batch-1 winner (EMA + warmup/cosine + unfreezing 8-16): +0.0155, +0.0121, +0.0170, mean +0.015. The unfreezing's +0.004 margin under seed 42 was seed noise.

Batch 2 (seed 42, on the EMA stack): unfreezing 4-12 +0.0172 ± 0.0038 and EMA 0.999 +0.0173 ± 0.0069 (both kept by the rule), unfreezing without the schedule +0.0134, projection at 0.3x +0.0149, EMA 0.995 +0.0132; on top of the winner, 512 tiles +0.0019 ± 0.0037, label smoothing 0.1 -0.0016 ± 0.0027, dropout 0.6 +0.0005 ± 0.0023. The offline verdict names ema_cos_unfreeze_sub512 (+0.0174, pooled SE 0.0056) the winner under the 2.93-SE bar, but every member of the family lies within +0.013 to +0.017 and the additions are within ±0.003 of their parent.

Reading: on this task the whole training-only gain is weight EMA, +0.015 with a seed-to-seed spread of 0.004; schedule, unfreezing, subsampling, smoothing and dropout move it by less than 0.003 either way. Cell closed at 17 attempts; recipe v1 for ABMIL on a strong-signal task = EMA 0.998-0.999 (warmup/cosine optional).
