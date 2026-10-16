# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__abmil

## 2026-10-16 batch 1 (8 attempts): winner ema_cos_unfreeze, +0.0155

Root 0.8828 macro AUC over 6 classes (strong signal; paired SEs 0.004-0.007, a quarter of CCRCC's).

- EMA 0.998 alone: +0.0112 ± 0.0039. Warmup + cosine alone: +0.0046 ± 0.0036. EMA + warmup + cosine (KRAS v6 leader): +0.0110 ± 0.0051; without warmup +0.0118 ± 0.0040; + 512 tiles +0.0118 ± 0.0044; + label smoothing 0.1 +0.0110 ± 0.0055. Every EMA variant sits at +0.011 to +0.012: EMA is the effect, the additions are neutral.
- EMA + warmup/cosine + projection unfreezing (0.1x to epoch 8, ramp to 1x by 16): +0.0155 ± 0.0068, kept; clears the frozen 2.93-SE bar (0.0143) and the best-of-7 Sidak bar (0.0119). The winner under the protocol's rule.
- HNSC recipe (lr 3e-5, dropout 0.6, 2048 tiles, EMA): +0.0164 ± 0.0046 but val_bacc -0.025, guard fail.

Reading: on a strong-signal morphology task the EMA gain is about +0.011 with a tight SE, and the only piece that adds to it is gradual projection unfreezing (+0.004 more). Decision: replicate root + ema998 + the winner under seeds 43 and 44; batch 2 stacks on the winner (unfreezing dose and schedule-free form, 512 tiles, LS 0.1, EMA dose, dropout 0.6).
