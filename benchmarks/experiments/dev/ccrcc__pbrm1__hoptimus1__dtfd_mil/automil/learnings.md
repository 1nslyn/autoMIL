# Cell-local learnings: ccrcc__pbrm1__hoptimus1__dtfd_mil

## 2026-10-16 batch 1 (8 attempts, all discarded)

Root 0.6044 (folds 0.50, 0.61, 0.59, 0.71, 0.61; marginal SE 0.033).

- Every EMA variant fails the val_bacc guard: ema998 (-0.029 bacc), ema_ls20 (-0.038), ema_ls20_d07 (-0.046), ema_ls20_d05 (-0.046), ema_step9 (-0.054), while their AUC holds (ema_ls20_d07 would be +0.0099 ± 0.0167). EMA weights shift the operating point at the fixed decision threshold on this task; on KRAS the same recipe passed the guard. The one EMA attempt that passes carries class-balanced bag weights (ema_ls20_d07_bal, +0.0015 ± 0.0173, bacc -0.016): the weighting offsets the shift.
- Without EMA: dropout 0.7 alone +0.0050 ± 0.0145 (leader), LS 0.2 + dropout 0.7 +0.0029 ± 0.0200. Dropout 0.5 inside the EMA stack: -0.0175.
- Nothing approaches the keep bar; the KRAS v6 leader recipe does not transfer.

Decision: replicate root + ema998 + d07 under seeds 43 and 44 to see whether the guard failure of EMA is systematic on this task and to measure the seed noise floor. Batch 2 after nnMIL and CLAM batch 1 read out.

## 2026-10-16 seed replicates (seeds 43, 44; same folds, training seed only)

Root by seed: 0.6044 (42), 0.5994 (43), 0.6051 (44); SD 0.003 across seeds, so DTFD's baseline is seed-stable here (ABMIL's is not).
- EMA 0.998: +0.0072 (42, guard fail), +0.0023 (43), +0.0012 (44); mean +0.004. val_bacc delta -0.029, -0.007, -0.003: the guard failure was largest under seed 42 and is not systematic in size, but the sign is consistent.
- Dropout 0.7: +0.0050, -0.0060, +0.0080; mean +0.002, not distinguishable from zero.
Reading: no piece gives DTFD a measurable gain on this task. Batch 2a (submitted): EMA 0.998 + class-balanced bag weights (does the weighting rescue the guard, as in ema_ls20_d07_bal), EMA 0.995, dropout 0.8, dropout 0.7 + balanced weights.
