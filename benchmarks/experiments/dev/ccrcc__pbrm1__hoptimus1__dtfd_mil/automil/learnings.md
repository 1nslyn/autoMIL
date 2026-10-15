# Cell-local learnings: ccrcc__pbrm1__hoptimus1__dtfd_mil

## 2026-10-16 batch 1 (8 attempts, all discarded)

Root 0.6044 (folds 0.50, 0.61, 0.59, 0.71, 0.61; marginal SE 0.033).

- Every EMA variant fails the val_bacc guard: ema998 (-0.029 bacc), ema_ls20 (-0.038), ema_ls20_d07 (-0.046), ema_ls20_d05 (-0.046), ema_step9 (-0.054), while their AUC holds (ema_ls20_d07 would be +0.0099 ± 0.0167). EMA weights shift the operating point at the fixed decision threshold on this task; on KRAS the same recipe passed the guard. The one EMA attempt that passes carries class-balanced bag weights (ema_ls20_d07_bal, +0.0015 ± 0.0173, bacc -0.016): the weighting offsets the shift.
- Without EMA: dropout 0.7 alone +0.0050 ± 0.0145 (leader), LS 0.2 + dropout 0.7 +0.0029 ± 0.0200. Dropout 0.5 inside the EMA stack: -0.0175.
- Nothing approaches the keep bar; the KRAS v6 leader recipe does not transfer.

Decision: replicate root + ema998 + d07 under seeds 43 and 44 to see whether the guard failure of EMA is systematic on this task and to measure the seed noise floor. Batch 2 after nnMIL and CLAM batch 1 read out.
