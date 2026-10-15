# Cell-local learnings: ccrcc__pbrm1__hoptimus1__abmil

## 2026-10-15 bag-cache identity check (before any attempt)

ABMIL devcv fold 0, native recipe, run twice outside the orchestrator on aihub GPU 0 (RTX 6000 Ada), each under its own `AUTOMIL_RESULTS_DIR`: without `AUTOBENCH_BAG_CACHE` (407 s) and with the cache at `/tmp/automil-bagcache` (63 s). Same prediction hash `b2c5ddd6…b6107`, same fold metrics (val_auc_smooth 0.4921, val_auc 0.5292, val_bacc 0.5757). The cache changes nothing but the read time, so every ABMIL and DTFD attempt in the dev cells runs with it. Note for later: this one native fold sits at chance; the five-fold root decides whether PBRM1 carries enough signal to measure a recipe on.

## 2026-10-16 batch 1 (8 attempts, all discarded)

Root 0.6115 (folds 0.49, 0.65, 0.59, 0.71, 0.62; marginal SE 0.035). Fixed 20-epoch runs; the native run selects epoch 0 on fold 4 and epoch 4 on fold 0, so two folds carry almost no usable validation signal.

- EMA 0.998 alone: +0.0073 ± 0.0086 paired. It cut the per-fold epoch jitter from 0.003-0.076 to 0.002-0.026; that is the whole effect.
- KRAS v6 leader (EMA + 1-epoch warmup + cosine to 0): +0.0015 ± 0.0149. The schedule adds nothing here. Warmup + cosine without EMA: -0.0096, guard fail. EMA + cosine without warmup: +0.0033 ± 0.0172.
- 512-tile bags on the EMA stack: -0.0225 ± 0.0182. Class-balanced weights: -0.0013. Projection unfreezing: -0.0306, guard fail. HNSC recipe (lr 3e-5, dropout 0.6, 2048 tiles, EMA): +0.0013 ± 0.0302.
- Paired SEs run 0.009-0.030 (KRAS devcv: about 0.01), so nothing here can clear the keep bar; the best-of-6 Sidak bar is 0.044.

Decision: before batch 2, replicate root + ema998 + ema_cos_nowarm under seeds 43 and 44 (cells `__seed43`, `__seed44`) to measure the seed noise floor of this task and whether EMA's +0.007 persists. Batch 2 (EMA dose, directions not yet tested) is designed once nnMIL and CLAM batch 1 read out.
