# Cell-local learnings: ccrcc__pbrm1__hoptimus1__abmil

## 2026-10-15 bag-cache identity check (before any attempt)

ABMIL devcv fold 0, native recipe, run twice outside the orchestrator on aihub GPU 0 (RTX 6000 Ada), each under its own `AUTOMIL_RESULTS_DIR`: without `AUTOBENCH_BAG_CACHE` (407 s) and with the cache at `/tmp/automil-bagcache` (63 s). Same prediction hash `b2c5ddd6…b6107`, same fold metrics (val_auc_smooth 0.4921, val_auc 0.5292, val_bacc 0.5757). The cache changes nothing but the read time, so every ABMIL and DTFD attempt in the dev cells runs with it. Note for later: this one native fold sits at chance; the five-fold root decides whether PBRM1 carries enough signal to measure a recipe on.
