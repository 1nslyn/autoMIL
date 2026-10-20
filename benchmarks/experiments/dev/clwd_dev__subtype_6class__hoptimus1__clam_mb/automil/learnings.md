# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__clam_mb

## 2026-10-20 batch 1 (8 of 9 attempts complete; EMA + 1024 tiles still running): every EMA variant fails the balanced-accuracy guard

Root 0.8805 macro AUC (folds 0.94, 0.85, 0.83, 0.88, 0.90; paired SEs 0.001–0.005; 3.5–4.4 h per attempt).

- EMA 0.9995: +0.0060 ± 0.0008 in AUC, every fold up, but val_bacc −0.043 ± 0.019: guard fail. The same shift DTFD showed on PBRM1 and the 6-class task; on the binary CLAM cells (PBRM1, high_grade) the guard passed.
- Label smoothing 0.2 alone: +0.0024 ± 0.0049 (guard pass; the leader). 512 random tiles: −0.0031 ± 0.0043, guard fail.
- Stacks on EMA: + LS 0.2 +0.0074, + 512 tiles +0.0070, + LS 0.2 + 512 tiles +0.0085 ± 0.0030, + LS 0.1 +0.0064; all with val_bacc −0.02 to −0.06, all guard fail.
- Offline verdict: one admissible attempt (LS 0.2, +0.0024); baseline kept.

Reading: the KRAS v6 CLAM recipe (EMA + LS 0.2 or 512 tiles, +0.021 to +0.025 on KRAS) is worth +0.007 to +0.009 in AUC on this 6-class task and is excluded by the guard: the averaged weights rank better and classify worse at the fixed threshold. EMA + 1024 tiles pending; then the cell closes.

## 2026-10-20 batch 1 complete (9 attempts): cell closed

- EMA 0.9995 + 1024 random tiles: +0.0070 ± 0.0020 in AUC, val_bacc −0.038 ± 0.021, guard fail, like every other EMA variant here.
- Final offline verdict: one admissible attempt (LS 0.2, +0.0024 ± 0.0049); baseline kept.
- Held-out sanity reading (split 0 test slides, taken once): root 0.873, EMA 0.873, LS 0.2 0.869, the three-piece stack 0.872. Nothing separates them on 80 slides.

Reading: on this 6-class task CLAM's EMA gains are small in AUC (+0.006 to +0.009), excluded by the balanced-accuracy guard, and absent on the single test split. Recipe v1 for CLAM stays EMA 0.9995 for binary tasks, with the operating-point caveat for multi-class. CLOSED.
