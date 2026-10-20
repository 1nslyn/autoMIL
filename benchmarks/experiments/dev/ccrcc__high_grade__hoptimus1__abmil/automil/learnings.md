# Cell-local learnings: ccrcc__high_grade__hoptimus1__abmil

## 2026-10-19 transfer check (no search): recipe v1 = EMA 0.998, +0.031, kept

Root 0.7246 (folds 0.70, 0.68, 0.66, 0.80, 0.78; 219 training bags). The native curves are noisy on this task: per-fold epoch jitter 0.007–0.090, hindsight gaps up to 0.052, selections at epochs 1–12 of 20.

- EMA 0.998: +0.0309 ± 0.0132, val_bacc −0.016 ± 0.032 (guard pass), kept by the framework. Jitter after EMA 0.003–0.052; four folds up, fold 2 −0.006.

Reading: the recipe transfers with no search and its size tracks the native jitter (PBRM1 +0.008 at jitter 0.003–0.076, the 6-class task +0.015, here +0.031). One attempt by design; the cell is closed.
