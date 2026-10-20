# Cell-local learnings: ccrcc__high_grade__hoptimus1__dtfd_mil

## 2026-10-19 transfer check (no search): EMA 0.998 + lr ×0.2 at epoch 9, +0.011, under the margin

Root 0.7299 (folds 0.63, 0.76, 0.66, 0.83, 0.77; selections at epochs 0–10, jitter 0.02–0.05).

- EMA 0.998 + lr_decay_step 9: +0.0107 ± 0.0071, val_bacc −0.0015 ± 0.016 (guard pass, unlike PBRM1 and the 6-class task where every DTFD EMA variant failed it). Under the keep margin (0.015); discarded. Jitter after EMA 0.003–0.052.

Reading: the AUC gain of EMA on DTFD is +0.003 to +0.011 across three tasks; whether the balanced-accuracy guard passes depends on the task. One attempt by design; the cell is closed.
