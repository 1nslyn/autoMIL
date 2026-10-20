# Cell-local learnings: ccrcc__high_grade__hoptimus1__simple_mil

## 2026-10-19 transfer check (no search): batch 4, −0.006, no effect

Root 0.7105 (folds 0.67, 0.74, 0.73, 0.76, 0.65). With 219 training bags the plan again picks batch 24 (9 steps per epoch), but no fold stops early: the native selections are epochs 10, 10, 5, 21, 11 and the runs end at the patience after 16–32 epochs.

- Batch 4: −0.0060 ± 0.0152, val_bacc +0.003 (guard pass); selections move to epochs 2, 5, 2, 24, 15 with more jitter (0.008–0.035 against 0.003–0.014).

Reading: the batch-size fix is not a cohort-size rule. It recovers the folds the native run leaves at epoch 0 (PBRM1 fold 0, +0.034) and does nothing where every fold already improves within the patience window (here −0.006; the 6-class task +0.007). The recipe therefore says: read the baseline's selected epochs first. One attempt by design; the cell is closed.
