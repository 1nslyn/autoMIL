# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__simple_mil

## 2026-10-18 batch 1 (8 attempts): batch 4 is the best piece again, +0.007; nothing kept; cell closed

Root 0.8898 macro AUC (folds 0.94, 0.86, 0.83, 0.89, 0.93; paired SEs 0.002–0.005). With 254 training bags the plan again picks batch 24 (11 steps per epoch), but this task is not under-trained: the native folds select epochs 26–65 and the epoch jitter is 0.001–0.004.

- Batch 4 alone: +0.0071 ± 0.0036, every fold at or above the root, val_bacc +0.046; selected epochs move earlier (15–37). Under the keep margin (0.015), so discarded by the framework; the best of the eight.
- EMA 0.98: −0.0010 ± 0.0018 (nothing to smooth). Label smoothing 0.1: +0.0028 ± 0.0030.
- The KRAS v5 pieces: body lr ×0.1 with batch 4 −0.0044, the v5 recipe −0.0018, with batch 8 −0.0046, body lr ×0.2 with batch 6 +0.0014, the v5 recipe + LS 0.1 −0.0006. All within noise or below.
- Offline verdict: leader +0.0071, pooled SE 0.0036, frozen bar 0.0104, best-of-8 bar 0.0100: baseline kept.

Reading: the batch-size effect has the same sign on both tasks but its size follows the under-training: +0.034 where the native run selected epoch 0 on a fold (CCRCC), +0.007 where every fold already trains past epoch 25. Recipe v1 nnMIL (batch 4 at the native lr) stands; on a well-trained cohort expect +0.005 to +0.01, not +0.03. No second batch: nothing else moved and each attempt costs 4.4 h on this cohort. CLOSED.
