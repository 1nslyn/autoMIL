# Cell-local learnings: ccrcc__pbrm1__hoptimus1__clam_mb

## 2026-10-16 identity attempt (node_0002)

The all-off composite recipe (every switch at its default, `--policy-variant identity --hparams '{}'`) reproduced the root bit for bit: all five fold prediction hashes and all fold values equal the root's (0.5703 0.5913 0.7046 0.6509 0.6453; 0.6325). The policy seam path is the identity on CLAM, so every lift measured through a recipe policy is the recipe's own effect.

## 2026-10-16 batch 1 (8 attempts + identity, all discarded)

Root 0.6325 (folds 0.57, 0.59, 0.70, 0.65, 0.65; marginal SE 0.024). The native run's per-fold epoch jitter is 0.04-0.10, the largest of the four arms on this task.

- EMA 0.9995 alone: +0.0071 ± 0.0094 paired, guard pass. The only positive attempt.
- Label smoothing 0.2 alone: -0.0299 ± 0.0178, guard fail. 512-tile bags alone: -0.0254 ± 0.0155.
- The KRAS v6 leader (EMA + LS 0.2): -0.0419 ± 0.0336, guard fail. EMA + 512 tiles: -0.0393. EMA + balanced weights without the weighted sampler: -0.0293 (bacc -0.076). The three-piece and four-piece stacks: -0.0536 and -0.0298, guard fail.
- Every attempt but EMA alone lowered val_bacc by 0.04-0.08.

Reading: on this task label smoothing and bag subsampling hurt CLAM outright, and the KRAS stacks inherit that loss; EMA is again the whole recipe effect, and it does not collect the large jitter here (the smoothed-vs-raw gap stays). Decision: this cell is closed at 9 attempts (the EMA dose study is deferred while the machine runs one job at a time); the development effort moves to CLWD, a task with a strong signal.
