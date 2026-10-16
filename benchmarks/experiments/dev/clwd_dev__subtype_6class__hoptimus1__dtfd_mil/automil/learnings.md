# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__dtfd_mil

## 2026-10-16 batch 1 (8 attempts, all discarded)

Root 0.8856 macro AUC over 6 classes (paired SEs 0.002-0.007).

- EMA 0.998 alone: +0.0028 ± 0.0020 but val_bacc -0.015, guard fail. As on CCRCC, EMA lowers DTFD's balanced accuracy at the fixed threshold while AUC holds: the interaction is systematic across tasks.
- EMA + lr x0.2 from epoch 9: +0.0040 ± 0.0026, guard pass; the leader, 1.5 SE, far under the bars (frozen 0.0158, best-of-7 0.0132).
- EMA + LS 0.2: +0.0028 ± 0.0047; + dropout 0.5: +0.0016; the KRAS v6 leader (EMA + LS 0.2 + dropout 0.7): -0.0022 ± 0.0059. Dropout 0.7 alone: -0.0119 ± 0.0068 (bacc +0.025). EMA + dropout 0.7: -0.0026.

Reading: DTFD's native recipe (lr decay, dropout, pseudo-bags) leaves nothing for a training-only recipe to collect on either task; the only consistent effect is EMA's small AUC gain paid for in balanced accuracy. Cell closed at 8 attempts.
