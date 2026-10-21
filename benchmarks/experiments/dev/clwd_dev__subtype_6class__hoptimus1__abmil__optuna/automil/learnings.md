# Cell-local learnings: clwd_dev__subtype_6class__hoptimus1__abmil__optuna

## 2026-10-21 the Optuna arm: 60 TPE trials over the recipe menu (validation only, then one sanity read)
Root 88.28% (identical folds to the agent cell's root). Driver `dev_optuna.py run --n-trials 60 --in-flight 4` (no class balancing: six classes); the balanced-accuracy guard as a constraint. 9 of 60 trials failed the guard; the median trial was +1.4%; 56 of 60 were above the root.
- Best of the first 30 and of all 60: t026 +1.8% ± 0.4% = EMA 0.999 + LS 0.05 + MAX_TILES 1024 + COSINE_EPOCHS 40 + MIN_EPOCHS 12 + UNFREEZE (8, 16, 0.1) + lr 8.4e-4 + max_epochs 80. ACCEPTED under the v6 bar (1.6%) and the Sidak bar at 51 candidates (1.7%). The top six sit at +1.8% to +1.9% (t032 the same recipe at lr 5.3e-4, weight_decay 1e-3, native epochs).
- The loop's leader on this cell was EMA 0.998 at +1.5% ± 0.4% (seed-robust, under its bar).
- Seed replicates (cells `__seed43__optuna`, `__seed44__optuna`, roots 88.35% / 88.19%): t026 +1.7% ± 0.4% / +2.0% ± 0.6%, t032 +1.6% / +2.0%, EMA alone +1.5% / +1.9%. Mean over three seeds: t026 +1.8%, t032 +1.8%, EMA +1.6%. The TPE recipe adds 0.2% over EMA alone: nothing.
- Held-out sanity read (split 0's test slides; dev cohort, sanity only): test macro AUC root 88.0 / 88.1 / 88.7% under seeds 42 / 43 / 44; t026 88.3 / 88.8 / 88.7% (mean +0.3%); t032 88.2 / 88.1 / 88.7% (mean +0.1%); EMA 89.1 / 88.1% (mean +0.1%). As for the loop's EMA: the validation gain has no visible test counterpart.
Decision: the cell is CLOSED. Both searches found the same plateau (+1.8% vs +1.6% on validation, +0.3% vs +0.1% on the sanity split).
