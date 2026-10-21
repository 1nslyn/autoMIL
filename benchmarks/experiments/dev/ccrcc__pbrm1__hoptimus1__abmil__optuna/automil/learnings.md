# Cell-local learnings: ccrcc__pbrm1__hoptimus1__abmil__optuna

## 2026-10-21 the Optuna arm: 60 TPE trials over the recipe menu (validation only, then one sanity read)
Root 61.15% (identical folds to the agent cell's root). Driver `dev_optuna.py run --n-trials 60 --in-flight 4 --positive-share 0.446`;
menu `dev_optuna_space.py`; the balanced-accuracy guard as a constraint. 27 of 60 trials failed the guard; the median trial sat 0.8% below the root; 20 of 60 were above it.
- Best of the first 30 (the agent's budget): t026 +1.0% ± 2.6% (13 guard-passing in the pool); no bar cleared.
- Best of 60: t042 +3.1% ± 2.2% = EMA 0.995 + MAX_TILES 2048 + UNFREEZE (4, 12, 0.1) + lr 2.4e-4 + weight_decay 1e-3 + dropout 0.25 + max_epochs 80 (native 20).
  The whole top six is this family at +2.1% to +3.1% (t037 +2.6% ± 1.1% with cosine 40 and lr 6.7e-4). Not past the v6 bar (7.9%: the pooled SE over 33 candidates) nor the Sidak bar at 33 (8.0%).
- The loop's leader on this cell was EMA 0.998 at +0.8% ± 0.9%. The loop never varied ABMIL's epoch cap (20); every TPE top trial trains 40 or 80 epochs, so the gap is a search-space difference (training length), not a smarter search.
- Seed replicates (cells `__seed43__optuna`, `__seed44__optuna`, fresh roots 57.2% / 56.3%): t042 +2.5% ± 3.1% / +6.0% ± 3.0%, t037 +1.4% / +4.0%, t026 +1.6% / +4.2%, EMA alone +1.5% / +0.2% (guard fail under seed 44). Mean over three seeds: t042 +3.9%, t037 +2.7%, t026 +2.3%, EMA +0.8%. The family is seed-robust on validation.
- Held-out sanity read (split 0's own test slides, about 80, SE about 4%; dev cohort, recorded as a sanity check only): test AUC root 51.9 / 54.5 / 56.7% under seeds 42 / 43 / 44; t042 56.5 / 57.6 / 54.5% (+4.6, +3.1, −2.2; mean +1.8%); t037 46.3 / 53.6 / 55.7% (mean −2.5%); t026 50.8 / 54.7 / 54.2% (mean −1.1%); EMA 54.8 / 56.6% (mean +0.1%). The test split of this task is at chance for every model: nothing found on validation can be confirmed or refuted here.
Decision: the cell is CLOSED. For the comparison: at the agent's budget TPE equals the agent (+1.0% vs +0.8%); with twice the budget and the epoch cap in the menu it finds a seed-robust +3% to +4% validation family whose test counterpart is invisible on a chance-level test split.
