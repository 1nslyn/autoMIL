# program.md, sklearn-iris consumer

Minimal autoMIL second consumer (DEC-02). Demonstrates that the framework
plugs into a non-autobench training script via the documented contract.

## What the agent searches

The agent explores improvements to a sklearn `LogisticRegression` baseline
on the iris dataset. Mutations the agent may try:

- swap classifier (e.g. SVM, RandomForest) under
  `automil/variants/classifier_v0/`
- adjust hyperparameters (max_iter, C, solver)
- engineer features (e.g. polynomial expansion, scaling)

## Primary_value definition

`primary_value == accuracy` on a 30% held-out test split (seed=42). Higher is
better. Baseline primary_value ~0.95 with `LogisticRegression(max_iter=200)`.

## Constraints

- CPU-only; no GPU or env vars required.
- 60s budget per cell.
- Variants live under `automil/variants/classifier_v0/`.
