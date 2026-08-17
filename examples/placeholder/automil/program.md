# autoMIL - Project Template

Autonomous research loop for improving ML models via iterative experimentation.

## Restart protocol

On every session start or context reset:

1. **Run `graph.reconcile()`** to sync graph.json with orchestrator state
2. **Read `graph.json`** via `ExperimentGraph.load()` for tree state, proposals, scoring
3. **Read `learnings.md`** for consolidated insights
4. **Read `config.yaml`** for project-specific settings
5. **Read `train.py`** for current best code state
6. **Read `results.tsv`** for reference only

Then continue the experiment loop from the appropriate step.

## Context

TODO: Describe your benchmark results and baseline performance here.

## Current target

- **Task:** TODO
- **Encoder:** TODO
- **Model:** TODO
- **Baseline Test AUC:** TODO
- **Optimization target:** primary_value = val_auc (the VALIDATION selection signal — test metrics are sealed in `held_out` and never drive selection)

## Setup

1. **Agree on a run tag**. Branch: `autoMIL/<tag>`.
2. **Create the branch**: `git checkout -b autoMIL/<tag>` from main.
3. **Read the in-scope files**: `program.md`, `config.yaml`, `prepare.py`, `train.py`.
4. **Verify features exist**: Check that feature files exist at the path specified in `config.yaml`.
5. **Establish baseline**: Run `train.py` unmodified to get baseline metrics.
6. **Initialize results.tsv, learnings.md, graph.json**.
7. **Confirm and go**.

## What you CAN do

Modify files listed under `files.editable` in `config.yaml`:
- CONFIG section, preprocess_features(), augment_batch(), create_loss_fn(),
  create_optimizer(), create_lr_schedule(), the training loop, model architecture.

## What you CANNOT do

- Modify files listed under `files.readonly` in `config.yaml`.
- Change the fold split assignments.

## The experiment loop

See the main autoMIL documentation for the full experiment loop protocol.

**Prerequisites**: Ensure the orchestrator daemon is running:
```bash
uv run automil orchestrator start
```

**NEVER STOP**: Once the loop begins, do NOT pause to ask if you should continue.
Run until manually interrupted.

**Simplicity criterion**: A small improvement with ugly complexity is not worth it.
Equal results with simpler code is a win.
