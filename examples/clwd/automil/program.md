# autoMIL - CLWD Lung Adenocarcinoma Subtype Classification

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

The CLWD dataset contains lung adenocarcinoma whole-slide images with 7 histological
subtypes. The task is multiclass classification from WSI-level features using
multiple instance learning (MIL).

## Current target

- **Task:** `subtype` (7-class lung adenocarcinoma subtype classification)
- **Encoder:** `hoptimus1` (1536d)
- **Model:** `clam_mb` (starting point, CLAM framework)
- **Baseline Test AUC:** TBD (run baseline first)
- **Optimization target:** primary_value = (test_auc + test_bacc) / 2

Available `MODEL_TYPE` options:
- **nnMIL models:** `vision_transformer`, `ab_mil`, `trans_mil`, `ilra_mil`, etc.
- **CLAM models:** `clam_sb`, `clam_mb`, `mil_fc`

Training config:
```
learning_rate: 3e-4, weight_decay: 1e-4, dropout: 0.25,
hidden_dim: 512, num_epochs: 100, warmup_epochs: 5,
patience: 10, batch_size: 32, max_seq_length: 4096
```

## Setup

1. **Agree on a run tag** (e.g. `clwd-mar10`). Branch: `autoMIL/<tag>`.
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
The same graph-based tracking, orchestrator submission, and learnings
consolidation apply to this project.

**Prerequisites**: Ensure the orchestrator daemon is running:
```bash
uv run automil orchestrator start
```

**NEVER STOP**: Once the loop begins, do NOT pause to ask if you should continue.
Run until manually interrupted.

**Simplicity criterion**: A small improvement with ugly complexity is not worth it.
Equal results with simpler code is a win.
