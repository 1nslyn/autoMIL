---
name: automil-setup
description: Set up autoMIL in an existing project. Scopes codebase, configures experiment framework, validates setup.
---

# autoMIL Setup

One-time setup that prepares a project for autonomous experimentation.

## Architecture

autoMIL overlays onto an existing git repo. Key concepts:

- **automil/ directory** can live anywhere in the repo (a subdirectory or the root).
  The framework finds it by walking up from cwd looking for `automil/config.yaml`.
- **File paths** in `files.editable`, `files.readonly`, and `uv run automil submit` are
  **relative to the git repo root**, not to where automil/ lives. This allows the
  agent to edit files anywhere in the repo.
- **Worktrees** are full repo checkouts created from the git root. Overlaid changes
  land at the correct paths because file paths are repo-root-relative.
- **run.command** executes from the worktree root (= git repo root). Use
  repo-relative paths in the command.

## Steps

### 1. Ask the user

Before doing anything, ask the user:
- Where should the automil/ overlay live?
- What training script or command runs a single experiment?
- Are there existing results to use as a baseline?

### 2. Initialize

Navigate to the target directory and run init:

```bash
cd <target_directory>
uv run automil init
```

This creates `automil/` with config.yaml, program.md, learnings.md, and
orchestrator directories. Works from any subdirectory of a git repo.

### 3. Scope the codebase

Read the project structure thoroughly. Identify:

- **Training entry point**: the script or command that trains one model and
  evaluates it. This must be a single-experiment command, NOT a batch/grid runner.
- **Model architecture**: files defining the model (layers, attention, pooling)
- **Training loop**: files controlling loss, optimizer, training logic
- **Data loading**: dataset classes, data loaders, preprocessing
- **Evaluation**: metrics computation, cross-validation setup
- **Configuration**: hyperparameters, constants, config files

### 4. Ensure single-experiment execution

autoMIL runs **one experiment at a time** — the agent makes a code change,
submits it, and the orchestrator trains once to measure the effect.

The training command must:
1. Run a single training run (not a grid/sweep)
2. Write `result.json` to the working directory before exiting

If the project only has a batch/grid runner, create a single-experiment
entry point that calls the existing pipeline for one configuration and
writes result.json.

Required result.json schema:
```json
{
  "status": "completed",
  "metrics": {
    "val_auc": 0.870,
    "val_bacc": 0.810,
    "test_auc": 0.872,
    "test_bacc": 0.830
  },
  "composite": 0.851,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500
}
```

### 5. Configure automil/config.yaml

Update every field. File paths in `files.editable` and `files.readonly` must
be **relative to the git repo root**:

- `run.command`: full command to run one experiment (repo-root-relative paths)
- `run.script`: set to `null` when using `run.command`
- `data.*`: paths to features, splits, metadata
- `encoders.*`: available encoders and dimensions
- `baseline.*`: best existing performance metrics
- `files.editable`: repo-root-relative paths to files the agent may modify
- `files.readonly`: repo-root-relative paths to files that must not change
- `metrics.*`: what metrics to track and how composite is computed
- `training.*`: current hyperparameter values (for agent reference)

### 6. Validate

```bash
uv run automil check
```

Fix any issues reported. All checks should pass before starting experiments.

### 7. Establish baseline

If results already exist, populate the baseline from those metrics (no re-run
needed). Otherwise, submit the unmodified code as the first experiment:

```bash
uv run automil submit --node node_0001 --desc "baseline" --files <editable_files>
uv run automil reconcile
uv run automil status
```

### 8. Done

Setup is complete. Use `/automil` to start the experiment loop.
