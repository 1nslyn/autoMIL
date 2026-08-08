# autoMIL Experiment Loop

## Project: OvaBench

Autonomous research loop for improving ML models via iterative experimentation.

## Phase 1: Setup (run once)

Before starting experiments, the agent must complete setup:

### 1. Scope the codebase

Read the project structure and identify:
- The training script (update `automil/config.yaml` field `run.script`)
- Model architecture files
- Data loading code
- Loss functions, optimizers, augmentation code
- Evaluation and metrics code

### 2. Configure automil/config.yaml

Fill in ALL fields:
- `run.script`: the training script filename (e.g., "train.py", "main.py", "src/train.py")
- `data.*`: paths to features, splits, and metadata
- `encoders.*`: available encoders and their dimensions
- `baseline.*`: your starting model and its performance
- `files.editable`: list of files the agent may modify (e.g., ["train.py", "models/clam.py"])
- `files.readonly`: list of files that should not be modified (e.g., ["prepare.py", "data/*.py"])

### 3. Ensure the result.json contract

The training script MUST write a `result.json` file to its working directory
before exiting. This is the only contract between your code and autoMIL.

Required fields:
```json
{
  "status": "completed",
  "metrics": {
    "val_auc": 0.870,
    "val_bacc": 0.810
  },
  "held_out": {
    "test_auc": 0.872,
    "test_bacc": 0.830
  },
  "composite": 0.851,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500
}
```

If the training script crashes before writing result.json, the experiment
is marked as crashed. The agent should verify this contract works by
checking the training script's output.

### 4. Establish baseline

Run the unmodified training script to confirm it produces valid results.
Use `automil submit` to record this as the baseline experiment:

```bash
uv run automil submit --node node_0001 --desc "baseline" --files <training_script>
```

### 5. Validate setup

Run `automil check` to verify everything is configured correctly.

## Phase 2: Experiment Loop (runs forever)

**Prerequisites:** `automil orchestrator start`

> The current loop shape (Research → Diagnose → Plan → Execute), mode rules,
> and budget rules are defined by the `/automil` skill; this file records
> project-specific facts only.

**LOOP FOREVER:**

1. Run `automil reconcile` to sync graph with orchestrator state
2. Run `automil rank` to get top proposals. If none, brainstorm new ones.
3. Read `automil/learnings.md` to avoid repeating failed approaches.
4. For each selected proposal:
   a. Edit project files to implement the idea
   b. Run `automil submit --node <id> --desc "..." --files <changed files>`
   c. Clean up only the changes created for that proposal. Do not use
      destructive restore commands that may discard unrelated local work.
5. Wait for completion notifications in `automil/orchestrator/completed/`
6. Read results, update graph via `automil reconcile`
7. Update `automil/learnings.md` with insights
8. If composite improved: commit the winning changes
9. If no proposals remain: brainstorm and use `automil propose`
10. Repeat

**NEVER STOP** while `.automil_active` exists.

## Restart Protocol

On every session start or context reset:

1. Run `automil reconcile` to sync graph with orchestrator state
2. Read `automil/graph.json` for experiment tree state
3. Read `automil/learnings.md` for accumulated insights
4. Read `automil/config.yaml` for project settings
5. Read the training script and key source files
6. Continue the experiment loop from step 2

## Environment Variables

The orchestrator provides these to your training script:
- `CUDA_VISIBLE_DEVICES`: physical GPU ID (masked)
- `AUTOMIL_GPU`: logical device, always `0` (use this or `cuda:0`)
- `AUTOMIL_NODE_ID`: experiment node ID (for logging)
- `AUTOMIL_DESC`: experiment description

## Keep/Discard Rules

The framework decides keep/discard on the single **validation** composite , 
never test:
- **Keep**: child's composite beats the parent's by more than the
  predeclared `accept_margin` δ (`graph.json` `meta.scoring.accept_margin`,
  default 0.0 = strict dominance)
- **Discard**: composite is within the margin of, or worse than, the parent

Optimize the **validation** composite only. Test is sealed into a
`held_out` block and must NOT be referenced, inspected, or optimized during
the loop, it is revealed exactly once, at the end, via `automil certify`.

The training script does not make this decision. It only writes raw metrics
to `result.json`.

## Debugging

- Experiment logs: `automil/orchestrator/archive/<node_id>/run.log`
- Orchestrator log: `automil/orchestrator/orchestrator.log`
- Completion notifications: `automil/orchestrator/completed/<node_id>.json`
- Graph state: `automil/graph.json`
