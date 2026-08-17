# autoMIL

Autonomous research loop for improving ML models via iterative experimentation.
Adapted from Karpathy's autoresearch concept, with persistent learnings and
robust session restart (inspired by Ralph and EvoScientist patterns).

## Restart protocol

On every session start or context reset:

1. **Run `graph.reconcile()`** to sync graph.json with orchestrator state
   (promotes completed experiments, recovers missing nodes, resets stale running)
2. **Read `graph.json`** via `ExperimentGraph.load()` for tree state, proposals, scoring
3. **Read `learnings.md`** for consolidated insights ("What Works" / "What Doesn't Work")
4. **Read `config.yaml`** for project-specific settings
5. **Read `train.py`** for current best code state
6. **Read `results.tsv`** for reference only (graph.json is the primary tracking)

Then continue the experiment loop from the appropriate step.

## Context

We already have a completed benchmark: 7 pathology encoders x 9 MIL architectures
x 2 tasks (BRCA mutation, HRD status), all on 5-fold stratified CV. The baselines:

| Task | Best Encoder | Best MIL Model     | Test AUC-ROC |
| ---- | ------------ | ------------------ | ------------ |
| BRCA | hibou_l      | ilra_mil           | 0.722        |
| HRD  | hoptimus1    | clam_mb            | 0.865        |

The goal: **beat these baselines** by any means that improve evaluation metrics
without modifying the evaluation methods. This includes improving the training
recipe, feature preprocessing, data augmentation, multi-encoder fusion, custom
model architectures, fusing ideas from existing architectures, or building
entirely new models inline in `train.py`.

### HRD top-10 leaderboard (test AUC-ROC, 5-fold CV)

| Rank | Encoder      | Model          | Framework | AUC   | 95% CI        |
| ---- | ------------ | -------------- | --------- | ----- | ------------- |
| 1    | h_optimus_1  | clam_mb        | CLAM      | 0.865 | 0.823 - 0.907 |
| 2    | uni_v2       | vision_transformer | nnMIL | 0.858 | 0.765 - 0.951 |
| 3    | uni_v2       | clam_mb        | CLAM      | 0.850 | 0.805 - 0.896 |
| 4    | hibou_l      | rrt_mil        | nnMIL     | 0.846 | 0.791 - 0.901 |
| 5    | h_optimus_1  | dtfd_mil       | nnMIL     | 0.845 | 0.757 - 0.934 |
| 6    | h0_mini      | vision_transformer | nnMIL | 0.842 | 0.738 - 0.946 |
| 7    | h_optimus_1  | clam_sb        | CLAM      | 0.841 | 0.772 - 0.911 |
| 8    | h_optimus_1  | ab_mil         | nnMIL     | 0.839 | 0.783 - 0.896 |
| 9    | h0_mini      | clam_sb        | CLAM      | 0.836 | 0.797 - 0.874 |
| 10   | uni_v2       | clam_sb        | CLAM      | 0.834 | 0.782 - 0.886 |

CLAM models hold 5 of the top 10 slots. The overall best (h_optimus_1 + clam_mb,
0.865) outperforms the nnMIL best (uni_v2 + vision_transformer, 0.858) by 0.007.

## Current target

**HRD only.** BRCA is out of scope for now.

- **Task:** `hrd` (HRD status prediction, binary)
- **Encoder:** `hoptimus1` (1536d, best HRD encoder)
- **Model:** `clam_mb` (best HRD MIL model, CLAM framework)
- **Baseline Test AUC:** 0.865
- **Optimization target:** primary_value = (test_auc + test_bacc) / 2

Available `MODEL_TYPE` options:
- **nnMIL models:** `vision_transformer`, `ab_mil`, `trans_mil`, `ilra_mil`, etc.
  (created via `create_mil_model()` from nnMIL model_factory)
- **CLAM models:** `clam_sb`, `clam_mb`, `mil_fc`
  (created from CLAM models; process one slide at a time)

Training config matching the benchmark:
```
learning_rate: 3e-4, weight_decay: 1e-4, dropout: 0.25,
hidden_dim: 512 (hardcoded by model_factory), num_epochs: 100,
warmup_epochs: 5, patience: 10, batch_size: 32, max_seq_length: 4096
```

CLAM-specific CONFIG keys (ignored for nnMIL models):
```
model_size: "small"       # "small" or "big" (CLAM hidden layer sizes)
k_sample: 8               # patches sampled for instance-level eval
bag_weight: 0.7           # weight for bag loss; (1 - bag_weight) for instance loss
instance_eval: True       # enable instance-level supervision (clam_sb/clam_mb only)
```

Note: CLAM models process slides individually in a per-sample loop, so
DataParallel provides no benefit. Use gradient accumulation instead.

Dataset: 206 slides (144 pos, 62 neg), UHN cohort, 5-fold stratified CV.

## Setup

Work with the user to:

1. **Agree on a run tag** (e.g. `hrd-mar10`). Branch: `autoMIL/<tag>`.
2. **Create the branch**: `git checkout -b autoMIL/<tag>` from main.
3. **Read the in-scope files**:
   - This file (`program.md`) for instructions.
   - `config.yaml` for project-specific settings.
   - `prepare.py` (read-only) for data loading, evaluation, constants.
   - `train.py` (editable) for training recipe, preprocessing, augmentation.
4. **Verify features exist**: Check that feature files exist at the path specified in `config.yaml`.
5. **Establish baseline**: Run `train.py` unmodified to confirm the baseline reproduces AUC ~0.858.
6. **Initialize results.tsv**: Create with just the header row.
7. **Initialize state.json and learnings.md**: Create initial versions.
8. **Confirm and go**.

## What you CAN do

Modify files listed under `files.editable` in `config.yaml`. Everything is fair game:

- **CONFIG section**: learning rate, weight decay, dropout, hidden dim, batch size,
  max sequence length, number of epochs, patience, warmup.
- **`preprocess_features()`**: L2 normalization, standardization, PCA, feature
  selection, multi-encoder fusion (load from multiple H5 dirs and concatenate).
- **`augment_batch()`**: stochastic patch dropout, Gaussian noise on features,
  feature-space mixup, random patch masking.
- **`create_loss_fn()`**: focal loss, label smoothing, class-weighted CE.
- **`create_optimizer()`**: SAM, muon, different AdamW configs, separate LR per group.
- **`create_lr_schedule()`**: linear warmup/decay, step LR, no schedule.
- **The training loop** (`train_single_fold`): gradient clipping, accumulation,
  hard instance mining, test-time augmentation.

## What you CANNOT do

- Modify files listed under `files.readonly` in `config.yaml`.
- Change the 5-fold split assignments.

You CAN modify or replace the model. `create_mil_model()` from the model factory
is the starting point, but you are free to define custom architectures inline in
`train.py`, modify model hyperparameters (layers, heads, hidden dims), or swap
in a completely different model. The only constraint is that the evaluation
(splits, metrics) stays fixed.

**Architecture modifications are encouraged.** Beyond hyperparameter tuning and
loss/augmentation changes, you should actively look for deficiencies in the
original model architectures and improve them.
This includes modifying attention mechanisms, adding layers, changing pooling
strategies, or building custom architectures inline in `train.py`. Architecture
improvements are preferred over pure hyperparameter tuning when possible.

## Experiment graph

Experiments are tracked as a directed tree in `graph.json` via the `ExperimentGraph`
class in `graph.py`. Each experiment is a node with a parent edge encoding "built upon."

### Node types
- **Executed**: An experiment that has been run, with metrics and keep/discard status
- **Proposed**: A brainstormed idea anchored to a specific parent config, not yet run

### Scoring
Each node has a `potential` score for prioritization:
- Executed: `primary_value + exploration_bonus(child_count)`
- Proposed: `parent.primary_value + exploration_bonus(siblings) + technique_novelty`
The agent uses `graph.rank_proposals(n=6, max_per_branch=2)` to pick diverse
experiments across multiple branches. Scores are advisory; the agent can override.

### Brainstorm step

When no proposals remain pending, brainstorm new ideas:
1. Review learnings.md for patterns (what worked, what failed, near-misses)
2. Review the graph for under-explored branches and near-miss nodes
3. Research recent MIL literature for new techniques
4. Consider fusing ideas from multiple architectures
5. Add proposals via `graph.add_proposed()` on high-potential branches

### Performance context

DeepHRD (JCO 2024) achieved AUC 0.74 on ovarian HRD from WSI. HRDPath (2025)
achieved AUC 0.846 with a multi-task dual-model architecture. Our current
primary_value of 0.850 is competitive but room for improvement likely exists
through architectural innovation and data augmentation strategies.

## Output format

The script prints a parseable summary:

```
---
val_auc_roc: 0.735000 (+/- 0.0500)
val_bacc: 0.680000 (+/- 0.0400)
val_f1: 0.700000 (+/- 0.0350)
test_auc_roc: 0.720000 (+/- 0.0600)
test_bacc: 0.670000 (+/- 0.0500)
test_f1: 0.690000 (+/- 0.0450)
baseline_delta: +0.013000
elapsed_seconds: 1200.5
peak_vram_mb: 4500.0
---
```

Extract the key metric: `grep "^test_auc_roc:" run.log`

## Logging results

Results are auto-logged by `train.py` to `results.tsv` (tab-separated):

```
commit	val_auc	val_bacc	test_auc	test_bacc	primary_value	delta	vram_gb	elapsed_min	status	description
```

The `status` (keep/discard) and `primary_value` are computed automatically by train.py.

## The experiment loop

Run on a dedicated branch (e.g. `autoMIL/hrd-mar10`).

**Prerequisites**: Ensure the orchestrator daemon is running:
```bash
uv run automil orchestrator start
```

**LOOP FOREVER:**

1. **Run `graph.reconcile()`** to sync with orchestrator state (promotes any
   completed experiments, recovers missing nodes, resets stale running).
2. **Call `graph.rank_proposals(n=6, max_per_branch=2)`** to get top proposals
   across different branches. If no proposals remain, trigger brainstorm step.
3. **Read `learnings.md`** to avoid repeating failed approaches.
4. **Design experiments**: For each selected proposal, generate a modified
   train.py content string with hardcoded `PROJECT_ROOT` and `results_tsv` paths.
   Compute `config_hash` via `ExperimentGraph.compute_config_hash()` and check
   `graph.has_config()` to skip duplicates. Do NOT modify `train.py` yet.
5. **Submit to orchestrator**: Call `graph.mark_running(node_id)` for each proposal.
   Write spec JSON files to the orchestrator queue with `graph_metadata`
   block containing `parent_id`, `techniques`, `config_hash`. Submit up to 6 at once.
6. **Wait for results**: Check the orchestrator completed directory for completion
   files. Use background tasks, not sleep-poll loops.
7. **Read results** from `completed/{node_id}.json`: extract status, primary_value,
   delta, test_auc, test_bacc.
8. **Update graph**: Call `graph.promote(node_id, metrics)` or
   `graph.mark_failed(node_id, status, error)` for each result.
   Call `graph.recalculate_scores()` and `graph.save()`.
9. **Append to `learnings.md`** with experiment insights.
10. If primary_value improved: update `train.py` with the winning config
    and `git commit -m "try: <description>"`.
11. If primary_value is worse or equal: no git changes needed (experiments ran from
    orchestrator archive, not from the working tree).
12. **Every 5 experiments**: consolidate learnings.md.
13. **If no proposals remain**: brainstorm and `graph.add_proposed()` on
    high-potential branches.
14. Repeat.

**GPU Management**: The agent does NOT set `AUTORESEARCH_GPU` or manage GPU
assignment. The orchestrator handles all scheduling via best-fit bin packing.
Set `estimated_vram_gb` in the spec (default 0.5 for CLAM_MB).

**Parallelism**: Submit multiple experiments per round. The orchestrator runs
one per GPU (3 simultaneously) and queues the rest. This maximizes throughput.

**Crashes**: The orchestrator detects OOM, crashes, and timeouts automatically.
Results appear in `completed/` with `status: "oom"`, `"crash"`, or `"timeout"`.

**NEVER STOP**: Once the loop begins, do NOT pause to ask if you should continue.
Run until manually interrupted. If you run out of strategies, trigger the
brainstorm step to research and add new ones.

**Simplicity criterion**: A small improvement with ugly complexity is not worth it.
Equal results with simpler code is a win. A 0.002 AUC improvement from deleting
code? Definitely keep. A 0.002 improvement from 50 lines of hacky augmentation?
Probably not worth it.

## Generalizing to other projects

To use this loop for a different experiment:

1. Copy this example directory to a new location
2. Edit `config.yaml` with your project settings (task, files, metrics, run command)
3. Replace `train.py` and `prepare.py` with your equivalents
4. Clear `results.tsv`, `learnings.md`, and `graph.json`
5. Update `program.md` context sections with your project's background
6. Everything else (hooks, skill, loop logic) works as-is
