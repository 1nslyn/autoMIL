---
name: automil-setup
description: Set up autoMIL in an existing project. Scopes codebase, configures experiment framework, validates setup.
---

# autoMIL Setup

One-time setup that prepares a project for autonomous experimentation. The skill
inspects the user's repo, drafts `automil/config.yaml` + `automil/program.md`,
scaffolds a `variants/` skeleton, picks defaults from a hardware probe, and
validates the result through a mandatory check + 1-minute dry-run experiment
before printing "Setup complete."

## Architecture

autoMIL overlays onto an existing git repo. Key concepts:

- **automil/ directory** can live anywhere in the repo (a subdirectory or the root).
  The framework finds it by walking up from cwd looking for `automil/config.yaml`.
- **File paths** in `files.editable`, `files.readonly`, and `automil submit` are
  **relative to the git repo root**, not to where automil/ lives. The agent edits
  files anywhere in the repo.
- **Worktrees** are full repo checkouts created from the git root. Overlaid changes
  land at the correct paths because file paths are repo-root-relative.
- **run.command** executes from the worktree root (= git repo root). Use repo-relative
  paths in the command.
- **Hardware report (D-189)** is a report, not a decision. The skill prints detected
  GPU count + per-GPU VRAM and asks the user if anything looks wrong before stamping
  defaults.

## Steps

The skill runs in this order. Steps 1 and 2 are interactive; steps 3+ run autonomously
with operator confirmation at each ambiguous decision point.

### 1. Confirm placement and entry point

Before doing anything, ask the operator:
- Where should the `automil/` overlay live? (default: project root)
- What single command runs ONE training experiment end-to-end? (NOT a grid runner)
- Is there a baseline in existing results to populate, or should the skill submit
  the unmodified code as `node_0001`?

### 2. Run hardware probe

```bash
cd <project_root>
automil init
```

`automil init` invokes `LocalBackend.healthcheck()` between the existing `--update`
guard and template render. The probe order is CUDA, then ROCm, then CPU; the report
is printed before any defaults stamp. If detection fails AND the operator declines
conservative defaults via `click.confirm`, init aborts with a recovery hint.

For CI / smoke-test paths, pass `--no-healthcheck` to skip the probe and use
conservative defaults (max_concurrent_per_gpu=4, default_vram_estimate_gb=8.0).

### 3. Scope the codebase

Read the project structure thoroughly using the Inspection Heuristics (next section).
Identify the training entry point, the model architecture files, the training loop,
data loading, evaluation, and configuration files. The skill never executes user
code during inspection; AST parse + regex grep only.

### 4. Draft scaffolding

Per the Drafting Conventions section, generate `automil/config.yaml`,
`automil/program.md`, and a `automil/variants/` skeleton (one starter variant per
discovered model class, marked with `# TODO: implement`).

### 5. Apply idempotency check

If any drafted artifact already exists, run the Idempotency Protocol (next-next
section) before writing. Never silently overwrite. Never silently skip.

### 6. Setup-done gate

Run the validation gate (see Setup-Done Gate section). Both `automil check` and
the 1-minute dry-run submit must pass before the skill prints "Setup complete."

### 7. Establish baseline

If results already exist, populate the baseline from those metrics (no re-run
needed). Otherwise, the agent loop's first run becomes `node_0001`.

## Inspection Heuristics

Heuristics in priority order (D-193). Single match = autonomous; multiple matches
or zero matches = ask the operator.

### Heuristic 1: training script discovery

Glob the repo for `train.py`, `main.py`, `run.py`, `training/*.py`, `scripts/train*.py`.
Single match = use it. Multiple matches = ask the operator to pick. Zero matches =
ask the operator to provide the path.

### Heuristic 2: framework detection

Read the first 50 lines of the chosen training script. Grep for `import torch`,
`import tensorflow`, `import jax`, `import sklearn`, `import lightning`. Report
detected framework. If none match, mark `framework: unknown` in the drafted
config and proceed.

### Heuristic 3: model class detection (AST-walk)

Parse the training script with `ast.parse` (NEVER `importlib.import_module`,
NEVER `exec`). Walk top-level `ClassDef` nodes. Check `bases` for one of:
`nn.Module`, `Module`, `torch.nn.Module`, `tf.keras.Model`, `Model`,
`BaseEstimator`, `pl.LightningModule`, `LightningModule`. Single match =
autonomous. Multiple matches = ask the operator. Zero matches = mark
`model_class: unknown` and proceed with framework label only.

Walk DOES NOT recurse into imports. The skill walks only the file the operator
pointed at. The user's intent is captured by which file they pointed at, not by
the import graph.

### Heuristic 4: env-var detection

Grep the training script + entry-point modules for `os.environ.get(...)` and
`os.environ[...]`. List discovered keys. Ask the operator which are required.
The drafted `automil/config.yaml: env.required` records the answer; `automil check`
validates the keys are present at runtime.

### Heuristic 5: result.json adapter check

Does the training script write `result.json`? Grep for `result.json`,
`peak_vram_mb`, `composite`. If zero matches, mark "result.json adapter required"
and emit an example adapter snippet in `program.md`. The training script either
writes `result.json` directly or wraps an upstream output via the snippet.

## Drafting Conventions

The skill drafts EXACTLY these artifacts (D-192). It does NOT run experiments,
choose hyperparameters, or modify the training script.

### automil/config.yaml

Stamp values from the hardware probe (D-191):

- `cap.default_vram_estimate_gb`: from `numpy.quantile(.95)` of the `vram_gb`
  column in `results.tsv` if it has at least 10 rows; otherwise
  `max(8.0, min(gpu_vram_gb) / 8.0)`. Note the column name is `vram_gb` (the
  orchestrator converts MB to GB at write time).
- `cap.max_concurrent_per_gpu`: derived from the empirical or conservative
  estimate above; floored at 1.
- `hardware.accelerator`, `hardware.gpu_count`, `hardware.min_vram_gb`: stamped
  for operator visibility (D-191 says stamp values, not comments; operator can
  edit afterwards).

### automil/program.md

A repo-inspection summary: training entry point, what it does, where logs land,
which env vars are required, whether result.json is written natively or via an
adapter snippet.

### automil/variants/ skeleton

One starter variant per discovered model class, located at
`automil/variants/<model_class>/<model_class>_v0.py`, marked `# TODO: implement`.
Forces the user-agent to discover the variant lattice via interactive search
rather than pre-bake.

## Idempotency Protocol

Re-running `/automil-setup` on an already-initialised project (D-194):

1. Detect existing files. If `automil/config.yaml` is present, load it via
   `yaml.safe_load`. If absent, write the drafted file directly (no diff needed).

2. Compute the value-tree diff using the OQ-4 stdlib path:

```python
import yaml, difflib, pprint
existing = yaml.safe_load(existing_text) or {}
drafted = yaml.safe_load(drafted_text) or {}
existing_repr = pprint.pformat(existing, sort_dicts=True, width=120).splitlines()
drafted_repr = pprint.pformat(drafted, sort_dicts=True, width=120).splitlines()
diff = list(difflib.unified_diff(existing_repr, drafted_repr,
                                 fromfile="existing", tofile="drafted", lineterm=""))
```

This compares parsed dict structures, NOT textual YAML. Comments and key ordering
are ignored, which is the correct behaviour: comments are preserved on disk,
diff surfaces only meaningful changes.

3. For each top-level key where `existing[k] != drafted[k]` (sections include
   `run`, `data`, `encoders`, `baseline`, `files`, `metrics`, `training`, `cap`,
   `hardware`), present the diff for THAT subtree and ask the operator:
   `[k]eep existing | [o]verwrite | [m]erge interactively | [s]how full diff`.

4. If `existing == drafted` value-tree-wise, this is a silent no-op. Idempotency
   invariant: same inputs produce zero on-disk changes (mtime advance is
   acceptable; byte-equal content is required).

Never silently overwrite. Never silently skip.

## Setup-Done Gate

Per D-195 the skill runs both stages. Both must pass before printing
"Setup complete."

### Stage 1: automil check

```bash
automil check
```

If exit code is non-zero, surface the specific failures and abort. Common
issues: protected files dirty, env vars missing, registry inconsistent. Fix
each issue and rerun the gate.

### Stage 2: 1-minute dry-run experiment

```bash
automil submit --node node_setup_validation --desc "setup-validation" --files <minimal-edit-set> --max-time 60
```

The `--max-time 60` flag (added in plan 07-02) caps the experiment at 60 seconds
wall-clock; the local backend rounds up to the 1-minute floor. The training
script's responsibility to honour the cap; if it cannot, this gate emits a
warning before submit.

Poll the orchestrator until terminal:

```bash
for i in $(seq 1 18); do
    sleep 5
    automil status | grep -q "node_setup_validation.*\(executed\|crashed\)" && break
done
```

The 90-second polling budget (18 iterations of 5-second sleep) matches D-195.

If the node reaches `executed` (NOT `crashed`), the gate passes. On `crashed`,
investigate the failure (typically: training script raises early, env vars
missing, paths wrong) and fix BEFORE printing "Setup complete." A "crashed"
result does NOT count as "done."

## Failure Modes

The skill is interactive and refuses ambiguity. Per Pitfall 9 from
`research/PITFALLS.md`, six mitigations are applied:

### Multiple training-script candidates

If glob (Heuristic 1) returns multiple matches (e.g. monorepo with `train.py`
AND `scripts/train.py`), the skill MUST ask. Picking the first match is a known
mis-scaffold mode for monorepos.

### Multiple model class candidates

If AST walk (Heuristic 3) returns multiple `nn.Module` subclasses, the skill
MUST ask. Picking the first defined class often yields the wrong target on
projects with helper modules.

### Zero training scripts found

If the repo has no Python file matching the glob, the skill exits with a
helpful message: "Cannot detect a training entry point. Provide one of: a
single-file train.py, a `automil/run.command` value pointing at the entry,
or a custom adapter writing result.json. See program.md template for the
contract."

### Drafted config contains the literal substring TODO

If the drafted config or program.md contains the literal string `TODO`, abort
the setup. `automil check` will reject it; surface the substring and the file
path now rather than waiting.

### Setup-done gate failure

If `automil check` exits non-zero OR the 1-minute submit reaches `crashed`,
the skill MUST NOT print "Setup complete." Print the specific failure and
the recovery action.

### Reasoning trace

Write a short reasoning trace to `.planning/setup-trajectory.md` (created on
demand) noting which heuristic matched the training script, which model class
was detected, which env vars were detected, and what user inputs were collected.
This is the audit trail for debugging "scaffold is wrong, why?"
