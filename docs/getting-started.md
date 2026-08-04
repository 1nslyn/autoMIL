# Getting Started with autoMIL

autoMIL automates the **research iteration**, not the hyperparameter: a coding
agent reads your codebase, implements **source-level** changes (architecture +
training recipe), and runs each as an isolated reproducible experiment in a
persistent tree that informs later proposals — a different level of automation
from grid/menu AutoML, which only searches configurations within a fixed
pipeline. autoMIL is the framework; your training code is the consumer. The seam
between them is the [training-script contract](training-script-contract.md):
write a `result.json`, honor `SIGTERM`, declare your env vars. Anything
beyond that is up to you.

## Prerequisites

- Python 3.10+
- Git
- A coding agent (Claude Code recommended; Codex / OpenCode / DeepSeek-via-X
  also first-class via per-runtime overlays)
- Optional: NVIDIA GPU(s) with CUDA, or a ROCm system, or just CPU. autoMIL
  detects accelerators at `automil init` time and stamps defaults accordingly.

## Installation

```bash
# Install as a global CLI tool (recommended)
uv tool install git+https://github.com/leoyin1127/autoMIL.git

# Or install from a local clone
git clone https://github.com/leoyin1127/autoMIL.git
cd autoMIL
uv tool install -e .
```

> **Note:** If `automil` is not installed globally, prefix every command with
> `uv run` (e.g. `uv run automil init`). The repo's CLAUDE.md and the in-repo
> `_shared/` skill use the `uv run` form throughout.

### Optional extras

```bash
pip install -e '.[slurm]'         # SLURMBackend (submitit AutoExecutor)
pip install -e '.[ray]'           # RayBackend (raw @ray.remote)
pip install -e '.[examples-iris]' # scikit-learn + pyyaml for the sklearn-iris example
```

## Adding autoMIL to Your Project

`automil init` overlays an `automil/` directory onto an existing git repo.

```bash
cd /path/to/your/project   # must be a git repo

# Auto-detect runtime from existing .claude/, .codex/, .opencode/ dirs
automil init

# Or pin a runtime explicitly
automil init --runtime claude
automil init --runtime codex
automil init --runtime opencode
automil init --runtime deepseek-via-opencode
automil init --runtime deepseek-via-codex
automil init --runtime all                  # install assets for every supported runtime

# CI / smoke-test path: skip hardware probe, use conservative defaults
automil init --no-healthcheck

# Re-render skills/hooks/AGENTS.md after upgrading autoMIL without re-scaffolding
automil init --update
```

`automil init` creates an `automil/` subdirectory with:

- `config.yaml`, project settings (run, files, env, scoring, cap, gate, backend, hardware, registry)
- `program.md`, agent instructions for the experiment loop
- `learnings.md`, accumulated insights (starts empty)
- `variants/`, registered variant modules (Phase 1); empty until your first `automil port-variant`
- `cells/`, cell budget state (Phase 4)
- `orchestrator/`, runtime queue / running / archive / completed directories
- `.gitignore`, excludes runtime files (`graph.json`, `orchestrator/`, `trajectory.jsonl`, `cells/`)

Per-runtime assets are written under `.claude/skills/automil/`,
`.codex/skills/automil-setup/`, etc., depending on selected runtime(s).
`AGENTS.md` is regenerated at the project root.

By default, `automil init` runs `LocalBackend.healthcheck()` to probe
CUDA / ROCm / CPU and writes detected GPU count, VRAM, and concurrency
defaults into `config.yaml`'s `hardware:` and `cap.default_vram_estimate_gb`
sections. Pass `--no-healthcheck` for CI environments without GPUs.

Your existing codebase is untouched. The agent will scope it and determine
what to edit.

## Configuration

After `automil init`, edit `automil/config.yaml`. The fields that you must
touch:

```yaml
run:
  script: "train.py"             # your training script
  command: null                  # optional full override, e.g. "python src/train.py --config config.yaml"

files:
  editable: ["train.py", "models/"]   # files, dirs, or globs the agent may modify
  readonly: ["evaluate.py"]           # files the agent must not touch

env:
  required: []                   # vars that MUST be set before submit; automil check enforces
  passthrough: [AUTOMIL_*]       # vars forwarded into experiment subprocesses

scoring:
  formula: ""                    # documentation-only; describe your composite recipe

baseline:
  composite: 0.0                 # your starting performance

cap:
  # cap is consumer-supplied — 6h is just the autoMIL-paper default, not
  # special. Durations accept 6h / 30m / 90s / 2d (or a bare number = seconds).
  # Change anytime with `automil budget set 6h`. See note below on precedence.
  budget: 6h                     # agent-active working budget
  safety_buffer: 30m             # refuse-new buffer; must be < budget
  mode: agent_active             # bill only while the agent is working (see below)
  idle_grace_seconds: 300        # agent_active: gap after the last action that
                                 # still counts as "working" before the clock pauses

backend:
  name: "local"                  # "local" | "slurm" | "ray"
```

A few notes on each:

- **`run.script` / `run.command`**, the orchestrator runs this script for
  each experiment. Defaults to `train.py`; override with any script name or
  full command.
- **`env.required` is mandatory in v1.0.** `automil check` fails with
  `Missing required env var: <name>` if anything declared here is unset
  in the orchestrator's environment. Empty list is fine for self-contained
  consumers (e.g. sklearn-iris).
- **`env.passthrough`** controls what the orchestrator forwards into each
  experiment subprocess. `AUTOMIL_*` matches all framework variables
  including `AUTOMIL_RUNTIME` (declared, never inferred, D-87).
- **`scoring.formula`** is documentation-only. Your training script
  computes the composite scalar and writes it to `result.json`. State the
  formula here so collaborators can read the recipe at a glance.
- **`cap.budget` / `cap.safety_buffer`** are consumer-supplied durations
  (`6h`, `30m`, `90s`, `2d`, or a bare number of seconds; the legacy
  `cap.budget_seconds` integer keys still work). The framework provides the
  per-cell cap *mechanism*: at `T - safety_buffer` the cell enters
  `refusing-new` (no new submits accepted into this cell); at `T` the cell
  enters `terminating` and SIGTERM is sent to running experiments. The
  precedence chain is `CLI flag > cap.<key> duration > legacy
  cap.<key>_seconds > framework fallback (6h / 30m)`. Set it the easy way with
  `automil budget set 6h` (or `automil budget show` to inspect), or override
  per-cell via `automil submit --budget-seconds N` (D-134; honored ONLY on the
  submit that creates the cell — later submits joining the same cell log INFO
  and keep the established value, preventing sandbagging).
- **`cap.mode`** chooses how the budget clock is metered:
  - **`agent_active`** (default) bills only the time the agent is *actually
    working*. A Claude Code `PostToolUse` hook (installed by `automil init`)
    stamps `automil/.last_action_at` on every tool call; the orchestrator
    advances the budget only while that marker is fresher than
    `cap.idle_grace_seconds`. Because the agent is quiescent while waiting on
    experiments, GPU/wait time does **not** count — you get more proposing time
    per budget. The counter is driven by harness-observed actions, not an
    agent-reported value, so it can't be padded without doing real work.
  - **`wall_clock`** is the legacy continuous clock (counts GPU + idle time
    since cell creation). Use it to reproduce pre-activity-gated behavior.
- **`backend.name`**, `local` works on any machine. `slurm` requires
  `pip install -e '.[slurm]'` and valid SLURM directives (`backend.slurm.directives.partition`,
  `account`, `cpus_per_task`, `mem_gb`). `ray` requires `pip install -e '.[ray]'`
  and a `RAY_ADDRESS` (or local fallback if `backend.ray.allow_local_fallback: true`).

For the full annotated template, see
[`src/automil/templates/config.yaml.j2`](../src/automil/templates/config.yaml.j2).

## Training Script Contract

Your training script must honor the 6 items documented in
[training-script-contract.md](training-script-contract.md):

1. Read `automil/config.yaml` (or honor a `--config` flag).
2. Honor the framework-owned CUDA/ROCm device masks; never override them.
3. Branch on `AUTOMIL_ACCELERATOR` (`cpu`, `cuda`, or `rocm`) and treat
   `AUTOMIL_GPU=0` only as the backward-compatible logical slot.
4. Exit cleanly on `SIGTERM` with a partial `result.json`.
5. Write `result.json` matching `automil/schemas/result.schema.json`.
6. Declared env vars (under `env.required`) must be present at startup.

The minimum valid `result.json` is:

```json
{"composite": 0.912}
```

`composite` is the single scalar the experiment tree uses for ranking
(higher is always better; for loss minimization, negate). All other
fields are optional. Under the **val-firewall**, `metrics` is
validation-only and is what `composite` is computed from; any test metrics go
in a sealed `held_out` block that the orchestrator quarantines under
`archive/<node>/certify/` and reveals once via `automil certify` — the agent
never sees test during search. A full example payload from the autobench
consumer:

```json
{
  "status": "completed",
  "metrics": {"val_auc": 0.870, "val_bacc": 0.810},
  "held_out": {"test_auc": 0.872, "test_bacc": 0.830},
  "composite": 0.840,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500,
  "fold_results": [...],
  "partial": false
}
```

The sklearn-iris consumer writes `{"composite": <accuracy>, "metrics": {"accuracy": 0.95, "f1": 0.94}, "status": "completed"}`.
Both shapes validate against the same schema; the framework imposes no
hardcoded metric keys.

The orchestrator validates this at ingest via JSON Schema. Malformed payloads
transition the node to `crashed` with a schema-location pointer. The
`composite` scalar is the single field the experiment tree uses for ranking
(UCB scoring, composite-dominance gated by the Ladder keep-margin, higher is
always better; for loss minimization, negate).

## Running the Loop

Use **tmux** so processes survive terminal disconnects.

### 1. Start the orchestrator

```bash
tmux new -s orchestrator
automil orchestrator start
# Ctrl-b d to detach
```

The orchestrator runs as a daemon, polls the queue, schedules experiments
across GPUs via best-fit bin packing, drains backend logs to `archive/<id>/run.log`,
validates each `result.json` against the schema at ingest, and updates the
graph.

### 2. Start the visualization dashboard (optional)

```bash
tmux new -s viz
automil viz start
# Open http://localhost:8420 in your browser
# Ctrl-b d to detach
```

Live SSE updates show the experiment tree, promotion-rate metric, and per-cell
budget consumption.

### 3. Launch your coding agent

**Claude Code (recommended):**

```bash
tmux new -s automil
claude --dangerously-skip-permissions   # autonomous mode, no permission prompts
# Then type: /automil-setup (first time) or /automil (subsequent runs)
```

`--dangerously-skip-permissions` lets the agent run unattended. The
`/automil-setup` skill runs once: it scopes the codebase, configures
`automil/config.yaml`, verifies the contract via a 1-minute dry-run gate,
and establishes a baseline. `/automil` then runs the experiment loop.

**Other runtimes:** install via `automil init --runtime <name>` (or
`--runtime all`), then point the agent at `automil/program.md` and tell it
to follow the loop. See [agent-compatibility.md](agent-compatibility.md).

### 4. Monitor progress

```bash
automil status              # quick summary
automil rank                # top-ranked proposals (UCB)
automil cell list           # cell budget state
automil orchestrator status # daemon health
# Open http://localhost:8420 for the 3D dashboard
```

## How Experiments Run

1. The agent edits files in your repo (any files, not just one).
2. Runs `automil submit --node node_0001 --desc "try focal loss" --mil-model clam_mb --files train.py models/clam.py`.
   Optional: `--max-time 60` for seconds-precision timeout, `--budget-seconds N`
   to override the cell budget on cell creation.
3. The CLI snapshots only the changed files into `automil/orchestrator/archive/node_0001/`
   and validates them against `registry.protected` globs (rejects edits to
   shared library code; those must ship as registered variant modules).
4. The orchestrator picks up the spec, creates a git worktree at the base
   commit, and overlays the changed files. Each experiment runs in isolation;
   only its diff is stored.
5. The experiment runs on the assigned backend (local, SLURM, or Ray) in a
   typed CPU/CUDA/ROCm execution slot. The orchestrator owns the matching
   device-isolation variables and maps any accelerator to logical device 0.
6. The agent (and a SIGTERM handler if installed) writes `result.json`.
7. The daemon validates the payload, ingests it, and updates the graph.
8. The agent runs `automil reconcile` to sync state, reads `learnings.md`,
   and designs the next experiment.

## Variants and Reproduction (Phase 1)

For architectural changes that touch shared library code, the v1.0 path is
to register the change as a variant module:

```bash
# After an experiment that edited shared code:
automil port-variant <node_id>           # convert overlay into automil/variants/<parent>/<name>.py
automil refresh-registry                 # regenerate per-kind variants/__init__.py
automil apply <node_id>                  # apply this node's variant selection to config.yaml
automil verify-repro <node_id>           # reproduce via the registry path; assert |actual - expected| < tolerance
```

`registry.protected` in `config.yaml` lists glob patterns the agent's overlay
must NOT touch directly (e.g. `lib/CLAM/**`, `src/models/clam/base.py`).
Variants live as committed code modules; the search-scope mode is `free`
by default (interface + purity validators) or `architecture-preserving`
(adds identity-strict validation per `identity_constraints`).

## Generalization Gate (Phase 5)

A keep-status node can be nominated as a candidate and promoted via a
pre-registered held-out manifest:

```bash
automil nominate <node_id>               # mark as candidate
automil gate register-manifest           # write & git-commit the held-out manifest BEFORE search
automil promote <candidate_id>           # paired Wilcoxon + bootstrap CI + Bonferroni alpha/K
```

Manual nomination is the v1.0 default (`gate.auto_nominate: false`).
Promotion-rate is exposed in viz `/api/promotion-rate` SSE and
`automil status`.

## Trajectory Recording (Phase 3)

Per-submit JSONL trajectories using OpenTelemetry `gen_ai.*` keys are
written to `archive/<node_id>/trajectory.jsonl`. Secrets (`sk-…`, `hf_…`,
`ghp_…`, AWS keys, `*_API_KEY=…`, `*_TOKEN=…`) are redacted on capture; per-event 8 KB
cap, per-file 5 MB soft / 50 MB hard rotate. Trajectories are gitignored
by default. Export a redacted, schema-validated bundle:

```bash
automil trajectory export <node_id> --out trajectory_bundle.tgz
```

## Revealing Held-Out Test Metrics (the val-firewall)

Selection runs entirely on the **validation** composite. Test metrics are sealed
at ingest into a `held_out` block that the orchestrator quarantines under
`archive/<node>/certify/`, out of every agent-facing surface during search. Once
search is done, reveal them exactly once:

```bash
automil certify                 # sealed held-out test for the val-selected best node
automil certify --top-k 3       # or the top-K keep nodes by validation composite
automil certify --node node_0042
```

This is the only sanctioned read of test: the val→test gap is the honest cost of
search, and it must never feed back into selection. Reporting the certified
held-out numbers is what keeps an encoder-vs-aggregator comparison free of
test-selection bias — the rigor the framework exists to provide.

## CLI Reference

| Command | Description |
|---------|-------------|
| `automil init [--runtime <r>] [--no-healthcheck]` | Add autoMIL to current git repo |
| `automil check` | Validate setup (protected files, env.required, backend, registry) |
| `automil show-skill --runtime <r>` | Render merged per-runtime skill file to stdout |
| `automil submit --node <id> --desc "..." --mil-model <m> [--files <f>] [--max-time SEC]` | Snapshot changed files, queue experiment. `--mil-model` is required unless `run.mil_model` is set in `config.yaml` |
| `automil cancel <node_id>` | Cancel a running experiment |
| `automil resubmit <node_id>` | Re-queue a terminal experiment as a new node |
| `automil rank` | Show top-ranked proposals (UCB) |
| `automil propose --parent <id> --desc "..."` | Add a brainstormed proposal |
| `automil reconcile [--recompute-best]` | Sync graph with orchestrator state |
| `automil status` | Show experiment summary |
| `automil port-variant <node_id>` | Convert overlay into a registered variant |
| `automil promote-variant <variant_id>` | Move a gate-passing variant to canonical |
| `automil refresh-registry` | Regenerate per-kind variants/__init__.py |
| `automil apply <node_id>` | Apply a node's variant selection to config.yaml |
| `automil revert-baseline` | Reset registry.protected paths to base_commit |
| `automil verify-repro <node_id>` | Reproduce a node via the registry path |
| `automil cell status [<id>] / list` | Inspect cell budget state and consumed seconds |
| `automil nominate <node_id>` | Mark keep-status node as gate candidate |
| `automil promote <candidate_id>` | Run Stage B gate |
| `automil certify [--node <id>] [--top-k N]` | Reveal sealed held-out TEST metrics for val-selected node(s) — once, post-search |
| `automil gate register-manifest / retire-manifest / status / stats` | Manage / inspect the gate manifest |
| `automil trajectory record / export` | JSONL trajectory capture + redacted export bundle |
| `automil start-loop` / `automil stop-loop` | Control agent loop flag |
| `automil orchestrator start / stop / status` | Manage GPU scheduler daemon |
| `automil viz start / stop / status` | Manage 3D dashboard |

Run `automil <command> --help` for the full flag listing.

## Examples

See [`examples/`](../examples/) for reference configurations:

- [`sklearn-iris/`](../examples/sklearn-iris/), ~80-line second consumer
  demonstrating the contract on a non-autobench pipeline (CPU-only, no env
  vars). Used as the Phase 8 final-acceptance sub-gate B in CI.
- [`ovarian_hrd/`](../examples/ovarian_hrd/), pre-v1.0 autonomous run
  with 189 experiments and accumulated `learnings.md`.
- [`clwd/`](../examples/clwd/), multi-class lung adenocarcinoma subtyping skeleton.
- [`placeholder/`](../examples/placeholder/), minimal template emitted by
  `automil init`.

## Further Reading

- [Training-Script Contract](training-script-contract.md), the seam between framework and consumer
- [Agent Compatibility](agent-compatibility.md), per-runtime setup, overlay merge model
- [Implementation Report](implementation-report.md), v1.0 architecture and design decisions
- [CHANGELOG](../CHANGELOG.md), v1.0 milestone release notes (BREAKING migration paths)
