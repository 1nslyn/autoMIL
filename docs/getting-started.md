# Getting Started with autoMIL

autoMIL automates the research iteration of an ML project. A coding agent reads
your codebase, implements source-level changes (architecture and training
recipe), and runs each change as an isolated, reproducible experiment in a
persistent tree that informs later proposals. Grid and menu AutoML only search
configurations inside a fixed pipeline; autoMIL searches over the code itself.

autoMIL is the framework and your training code is the consumer. The two meet at
the [training-script contract](training-script-contract.md): write a
`result.json`, exit cleanly on `SIGTERM`, and declare the environment variables
the script needs.

## Prerequisites

- Python 3.10 or newer
- Git
- A coding agent. Claude Code is the primary runtime. Codex, OpenCode, and
  DeepSeek (through OpenCode or Codex) are also supported; see
  [Other runtimes](#other-runtimes).
- Optional: NVIDIA GPUs with CUDA, a ROCm system, or CPU only. `automil init`
  detects the accelerator and writes matching defaults.

## Installation

```bash
# Install as a global CLI tool (recommended)
uv tool install git+https://github.com/1nslyn/autoMIL.git

# Or install from a local clone
git clone https://github.com/1nslyn/autoMIL.git
cd autoMIL
uv tool install -e .
```

> **Environment convention:** work inside the autoMIL source repository with
> `uv run automil ...`. The bare `automil` commands below assume the global
> install from `uv tool install` and are meant for your own project.

Optional backends are extras. Install them into the same tool environment:

```bash
uv tool install "automil[slurm] @ git+https://github.com/1nslyn/autoMIL.git"   # SLURM backend (submitit)
uv tool install "automil[ray] @ git+https://github.com/1nslyn/autoMIL.git"     # Ray backend
```

With `uv tool install` the orchestrator runs your script with the tool's own
Python unless `run.command` is set, so set `run.command` to the command that
runs your script in its own environment (for example `uv run python train.py`).

## Adding autoMIL to your project

`automil init` overlays an `automil/` directory onto an existing git repository.
Your code is not modified.

```bash
cd /path/to/your/project   # must be inside a git repo

automil init                       # detect the runtime from .claude/, .codex/, .opencode/
automil init --runtime claude      # or pin one (see "Other runtimes" for the full list)
automil init --no-healthcheck      # skip the hardware probe (CI, machines without GPUs)
automil init --update              # re-render skills, hooks, and AGENTS.md after upgrading autoMIL
```

`--update` also re-renders `config.yaml` from the template unless you pass
`--no-healthcheck`, which replaces your edits in that file. Use
`automil init --update --no-healthcheck` to keep `config.yaml` as it is.

Init creates:

- `automil/config.yaml`: project settings (run, files, env, scoring, cap, gate, backend, hardware, registry)
- `automil/program.md`: instructions for the agent's experiment loop
- `automil/plan.md`: the agent's plan for the current batch of experiments
- `automil/learnings.md`: accumulated insights, empty at the start
- `automil/variants/`: registered variant modules, empty until your first `automil port-variant`
- `automil/orchestrator/`: the `queue`, `running`, `archive`, and `completed` directories
- `automil/.gitignore`: excludes runtime files such as `graph.json`, `orchestrator/`, `sessions/`, and trajectories
- `AGENTS.md` at the project root, plus per-runtime assets (for Claude: `.claude/skills/`, `.claude/hooks/`, `.claude/settings.json`)

Cell budget state is created lazily under `automil/cells/` on the first submit,
not by `init`. Unless you pass `--no-healthcheck`, init probes CUDA, ROCm, or CPU
and writes the detected GPU count, VRAM, and concurrency into the `hardware:` and
`orchestrator:` sections of `config.yaml`, including
`orchestrator.default_vram_estimate_gb`.

## Configuration

Edit `automil/config.yaml`. The fields you normally set:

```yaml
run:
  script: "train.py"             # your training script
  command: null                  # optional full command, e.g. "python src/train.py --config config.yaml"

files:
  editable: ["train.py", "models/"]   # files, dirs, or globs the agent may modify
  readonly: ["evaluate.py"]           # files the agent must not touch

env:
  required: []                   # variables that must be set; `automil check` enforces this
  passthrough: [AUTOMIL_*]       # variables forwarded into experiment subprocesses

scoring:
  formula: ""                    # see below
  accept_margin: 0.0             # predeclared keep-margin
  se_multiplier: 1.0

cap:
  budget: 6h                     # working budget per cell
  safety_buffer: 30m             # must be shorter than budget
  mode: agent_active             # or wall_clock
```

The annotated template is
[`src/automil/templates/config.yaml.j2`](../src/automil/templates/config.yaml.j2).

- **`env.required`**: `automil check` fails with `Missing required env var:
  <name>` if a listed variable is unset in the orchestrator's environment. An
  empty list is fine for self-contained scripts.
- **`env.passthrough`**: variables forwarded into each experiment subprocess.
  `AUTOMIL_*` covers all framework variables, including `AUTOMIL_RUNTIME`, which
  you declare and autoMIL never infers.

### Scoring and keep/discard

The framework recomputes `primary_value` from the validation `metrics` block at
ingest and does not trust the value the script reports. `scoring.formula` picks
the reducer:

- `""` (default) or `mean`: the mean of all validation metrics
- `max` or `min`
- `val_<metric>`, for example `val_auc`: `primary_value` is exactly that one
  validation metric. The other metrics stay recorded but do not vote. The metric
  must exist in `metrics`, or the result fails ingest. This is the recommended
  setting once you know your metric names; set `metrics.primary` to the same
  selector.
- `trust_reported`: use the reported value as is. This weakens the validation
  firewall (see [Revealing held-out test metrics](#revealing-held-out-test-metrics)).

Arithmetic expressions are rejected by `automil check`.

A child node is kept only if

```
child.primary_value > parent.primary_value + margin
margin = max(accept_margin, se_multiplier x SE)
```

`SE` is the standard error of the per-fold child-minus-parent deltas when both
nodes carry primary values for the same folds (runs share folds under a locked
seed, so the fold effect cancels). Otherwise it is the parent's marginal
cross-fold SE. The margin can only raise the bar set by `accept_margin`, never
lower it. This is a screen during search. The significance test is the
[generalization gate](#generalization-gate).

### Budget

`cap.budget` and `cap.safety_buffer` are durations (`6h`, `30m`, `90s`, `2d`, or
a bare number of seconds). At `budget - safety_buffer` a cell stops accepting
new submits; at `budget` it sends `SIGTERM` to running experiments. The value in
effect is the CLI flag, then the `cap` key, then the framework fallback
(6h and 30m). Inspect or change it with `automil budget show` and
`automil budget set 6h`. `automil submit --budget-seconds N` overrides it, but
only on the submit that creates the cell, so later submits cannot change an
established budget. `cap.eval_budget` (default `null`) additionally caps how
many experiments a cell may launch, counting crashes and budget kills.

`cap.mode` chooses how the clock runs:

- **`agent_active`** bills only the time the agent is working, using Claude
  Code's cumulative `claude_code.active_time.total` metric (CLI and user series,
  idle time excluded). The orchestrator reads it from Claude's local Prometheus
  endpoint on every scheduling cycle, and `SessionStart` and `SessionEnd` hooks
  bind the Claude session to the cell. Claude Code reads telemetry settings only
  from the environment it starts in, and `automil init` does not set them. Start
  Claude like this (init prints the same line):

  ```bash
  CLAUDE_CODE_ENABLE_TELEMETRY=1 OTEL_METRICS_EXPORTER=prometheus \
  OTEL_METRICS_INCLUDE_SESSION_ID=true OTEL_EXPORTER_PROMETHEUS_PORT=9464 \
  claude --dangerously-skip-permissions
  ```

  The port is `activity.exporter_port` in `config.yaml` (default 9464).
- **`wall_clock`** counts continuous time since the cell was created, including
  GPU and idle time. It needs no telemetry setup and works with any runtime.

### Backend

`backend.name` is `local` (any machine), `slurm`, or `ray`. `slurm` needs the
`slurm` extra and valid `backend.slurm.directives` (`partition`, `account`,
`cpus_per_task`, `mem_gb`). `ray` needs the `ray` extra and a `RAY_ADDRESS`, or
`backend.ray.allow_local_fallback: true`.

## Training script contract

Your training script reads its config, honors the device masks the framework
sets, exits cleanly on `SIGTERM`, and writes a `result.json` that validates
against `result.schema.json`. The minimum valid result is:

```json
{"primary_value": 0.912}
```

`primary_value` is the single number used to rank experiments. Higher is always
better, so negate a loss. `metrics` holds validation metrics only. Test metrics
go in a separate `held_out` block, which the framework seals away from the
agent. Malformed results mark the node `crashed`.

The full contract, the SIGTERM patterns, the `validation_folds` per-fold
evidence, and examples are in [training-script-contract.md](training-script-contract.md).

## Running the loop

Use tmux so processes survive a dropped terminal.

### 1. Start the orchestrator

```bash
tmux new -s orchestrator
automil orchestrator start     # detach with Ctrl-b d
```

The orchestrator polls the queue, schedules experiments onto GPUs, writes each
run's log to `archive/<id>/run.log`, validates every `result.json`, and updates
the graph.

### 2. Start the dashboard (optional)

```bash
tmux new -s viz
automil viz start              # http://localhost:8420
```

The dashboard shows the experiment lineage tree (3D or flat) next to a chart of
primary value by attempt, a discovery timeline, the agent's transcript, a table
of node records, and the agent's notes. It binds to 127.0.0.1 by default; set
`viz.host` in `config.yaml` to expose it on a trusted network. It has no
authentication. Over SSH, forward the port with
`ssh -N -L 8420:127.0.0.1:8420 user@host`. `automil viz export` writes the record
as a static site.

### 3. Launch your coding agent

```bash
tmux new -s automil
claude --dangerously-skip-permissions   # for agent_active, add the telemetry variables above
# then type /automil-setup the first time, and /automil afterwards
```

`--dangerously-skip-permissions` lets the agent run unattended. The
`/automil-setup` skill runs once: it scopes the codebase, fills in
`config.yaml`, checks the contract with a one-minute dry run, and records a
baseline. `/automil` then runs the experiment loop. For other runtimes see
[Other runtimes](#other-runtimes).

### 4. Monitor progress

```bash
automil status                # summary, including the 30-day promotion rate
automil rank                  # top-ranked proposals
automil cell list             # budget state per cell
automil orchestrator status   # daemon health
```

## How experiments run

1. The agent edits files and submits, for example
   `automil submit --node node_0001 --desc "try focal loss" --mil-model clam_mb --files train.py models/clam.py`.
   `--mil-model` names the budget cell and can be omitted when `run.mil_model` is
   set. `--max-time SECONDS` sets the experiment timeout.
2. The CLI snapshots only the changed files into
   `automil/orchestrator/archive/node_0001/` and rejects edits to files matching
   `registry.protected`; those changes must ship as registered variant modules.
3. The orchestrator creates a git worktree at the base commit, overlays the
   changed files, and runs the script on the configured backend in a CPU, CUDA,
   or ROCm slot. It owns the device-mask variables and exposes the accelerator as
   logical device 0.
4. The daemon validates `result.json`, ingests it, and updates the graph. The
   agent runs `automil reconcile`, reads `learnings.md`, and designs the next
   experiment.

## Variants and reproduction

For architectural changes that touch shared library code, register the change as
a variant module:

```bash
automil port-variant <node_id>      # overlay -> automil/variants/<parent>/<name>.py
automil refresh-registry            # regenerate the per-kind variants/__init__.py
automil apply <node_id>             # write this node's variant selection into config.yaml
automil verify-repro <node_id>      # reproduce through the registry and compare
```

`registry.protected` lists globs the overlay may not touch (for example
`lib/CLAM/**`). The search scope is `free` by default, or
`architecture-preserving`, which adds the checks in `registry.identity_constraints`.

## Generalization gate

A node with status `keep` can be nominated as a candidate and tested on cells
that were declared in advance:

```bash
automil nominate <node_id>
automil gate register-manifest <parent_id> --strategy operator-curated --held-out-cells "<cell_id:dataset:encoder:task>,..."
                                             # writes and git-commits the held-out manifest before the test
automil promote <candidate_id>               # paired Wilcoxon, bootstrap CI, Bonferroni alpha/K
```

Nomination is always manual (`gate.auto_nominate` is reserved and has no effect).
`automil gate status` and `automil gate stats` show the manifest and the
promotion rate. See [calibration-pilot.md](calibration-pilot.md) for choosing `K`.

## Trajectory recording

Each submit can record a JSONL trajectory of the agent's events, using
OpenTelemetry `gen_ai.*` field names, at `archive/<node_id>/trajectory.jsonl`.
Secrets (`sk-...`, `hf_...`, `ghp_...`, AWS keys, `*_API_KEY=...`, `*_TOKEN=...`)
are redacted on capture. Each event is capped at 8 KB, and files rotate at 5 MB
(soft) and 50 MB (hard). Trajectories are gitignored. Export a redacted,
schema-validated bundle with:

```bash
automil trajectory export <node_id> --out trajectory_bundle.tgz
```

How events are captured depends on the runtime; see [Other runtimes](#other-runtimes).

## Revealing held-out test metrics

Selection uses the validation `primary_value` only. Test metrics are sealed at
ingest under `archive/<node>/certify/`, outside everything the agent can read.
After the search ends, reveal them once:

```bash
automil certify                 # the validation-selected best node
automil certify --node node_0042
```

This is the only supported read of test. `automil certify --top-k K` with K above
1 needs `--unseal-multiple`, because comparing several test results is itself
selection on test. Use it only for a table where all K rows are reported together.

## Other runtimes

autoMIL works through files and the CLI. An agent needs to read files, edit code,
and run shell commands. The shared instructions live once in
`src/automil/agent_assets/_shared/`, and per-runtime directories add only what
differs. `automil init --runtime <r>` installs the assets for one runtime:

| Runtime | `--runtime` value | What init installs |
|---------|-------------------|--------------------|
| Claude Code | `claude` | Skills `automil` and `automil-setup` under `.claude/skills/`, the `on_stop.sh` hook (blocks stopping while the loop is active) and the activity hooks registered in `.claude/settings.json`, and `.claude/CLAUDE.md` importing `AGENTS.md` |
| Codex | `codex` | `.codex/instructions.md`, a copy of the shared `AGENTS.md` |
| OpenCode | `opencode` | `.opencode/AGENTS.md` and the trajectory plugin `.opencode/plugins/automil-trajectory.ts` |
| DeepSeek through OpenCode | `deepseek-via-opencode` | The OpenCode assets |
| DeepSeek through Codex | `deepseek-via-codex` | The Codex assets |

`--runtime all` installs Claude, OpenCode, and Codex. Without `--runtime`, init
detects existing `.claude/`, `.codex/`, and `.opencode/` directories and installs
those, and falls back to Claude if none exist. All runtimes also get `AGENTS.md`
at the project root.

DeepSeek is a model, not a runtime, so it runs inside OpenCode or Codex. Set
`AUTOMIL_RUNTIME=deepseek-via-opencode` or `deepseek-via-codex` so trajectories
carry the right tag. In general, declare `AUTOMIL_RUNTIME` in the agent's
environment; autoMIL never infers it.

To see exactly what an agent will read, render the skill or the `AGENTS.md`:

```bash
automil show-skill --runtime claude > /tmp/claude-skill.md
automil show-skill --runtime codex --asset AGENTS
```

No runtime currently overrides the shared text, so the output is the same for
every runtime.

**Trajectory capture by runtime.**

- Claude Code: the `on_stop.sh` hook records events when `AUTOMIL_NODE_ID` and
  `AUTOMIL_RUNTIME` are set.
- OpenCode: the installed plugin records events automatically.
- Codex and DeepSeek through Codex: no stable hook API exists, so add
  `automil trajectory record '<event-json>'` to the agent's instructions after
  each tool call. See
  [`agent_assets/codex/README.md`](../src/automil/agent_assets/codex/README.md).

**Any other agent** (Cursor, Aider, Windsurf, and so on) can follow the loop if
it can read files, edit code, and run shell commands. Run `automil init
--no-healthcheck`, skip the per-runtime assets, and point the agent at
`automil/program.md` and the [training-script contract](training-script-contract.md).
It reads `config.yaml`, `graph.json`, `learnings.md`, and `program.md`, edits
your source files, and drives everything else through `automil` commands. The
agent needs to remember what it has tried, and `learnings.md` is where it
records that between submissions.

## Command reference and examples

Run `automil --help` for the command list and `automil <command> --help` for the
options of each command.

[`examples/`](../examples/) has reference layouts: `sklearn-iris` (a short
CPU-only consumer that CI runs end to end), `ovarian_hrd` (an autonomous run with
189 experiments and its `learnings.md`), `clwd` (multi-class subtyping), and
`placeholder` (the template `automil init` produces). Related documents:
[training-script contract](training-script-contract.md),
[gate calibration pilot](calibration-pilot.md), and the
[CHANGELOG](../CHANGELOG.md).
