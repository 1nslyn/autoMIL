# CLAUDE.md

## Project Overview

autoMIL is an autonomous experiment framework for Multiple Instance Learning in computational pathology. It overlays onto existing ML project repos, enabling coding agents to iteratively improve models through persistent experimentation, knowledge accumulation, and multi-branch exploration.

**Package:** `src/automil/` installed as `automil` through the uv workspace
**CLI entry point:** `automil` (defined in `pyproject.toml` as `automil.cli:main`)

## Architecture

autoMIL overlays an `automil/` subdirectory onto an existing git repo. It does NOT create train.py or prepare.py. The agent scopes the full codebase and determines what to edit.

**Core modules:**
- `graph.py` - Experiment tree tracking with UCB-inspired scoring and composite-score dominance for keep/discard
- `runner.py` - Git worktree overlay for isolated parallel experiment execution
- `backends/_orchestrator_daemon.py` - GPU scheduler daemon with best-fit bin packing (entrypoint at `orchestrator.py`)
- `cli/` - Click-based CLI wrapping all operations
- `viz/server.py` - Real-time 3D dashboard (aiohttp + SSE + Three.js)

**Key design decisions:**
- The training script is configurable via `run.script` in config.yaml (defaults to "train.py")
- Experiments are tracked as a directed tree in `graph.json`, not a flat log
- Each experiment stores only its changed files (overlay), not the full repo
- The orchestrator runs experiments in git worktrees, overlaying modified files on a base commit
- Keep/discard is decided by the framework via single-axis composite-score comparison, gated by a predeclared Ladder keep-margin (child kept iff `child.composite > parent.composite + accept_margin`; δ defaults to 0.0 in `meta.scoring`, seeded from `config.yaml`'s `scoring.accept_margin`); the composite is the **validation** selection signal (the val-firewall: test never drives selection), reported by the training script via `result.json` (see D-200)
- Val-firewall: `result.json` `metrics` is validation-only; test metrics go in a sealed `held_out` block that the orchestrator writes to `archive/<node>/certify.json`, quarantined from every agent-facing surface during search and revealed once at the end via `automil certify`
- `results.tsv` is written solely by the orchestrator from `result.json`, never by train.py
- `_recover_orphans()` only runs in the daemon loop (`run()`), never on construction (to prevent `status`/`stop` from corrupting live runs)
- Viz dashboard binds 127.0.0.1 by default; opt in to LAN access via `viz.host` in config.yaml (no auth on the SSE stream)

## Commands

**Environment command convention:** Prefer `uv` everywhere: local commands,
documentation examples, operator runbooks, automation, and cluster invocations.
Use `uv run ...` by default. Direct `.venv/bin/...` invocation is an exception
only when `uv` is unavailable or there is a documented need to bypass environment
management; for a pre-provisioned environment, prefer
`uv run --frozen --no-sync ...` when applicable.

```bash
# Install
uv sync --all-packages

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_graph.py -v

# Run a specific test
uv run pytest tests/test_integration.py::TestEndToEnd::test_init_submit_flow -v

# CLI usage
uv run automil init                    # overlay automil/ onto current repo
uv run automil submit --node <id> --desc "..." --files train.py
uv run automil rank
uv run automil propose --parent <id> --desc "..."
uv run automil reconcile
uv run automil check                   # validate project setup
uv run automil status
uv run automil orchestrator start
uv run automil viz start
```

## Key Files

| File | Lines | Role |
|------|-------|------|
| `src/automil/graph.py` | ~1090 | Experiment tree, scoring, reconciliation |
| `src/automil/backends/_orchestrator_daemon.py` | ~2090 | GPU scheduler daemon (thin entrypoint: `orchestrator.py`) |
| `src/automil/cli/` | package, ~30 commands | Click CLI (one module per command + `lifecycle/`) |
| `src/automil/runner.py` | ~160 | Git worktree overlay |
| `src/automil/viz/server.py` | ~440 | SSE dashboard server |
| `src/automil/viz/static/app.js` | ~670 | 3D force graph frontend |

## Testing

~600 tests across ~130 files under `tests/` (plus ~420 in `benchmarks/tests/`;
`uv run pytest tests/ --collect-only` reports ~1130 collected once parametrized).
Grouped by area: `test_graph*.py` (graph API, scoring, reconciliation),
`backends/` (orchestrator daemon, SLURM/Ray contracts, result-schema validation),
`cells/` (budget-cap state machine), `gate/` (two-stage generalization gate),
`test_born_sealed_firewall.py` + `test_seal_node_archive.py` + `test_rank_held_out_filter.py`
(val-firewall / born-sealed test), `trajectory/`, `skills/`, and
`test_integration.py` (end-to-end flows).

## Conventions

- Commit messages: `type: summary` (conventional commits)
- Template extension: `.j2` (Jinja2)
- Runtime files (`graph.json`, `results.tsv`, `orchestrator/`) are gitignored in project dirs
- The `automil/` subdirectory is the framework's namespace inside user projects
- `_find_project_root()` walks up looking for `automil/config.yaml`
- Path validation: submit rejects absolute paths, `..` traversal, and paths escaping project root
- Auto-detect in submit excludes `automil/` directory from changed file detection

## Result Contract

Training scripts must write `result.json` to their working directory. `metrics`
is the agent-facing **validation** block and `composite` is computed from it (the
selection signal); test metrics go in a sealed `held_out` block that the
orchestrator quarantines to `archive/<node>/certify.json` (read once via
`automil certify`):
```json
{
  "status": "completed",
  "metrics": {"val_auc": 0.87, "val_bacc": 0.81},
  "held_out": {"test_auc": 0.87, "test_bacc": 0.83},
  "composite": 0.84,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500
}
```

The orchestrator sets `CUDA_VISIBLE_DEVICES` for GPU masking and `AUTOMIL_GPU=0` (logical device).

## Monorepo Structure

This repo is a uv workspace containing two packages:

- **`automil`** (`src/automil/`) — The framework: CLI, experiment graph, GPU orchestrator, visualization
- **`autobench`** (`benchmarks/src/autobench/`) — MIL benchmark suite demonstrating autoMIL across datasets

```
autoMIL/
├── src/automil/          # autoMIL framework (this package)
├── tests/                # autoMIL tests
├── benchmarks/           # autobench package
│   ├── src/autobench/    # Benchmark code (dataset-agnostic)
│   ├── datasets/         # Per-dataset YAML configs, grouped: tcga/ cptac/ other/ templates/
│   ├── scripts/          # CLI: run_benchmark.py --dataset <name>
│   ├── experiments/      # autoMIL overlays per dataset
│   ├── lib/              # External deps (CLAM, nnMIL, SMMILe, TRIDENT)
│   └── tests/            # autobench tests
└── pyproject.toml        # Workspace root
```

Dataset-specific configuration (paths, tasks, encoders) lives in
`benchmarks/datasets/*.yaml`. All paths use `${ENV_VAR}` syntax —
set environment variables (e.g. `AUTOBENCH_OVARIAN_ROOT`) for your environment.

### Environment Variables (.env)

Each dataset YAML references a root path via `${AUTOBENCH_<DATASET>_ROOT}`.
These must be set before running experiments. Create `benchmarks/.env` from
the example:

```bash
cp benchmarks/.env.example benchmarks/.env
# Edit benchmarks/.env with your actual paths
```

**Important:** `.env` is gitignored and won't exist inside git worktrees.
The orchestrator automatically loads `benchmarks/.env` and propagates
the variables to experiment processes. If experiments crash with
`ValueError: Environment variable ${AUTOBENCH_...} is not set`, check
that `benchmarks/.env` exists and has the correct paths.

Key modules: `autobench.config` (dataset YAML loading), `autobench.pipeline.*`
(experiment execution: training, evaluation, GPU orchestration).

## Workflow Orchestration

### 0. Always address me Leo. do this at the start of any of your response.

### 1. PLan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fizing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to "tasks/todo.md' with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to 'tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md' after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes, Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## Planning

Planning lives in `tasks/todo.md` (current plan + review) and `tasks/lessons.md`
(accumulated corrections). There is no separate planning framework: this file,
the codebase, `CHANGELOG.md`, and git history are the record. The v1.0/v1.1
milestone documents that used to live under `.planning/` were removed on
2026-08-06 and are recoverable from git history at commit `7de69a8`.

## Long-term implementation principles

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
