# Codebase Structure

**Analysis Date:** 2026-04-30

## Directory Layout

```
autoMIL/                                    # uv workspace root
├── pyproject.toml                          # workspace + automil package metadata
├── uv.lock                                 # uv dependency lockfile
├── README.md
├── LICENSE
├── CLAUDE.md                               # project instructions for Claude
├── .gitignore                              # ignores datasets/, paper/, ref/, tasks/, .env, *.log, *.pid, .venv
├── .env                                    # NOT committed (workspace-level)
├── .automil_worktrees/                     # runtime: per-experiment git worktrees (gitignored implicitly via /datasets pattern? — see Runtime Files below)
│   └── <node_id>/                          # detached checkout at base_commit + overlay
│
├── src/                                    # autoMIL framework package
│   └── automil/
│       ├── __init__.py                     # __version__ = "0.1.0"
│       ├── cli.py                          # ~726 lines — Click CLI surface
│       ├── graph.py                        # ~758 lines — ExperimentGraph + scoring + reconcile
│       ├── orchestrator.py                 # ~884 lines — GPU scheduler daemon
│       ├── runner.py                       # ~95 lines — git worktree primitives
│       ├── templates/                      # Jinja2 scaffolding (.j2)
│       │   ├── config.yaml.j2
│       │   ├── program.md.j2
│       │   └── learnings.md.j2
│       ├── claude_assets/                  # installed into <project>/.claude/ on init
│       │   ├── skills/
│       │   │   ├── automil/SKILL.md
│       │   │   └── automil-setup/SKILL.md
│       │   └── hooks/
│       │       └── on_stop.sh
│       └── viz/                            # 3D dashboard
│           ├── __init__.py
│           ├── server.py                   # ~346 lines — aiohttp + SSE + watchdog
│           └── static/
│               ├── index.html
│               ├── app.js                  # ~630 lines — 3D force graph
│               ├── style.css
│               └── vendor/                 # vendored d3, three, three-spritetext, 3d-force-graph
│
├── tests/                                  # automil framework tests (48 total)
│   ├── test_graph.py                       # 26 tests
│   ├── test_runner.py                      # 7 tests
│   ├── test_cli.py                         # 5 tests
│   └── test_integration.py                 # 10 tests
│
├── benchmarks/                             # autobench package (uv workspace member)
│   ├── pyproject.toml
│   ├── .env                                # NOT committed; AUTOBENCH_<DATASET>_ROOT vars
│   ├── .env.example                        # committed template
│   ├── src/
│   │   └── autobench/
│   │       ├── __init__.py
│   │       ├── config.py                   # dataset YAML loader, env-var resolution
│   │       ├── data.py
│   │       ├── encoders/
│   │       │   ├── __init__.py
│   │       │   └── h0_mini.py
│   │       └── pipeline/                   # experiment execution
│   │           ├── __init__.py
│   │           ├── prepare.py
│   │           ├── splits.py
│   │           ├── evaluate.py
│   │           ├── results.py
│   │           ├── orchestrator.py         # autobench-internal multi-fold orchestration
│   │           ├── _gpu_worker.py
│   │           ├── config.py
│   │           ├── clam/                   # CLAM-specific train/evaluate
│   │           ├── nnmil/                  # nnMIL-specific
│   │           └── smmile/                 # SMMILe-specific
│   ├── datasets/                           # per-dataset YAML configs (grouped by program)
│   │   ├── tcga/                           # TCGA configs (incl. the preprint roster)
│   │   ├── cptac/                          # CPTAC configs
│   │   ├── other/                          # clwd, hancock, ovarian
│   │   └── templates/                      # tcga_template, cptac_template, placeholder
│   ├── scripts/                            # CLI entry points for benchmarks
│   │   ├── run_experiment.py               # single experiment runner (writes result.json)
│   │   ├── run_benchmark.py                # multi-experiment sweep
│   │   ├── run_feature_extraction.py
│   │   ├── slurm/                          # SLURM submit wrappers (submit_*.sh)
│   │   └── manifests/                      # data-prep utilities (add_os_to_manifest, …)
│   ├── lib/                                # vendored external MIL libraries
│   │   ├── CLAM/
│   │   ├── nnMIL/
│   │   ├── SMMILe/
│   │   └── TRIDENT/
│   ├── experiments/                        # per-dataset autoMIL overlays + state
│   │   ├── ccrcc/
│   │   │   ├── automil/                    # the overlay autoMIL manages
│   │   │   │   ├── config.yaml
│   │   │   │   ├── graph.json              # gitignored runtime
│   │   │   │   ├── results.tsv             # gitignored runtime
│   │   │   │   ├── program.md
│   │   │   │   ├── learnings.md
│   │   │   │   └── orchestrator/           # gitignored runtime
│   │   │   │       ├── queue/<id>.json
│   │   │   │       ├── running/<id>.json
│   │   │   │       ├── completed/<id>.json
│   │   │   │       ├── archive/<id>/{spec,result,run.log,<overlay tree>}
│   │   │   │       ├── orchestrator.pid
│   │   │   │       ├── orchestrator.log
│   │   │   │       ├── gpu_state.json
│   │   │   │       ├── viz_server.pid
│   │   │   │       └── viz_server.log
│   │   │   ├── tasks/                      # ad-hoc task notes
│   │   │   └── .automil_active             # flag created by `automil start-loop`
│   │   ├── clwd/
│   │   │   ├── automil/                    # same shape as ccrcc
│   │   │   └── generalization_test/
│   │   ├── ovarian_hrd/
│   │   └── placeholder/
│   └── tests/
│       ├── conftest.py
│       ├── _helpers.py
│       └── test_*.py                       # autobench tests (config, data, encoders, splits, ...)
│
├── examples/                               # reference overlays (not actively used)
│   ├── ccrcc/
│   ├── clwd/
│   ├── ovarian_hrd/
│   └── placeholder/
│
├── datasets/                               # raw + extracted dataset roots (gitignored)
│   ├── CCRCC/
│   └── CLWD/
│
├── docs/
│   └── public/
│
├── ref/                                    # reference repos (ASI-Evolve, EvoScientist) — gitignored
├── tasks/                                  # working notes (gitignored)
├── .planning/
│   └── codebase/                           # this directory — codebase maps for GSD commands
├── .claude/                                # project Claude Code config
│   ├── settings.json
│   ├── skills/
│   │   ├── automil/
│   │   ├── automil-setup/
│   │   └── commit/
│   └── hooks/
└── .venv/                                  # uv-managed venv (gitignored)
```

## Directory Purposes

**`src/automil/`:**
- Purpose: The autoMIL framework package (installed as `automil` via `pip install -e .`).
- Contains: All framework code; templates and Claude assets shipped inside the wheel.
- Key files: `cli.py`, `graph.py`, `orchestrator.py`, `runner.py`, `viz/server.py`.

**`src/automil/templates/`:**
- Purpose: Jinja2 scaffolding rendered by `automil init` into the user repo.
- Contains: `config.yaml.j2` (orchestrator settings, baseline metrics, file scopes), `program.md.j2`, `learnings.md.j2`.
- Key files: `src/automil/templates/config.yaml.j2`.

**`src/automil/claude_assets/`:**
- Purpose: Claude Code assets installed into `<project>/.claude/` on init.
- Contains: `skills/automil/SKILL.md`, `skills/automil-setup/SKILL.md`, `hooks/on_stop.sh` (registered as a Claude `Stop` hook in `<project>/.claude/settings.json` by `cli.py:155-179`).

**`src/automil/viz/`:**
- Purpose: Real-time 3D dashboard.
- Contains: `server.py` (aiohttp + SSE), `static/index.html`, `static/app.js` (3D force graph), `static/vendor/` (vendored d3, three, three-spritetext, 3d-force-graph).

**`tests/`:**
- Purpose: autoMIL framework tests (run via `uv run pytest tests/ -v`).
- Contains: `test_graph.py`, `test_runner.py`, `test_cli.py`, `test_integration.py`.

**`benchmarks/`:**
- Purpose: `autobench` workspace member — dataset-agnostic MIL benchmark suite that demonstrates autoMIL across datasets.
- Contains: `src/autobench/` (package), `datasets/` (YAML configs), `scripts/` (CLI entry), `lib/` (vendored MIL libraries), `experiments/<dataset>/automil/` (per-dataset autoMIL overlays + runtime state), `tests/`.

**`benchmarks/src/autobench/pipeline/`:**
- Purpose: Per-framework training/evaluation code.
- Contains: `clam/`, `nnmil/`, `smmile/` subpackages; shared `prepare.py`, `splits.py`, `evaluate.py`, `results.py`, `orchestrator.py`, `_gpu_worker.py`.

**`benchmarks/datasets/`:**
- Purpose: Per-dataset YAML configs with `${ENV_VAR}` path templating; resolved at load time by `autobench.config`.

**`benchmarks/lib/`:**
- Purpose: Vendored upstream MIL libraries (CLAM, nnMIL, SMMILe, TRIDENT).
- Modification policy: Agent may edit these as part of an experiment overlay (e.g., `benchmarks/lib/CLAM/models/model_clam.py`).

**`benchmarks/experiments/<dataset>/automil/`:**
- Purpose: The autoMIL state for a particular dataset (one autoMIL "site" per dataset).
- Contains: `config.yaml` (dataset-specific run script, file scopes, baseline), `graph.json` (experiment tree), `results.tsv` (orchestrator-only writer), `program.md`, `learnings.md`, `orchestrator/{queue,running,archive,completed,*.pid,*.log,gpu_state.json}`.
- Key invariant: `_find_automil_dir()` walks up from cwd looking for `automil/config.yaml`, so the agent must `cd benchmarks/experiments/<dataset>/` before invoking the CLI.

**`.automil_worktrees/`:**
- Purpose: Runtime — detached git worktrees, one per running experiment.
- Lifecycle: Created by `Runner.create_worktree()` at `git worktree add --detach <path> <base_commit>`; removed by `Runner.cleanup_worktree()` after `result.json` collection.
- Location: At the project_root (the git root), not under `automil/`. So the workspace root has `.automil_worktrees/` and each dataset's `automil/` does not.

**`examples/`:**
- Purpose: Reference templates / sample overlays. Not used by runtime.

**`datasets/`:**
- Purpose: Raw + extracted feature roots (e.g., CCRCC features, TRIDENT outputs). Large, gitignored.

**`tasks/` and `ref/`:**
- Purpose: Working notes and reference repos. Both are gitignored at the workspace level (`.gitignore` lists `tasks/`, `ref/`).

## Key File Locations

**Entry Points:**
- `src/automil/cli.py:67` — `main` Click group; pyproject.toml registers `automil = "automil.cli:main"`.
- `src/automil/orchestrator.py:701` — `ExperimentOrchestrator.run()` daemon loop (entered via `automil orchestrator start`).
- `src/automil/viz/server.py:222` — `cmd_start()` aiohttp server (entered via `automil viz start`).
- `benchmarks/scripts/run_experiment.py` — single-experiment runner used as the `run.script` target by the `autobench` overlays.

**Configuration:**
- `pyproject.toml` — workspace root, automil package metadata, `[project.scripts] automil = "automil.cli:main"`, `[tool.uv.workspace] members = ["benchmarks"]`.
- `benchmarks/pyproject.toml` — autobench package.
- `benchmarks/experiments/<dataset>/automil/config.yaml` — per-dataset autoMIL config (run script, file scopes, baseline, orchestrator settings).
- `benchmarks/datasets/<dataset>.yaml` — per-dataset paths/encoders/tasks (with `${AUTOBENCH_<DATASET>_ROOT}` templating).
- `benchmarks/.env` — environment variables for dataset roots (gitignored; loaded by orchestrator at startup).

**Core Logic:**
- `src/automil/graph.py:18` — `ExperimentGraph` class (tree, scoring, reconcile).
- `src/automil/graph.py:303` — `recalculate_scores()` (UCB potential).
- `src/automil/graph.py:405` — `reconcile()` (sync with orchestrator state).
- `src/automil/orchestrator.py:130` — `ExperimentOrchestrator` class.
- `src/automil/orchestrator.py:339` — `_find_best_gpu()` (best-fit bin packing).
- `src/automil/orchestrator.py:374` — `_launch()` (env masking, subprocess launch).
- `src/automil/runner.py:11` — `Runner` class (git worktree lifecycle).
- `src/automil/cli.py:188` — `submit` command (snapshot + queue spec).
- `src/automil/cli.py:472` — `propose` command.
- `src/automil/cli.py:510` — `reconcile` command.

**Visualization:**
- `src/automil/viz/server.py:47` — `GraphWatcher` (watchdog inotify).
- `src/automil/viz/server.py:159` — `sse_handler` (Server-Sent Events).
- `src/automil/viz/static/app.js` — 3D force-graph frontend.

**Testing:**
- `tests/test_graph.py` — 26 tests covering graph API, scoring, reconciliation, migration.
- `tests/test_runner.py` — 7 tests covering worktree create/cleanup, overlay, result collection.
- `tests/test_cli.py` — 5 tests covering init, submit, rank.
- `tests/test_integration.py` — 10 tests covering end-to-end flows, path sanitization, deletions, scoring.
- `benchmarks/tests/` — autobench tests (config, data, encoders, splits, prepare, train, evaluate, integration).

**Templates and Assets:**
- `src/automil/templates/config.yaml.j2` — rendered to `<project>/automil/config.yaml` by `automil init`.
- `src/automil/templates/program.md.j2`, `learnings.md.j2` — rendered to the project's `automil/` dir.
- `src/automil/claude_assets/skills/automil/SKILL.md` — installed to `<project>/.claude/skills/automil/SKILL.md`.
- `src/automil/claude_assets/hooks/on_stop.sh` — installed to `<project>/.claude/hooks/on_stop.sh` and registered in `settings.json`.

## Naming Conventions

**Files:**
- Python: `snake_case.py` (e.g., `graph.py`, `orchestrator.py`, `run_experiment.py`).
- Templates: `<target_name>.j2` (Jinja2 extension); rendered to `<target_name>` (`config.yaml.j2 → config.yaml`).
- Test files: `test_<module>.py` (pytest discovery via `[tool.pytest.ini_options] testpaths = ["tests"]`).
- Shell scripts: `submit_*.sh` for cluster wrappers; `on_stop.sh` for Claude hooks.

**Directories:**
- Lowercase with underscores for Python packages (`autobench`, `automil`).
- Lowercase with hyphens for Claude skill subdirectories (`automil-setup`).
- Per-experiment directories: `node_NNNN` (4-digit zero-padded; minted by `ExperimentGraph.next_id()` at `graph.py:75-78`). All artifacts for that experiment use this id (queue file, running file, completed file, archive subdir).

**Identifiers:**
- Experiment node ids: `node_0001`, `node_0042`, ... (zero-padded to 4 digits).
- Spec ids assigned by orchestrator counter when no id present: `0001`, `0002`, ... (also 4-digit, no `node_` prefix; auto-assigned in `cmd_submit` at `orchestrator.py:836-843`).
- Config hashes: 16-char hex SHA-256 prefix (`graph.py:371`).
- Technique tags: lowercase snake_case (e.g., `focal_g1`, `grad_clip`, `topk_attn`); see `DEFAULT_TECHNIQUE_MAP` in `graph.py:19-30`.

**Runtime files (gitignored within each `automil/` site):**
- `graph.json` — the experiment tree.
- `results.tsv` — TSV with `node_id\tval_auc\tval_bacc\ttest_auc\ttest_bacc\tcomposite\tvram_gb\telapsed_min\tstatus\tdescription` header.
- `orchestrator/{queue,running,archive,completed}/<id>.json|<id>/`.
- `orchestrator/{orchestrator,viz_server}.pid` and `.log`.
- `orchestrator/gpu_state.json` (rewritten every poll cycle).
- `.automil_active` — flag created by `automil start-loop`, removed by `stop-loop`.

## Where to Add New Code

**New CLI command:**
- Primary code: `src/automil/cli.py` — add `@main.command()` decorated function (or `@orchestrator.command()` / `@viz.command()` for subgroup commands).
- Tests: `tests/test_cli.py` (uses `click.testing.CliRunner`).

**New graph operation (e.g., new scoring rule, new node attribute):**
- Implementation: `src/automil/graph.py` — add method on `ExperimentGraph`. Mutate `node` dict directly; bump `meta` counters as needed; call `self.save()` only from the CLI layer.
- Tests: `tests/test_graph.py` and `tests/test_integration.py`.

**New orchestrator behavior (e.g., new launch env var, new completion classifier):**
- Implementation: `src/automil/orchestrator.py` — modify `_launch()` for env vars, `_handle_completion()` for status classification, `tick()` for scheduling logic, `_reload_orchestrator_config()` for hot-reloadable config keys.
- Tests: `tests/test_runner.py` for worktree-side concerns; `tests/test_integration.py` for end-to-end.

**New worktree primitive:**
- Implementation: `src/automil/runner.py` (keep this module thin — only git/shutil primitives).
- Tests: `tests/test_runner.py`.

**New dashboard data:**
- Backend: `src/automil/viz/server.py` — extend `_overlay_running_status()` or `_notify()` to enrich the payload.
- Frontend: `src/automil/viz/static/app.js`.

**New benchmark dataset:**
- Add `benchmarks/datasets/<dataset>.yaml` with `${AUTOBENCH_<DATASET>_ROOT}` placeholders.
- Append `AUTOBENCH_<DATASET>_ROOT=...` to `benchmarks/.env`.
- Create `benchmarks/experiments/<dataset>/automil/` by `cd`-ing there and running `automil init` (or copy from `examples/`).
- Tests: extend `benchmarks/tests/test_config.py` if loader logic changes.

**New MIL framework integration:**
- Vendor library under `benchmarks/lib/<framework>/`.
- Add per-framework subpackage under `benchmarks/src/autobench/pipeline/<framework>/` with `train.py`, `evaluate.py`.
- Wire into `benchmarks/scripts/run_experiment.py` dispatch.

**New experiment proposal (agent action):**
- No code edit required. `automil propose --parent <id> --desc "..." --techniques <tag>...` adds a node of `type=proposed`.
- To realize the proposal: edit host-repo files, then `automil submit --node <new_id> --desc "..." --parent <id>`.

## Special Directories

**`.automil_worktrees/`:**
- Purpose: Runtime per-experiment git worktrees.
- Generated: Yes — by `Runner.create_worktree()` at orchestrator launch.
- Committed: No — should be ignored. Currently NOT in root `.gitignore`; relies on git's own knowledge that worktrees are auxiliary. Add a `/.automil_worktrees/` line to `.gitignore` if it ever shows up in `git status`.
- Cleanup: `Runner.cleanup_worktree()` removes via `git worktree remove --force` after each experiment; `prune_stale_worktrees()` at daemon startup cleans dangling refs.

**`benchmarks/experiments/<dataset>/automil/orchestrator/`:**
- Purpose: All orchestrator runtime state.
- Generated: `automil init` creates the empty subdirs; orchestrator and CLI write files.
- Committed: Per-site `automil/.gitignore` (rendered from template) excludes `graph.json`, `results.tsv`, `orchestrator/`. The `automil/config.yaml`, `program.md`, and `learnings.md` ARE committed.

**`datasets/`, `ref/`, `tasks/`, `paper/`, `logs/`:**
- Purpose: Large data, reference repos, working notes, output logs.
- Committed: No — explicitly listed in root `.gitignore`.

**`.venv/`:**
- Purpose: uv-managed Python virtualenv.
- Committed: No.

**`docs/public/`:**
- Purpose: Public documentation for the project.
- Committed: Yes.

---

*Structure analysis: 2026-04-30*
