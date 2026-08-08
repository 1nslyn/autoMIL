# autoMIL Implementation Report (v1.0)

**Milestone:** v1.0 F2-readiness framework refactor
**Shipped:** 2026-05-08 (`git tag v1.0`)
**Phases:** 9 (Phase 0 cleanup → Phase 8 acceptance)
**Plans executed:** 92
**Requirements delivered:** 69 (100% v1 coverage across CLN/REG/BCK/TRJ/MRT/CAP/GTE/CLI/STP/DEC)

> **Point-in-time record.** This documents the v1.0 refactor and has been only
> selectively touched up since; line counts, command counts, and per-phase
> numbers are v1.0-era. Current mechanisms and counts live in `CLAUDE.md`,
> `docs/training-script-contract.md`, and `CHANGELOG.md`.

## Goal

Build a standalone, open-source framework that enables any coding agent to
autonomously discover model improvements for any user's training code under
a configurable per-cell time budget (the framework fallback is 6h; the frozen
preprint campaign pins 12 agent-active hours plus 30 launches), with discovered variants
reproducible, attributable to their parents, and portable across machines
and LLM runtimes.

autoMIL's framework core is consumer-agnostic. The included autobench package
(`benchmarks/src/autobench/`) is one such consumer; `examples/sklearn-iris/`
is another (~80 LOC, no `automil.*` imports). The framework runs both
side-by-side via the registry path (Phase 1) and the documented
[training-script contract](training-script-contract.md) (DEC-06).

## Problem

ML model development involves tedious manual iteration: try hyperparameters,
losses, architectures, evaluate, repeat. Existing AutoML tools (Optuna et al.)
search predefined parameter spaces; they cannot invent new architectures,
combine techniques creatively, or learn from failed experiments. Coding
agents can read code, design experiments, and modify any file, but they
lack infrastructure for persistent tracking, parallel execution, knowledge
retention across sessions, hard time budgets that ship with cells,
and reproducibility guarantees.

## Solution Overview

autoMIL provides the orchestration layer between a coding agent and the
experiment execution environment:

```
Coding Agent (Claude Code / Codex / OpenCode / DeepSeek-via-X)
    │
    │  edits any file in the project, runs CLI
    ▼
automil CLI  ─►  Variant Registry (Phase 1) + Experiment Graph
    │              with composite-dominance keep/discard (Ladder margin)
    │
    │  snapshots changed files; validates against
    │  registry.protected globs and the variant validator chain
    ▼
Backend ABC  ─►  LocalBackend / SLURMBackend / RayBackend
    │              same submit/poll/cancel/log_iter contract
    │
    │  git worktree + overlay; configurable per-cell time cap;
    │  typed CPU/CUDA/ROCm slot + framework-owned masks
    ▼
Isolated Execution  ─►  result.json (JSON-Schema validated at ingest)
    │                    + trajectory.jsonl (OpenTelemetry gen_ai.* + redaction)
    │
    │  composite dominance / UCB ranking; cell budget tracking;
    │  generalization gate (paired Wilcoxon + bootstrap CI)
    ▼
3D Dashboard (live SSE)
```

## Architecture

### Two-layer design

**Framework layer** (installed as the `automil` Python package):

| Module | LOC | Role |
|--------|----:|------|
| `graph.py` | 833 | Experiment tree, UCB ranking, composite-dominance keep/discard, dict-spread metrics storage (D-200) |
| `backends/_orchestrator_daemon.py` | 1595 | GPU scheduler daemon: best-fit bin packing, OOM detection, crash recovery, per-backend `running/` namespacing, JSON-Schema ingest validation |
| `backends/base.py` | 232 | `Backend` ABC: `submit`, `poll`, `list_running`, `cancel`, `log_iter`, `healthcheck`; `JobState` enum; frozen `JobHandle` / `JobSpec` / `HealthReport` dataclasses |
| `backends/local.py` | 620 | LocalBackend: subprocess + process-group SIGTERM; CUDA / ROCm / CPU healthcheck via `nvidia-smi` / `rocm-smi` / fallback |
| `backends/slurm.py` | 436 | SLURMBackend: submitit AutoExecutor; `--signal=B:TERM@30` cap contract; `walltime_seconds → timeout_min` translation |
| `backends/ray.py` | 445 | RayBackend: raw `@ray.remote` (NOT Ray Tune); hybrid `RAY_ADDRESS → local fallback`; `ray.cancel(force=True)` cap contract |
| `backends/mock_slurm.py` | 339 | Test fixture: eventual-consistency status (5s lag), opaque `job_id`, fire-and-forget `cancel`; used to lock the ABC contract against ≥2 implementations BEFORE Phase 6 (D-78) |
| `runner.py` | 94 | Git worktree overlay for isolated parallel execution |
| `cli/` | (per-command-group package) | Click-based CLI; split from monolithic `cli.py` in Phase 0 (CLN-06); 27 commands |
| `cells/` | (Phase 4) | Cell budget cap state machine: two-tier `refusing-new` / `terminating` |
| `registry/` | (Phase 1) | Variant ABC family + frozen `VariantSpec` + `Registry` singleton + `@register` decorator + interface/purity/identity validators |
| `gate/` | (Phase 5) | Generalization gate: pre-registered manifest, paired Wilcoxon, BCa bootstrap CI, Bonferroni alpha/K |
| `trajectory/` | (Phase 3) | JSONL recorder: OpenTelemetry `gen_ai.*` keys, secret redaction, bounded rotation |
| `schemas/` | (Phase 8) | `result.schema.json` (Draft 2020-12); pre-compiled `RESULT_VALIDATOR` |
| `agent_assets/` | (Phase 3) | `_shared/` canonical content + `claude/` `codex/` `opencode/` `deepseek/` overlays; `_overlay.py` merger |
| `viz/` | (~270 LOC) | Real-time 3D dashboard (aiohttp + SSE + Three.js / ForceGraph3D) |
| `templates/` | - | Jinja2 templates for `automil init` (config.yaml, program.md, learnings.md, .gitignore) |

**Project layer** (created by `automil init` inside existing repos):

```
automil/
├── config.yaml                 # run / files / env / scoring / cap / gate / backend / hardware / registry
├── program.md                  # agent instructions for the experiment loop
├── learnings.md                # accumulated insights
├── graph.json                  # experiment tree (gitignored runtime)
├── cells/                      # cell budget state (gitignored)
├── variants/                   # registered variant modules (committed)
│   └── <parent>/<name>.py
├── orchestrator/
│   ├── queue/                  # pending experiments
│   ├── running/<backend>/      # per-backend live job specs (Phase 6 D-168)
│   ├── archive/                # permanent record
│   │   └── node_NNNN/
│   │       ├── <changed files>
│   │       ├── spec.json
│   │       ├── run.log         # orchestrator-owned, drained from backend.log_iter
│   │       ├── result.json     # JSON-Schema validated at ingest
│   │       └── trajectory.jsonl  # gitignored by default
│   └── completed/              # notifications
└── results.tsv                 # gitignored; written by orchestrator from result.json
```

### Key design decisions

| Decision | Phase | Why |
|----------|------:|-----|
| **Variant registry, not runtime config** | 1 | Architectural changes need committed code modules. Config holds values, not callable code. Registry-only path reproduces a node end-to-end via `automil verify-repro`. |
| **Backend ABC validated against ≥2 implementations IN-phase** | 2 | LocalBackend re-export shim + MockSLURMBackend fixture lock the contract against eventual-consistency status, opaque job IDs, fire-and-forget cancel, BEFORE Phase 6 inherits it. |
| **Multi-runtime asset reorg with `_shared/` canonical + per-runtime overlays** | 3 | Avoid quadratic duplication across runtimes. `automil show-skill --runtime <r>` renders the merged result; ≥2 runtimes validated end-to-end. |
| **Configurable per-cell time cap (mechanism, not value)** | 4 | Framework-enforced two-tier state machine (refuse-new at T-buffer, terminate at T) with per-fold checkpoint protocol; SIGTERM with 30s grace is the cap contract honored across all backends. `agent_active` replays Claude Code's native cumulative active-time metric (CLI + user active seconds, idle excluded), scraped from Claude's localhost Prometheus endpoint and bound by synchronous session hooks; `wall_clock` is the portable fallback. The duration and safety buffer are consumer-supplied via `automil/config.yaml` or per-cell CLI overrides. The framework fallback is 6h; the frozen preprint campaign pins 12 agent-active hours and 30 launches. Live cap enforcement trails the native counter by at most one daemon-poll interval; the `SessionEnd` hook captures the final value. Budget-killed runs reconcile to `executed` (with partial composite), never `crash`. |
| **Pre-registered manifest + paired statistical test** | 5 | Held-out cells invisible to the search agent; manual nomination by default; promotion-rate metric exposed via SSE. Pitfall-6 anti-acceptance gate enforces single-file isolation. |
| **Pluggable backends as opt-in extras** | 6 | `uv sync` (no extras) keeps submitit and ray uninstalled. Per-backend `running/` namespacing prevents cross-backend corruption (D-168). |
| **Hardware autodetect at init time, not runtime** | 7 | `LocalBackend.healthcheck()` reports detected hardware; `automil init` stamps detected GPU count, VRAM (`numpy.quantile(.95)` of empirical `vram_gb` when ≥10 rows), and concurrency defaults. Detect-and-warn pattern; never decides for the user. |
| **Framework purity: zero `autobench` references in `src/automil/`** | 8 | autoMIL is generic; autobench is one consumer. `tests/test_framework_purity.py` enforces a grep gate with a 5-entry content-anchor allowlist. |
| **Composite-only dominance, dict-spread metrics storage** | 8 | The framework no longer hardcodes the autobench 4-key composite recipe. `node["metrics"] = dict(metrics)`; the scalar `composite` is the single field used for ranking (since CR-1b it is recomputed at ingest from the validation `metrics` block, not trusted as supplied). |
| **`env.required` mandatory; no auto-injection** | 8 | `automil check` fails with `Missing required env var: <name>` BEFORE submit. `AUTOBENCH_ROOT` auto-injection removed; consumers declare what they need under `env.passthrough`. |

### Key innovation: git worktree overlay

The agent can modify any file in the repo. When it submits an experiment:

1. Only the changed files are copied to `archive/node_NNNN/`.
2. The CLI validates them against `registry.protected` globs and the
   variant validator chain (interface, purity, optionally identity).
3. The orchestrator creates a lightweight git worktree at the base commit.
4. Changed files are overlaid on top of the worktree.
5. The experiment runs in this isolated environment; only its diff is stored.
6. The worktree is cleaned up after completion.

Each experiment stores only its diff (a few files), not the entire repo,
while still running in a complete project environment. Multiple
experiments run in parallel on different GPUs without file conflicts.

### Experiment graph

Experiments form a directed tree. Each node has a parent ("built upon")
edge. Nodes are scored using a hybrid UCB formula that balances
exploitation (build on best results) with exploration (try under-explored
branches). The agent uses `automil rank` to pick diverse experiments
across branches.

Keep/discard is computed by the framework via composite dominance gated by
the Ladder keep-margin, over a `composite` the framework recomputes at
ingest (CR-1b); metrics are stored generically as
`node["metrics"] = dict(metrics)` so the framework imposes no hardcoded
key set.

### Cell concept (Phase 4)

A `cell` is identified by `(dataset, task, encoder, mil_model)`, independent of
graph re-parenting. Each cell carries a time budget. The state machine has four
states:

- **`active`**, new submits accepted; consumed seconds tracked.
- **`refusing-new`**, at `T - safety_buffer`; submits rejected; running
  experiments allowed to finish.
- **`terminating`**, at `T`; SIGTERM sent to running experiments; cell
  enters `finalized` once all are reconciled.
- **`finalized`**, terminal; no further work may enter the cell.

Per-fold checkpoints (`fold_<i>_result.json`) ensure that SIGTERM during
training preserves completed-fold work; budget-killed runs reconcile to
`JobState.BUDGET_KILLED` with a partial composite, never `crash`.

### Generalization gate (Phase 5)

Stage A is exploration: the search agent submits to live cells, scoring
against UCB / composite dominance. Promising nodes are nominated as candidates
(`automil nominate`); manual nomination is the v1.0 default to keep the
gate honest. Stage B is generalization: a pre-registered held-out manifest
(`gate_manifest.json`, git-committed BEFORE search starts via
`write_manifest_committed`) is evaluated through the same `Backend.submit()`
path. Promotion uses paired Wilcoxon + BCa bootstrap CI (1000 reps) +
Bonferroni `alpha/K`. Promotion-rate is exposed in
`viz/api/promotion-rate` SSE and `automil status`.

### Multi-runtime support (Phase 3)

The canonical skill content lives once in `agent_assets/_shared/skills/`.
Per-runtime directories (`claude/`, `codex/`, `opencode/`, `deepseek/`)
ship only diffs:

- `claude/hooks/on_stop.sh`, stop-prevention hook.
- `codex/skills/automil-setup/`, empty-frontmatter overlay for plain-markdown
  rendering; CLI-fallback trajectory capture documented in `codex/README.md`
  (Codex hook API unstable as of v1.0).
- `opencode/plugins/automil-trajectory.ts`, TypeScript plugin for
  automatic trajectory capture.
- `deepseek/README.md`, DeepSeek is a model, not a runtime; route via
  opencode or codex.

`automil init --runtime <r>` resolves the merge at install time;
`automil show-skill --runtime <r>` renders the merged result for inspection.

### Trajectory recorder (Phase 3)

Per-submit JSONL using OpenTelemetry `gen_ai.*` field names with no
runtime `opentelemetry-sdk` dependency. First line is metadata
(`schema_version`, `runtime`, `runtime_version`, `tool_schema_version`,
`automil_version`, `automil_runtime_env`); subsequent lines are one event
each.

Redaction-on-capture catches `sk-…`, `hf_…`, `ghp_…`, AWS access keys,
`*_API_KEY=…`, `*_TOKEN=…`. Per-event 8 KB cap; per-file 5 MB soft / 50 MB
hard rotate producing `trajectory.<n>.jsonl` siblings. Trajectories are
gitignored by default; `automil trajectory export` produces a redacted,
schema-validated bundle.

## Test Suite

| Surface | Approx tests |
|---------|------:|
| Total collected | 950 |
| Phase 7 D-198 11-clause acceptance | 11 |
| Phase 8 D-208 11-clause acceptance | 11 |
| Phase 6 D-179 11-clause acceptance (extras-gated) | 9 PASS + 2 SKIP |
| Phase 8 final acceptance sub-gates | A (workstation-deferred, `requires_ccrcc_data`), B (CI: sklearn-iris end-to-end via real orchestrator subprocess), C (workstation-deferred) |
| Phase 8 framework purity grep gate | 3-test gate, 5-entry content-anchor allowlist |
| Pitfall-6 single-file gate (held-out isolation) | 3 tests, 35 D-149 assertions |

`tests/acceptance/` carries the milestone-level gates. Per-phase tests
live alongside the modules they cover (`tests/cli/`, `tests/backends/`,
`tests/cells/`, `tests/gate/`, `tests/trajectory/`, etc.).

### Known tech debt at v1.0 close

| Item | Phase | Disposition |
|------|------:|-------------|
| 3 pre-existing `tick_cells` failures (Phase 4-origin) | 4 | Tracked as Phase 6 follow-up #1 in STATE.md |
| `tests/test_per_fold_writer.py` collection error (autobench import) | 4/8 | Pre-existing since Phase 7 verification |
| Phase 5 calibration pilot K-determination | 5 | Workstation; scaffold ready |
| Real SLURM cluster + multi-node Ray verification | 6 | `@pytest.mark.requires_slurm` / `requires_ray`; nightly only |
| External hardware shape verification (CPU-only laptop, ROCm) | 7 | D-197 documented-MEDIUM portability with override path |
| Phase 8 sub-gate A (CCRCC `node_0176` ±0.005) | 8 | Workstation; needs `AUTOBENCH_CCRCC_ROOT` |
| Phase 8 sub-gate C (heterogeneous consumers in same project) | 8 | Body is `pytest.skip()` per 08-09; workstation completion needed |
| graph.json schema_version bump for D-200 dict-spread | 8 | v1.1 |
| results.tsv generalization (4-key autobench shape retained) | 8 | Generalize when third consumer surfaces |
| viz dashboard generic-metric rendering | 8 | Post-v1 |

The full cross-phase integration audit lived at
`.planning/milestones/v1.0-MILESTONE-AUDIT.md`, removed from the working tree
on 2026-08-06; recover it with
`git show 7de69a8:.planning/milestones/v1.0-MILESTONE-AUDIT.md`.

## Pitfall-Driven Anti-Acceptance

Each phase ships a defense against a specific failure mode:

| Pitfall | Defense | Phase |
|---------|---------|------:|
| 1. Still uses old path | Disable-old gate + protected-files validator | 1 |
| 2. Leaky backend ABC | MockSLURM in parallel with LocalBackend; ABC locked against ≥2 implementations | 2 |
| 3. Multi-runtime untested-but-claimed | ≥2 runtimes end-to-end smoke test | 3 |
| 4. Mid-fold guillotine | Per-fold checkpoint protocol ships WITH cap | 4 |
| 5. Trajectory leak/bloat/fossilize | Redaction-on-capture + bounded JSONL + schema-version metadata | 3 |
| 6. Gate calibration | Pre-registered held-out manifest + paired statistical test | 5 |
| 7. Decoupling shipped wrong | sklearn-iris second consumer + end-to-end | 8 |
| 8. Hardware mis-detect | Detect-and-warn pattern + ≥3 hardware shapes | 7 |
| 9. Setup skill mis-scaffold | Mandatory `automil check` + 1-min dry-run gate | 7 |

## What v1.0 Means

`v1.0` is the **milestone tag** marking the F2-readiness framework refactor.
The `pyproject.toml` package version (`0.1.0` at v1.0 close) is independent;
external publication of a `1.0.0` release is the next decision after the
deferred workstation UAT items resolve.

The core value statement is the contract:

> An agent can autonomously discover model improvements for any user's
> training code under a configurable per-cell time budget (the framework
> fallback is 6h; the frozen preprint campaign pins 12 agent-active hours plus
> 30 launches), with discovered variants reproducible, attributable to
> their parents, and portable across machines and LLM runtimes.

All 9 phases shipped, 92 plans executed, 69 v1 requirements delivered.
The D-208 11-clause acceptance gate green; sklearn-iris sub-gate B green
in CI; CCRCC `node_0176` reproduction sub-gate A workstation-deferred but
fixture-wired; framework purity grep gate green.

## See Also

- [Getting Started](getting-started.md), installation, configuration, first submit
- [Training-Script Contract](training-script-contract.md), the seam between framework and consumer
- [Agent Compatibility](agent-compatibility.md), per-runtime setup, overlay merge model
- [CHANGELOG](../CHANGELOG.md), v1.0 milestone release notes

The milestone audit, roadmap, and REQUIREMENTS traceability that this report
cross-referenced lived under `.planning/` and were removed on 2026-08-06.
Recover any of them from git history, e.g.
`git show 7de69a8:.planning/milestones/v1.0-REQUIREMENTS.md`.
