# External Integrations

**Analysis Date:** 2026-04-30

## APIs & External Services

**Model registries (HuggingFace Hub):**
- HuggingFace Hub — pathology foundation model downloads via `transformers.AutoModel.from_pretrained` and `timm.create_model("hf-hub:...")`
  - SDK/Client: `huggingface-hub 0.36.2`, `transformers 4.57.6`, `timm 0.9.16`
  - Auth: `HF_TOKEN` env var (set in `benchmarks/.env`, declared in `benchmarks/.env.example:5`)
  - Models referenced from dataset YAMLs (`benchmarks/datasets/*.yaml` `encoders.models`):
    - `paige-ai/Virchow2` → `virchow2` (dim 2560)
    - `MahmoodLab/UNI2-h` → `uni_v2` (dim 1536)
    - `bioptimus/H-optimus-1` → `hoptimus1` (dim 1536)
    - `histai/hibou-L` → `hibou_l` (dim 1024)
    - `MahmoodLab/conchv1_5` → `conch_v15` (dim 768)
    - `kaiko-ai/midnight` → `midnight12k` (dim 1536)
    - `bioptimus/H0-mini` → `h0_mini` (dim 768)
  - Custom encoder loader: `benchmarks/src/autobench/encoders/h0_mini.py` uses `timm.create_model("hf-hub:bioptimus/H0-mini", pretrained=True)`
  - Standard loader path: `benchmarks/scripts/run_feature_extraction.py:50` calls `trident.patch_encoder_models.encoder_factory(encoder_key)`
  - CLAM also references `MahmoodLab/TITAN` (`benchmarks/lib/CLAM/models/builder.py:62`) and the `conch` open-clip variant (CONCH ViT-B-16 from `benchmarks/lib/CLAM/models/builder.py:14, 54`)

**Experiment tracking:**
- Weights & Biases — optional run logging during CLAM training
  - SDK/Client: `wandb 0.25.1` (lazy-imported inside `benchmarks/src/autobench/pipeline/clam/train.py:136`)
  - Auth: `WANDB_API_KEY` env var (declared in `benchmarks/.env.example:8`)
  - Project name set per-dataset: `f"{ds.name}-automil"` in `benchmarks/scripts/run_experiment.py:193`; disabled with `--no_wandb`
  - Captures CLAM's tensorboard writes automatically (see comment at `benchmarks/src/autobench/pipeline/clam/train.py:133`)
- TensorBoard — local logging
  - SDK: `tensorboard 2.20.0`, `tensorboardx 2.6.4`
  - No remote server; logs written next to results in `automil/orchestrator/archive/<node_id>/results/`

## Data Storage

**Databases:**
- None — this project does not use a relational/document database.

**Filesystem-based experiment store (the framework's "database"):**
- `automil/graph.json` — experiment tree (single JSON file, sole writer is `ExperimentGraph.save()` in `src/automil/graph.py`)
- `automil/results.tsv` — append-only results table; sole writer is `ExperimentOrchestrator._append_results_tsv` (`src/automil/orchestrator.py:611`); never written by training scripts (per `CLAUDE.md` "Result Contract")
- `automil/orchestrator/queue/<node>.json` — pending experiment specs
- `automil/orchestrator/running/<node>.json` — in-flight specs (used for orphan recovery on restart, `src/automil/orchestrator.py:295`)
- `automil/orchestrator/archive/<node>/` — per-experiment artifacts: `spec.json`, `result.json`, `run.log`, `results/` (per-fold metrics), plus snapshotted overlay files
- `automil/orchestrator/completed/<node>.json` — completion notifications consumed by `automil reconcile`
- `automil/orchestrator/{orchestrator.pid, orchestrator.log, gpu_state.json, viz_server.pid, viz_server.log}` — daemon state
- `.automil_active` flag at project root — set by `automil start-loop`, prevents Claude Code Stop hook from terminating the agent (`src/automil/cli.py:549`, `src/automil/claude_assets/hooks/on_stop.sh`)
- `.automil_worktrees/<node>/` — per-experiment git worktrees created by `Runner.create_worktree` (`src/automil/runner.py:23`)

**File Storage:**
- Local filesystem only — no S3 / GCS / Azure Blob
- WSI images (`.svs`): user-provided per dataset (`${AUTOBENCH_<DATASET>_ROOT}/wsi/`)
- Extracted features: `${data_root}/trident_output/20x_224px_0px_overlap/` (HDF5 + PyTorch tensors), produced by TRIDENT in `benchmarks/scripts/run_feature_extraction.py`

**Caching:**
- HuggingFace Hub model cache — default `~/.cache/huggingface/`
- TRIDENT local checkpoint cache — `benchmarks/lib/TRIDENT/trident/{patch_encoder_models,slide_encoder_models,segmentation_models}/local_ckpts.json`
- Features cached on disk per encoder (re-extraction skipped if present)

## Authentication & Identity

**Auth Provider:**
- None — autoMIL has no user authentication. Single-user research framework.
- Viz server binds to `0.0.0.0:8420` (`src/automil/viz/server.py:260`) with `Access-Control-Allow-Origin: *` for SSE (`src/automil/viz/server.py:166`); intended for trusted local/LAN access only.

**Service-account-style credentials:**
- `HF_TOKEN` — required for gated HuggingFace models (Virchow2, UNI2-h, etc.); read from environment after `dotenv` load
- `WANDB_API_KEY` — required if wandb logging is enabled

## Monitoring & Observability

**Error Tracking:**
- None — no Sentry, no Bugsnag. (`sentry-sdk` appears in `uv.lock` only as a wandb transitive dependency, not used directly.)

**Logs:**
- Python `logging` module — orchestrator logs to `automil/orchestrator/orchestrator.log` and stdout (`src/automil/orchestrator.py:759`)
- Viz server logs to `automil/orchestrator/viz_server.log` (`src/automil/viz/server.py:244`)
- Per-experiment training logs written to `automil/orchestrator/archive/<node>/run.log` (subprocess stdout+stderr captured by `subprocess.Popen` at `src/automil/orchestrator.py:440`)
- No structured JSON logging; plain text with `%(asctime)s [%(levelname)s] %(message)s` format

**Metrics:**
- GPU state polled every `poll_interval_sec` (default 5s) via `nvidia-smi` (`src/automil/orchestrator.py:98`); written to `automil/orchestrator/gpu_state.json` for the viz server
- No Prometheus / OpenTelemetry / StatsD integration

**Health checks:**
- `automil status` and `automil orchestrator status` print state to stdout
- `automil check` (`src/automil/cli.py:569`) validates project setup: training script presence, data paths, GPU availability, orchestrator dirs

## CI/CD & Deployment

**Hosting:**
- Self-hosted only — runs on a workstation or HPC node
- No deployment target (no Dockerfile, no Kubernetes manifest, no Helm chart in the repo)
- Slurm-style submission scripts under `benchmarks/scripts/slurm/submit_*.sh` for HPC clusters (`submit_benchmark.sh`, `submit_feature_extraction.sh`, `submit_feature_extraction_mig.sh`, `submit_3dataset_benchmark.sh`, `submit_test.sh`, `submit_test_extraction.sh`, `submit_gdc_download.sh`)

**CI Pipeline:**
- None — no `.github/workflows/`, no `.gitlab-ci.yml`, no Jenkinsfile detected at repo root.

## Environment Configuration

**Required env vars (set in `benchmarks/.env`, template at `benchmarks/.env.example`):**

| Var | Purpose | Required |
|-----|---------|----------|
| `HF_TOKEN` | HuggingFace gated model access | Yes (for foundation encoders) |
| `WANDB_API_KEY` | W&B run logging | Optional |
| `AUTOBENCH_OVARIAN_ROOT` | Ovarian BRCA/HRD dataset root | Per-dataset |
| `AUTOBENCH_CLWD_ROOT` | CLWD lung adeno dataset root | Per-dataset |
| `AUTOBENCH_CCRCC_ROOT` | CPTAC CCRCC dataset root | Per-dataset |
| `AUTOBENCH_HANCOCK_ROOT` | Hancock dataset root | Per-dataset |
| `AUTOBENCH_TCGA_LUAD_ROOT` | TCGA-LUAD root (preprint roster — KRAS, binary) | Per-dataset |
| `AUTOBENCH_TCGA_LGG_ROOT` | TCGA-LGG root (preprint roster — IDH1, binary) | Per-dataset |
| `AUTOBENCH_TCGA_HNSC_ROOT` | TCGA-HNSC root (preprint roster — tumor grade, 3-class) | Per-dataset |
| `AUTOBENCH_CPTAC_GBM_ROOT` | CPTAC-GBM root (preprint roster — TP53, binary) | Per-dataset |
| `AUTOBENCH_CPTAC_PDAC_ROOT` | CPTAC-PDAC root (preprint roster — immune_class, 3-class) | Per-dataset |
| `AUTOBENCH_TCGA_GBM_ROOT` | TCGA-GBM root (Yeonwoo; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_BRCA_ROOT` | TCGA-BRCA root (Yeonwoo; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_BLCA_ROOT` | TCGA-BLCA root (Yeonwoo; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_COAD_ROOT` | TCGA-COAD root (Keishi; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_UCEC_ROOT` | TCGA-UCEC root (Keishi; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_SKCM_ROOT` | TCGA-SKCM root (Keishi; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_THCA_ROOT` | TCGA-THCA root (Keishi; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_STAD_ROOT` | TCGA-STAD root (Keishi; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_CESC_ROOT` | TCGA-CESC root (Ryan; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_TGCT_ROOT` | TCGA-TGCT root (Ryan; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_PAAD_ROOT` | TCGA-PAAD root (Ryan; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_PCPG_ROOT` | TCGA-PCPG root (Ryan; catalog, not preprint roster) | Per-dataset |
| `AUTOBENCH_TCGA_UCS_ROOT` | TCGA-UCS root (Ryan; catalog, not preprint roster) | Per-dataset |

**Loading order:**
1. `benchmarks/scripts/run_experiment.py:30` — `dotenv.load_dotenv("../.env")` at script start
2. `benchmarks/scripts/run_feature_extraction.py` — same
3. `src/automil/orchestrator.py:222` — `_load_dotenv()` reads `<project_root>/.env` and `<project_root>/benchmarks/.env` and propagates to subprocess `env=` (worktrees do not inherit gitignored `.env`)

**Resolution:**
- `benchmarks/src/autobench/config.py:30` `_resolve_env_vars()` resolves `${VAR}` / `${VAR:default}`; raises `ValueError("Environment variable ${VAR} is not set and no default provided")` on missing required vars

**Orchestrator-injected vars (per-experiment, set in `src/automil/orchestrator.py:419`):**
- `CUDA_VISIBLE_DEVICES=<gpu_id>` — masks physical GPU; the worker sees logical device 0
- `AUTOMIL_GPU=0` — logical GPU index for training scripts
- `AUTOMIL_DESC=<spec.description>`
- `AUTOMIL_NODE_ID=<node_id>`
- `AUTOMIL_RESULTS_DIR=<archive_dir_absolute>` — used by `benchmarks/scripts/run_experiment.py:171` to redirect per-fold checkpoints out of the shared `benchmark_dir`
- `AUTOBENCH_ROOT=<worktree>/benchmarks` — overrides the editable-install pointer so worktree overlays of `benchmarks/lib/` and `benchmarks/src/autobench/` are respected (`benchmarks/src/autobench/__init__.py:12`)
- `PYTHONPATH=<worktree>/benchmarks/src:<existing_PYTHONPATH>`

**Secrets location:**
- `benchmarks/.env` (gitignored — see `.gitignore` line 11)
- Also gitignored: `.env`, `*.log`, `*.pid` (`.gitignore` lines 7–10)
- No secrets manager (Vault, AWS Secrets Manager, etc.)

## Webhooks & Callbacks

**Incoming:**
- None — autoMIL exposes no webhook endpoints.

**Outgoing:**
- W&B run lifecycle calls (`wandb.init`, `wandb.log`, `wandb.finish`) — only when `wandb_project` is non-None

## Cross-Process / Cross-Boundary Surfaces

**OS subprocess boundaries (the orchestrator's primary integration surface):**
- `git` CLI — invoked by `src/automil/cli.py:307` (`git diff --name-only`, `git ls-files --others`, `git rev-parse HEAD`) and `src/automil/runner.py` (`git worktree add --detach`, `git worktree remove --force`, `git worktree prune`)
- `nvidia-smi` — `src/automil/orchestrator.py:101` polls `--query-gpu=index,memory.total,memory.free,utilization.gpu`; `src/automil/cli.py:619` for `automil check`
- Training-script subprocesses — `subprocess.Popen([sys.executable, run_script], cwd=worktree, env=...)` in `src/automil/orchestrator.py:440`
- Signal handling — `SIGTERM`/`SIGINT` for graceful drain (`src/automil/orchestrator.py:712`, `src/automil/viz/server.py:267`)

**Git worktrees as an integration boundary:**
- Each experiment runs in `<project_root>/.automil_worktrees/<node_id>/` (detached worktree at `spec.base_commit`)
- Overlay applied from `automil/orchestrator/archive/<node>/` (modified files + deletions list) — `src/automil/runner.py:37`
- `result.json` collected back from worktree to archive — `src/automil/runner.py:62`
- Worktrees are `.gitignored` (`.gitignore` line 17 implicitly via `tasks/` etc.; `.automil_worktrees/` is created by `Runner.__init__` at `src/automil/runner.py:17`)

**Browser frontend (HTTP/SSE):**
- Server: `src/automil/viz/server.py` — aiohttp on port 8420
- Endpoints:
  - `GET /` — serves `src/automil/viz/static/index.html`
  - `GET /events` — Server-Sent Events stream of `graph_update` JSON envelopes (`src/automil/viz/server.py:159`)
  - `GET /static/*` — vendored JS/CSS
- Client uses `EventSource` (browser native) plus vendored `d3.v7`, `three`, `three-spritetext`, `3d-force-graph`
- Update trigger: `watchdog.Observer` watches `automil/` and `automil/orchestrator/` for changes to `graph.json` and `gpu_state.json` (`src/automil/viz/server.py:250`)
- No WebSocket; SSE only. No CSRF/auth; trusted-network assumption.

**Claude Code integration:**
- Skills installed by `automil init` from `src/automil/claude_assets/skills/` into `<project_root>/.claude/skills/{automil,automil-setup}/SKILL.md`
- Stop hook registered in `<project_root>/.claude/settings.json` pointing at `src/automil/claude_assets/hooks/on_stop.sh` (copied during init); prevents Claude from exiting while `.automil_active` flag is set

## Vendored MIL Libraries (sys.path overlays, not pip packages)

These live under `benchmarks/lib/` and are loaded via `sys.path` insertion rather than as installed packages. Each has its own license/origin and is treated as an external dependency.

| Library | Path | Loaded via | License | Use |
|---------|------|------------|---------|-----|
| CLAM | `benchmarks/lib/CLAM/` | `benchmarks/src/autobench/pipeline/clam/_imports.py:14` (prepends `LIB_ROOT/CLAM` to sys.path) | `benchmarks/lib/CLAM/LICENSE.md` | CLAM_SB / CLAM_MB / MIL_fc models, training utilities, dataset_modules; original Mahmood Lab CLAM repo |
| nnMIL | `benchmarks/lib/nnMIL/` | `benchmarks/src/autobench/pipeline/nnmil/_imports.py:13` (prepends `LIB_ROOT` so `from nnMIL.training...` resolves) | (vendored) | `ClassificationTrainer`, `ExperimentPlanner`, `load_plan`; backbone for ab_mil, trans_mil, ds_mil, dtfd_mil, ilra_mil, wikg_mil, simple_mil, vision_transformer, rrt |
| SMMILe | `benchmarks/lib/SMMILe/` | `benchmarks/src/autobench/pipeline/smmile/_imports.py:13` (prepends `LIB_ROOT/SMMILe/single`) | `benchmarks/lib/SMMILe/LICENSE` | `SMMILe_SINGLE` model, single-task superpixel-aware MIL |
| TRIDENT | `benchmarks/lib/TRIDENT/` | Installed as a `file://` path dep in `benchmarks/pyproject.toml:11` (real pip install, unlike the others) | CC BY-NC-ND 4.0 (`benchmarks/lib/TRIDENT/LICENSE`) | WSI segmentation, patching, and patch-encoder factory; Mahmood Lab |

**Why three of these are sys.path overlays (not pip installs):**
- They use bare module names (e.g. `from models.model_clam import CLAM_SB`) that would collide if installed as packages.
- The `_imports.py` files are the single chokepoint for sys.path manipulation per the comment in each file: "no other module has to touch sys.path".
- `AUTOBENCH_ROOT` env var (set by orchestrator) overrides `BENCHMARKS_ROOT` so worktrees see their own `lib/` instead of the editable-installed parent (`benchmarks/src/autobench/__init__.py:12`).

## Result Contract (cross-boundary)

Training scripts must write `result.json` to their worktree CWD; the orchestrator collects it via `Runner.collect_result` (`src/automil/runner.py:62`). Schema (per `CLAUDE.md` and `benchmarks/scripts/run_experiment.py:85`):

```json
{
  "status": "completed",
  "metrics": {"val_auc": 0.87, "val_bacc": 0.81, "test_auc": 0.87, "test_bacc": 0.83},
  "composite": 0.85,
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500
}
```

Orchestrator-derived statuses on missing/non-zero exit: `oom` (log contains `CUDA out of memory` / `OutOfMemoryError`), `timeout`, `crash`, `completed` — set in `src/automil/orchestrator.py:510`.

---

*Integration audit: 2026-04-30*
