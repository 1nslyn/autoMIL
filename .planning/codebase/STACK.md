# Technology Stack

**Analysis Date:** 2026-04-30

## Languages

**Primary:**
- Python 3.10+ (`requires-python = ">=3.10"` in `pyproject.toml`) — all framework code (`src/automil/`) and benchmark code (`benchmarks/src/autobench/`)

**Secondary:**
- JavaScript (ES) — viz frontend in `src/automil/viz/static/app.js` (vanilla, no build step)
- HTML/CSS — viz dashboard in `src/automil/viz/static/index.html`, `src/automil/viz/static/style.css`
- Bash — Slurm submission scripts in `benchmarks/scripts/slurm/submit_*.sh`, Claude Code hook in `src/automil/claude_assets/hooks/on_stop.sh`
- Jinja2 templates (`.j2`) — `src/automil/templates/{config.yaml.j2, program.md.j2, learnings.md.j2, .gitignore.j2}`
- YAML — dataset configs in `benchmarks/datasets/*.yaml`, project config in `automil/config.yaml`

## Runtime

**Environment:**
- CPython 3.10+ (target — `>=3.10` baseline; uv.lock has wheels for 3.10–3.14)
- Linux (developed/run on `Linux 6.17.0-14-generic`; CUDA-only paths)

**Package Manager:**
- `uv` (Astral) — workspace root at `/home/jma/Documents/yinshuol/autoMIL/`
- Lockfile: `uv.lock` present (≈100k lines, full transitive resolution)
- `uv` workspace declaration in root `pyproject.toml`:
  ```toml
  [tool.uv.workspace]
  members = ["benchmarks"]
  ```
- Both packages installed editable via `pip install -e .` / `uv sync`

**Workspace layout:**
- `automil` package — `src/automil/` (framework: CLI, orchestrator, graph, viz)
- `autobench` package — `benchmarks/src/autobench/` (MIL benchmark suite, separate `pyproject.toml` at `benchmarks/pyproject.toml`)

## Frameworks

**Core (automil framework — `src/automil/pyproject.toml`):**
- `click >= 8.1` (resolved 8.3.1) — CLI grouping/commands in `src/automil/cli.py`
- `aiohttp >= 3.9` (resolved 3.13.3) — async HTTP server with SSE in `src/automil/viz/server.py`
- `watchdog >= 4.0` (resolved 6.0.0) — inotify-based filesystem watcher for `graph.json`/`gpu_state.json` updates (`src/automil/viz/server.py`)
- `jinja2 >= 3.1` (resolved 3.1.6) — template rendering for `automil init` (`src/automil/cli.py:79`)
- `pyyaml >= 6.0` (resolved 6.0.3) — config parsing in `src/automil/cli.py`, `src/automil/orchestrator.py`, `benchmarks/src/autobench/config.py`
- `torch >= 2.10.0` (resolved 2.10.0) — declared as a hard dependency on automil itself; actually used inside autobench

**Core (autobench package — `benchmarks/pyproject.toml`):**
- `trident @ file:///home/jma/Documents/yinshuol/autoMIL/benchmarks/lib/TRIDENT` (resolved 0.2.3) — Mahmood Lab WSI preprocessing/feature extraction, vendored as a path dependency
- `transformers >= 4.35, < 5` (resolved 4.57.6) — pathology foundation model loading
- `python-dotenv` — `.env` file loading in `benchmarks/scripts/run_experiment.py`, `benchmarks/scripts/run_feature_extraction.py`
- `scikit-learn >= 1.3` (resolved 1.7.2/1.8.0) — `StratifiedKFold` splits, `sklearn.metrics` evaluation
- `scipy >= 1.10`, `h5py >= 3.0` (resolved 3.16.0), `tqdm >= 4.60`
- `pandas >= 2.0`, `pyyaml >= 6.0`
- `wandb >= 0.25.0` (resolved 0.25.1) — experiment logging, optional via `--no_wandb`
- `tensorboardx >= 2.6.4`, `tensorboard >= 2.20.0` — CLAM tensorboard writes captured by wandb
- `beartype >= 0.22.9`, `jaxtyping >= 0.3.7` — runtime type checking (used by TRIDENT)
- `torch-geometric >= 2.7.0` — graph models (e.g. `wikg_mil`)
- `scikit-image >= 0.25.2` — image preprocessing

**Testing:**
- `pytest >= 9.0.2` — single test runner for both packages (declared in `[dependency-groups].dev` in both `pyproject.toml` files)
- Test discovery: `[tool.pytest.ini_options].testpaths = ["tests"]` in both packages
- 48 tests total (per `CLAUDE.md`): `tests/test_graph.py` (26), `tests/test_runner.py` (7), `tests/test_cli.py` (5), `tests/test_integration.py` (10)
- autobench tests in `benchmarks/tests/` (15+ files: `test_benchmark_*.py`, `test_config.py`, `test_data.py`, `test_encoders.py`, etc.)

**Build/Dev:**
- `hatchling` — build backend for both packages (`[build-system].requires = ["hatchling"]`)
- Wheel layout: `[tool.hatch.build.targets.wheel].packages = ["src/automil"]` and `["src/autobench"]`
- `[tool.hatch.metadata].allow-direct-references = true` in `benchmarks/pyproject.toml` to permit the `file:///` TRIDENT URL

## Key Dependencies

**Critical (framework core):**
- `click 8.3.1` — without this `automil` CLI doesn't load
- `aiohttp 3.13.3` — viz dashboard cannot start (`src/automil/viz/server.py:24` hard-fails import)
- `watchdog 6.0.0` — viz cannot detect graph updates (`src/automil/viz/server.py:30`)
- `pyyaml 6.0.3` — `automil/config.yaml` and dataset YAMLs unparseable without it
- `jinja2 3.1.6` — `automil init` template rendering

**Critical (ML pipeline):**
- `torch 2.10.0` + `torchvision 0.25.0` — every training pipeline (`autobench.pipeline.{clam,nnmil,smmile}.train`)
- `trident 0.2.3` (vendored, path dep) — patch encoder factory, segmentation, feature extraction; consumed in `benchmarks/scripts/run_feature_extraction.py:50` and `benchmarks/src/autobench/encoders/h0_mini.py:2`
- `transformers 4.57.6` — `AutoModel.from_pretrained` for HuggingFace pathology encoders (Virchow2, UNI2-h, H-optimus-1, hibou-L, conchv1_5, midnight, H0-mini)
- `timm 0.9.16` — pinned by TRIDENT; ViT backbones for foundation encoders (`benchmarks/src/autobench/encoders/h0_mini.py:12`)
- `huggingface-hub 0.36.2` — model download/cache; gated downloads need `HF_TOKEN`
- `h5py 3.16.0` — patch feature storage (.h5/.pt artifacts in `${data_root}/trident_output/`)
- `openslide-python 1.4.3` + `openslide-bin 4.0.0.13` — WSI .svs reading (transitive via TRIDENT)
- `opencv-python 4.13.0.92` — image ops (transitive via TRIDENT)
- `pandas 2.3.3`, `numpy 2.2.6`, `scipy 1.15.3`, `scikit-learn 1.7.2` — data/eval

**Critical (CUDA stack — pulled in by torch):**
- `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cufft-cu12`, `nvidia-curand-cu12`, `nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusparselt-cu12`, `nvidia-nccl-cu12`, `nvidia-nvjitlink-cu12`, `nvidia-nvshmem-cu12`, `nvidia-nvtx-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-cupti-cu12`, `cuda-bindings`, `cuda-pathfinder`, `triton`
- All resolved in `uv.lock`; CPU-only execution is not exercised in this codebase

**Infrastructure:**
- `wandb 0.25.1` — optional experiment tracking (training scripts gate on `wandb_project is not None`); `WANDB_API_KEY` env var
- `tensorboard 2.20.0` + `tensorboardx 2.6.4` — auto-captured by wandb in `benchmarks/src/autobench/pipeline/clam/train.py`
- `pytest 9.0.2` — dev-only

**Vendored frontend (no npm/build step):**
- `src/automil/viz/static/vendor/d3.v7.min.js` (273 KB)
- `src/automil/viz/static/vendor/three.min.js` (619 KB)
- `src/automil/viz/static/vendor/3d-force-graph.min.js` (1.18 MB)
- `src/automil/viz/static/vendor/three-spritetext.min.js` (9 KB)
- Loaded via `<script src="...">` tags in `src/automil/viz/static/index.html`; committed to repo (see commit `137aa70 chore(viz): vendor d3, three, three-spritetext, 3d-force-graph`).

## Configuration

**Environment:**
- Per-project config rendered by `automil init` from `src/automil/templates/config.yaml.j2` into `automil/config.yaml` (kept in user project, not in this repo)
- Per-dataset config: `benchmarks/datasets/{ovarian,clwd,ccrcc,hancock,tcga_luad,tcga_template,placeholder}.yaml`
- Path interpolation: `${ENV_VAR}` and `${ENV_VAR:default}` plus `${field_name}` cross-references, implemented in `benchmarks/src/autobench/config.py:30` (`_resolve_env_vars`) and `_resolve_paths`
- `.env` loading:
  - `benchmarks/.env` (gitignored, present locally) loaded by `python-dotenv` in `benchmarks/scripts/run_experiment.py:30` and `benchmarks/scripts/run_feature_extraction.py`
  - Also loaded by the orchestrator daemon's `_load_dotenv()` at `src/automil/orchestrator.py:222` (handles `<project_root>/.env` and `<project_root>/benchmarks/.env`) — necessary because git worktrees don't inherit `.env`
- Example env vars in `benchmarks/.env.example`: `HF_TOKEN`, `WANDB_API_KEY`, `AUTOBENCH_OVARIAN_ROOT`, `AUTOBENCH_CLWD_ROOT`, `AUTOBENCH_CCRCC_ROOT`, `AUTOBENCH_HANCOCK_ROOT`, plus 13 total `AUTOBENCH_TCGA_*_ROOT` vars (3 preprint roster — luad/lgg/hnsc — + 10 additional TCGA catalog: blca/coad/skcm/ucec/thca/stad/tgct/paad/pcpg/ucs) and 2 `AUTOBENCH_CPTAC_*_ROOT` preprint roster vars (gbm/pdac)

**Build:**
- Root `pyproject.toml` declares the workspace and the `automil` package
- `benchmarks/pyproject.toml` declares the `autobench` package and its TRIDENT path dep
- No `setup.py`, no `requirements.txt`, no `Pipfile` — uv-only
- No `.python-version` file; Python pinned via `requires-python` only

## Platform Requirements

**Development:**
- Linux (Ubuntu-class) recommended (only platform exercised)
- NVIDIA GPU(s) — `nvidia-smi` queried by orchestrator (`src/automil/orchestrator.py:99`) and `automil check` (`src/automil/cli.py:619`); fallback to GPU 0 if missing
- CUDA toolkit + drivers compatible with torch 2.10.0
- `git` ≥ 2.5 — git worktrees used for experiment isolation (`src/automil/runner.py:30`)

**Production:**
- Same as development; the framework runs as a long-lived daemon on a workstation/HPC node
- Orchestrator daemon: `src/automil/orchestrator.py` writes `automil/orchestrator/{orchestrator.pid, orchestrator.log, gpu_state.json}`
- Viz server: `src/automil/viz/server.py` listens on `0.0.0.0:8420` by default

---

*Stack analysis: 2026-04-30*
