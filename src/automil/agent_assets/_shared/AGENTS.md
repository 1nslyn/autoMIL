# AGENTS

This project uses autoMIL — an autonomous experiment framework for ML.

## How to work in this repo

- Read `automil/program.md` for the experiment goals.
- Read `automil/learnings.md` before submitting (avoid repeating dead-ends).
- Prefer `uv run ...` for commands whenever uv is available in the consumer
  project. Use the consumer project's declared environment runner only when uv
  is unavailable or the project explicitly requires it.
- Submit experiments via `uv run automil submit`. Never run training scripts
  directly.

## Constraints

- Budget: consumer-declared per cell in `automil/config.yaml` (`cap.budget`,
  time axis — 6h is only the framework fallback) and, when set,
  `cap.eval_budget` — a hard cap on **launched attempts**. Every launched
  attempt is charged, including crashes and budget-kills; check remaining
  with `uv run automil cell status`.
- Trajectories captured automatically (gitignored by default).

## Runtime

- Set `AUTOMIL_RUNTIME` to declare your runtime (e.g. `export AUTOMIL_RUNTIME=claude-code`).
- Use `uv run automil show-skill --runtime <name>` to view the runtime-specific
  setup guide.
