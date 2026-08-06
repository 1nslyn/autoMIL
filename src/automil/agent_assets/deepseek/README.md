# DeepSeek in autoMIL

DeepSeek is a **model**, not a runtime. It does not have its own agent runtime framework.

## How to use DeepSeek with autoMIL

DeepSeek models are routed through one of the following agent runtimes:

- **opencode** — use `uv run automil init --runtime opencode` (recommended; opencode supports
  DeepSeek models via its model selection configuration)
- **Codex** — use `uv run automil init --runtime codex` (CLI fallback; no native hook support
  in Phase 3)

When using DeepSeek via opencode, the opencode TypeScript plugin installed by
`uv run automil init --runtime opencode` handles trajectory capture automatically.
Set `AUTOMIL_RUNTIME=deepseek-via-opencode` to tag trajectories correctly.

## Why there is no deepseek/ runtime overlay

autoMIL's runtime overlays cover agent runtime frameworks (Claude Code, opencode, Codex),
not model providers. DeepSeek is a model choice within those frameworks.
See `agent_assets/_shared/AGENTS.md` for the universal instructions applicable to all runtimes.
