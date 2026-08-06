# Codex hook integration — CLI fallback

## Status (Phase 3)

Codex's hook API surface is **unstable as of 2026-05** (D-100). Phase 3 delivers
CLI-fallback trajectory capture only. Native Codex hook integration will be added
in Phase 4 or after Codex stabilises.

## How to capture trajectories with Codex

Since Codex does not have a stable hook API, use the CLI fallback directly:

1. After each tool call, invoke `uv run automil trajectory record` with the event JSON:

```bash
export AUTOMIL_NODE_ID="node_0001"
export AUTOMIL_RUNTIME="codex"
uv run automil trajectory record '{"gen_ai.provider.name":"codex","gen_ai.event.name":"tool_call","gen_ai.event.timestamp":"2026-05-03T00:00:00Z","gen_ai.tool.name":"bash"}'
```

2. Add this invocation to your Codex agent's task instructions (`AGENTS.md` or
   `.codex/instructions.md`): "After each tool use, record the event with
   `uv run automil trajectory record`."

## DeepSeek via Codex

If routing DeepSeek through Codex, set:
```bash
export AUTOMIL_RUNTIME="deepseek-via-codex"
```

The trajectory will be tagged with `deepseek-via-codex` as the runtime.
