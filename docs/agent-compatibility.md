# Agent Compatibility Guide

autoMIL's framework core is runtime-agnostic; the only requirements are
read access to files, write access to your project tree, and shell access
to invoke `automil` commands. Beyond that, autoMIL ships first-class
runtime overlays for **Claude Code**, **Codex**, **OpenCode**, and
**DeepSeek** (routed via opencode or codex).

The bare `automil` examples in this guide assume the CLI was installed with
`uv tool install`. When working from the autoMIL source repository, use
`uv run automil ...` instead.

The asset layout is `_shared/` (canonical SKILL/AGENTS content) plus
per-runtime directories that carry only diffs/overrides. `automil init`
merges them at install time; `automil show-skill --runtime <r>` renders
the merged result for inspection.

## Runtime Asset Layout

```
src/automil/agent_assets/
├── _shared/                    # canonical content
│   ├── AGENTS.md
│   └── skills/
│       ├── automil/SKILL.md           # the experiment loop
│       └── automil-setup/SKILL.md     # one-time setup
├── claude/
│   └── hooks/on_stop.sh        # stop-prevention hook (Claude-specific)
├── codex/
│   ├── README.md               # CLI-fallback trajectory capture (Codex hook API is unstable as of v1.0)
│   └── skills/automil-setup/   # empty-frontmatter overlay for Codex plain-markdown rendering
├── opencode/
│   └── plugins/automil-trajectory.ts   # TypeScript plugin for trajectory capture
└── deepseek/
    └── README.md               # DeepSeek is a *model*; route via opencode or codex
```

`automil init --runtime <r>` resolves the merge at install time. The
canonical text lives once; runtime overlays only touch the parts that
differ (e.g. Codex needs an empty-frontmatter rendering of `automil-setup/SKILL.md`
because plain-markdown renderers reject the frontmatter).

## Supported Runtimes

| Runtime | Status | Setup | What ships |
|---------|--------|-------|------------|
| **Claude Code** | First-class | `automil init --runtime claude` | `_shared/` skills merged into `.claude/skills/automil/` and `.claude/skills/automil-setup/`; `on_stop.sh` hook installed under `.claude/hooks/` to prevent the agent from stopping mid-loop |
| **Codex** | First-class (CLI fallback for trajectory) | `automil init --runtime codex` | `_shared/` skills merged into `.codex/skills/`; empty-frontmatter overlay applied; trajectory recorded via `automil trajectory record` invoked from agent task instructions (Codex hook API is not yet stable) |
| **OpenCode** | First-class | `automil init --runtime opencode` | `_shared/` skills merged into `.opencode/skills/`; TypeScript plugin `automil-trajectory.ts` installed under `.opencode/plugins/` for automatic trajectory capture |
| **DeepSeek** | First-class via routing | `automil init --runtime deepseek-via-opencode` or `deepseek-via-codex` | DeepSeek is a model, not a runtime. Choose a host runtime (opencode preferred); `AUTOMIL_RUNTIME=deepseek-via-<host>` tags trajectories correctly. See `agent_assets/deepseek/README.md` |
| **Cursor / Aider / Windsurf / others** | Compatible | Manual | Point the agent at `automil/program.md` and the [training-script contract](training-script-contract.md). Any agent that can read files, edit code, and run shell commands works via the file + CLI surface |

## Installing Runtime Assets

```bash
# Auto-detect from existing .claude/, .codex/, .opencode/ dirs
automil init

# Pin one runtime
automil init --runtime claude
automil init --runtime codex
automil init --runtime opencode
automil init --runtime deepseek-via-opencode
automil init --runtime deepseek-via-codex

# Install assets for every supported runtime
automil init --runtime all

# Re-render skills/hooks/AGENTS.md after upgrading autoMIL
automil init --update
```

## Inspecting the Merged Skill

```bash
automil show-skill --runtime claude > /tmp/claude-skill.md
automil show-skill --runtime codex --asset AGENTS
```

`--asset` selects between `SKILL` (default) and `AGENTS`. The output is
the fully merged content (canonical `_shared/` text + per-runtime
overlay applied), useful for verifying what the agent will actually see.

## Trajectory Capture

Per-submit trajectories are written to `archive/<node_id>/trajectory.jsonl`
using OpenTelemetry `gen_ai.*` field names. Capture is runtime-specific:

- **Claude Code**, captured via the `on_stop.sh` hook installed under
  `.claude/hooks/`.
- **OpenCode**, captured via the `automil-trajectory.ts` plugin installed
  under `.opencode/plugins/`.
- **Codex / DeepSeek-via-codex**, captured via CLI fallback. Add
  `automil trajectory record '<event-json>'` to the agent's task
  instructions after each tool call. See
  [`agent_assets/codex/README.md`](../src/automil/agent_assets/codex/README.md).

Secrets (`sk-…`, `hf_…`, `ghp_…`, AWS access keys, `*_API_KEY=…`,
`*_TOKEN=…`) are redacted on capture; per-event 8 KB cap; per-file 5 MB
soft / 50 MB hard rotate. Trajectories are gitignored by default. Export
a redacted, schema-validated bundle:

```bash
automil trajectory export <node_id> --out trajectory_bundle.tgz
```

`AUTOMIL_RUNTIME` is declared, never inferred, set it explicitly in the
agent's environment so trajectories are tagged correctly.

## Universal Requirements

Any compatible agent must be able to:

1. **Read files**, `automil/config.yaml`, `automil/graph.json`,
   `automil/learnings.md`, `automil/program.md`, and your project's source.
2. **Edit files**, any file in your project the agent determines is relevant.
   Files matching `registry.protected` globs in `config.yaml` are hard-rejected
   at submit; those changes must ship as registered variant modules under
   `automil/variants/`.
3. **Run shell commands**, `automil submit`, `automil rank`, `automil reconcile`,
   etc.
4. **Maintain context**, remember what experiments have been tried.
   `automil/learnings.md` is the persistence layer; agents append to it
   between submissions.

## Manual Setup for Non-Listed Runtimes

If your runtime isn't in the table above:

1. Install autoMIL with `uv tool install -e .` from a local clone.
2. Run `automil init --no-healthcheck` (or with healthcheck if you have a GPU).
3. Skip per-runtime asset installation; instead, point the agent at
   `automil/program.md` (instructions for the loop) and
   [`docs/training-script-contract.md`](training-script-contract.md) (the
   seam between framework and consumer).
4. Configure trajectory capture manually by invoking
   `automil trajectory record '<event-json>'` after each tool call (the
   Codex pattern).

## How It Works

The agent interacts with autoMIL entirely through:

- **Files**: `automil/config.yaml`, `automil/graph.json`, `automil/learnings.md`
  (read), `automil/program.md` (read; agent instructions), any project source
  file (edit).
- **CLI**: `automil` commands for all operations (submit, rank, propose,
  reconcile, status, port-variant, nominate, promote, …).
- **`automil/program.md`**: complete narrative for the experiment loop.

No agent-specific APIs or integrations are required beyond basic file and
shell access. Per-runtime overlays only enrich the experience (hooks,
trajectory plugins, skill rendering); the framework itself is runtime-agnostic.

## See Also

- [Getting Started](getting-started.md), installation, configuration, first submit
- [Training-Script Contract](training-script-contract.md), the seam between framework and consumer
- [Implementation Report](implementation-report.md), architecture details, multi-runtime asset layout rationale
- [`agent_assets/deepseek/README.md`](../src/automil/agent_assets/deepseek/README.md), DeepSeek routing
- [`agent_assets/codex/README.md`](../src/automil/agent_assets/codex/README.md), Codex CLI-fallback trajectory capture
