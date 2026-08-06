# autoMIL Setup, Codex notes

This file is the Codex per-runtime overlay for the automil-setup skill. Codex
renders the merged result as plain markdown without YAML frontmatter, per the
Codex AGENTS.md and Skills conventions.

The shared canonical content lives at
`src/automil/agent_assets/_shared/skills/automil-setup/SKILL.md`. _overlay.py
merges it with this file at install time and writes the result to
`.codex/instructions.md` (per cli/init.py codex branch).

## Codex-specific notes

When operating under Codex CLI:

- Use `bash` tool for the `automil` invocations called out in the shared
  Setup-Done Gate section.
- Codex's working directory model expects you to `cd` into the project root
  before running `uv run automil init`.
- Codex does not parse YAML frontmatter; the shared SKILL.md's frontmatter is
  intentionally absent from the rendered output for this runtime.
