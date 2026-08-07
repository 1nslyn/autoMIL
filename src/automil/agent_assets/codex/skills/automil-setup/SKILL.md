# autoMIL Setup, Codex notes

This file holds Codex-specific notes for the automil-setup skill. Honesty note
(claims-alignment C-d): **no code path currently consumes this file** — the
`init` codex branch copies `_shared/AGENTS.md` verbatim and never calls
`merge_skill` for skills, and `show-skill` looks for overlays at the runtime
directory root, not under `skills/`. It is retained as the D-196 acceptance
artifact (plain markdown, no YAML frontmatter — the format a future codex
merge would require) and as the reference text for operators wiring Codex by
hand. The shared canonical content lives at
`src/automil/agent_assets/_shared/skills/automil-setup/SKILL.md`.

## Codex-specific notes

When operating under Codex CLI:

- Use `bash` tool for the `automil` invocations called out in the shared
  Setup-Done Gate section.
- Codex's working directory model expects you to `cd` into the project root
  before running `uv run automil init`.
- Codex does not parse YAML frontmatter; the shared SKILL.md's frontmatter is
  intentionally absent from the rendered output for this runtime.
