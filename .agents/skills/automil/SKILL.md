---
name: automil
description: Run the autonomous MIL experiment loop. Requires setup first (use /automil-setup).
---

# autoMIL Experiment Loop

Run the autonomous experiment loop. Setup must be completed first via
`/automil-setup`.

## Pre-flight

1. `cd` to the directory containing `automil/config.yaml`
2. Verify setup: `uv run automil check` (must pass with no issues)
3. Start orchestrator in a **tmux** session (it must stay running):
   ```bash
   tmux new -s orchestrator
   uv run automil orchestrator start
   # Ctrl-b d to detach
   ```
4. Start the agent loop in another tmux session with `--dangerously-skip-permissions`
   so it can run autonomously without prompts:
   ```bash
   tmux new -s automil
   Codex --dangerously-skip-permissions
   # Then type: /automil
   ```
5. Start loop flag: `uv run automil start-loop`

## Important: File paths are git-root-relative

All file paths in `files.editable`, `uv run automil submit --files`, and `run.command`
are relative to the **git repo root**, not to where automil/ lives. The
orchestrator creates worktrees from the git root, so overlay paths must match.

## Run

1. Read `automil/config.yaml`, `automil/graph.json`, `automil/learnings.md`
2. Read the training script and key source files from `files.editable`
3. Run `uv run automil reconcile` to sync graph state

Then follow Phase 2 in `automil/program.md`:

**LOOP FOREVER:**

1. `uv run automil reconcile`
2. `uv run automil rank` to get top proposals. If none, brainstorm new ones.
3. Read `automil/learnings.md` to avoid repeating failures.
4. For each proposal:
   a. Edit project files to implement the idea
   b. `uv run automil submit --node <id> --desc "..." --files <changed files>`
   c. Restore working tree: `git checkout -- <files>`
5. Wait for completions in `automil/orchestrator/completed/`
6. `uv run automil reconcile` to update graph
7. Update `automil/learnings.md`
8. If improved: commit winning changes
9. If no proposals: brainstorm, `uv run automil propose`
10. Repeat

## Rules

- NEVER STOP while `.automil_active` exists
- Use `uv run automil submit` for every experiment (not manual runs)
- Use `uv run automil rank` to pick experiments (not random)
- Update `automil/learnings.md` after every result
- Commit winning experiments to git
- File paths in submit --files must be relative to git repo root

## Stopping

User runs `uv run automil stop-loop` to allow the agent to exit.
