---
name: commit
description: "Smart granular commit workflow that groups changes into separate, logically-grouped commits by feature, functionality, or type. Never commits everything at once. No co-author line. Uses Conventional Commits. Use when user says commit, /commit, or commit my changes, or asks to save/checkpoint work, or at the end of a task when changes need to be committed."
metadata:
  user-invokable: true
---

# Smart Commit

Commit all pending changes as **separate, logically-grouped commits** using
Conventional Commits format. Never lump everything into one commit.

## Commit Message Format

```
<type>(<scope>): <imperative summary>

<optional body — what and why, not how>
```

### Types

| Type       | When to use                                  |
|------------|----------------------------------------------|
| `feat`     | New feature or capability                    |
| `fix`      | Bug fix                                      |
| `refactor` | Code restructuring, no behavior change       |
| `test`     | Adding or updating tests                     |
| `docs`     | Documentation only                           |
| `chore`    | Build, config, CI, dependencies              |
| `perf`     | Performance improvement                      |
| `style`    | Formatting, whitespace, linting (no logic)   |

### Rules

- Subject line: imperative mood, lowercase, no period, max 72 chars
- Scope: short module/area name (e.g., `benchmark`, `config`, `gpu-worker`)
- No `Co-Authored-By` line — ever
- Body is optional; use only when the "why" isn't obvious from the subject

## Workflow

### 1. Analyze all changes

Run in parallel:
- `git status` (never use `-uall`)
- `git diff` (unstaged changes)
- `git diff --cached` (staged changes)
- `git log --oneline -10` (recent style reference)

### 2. Group changes into logical commits

Examine every changed and untracked file. Group them by:

1. **Feature/functionality** — files that implement the same feature together
2. **Type** — tests separate from source, docs separate from code, config separate from logic
3. **Module** — changes to different modules go in different commits

Ordering: commit foundational/dependency changes first, then dependents.

### 3. Create each commit

For each logical group:

```bash
git add <specific files>
git commit -m "$(cat <<'EOF'
type(scope): summary

Optional body explaining why.
EOF
)"
```

- Stage files by explicit path — never `git add .` or `git add -A`
- One `git commit` per logical group
- Verify with `git status` after all commits

### 4. Present summary

After all commits, show a table:

```
| # | Commit | Files | Message |
|---|--------|-------|---------|
| 1 | abc123 | 3     | feat(benchmark): add multi-strategy support |
| 2 | def456 | 2     | test(benchmark): update config and worker tests |
```

## Grouping Heuristics

- A source file and its corresponding test file → **two commits** (feat/fix + test)
- Multiple config files changed together → **one commit** (chore)
- New script + docs about that script → **two commits** (feat + docs)
- Unrelated one-line fixes across modules → **separate commits per module**
- Rename/move + functional change → **two commits** (refactor + feat/fix)

## Edge Cases

- **Pre-commit hook failure**: fix the issue, re-stage, create a NEW commit (never --amend)
- **No changes**: report "nothing to commit" and stop
- **Sensitive files** (.env, credentials, secrets): warn the user, do NOT commit
- **Large binary files**: warn the user before committing
