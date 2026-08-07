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
   claude --dangerously-skip-permissions
   # Then type: /automil
   ```
5. Start loop flag: `uv run automil start-loop`
6. **Start a persistent Monitor watcher on the orchestrator log** (critical —
   see "Event-driven loop" below)

## Event-driven loop — start a Monitor watcher

The loop is autonomous and long-running (experiments take 60–240 min
each). You **must** drive it from completion events, not polling.
Immediately after the first `automil submit` in any session, start a
persistent `Monitor` on `automil/orchestrator/orchestrator.log` filtered
to state-transition lines. Without this, GPUs go idle for hours between
submits and the loop stalls.

Use the `Monitor` tool with:

- `persistent: true` (lives for the whole session, not the 5-min default)
- `timeout_ms: 3600000` (ignored when persistent, but set for safety)
- An **absolute** path to the log (Monitor runs in an independent shell)
- `tail -n 0 -F` so you start at EOF and follow through rotation
- `grep --line-buffered` — without this, pipe buffering delays events by
  minutes
- A tight regex: `Completed node_|Launched node_|crash` is enough to
  stay oriented without flooding the chat

Example command:

```bash
tail -n 0 -F /abs/path/to/project/automil/orchestrator/orchestrator.log \
  2>/dev/null \
  | grep --line-buffered -E "Completed node_|Launched node_|crash"
```

When a `Completed` event arrives: reconcile, read the result, update
`learnings.md`, queue the next experiment. Never let the queue go empty
while `.automil_active` exists. If the monitor gets auto-stopped for
volume, restart it with a tighter regex.

For one-shot "wait until this one command finishes" (not the loop),
prefer `Bash(..., run_in_background=true)` — Monitor is for streaming.

## Important: File paths are git-root-relative

All file paths in `files.editable`, `uv run automil submit --files`, and `run.command`
are relative to the **git repo root**, not to where automil/ lives. The
orchestrator creates worktrees from the git root, so overlay paths must match.

## The loop is Research → Diagnose → Plan → Execute

Read `registry.mode` before proposing. In `free` mode, the loop explores
architecture and recipe and keeps the structural portfolio gate. In
`architecture-preserving` mode, the published model is fixed: explore declared
scalars plus executable train-only optimizer/update, scheduler, and stopping
policies, never architecture, data/sampling, or ensemble proposals.
Never propose a scalar named in `registry.identity_locked_hparams`; those keys
can erase the arm's defining mechanism and are rejected at submit and launch.
`automil portfolio` enforces the configured mode.

First, every session/restart: read `automil/config.yaml`, `automil/graph.json`,
`automil/learnings.md`, `automil/plan.md`, and the training + `files.editable`
source; then `uv run automil reconcile`.

**LOOP FOREVER — one batch =**

1. **RESEARCH.** Delegate a short research sub-agent (WebSearch + WebFetch) for
   current-year and prior-year methods relevant to this model class and the
   current bottleneck. Pick 1–2 tractable drop-ins that fit the existing
   pipeline (no full rewrites, no data-format changes). Log title + arXiv id in
   `learnings.md` for anything you try, so future sessions don't re-try it blind.

2. **DIAGNOSE.** Read `graph.json` + recent `archive/<node>/result.json` +
   `learnings.md` and name the **one primary failure mode** of the current best —
   overfit · underfit · attention-collapse · poor calibration · class-imbalance ·
   data/feature bottleneck — with evidence. This is what the batch attacks. Never
   propose from "which knob is untried"; propose from "what is actually limiting
   the model".

3. **PLAN.** Rewrite `automil/plan.md`: the diagnosis, then a table of this
   batch's proposals — each with `kind`, parent, and *hypothesis → expected
   mechanism*. Queue each with
   `uv run automil propose --parent <id> --kind <k> --desc "..."`
   (kinds: `architecture` · `regularization` · `hp` · `data` · `ensemble`).
   Then run `uv run automil portfolio`. In `free` mode, aim ≥50% structural
   (architecture/ensemble) and rebalance if it reports BELOW TARGET. In
   `architecture-preserving` mode, use only `regularization` or `hp`;
   `architecture`, `data`, and `ensemble` are forbidden. A batch may be
   HP-only if that is what the diagnosis supports, but actively consider the
   broader executable train-only recipe surface instead of enumerating knobs.

4. **EXECUTE.** `uv run automil rank`, then implement and submit — **prefer the
   variant-registry path**:
   a. In architecture-preserving mode, add a registered `PolicyVariant` at
      `automil/variants/_policies/<name>.py`, then submit it explicitly with
      `--files automil/variants/_policies/<name>.py --override
      "--policy-variant <name>"`. The protected consumer loop supplies only
      optimizer/scheduler/stopping seams; model, defining loss, measurement,
      and held-out outputs are unavailable through this interface. Keep module
      scope declarative: only import `automil.registry`, `__future__`, `typing`,
      or `collections.abc` at top level; import numerical libraries inside the
      policy methods that use them.
   b. In free mode, model/loss/policy variants may use their registered kind
      directory and normal selection lifecycle.
   c. Free-mode edit: edit project files, then
      `uv run automil submit --node <id> --desc "..." --mil-model <model> --files <changed files>`,
      then **selectively** restore ONLY those files:
      `git restore --source=HEAD -- <each-file>`. Never bulk-restore
      (`git checkout .`, `git restore .`, `git stash -k`) — it discards unrelated work.

   **Saturate every GPU** — submit until the VRAM bin-packer can't fit another
   run, not until each GPU has one. Measure a typical run's `peak_vram_mb`
   (`archive/<node>/result.json`) and set `orchestrator.max_concurrent_per_gpu`
   + `orchestrator.default_vram_estimate_gb` in `config.yaml` (hot-reloads live).
   Check `orchestrator/gpu_state.json` → `schedulable_free_gb` / `running`; if
   workers < cap while the queue is non-empty and free VRAM is large, the loop is
   running serially — submit more specs until the cap binds.

5. **WAIT** on Monitor completion events (do **not** poll). Agent-active time is
   Claude Code's native cumulative active-time metric (CLI + user active
   seconds; idle excluded). Monitor is the event-driven wait mechanism, not the
   budget clock. Research or diagnosis performed while a batch trains remains
   active work and is therefore billed normally.

6. **RECONCILE + LEARN.** `uv run automil reconcile`; read results; update
   `learnings.md` (what worked / failed / near-miss, with paper ids). Commit
   winning changes. Then loop back to RESEARCH for the next batch.

## The val-firewall — select on validation, never test

Every `result.json` `metrics` block is **validation-only**, and `composite` is
computed from it — that is the sole selection signal. Any test metrics the
consumer emits are sealed into a `held_out` block that the orchestrator
quarantines under `archive/<node>/certify/`; they are never surfaced to you
during search. Do **not** try to read, reconstruct, or optimize against test.
When the whole search is finished, reveal the held-out number exactly once:

```bash
uv run automil certify        # honest held-out test for the val-selected winner
```

The val→test gap is the honest cost of search; reporting the certified number
(not a test-selected one) is what keeps the comparison trustworthy.

## Rules

- NEVER STOP while `.automil_active` exists
- Every batch: DIAGNOSE a failure mode and rewrite `automil/plan.md` BEFORE proposing
- Always pass `--kind` to `automil propose` and run `automil portfolio`
  before execution. Free mode requires its structural quota; architecture-
  preserving mode forbids architecture/ensemble and has no structural quota.
- In architecture-preserving mode, edit only `files.editable`; never attempt
  to bypass admissibility with explicit `--files`, raw identity overrides, or
  a model/loss variant.
- NEVER read or optimize against test: `metrics`/`composite` are validation-only and test is sealed (val-firewall). Reveal held-out test once at the very end with `uv run automil certify`, never inside the loop
- Use `uv run automil submit` for every experiment (not manual runs)
- Use `uv run automil rank` to pick experiments (not random)
- Update `automil/learnings.md` after every result (paper title + arXiv id when from a paper)
- Commit winning experiments to git
- File paths in submit --files must be relative to git repo root
- Restore selectively after submit; never bulk-restore the working tree

## Stopping

User runs `uv run automil stop-loop` to allow the agent to exit.
