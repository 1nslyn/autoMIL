# Running your cells of the 130-cell agentic campaign (fir HPC)

This is the successor to
[`run_preprint_benchmark.md`](run_preprint_benchmark.md). That one told you how
to run the **static baseline grid**. This one tells you how to run the
**agentic campaign** — the actual experiment the preprint is about.

It is a different kind of work. In the baseline you ran three `sbatch` commands
and waited. Here you drive a staged, fail-closed controller one cell at a time,
and a coding agent does the research inside a budget you are responsible for not
breaking.

> **Read §1 before you touch anything.** The protocol rules are not style
> preferences — if you break one, that cell's data cannot go in the paper.

---

## 0. Status — what you can and cannot start today

[`benchmarks/campaigns/preprint_130/PROGRESS.md`](../../benchmarks/campaigns/preprint_130/PROGRESS.md)
is the single source of truth for launch gates. **Nobody runs a formal cell
until every item in its Gates section is green**, including exact 130/130
manifest coverage, the ten-regime real-GPU canary, the locked agent protocol,
and the approved allocation. Until then, use only a throwaway cell root.

Tracking lives in two places, and they are not interchangeable:

- **[The campaign tracker Sheet](https://docs.google.com/spreadsheets/d/1e79rsWlc8BOZoi6xFRWQvyi9M1uDuCexlQCsOTE5gOU/edit)**
  — one row per cell, 130 rows. Filter by `Owner` to get your 26. **This is
  where you update your own cells.**
- [`benchmarks/campaigns/preprint_130/PROGRESS.md`](../../benchmarks/campaigns/preprint_130/PROGRESS.md)
  — the overall view: gates, rolling totals, weekly snapshot. Leo keeps this current.

Before public release, Leo confirms that the Sheet's sharing policy matches the
repository's intended visibility. The tracked URL is deliberate; sharing state
is a release check recorded in `PROGRESS.md`, not a reason to copy gate status
into this runbook.

**Never put a test metric in either one.**

---

## 1. The five rules

These are enforced in code. You should still know them, because a fail-closed
error is much easier to read when you know what it is protecting.

**1. Validation only. You never see test.**
Test metrics are written straight into a sealed archive and are not visible to
you or to the agent during search. They are unsealed once, at the very end, for
the frozen winner only. If you ever find yourself looking at a test number
during search — stop and tell Leo.

**2. Exactly 30 launched attempts per cell, 12h agent-active.**
Crashes, OOMs, timeouts and budget-kills all consume the budget. That is
deliberate: equal effort means an equal cap on *launched attempts*, not on
successes. `freeze-discovery` refuses to run at 29 or 31.

**3. Train-only edits.**
The agent may change declared config values and files under
`automil/variants/_policies/`. The published model, the data pipeline, the
splits and all measurement code are protected. A candidate touching a protected
path is rejected at submit and re-checked before launch.

**4. One fresh agent session per cell, no cross-cell memory.**
Each cell gets its own isolated project root, its own graph, its own learnings
file. Do not carry findings from your cell into another. Do not reuse a
`session_id`.

**5. The baseline is outside the budget.**
The native-recipe five-fold baseline is run by you as operator, before the agent
starts, and does not consume any of the 30 attempts.

---

## 2. Your assignment

Same cohort ownership as the baseline phase. 26 cells each.

| Member  | Dataset    | Classification task    | `<dataset>` |
| ------- | ---------- | ---------------------- | ----------- |
| Leo     | TCGA-LUAD  | KRAS (binary)          | `tcga_luad` |
| Yeonwoo | TCGA-LGG   | IDH1 (binary)          | `tcga_lgg`  |
| Ryan    | TCGA-HNSC  | tumour grade (3-class) | `tcga_hnsc` |
| Keishi  | CPTAC-GBM  | TP53 (binary)          | `cptac_gbm` |
| Terry   | CPTAC-PDAC | immune_class (3-class) | `cptac_pdac`|

Your 26 = 13 classification + 13 survival, where 13 = (4 aggregators × 3 tile
encoders) + 1 TITAN arm. Cell ids look like:

```
tcga_luad__kras__uni_v2__clam__s42__preprint-v2
```

---

## 3. Setup, once

```bash
ssh <you>@fir.alliancecan.ca      # login2 / login3 if /home misbehaves
cd ~/scratch/autoMIL
git checkout main && git pull --ff-only
uv sync --all-packages
```

Make sure your dataset root is in `benchmarks/.env`. The campaign reads it and
propagates the values into the detached training worktree, which cannot see your
shell environment.

> `benchmarks/.env` is gitignored and **absent inside worktrees**. If it is
> missing entirely, `run-baseline` will not fail fast — it fails later, inside
> training, with a missing-path error. Check it exists before you start.

Every command below pins the workspace explicitly so it remains valid after
entering a materialized cell. Use this form as written:

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/<script>.py" ...
```

---

## 4. One cell, end to end

Pin the repository and cell roots once and reuse them. Export both because the
Claude session and every command run after `cd` must resolve the same workspace:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export CELL="$REPO_ROOT/benchmarks/campaigns/preprint_130/runtime/tcga_lgg__idh1__uni_v2__clam__s42__preprint-v2"
```

### 4a. Check where the cell is

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" status --cell-root "$CELL"
```

This is your main instrument. It reports the phase, attempts charged against the
budget, promoted candidates, and the winner — **validation only**. It will never
print a held-out value. Run it between every step.

The phase order never skips:
`discovery → promotion-ready → promotion → selection-ready → winner-frozen`.

### 4b. Run the native baseline

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" baseline-command --cell-root "$CELL"   # inspect, runs nothing
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" run-baseline    --cell-root "$CELL" --gpu 0
```

Five folds, the arm's own published recipe, outside the 30-attempt budget. It
takes a lock, so a second invocation refuses rather than double-running. If a
baseline is already registered it re-verifies and returns — safe to re-run.

### 4c. Start metering, then open the agent session

Start the cell orchestrator first so it can scrape Claude's localhost
Prometheus endpoint. Only one formal discovery Claude session may run on a host
at once (port 9464). Then start one fresh Claude session in the cell root. After
the first orchestrator scrape, copy `agent_session.template.json`, fill
in a globally unique `session_id` and timezone-aware ISO-8601 `started_at`, and
run this command from inside that same Claude session:

```bash
uv run --project "$REPO_ROOT" automil --project "$CELL" check
uv run --project "$REPO_ROOT" automil --project "$CELL" orchestrator start
cd "$CELL"
claude

# Inside that Claude session, after the first metric export:
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" open-agent-session \
  --cell-root "$PWD" --agent-session /abs/path/agent_session_start.json
```

This binds the hook-recorded, natively metered session to the cell. Every later
`automil submit` is stamped with it, and submit **refuses** if the session was
never opened or lacks an active-time sample.

It also refuses if the cell is not pristine: any prior experiment, any extra
file in `variants/_policies/`, an edited `plan.md` or `learnings.md`, a non-zero
attempt counter, or a `session_id` already used by another cell. That is the
"one fresh session, no cross-cell memory" rule, enforced.

### 4d. Run discovery

```bash
uv run --project "$REPO_ROOT" automil --project "$CELL" check
# The orchestrator and ONE fresh Claude session are already running from 4c.
uv run --project "$REPO_ROOT" automil --project "$CELL" status
uv run --project "$REPO_ROOT" automil --project "$CELL" rank
```

The agent proposes candidates and submits them with `automil submit`. It stops
at 30 charged attempts.

> `automil rank` shows validation only. **Never pass `--include-held-out`.**

### 4e. Freeze discovery

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" freeze-discovery --cell-root "$CELL"
```

Requires exactly 30 charged attempts and nothing still queued or running. It
audits all 30 attempts, deduplicates candidates, and freezes the top ≤10 by
validation mean.

Now end the coding-agent session before any promotion or selection step:

```bash
# Inside Claude: exit normally so SessionEnd captures and persists the final sample.
/exit

# Back in the operator shell. Keep agent_session_end.json for step 4h.
```

The controller refuses promotion and direct/zero-candidate winner selection
until the exclusive bound discovery session has both `SessionEnd` and its
durable final active-time sample. Do not bypass that refusal or leave Claude
open while continuing the controller.

### 4f. Promotion on folds 3 and 4

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" materialize-promotion --cell-root "$CELL"
uv run --project "$REPO_ROOT" automil --project "$CELL/promotion" orchestrator start
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" freeze-promotion --cell-root "$CELL"
```

The frozen top-10 are copied byte-exact and re-run on the two held-back
validation folds. **No agent runs here** — promotion is non-adaptive by design.
A candidate that fails is marked ineligible; that is not fatal to the cell.

### 4g. Select the winner

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" select-winner --cell-root "$CELL"
```

One winner, chosen on the five-fold **validation** mean, ties broken
deterministically with the baseline preferred. Immutable once written. Still no
test data anywhere.

### 4h. Finalize the session attestation

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" finalize-agent-session \
  --cell-root "$CELL" --agent-session /abs/path/agent_session_end.json
```

Claude already exited after step 4e and flushed its final native active-time
export. After the winner is frozen, fill
`agent_session_end.template.json` with `session_id`, `ended_at`,
`termination_reason` and the runtime `usage` block (tokens and cost). **Report
usage honestly — if a number is unavailable, say unavailable. Never write a
zero.** The analysis plan treats a coerced zero as a data-integrity failure.

Then set that cell's `Stage` to `W` in the tracker Sheet, with today's date.

### 4i. Shortcut

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" advance --cell-root "$CELL"
```

Performs one legal transition and stops. It will not run past `winner-frozen`,
and it inherits the same closed-discovery-session requirement before promotion
or winner selection.

---

## 5. Certification — Leo only, once, after all 130

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" freeze-selections
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" certify-all
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" report
```

This is the only point in the entire project where held-out data is opened. It
fails closed unless all 130 cells are frozen and consistent. Do not run these.

---

## 6. When something refuses

The controller is deliberately fail-closed. A refusal is information, not an
obstacle — **do not work around it.**

| Message shape | What it means | Do |
|---|---|---|
| `manifest differs...` | Your checkout drifted from the frozen roster | `git pull --ff-only`; do not regenerate the manifest |
| `native baseline is already running` | A lock is held | Wait; check for an orphaned process before retrying |
| `open the campaign agent session before the first submit` | Step 4c was skipped | Open the session, then restart the agent |
| `SessionEnd ... required before promotion or winner selection` | Claude is still open or its final native sample was not saved | Exit Claude normally; verify the SessionEnd hook succeeded; do not hand-edit the journal |
| attempts ≠ 30 at freeze | Budget not exhausted, or over-run | Report it — do not hand-edit state |
| `agent session finalization is immutable` | Already finalized with different content | Stop; tell Leo |
| candidate marked `inadmissible` | The agent touched a protected path | Expected and healthy. It is archived as a violation and does not enter the leaderboard |

Two standing rules:

- **Never hand-edit `campaign_state.json`.** It is integrity-checked on every
  read; editing it invalidates the cell.
- **Never `scancel` a queued GPU job to resubmit it.** Queue age drives
  priority; you will land behind hundreds of jobs. Fix the problem in place.

---

## 7. Things worth knowing before you start

- **A null result is a fine result.** We are measuring whether equal-effort
  recipe search moves the cross-method ranking. Stability and null are
  publishable outcomes. Nobody should be tuning toward a preferred answer.
- **GBM survival is our declared low-signal region** — 9 of its 13 survival
  cells sit at or near chance on validation. Run it like everything else.
- **`kappa` is not comparable across arms** (only nnMIL emits it), and
  `sensitivity`/`specificity` are undefined for the 3-class tasks. Do not build
  any cross-arm comparison on those.
- **Do not infer progress from directory listings.** Use `status`. Directories
  exist before they are valid.
- **There is no aggregate 130-cell status command yet.** Loop `status` over the
  cell roots, and keep the tracker Sheet current — right now it is the only
  cross-cell view we have.

---

## 8. Reference

- Protocol authority: [`benchmarks/campaigns/preprint_130/README.md`](../../benchmarks/campaigns/preprint_130/README.md)
- Frozen roster: [`benchmarks/campaigns/preprint_130/manifest.json`](../../benchmarks/campaigns/preprint_130/manifest.json)
- Pre-registered analysis: [`benchmarks/campaigns/preprint_130/analysis_plan.json`](../../benchmarks/campaigns/preprint_130/analysis_plan.json)
- Per-cell tracker: [campaign tracker Sheet](https://docs.google.com/spreadsheets/d/1e79rsWlc8BOZoi6xFRWQvyi9M1uDuCexlQCsOTE5gOU/edit)
- Overall progress: [`benchmarks/campaigns/preprint_130/PROGRESS.md`](../../benchmarks/campaigns/preprint_130/PROGRESS.md)
