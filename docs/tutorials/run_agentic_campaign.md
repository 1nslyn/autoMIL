# Running your cells of the 78-cell agentic campaign

This guide covers the **agentic campaign**, the experiment the preprint is
about: 3 cohorts (`tcga_luad`, `tcga_hnsc`, `cptac_pdac`) and 78 cells. The
static baseline grid is a separate workflow, described in
[benchmark_experiments_tutorial.md](benchmark_experiments_tutorial.md).

The work differs from the baseline grid. There you queue a few batch jobs and
wait. Here you drive a staged, fail-closed controller one cell at a time, and
a coding agent does the research inside a budget you are responsible for not
breaking.

A cell's identity is `dataset + task + encoder + arm + seed + protocol_version`,
never a machine or a git commit, but every run of a cell trains on one GPU
type: the type `reproduction_policy.json` declares for the cell's runtime set.
The official cells (`runtime/`) run on fir's full H100s. Another host runs only
a set of its own, with its own roster, its GPU type declared for that set and
baselines trained on that host (campaign README, Section 4c).

> **Read §1 before you touch anything.** The protocol rules are not style
> preferences — if you break one, that cell's data cannot go in the paper.

---

## 0. Status — what you can and cannot start today

[`benchmarks/campaigns/preprint_130/PROGRESS.md`](../../benchmarks/campaigns/preprint_130/PROGRESS.md)
is the single source of truth for launch gates. **Nobody runs a formal cell
until every item in its Gates section is green**, including exact
manifest coverage, the ten-regime real-GPU canary, the locked agent protocol,
and the compute plan derived from that canary's timings. Until then, use only a
throwaway cell root.

Tracking lives in two places, and they are not interchangeable:

- **[The campaign tracker Sheet](https://docs.google.com/spreadsheets/d/1e79rsWlc8BOZoi6xFRWQvyi9M1uDuCexlQCsOTE5gOU/edit)**
  — one row per cell. Filter by `Owner` to get your cells. **This is
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
you or to the agent during search. They are unsealed once, at the very end,
for each frozen winner paired with its native baseline — nothing else is ever
revealed. If you ever find yourself looking at a test number during search —
stop and tell Leo.

**2. Exactly 30 launched attempts per cell, 12h agent-active.**
Crashes, OOMs, timeouts and budget-kills all consume the budget. That is
deliberate: equal effort means an equal cap on *launched attempts*, not on
successes. `freeze-discovery` refuses to run at 29 or 31. If the 12h
agent-active budget exhausts before attempt 30, the refusal is one-way and
the freeze fails closed below 30 — that is the declared posture, sized to be
a tail event (typical per-attempt activity puts 30 attempts at 1.5–5h);
report it, do not work around it.

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

## 2. The cells

The active roster (`active_roster.json`, declared 2026-08-23) has three cohorts.
Who runs which cohort is coordinated separately.

| Dataset    | Classification task    | `<dataset>`   |
| ---------- | ---------------------- | ------------- |
| TCGA-LUAD  | KRAS (binary)          | `tcga_luad`   |
| TCGA-HNSC  | tumour grade (3-class) | `tcga_hnsc`   |
| CPTAC-PDAC | immune_class (3-class) | `cptac_pdac`  |

Each cohort has 26 cells: 13 classification + 13 survival, where 13 = (4
aggregators × 3 tile encoders) + 1 TITAN arm. Cell ids look like:

```
tcga_luad__kras__uni_v2__clam__s42__preprint-v4
```

---

## 3. Setup, once

Do all of this on the machine that will actually run your cells. Steps 3a–3c
are what the launcher checks; skipping one does not degrade gracefully, it
refuses to launch.

### 3a. Repository and environment

```bash
git clone https://github.com/1nslyn/autoMIL.git   # a full clone, not an export:
cd autoMIL                                            # baselines and the orchestrator
git checkout main && git pull --ff-only               # both run in git worktrees
uv sync --all-packages
```

Confirm the environment before trusting it:

```bash
nvidia-smi                                            # the orchestrator bin-packs on it
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run pytest tests/ -q                               # two separate invocations,
uv run pytest benchmarks/tests/ -q                    # from the repository root
```

`tmux` must be installed: the orchestrator runs in the **foreground**, and both
it and the formal session have to survive an SSH disconnect.

### 3b. Host hygiene the launcher enforces

The formal session's instruction surface must be exactly the frozen protocol,
so `campaign_launch.py` fails closed on any of the following. Fix the named
condition — never bypass it.

- `~/.claude/CLAUDE.md` does not exist, and no `CLAUDE.local.md` or
  `.claude/CLAUDE.md` sits on any directory between the cell root and `/`.
- `~/.claude/rules/` is absent or empty, and so is every `.claude/rules/` on
  that same walk: the runtime loads every file in a rules directory as
  instructions, so a synced dotfiles rules folder would join the surface.
- No **unpinned plain `CLAUDE.md`** sits on that same walk either. Only the
  repository `CLAUDE.md` is pinned, so a stray `~/CLAUDE.md`, or one in the
  directory holding your clone, refuses the launch — this catches people out
  more often than the `.local` variants do.
- `~/.claude/plugins` is absent or empty. On a shared machine this is a
  coordination step, not a private one: plugins installed for somebody else's
  project still load into your session.
- None of these are set in your shell: `ANTHROPIC_MODEL`,
  `ANTHROPIC_SMALL_FAST_MODEL`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_BEDROCK`,
  `CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_EFFORT_LEVEL`.
- The repository `CLAUDE.md` still hashes to the value pinned in the protocol
  (SHA-256 over the decoded text, so newline style is normalised first). Do not
  edit it during the campaign.
- The **first token** of `claude --version` equals the protocol's
  `runtime_version` — the pin is the bare version, `2.1.286`, not the whole
  line the CLI prints. Pin that
  version and **turn the CLI autoupdater off on the host** — the launcher sets
  `DISABLE_AUTOUPDATER=1` for the session it starts, which cannot protect you
  from a binary that drifted between cells.

  Do this at the filesystem, not in settings. `autoUpdates: false` is silently
  ignored on a native install: the resolver only honours it when
  `installMethod !== "native"` or `autoUpdatesProtectedForNative !== true`, and
  the native installer sets both against you. `DISABLE_AUTOUPDATER=1` in the
  host shell profile stops the background updater, but any session with a shell
  can unset it, and cells run under `bypassPermissions`. The only thing a
  non-root process cannot defeat is the immutable flag on the directory the
  updater must write into:

  ```bash
  sudo chattr +i ~/.local/share/claude/versions
  ```

  Confirm with `lsattr -d` (look for `i`), and clear it with `chattr -i` when
  you deliberately upgrade — then repin `runtime_version` and re-freeze. A drift
  caught here is a refused launch; a drift missed here is a cell that attests a
  runtime it did not run on.

### 3c. Ports and GPUs

Each cell meters on its own Prometheus port, assigned deterministically as
`9464 + <manifest row>`, so concurrent cells never contend for one endpoint.
Check that `9464–9593` is free (`ss -ltn`); `8421` is the optional viz server.
When several cells share a host, give each cell's orchestrator a disjoint GPU
partition — `AUTOMIL_VISIBLE_GPUS=0,1` in one tmux window, `2,3` in another. A
malformed value refuses startup rather than quietly scheduling on every GPU.

Concurrency is bounded by GPUs, not by hosts: budget at least one GPU per
parallel cell, plus its orchestrator/session pair.

### 3d. Dataset roots

Make sure your dataset root is in `benchmarks/.env`. Orchestrator-run
discovery training executes in a detached worktree with a whitelisted
environment — it cannot see your shell, and `benchmarks/.env` is how values
reach it. `run-baseline` is different: it inherits your full shell environment
and applies `.env` only as a fallback (shell values win), so keep the two
consistent.

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
export CELL="$REPO_ROOT/benchmarks/campaigns/preprint_130/runtime/tcga_luad__kras__uni_v2__clam__s42__preprint-v4"
```

The day-to-day driver is the operator CLI, `campaign_operate.py`. It adds
**zero protocol semantics**: every state transition still goes through the
audited controllers (`campaign_stage.py`, `campaign_launch.py`, `automil ...`)
as subprocesses — the operator CLI only sequences them, supervises the
long-running phases, and reports. Every controller refusal is printed
verbatim and exits non-zero; a refusal is information, never something to
work around. The underlying per-command forms are preserved in
[Appendix A](#appendix-a-recovery--manual-operation-the-audit-trail) — they
are the audit trail and the recovery path when you need to drive one
transition by hand.

Five subcommands cover the whole cell lifecycle:
`up → launch → bind → (watch) → finish`.

### 4a. Bring the cell up

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" up "$CELL" --gpu 0
```

`up` preflights the host, then arranges everything that must be running
before the formal session can start:

- **Preflight** (fail-closed): repo root sanity, `benchmarks/.env` present,
  the host's `claude --version` first token equals the frozen protocol's
  `runtime_version`, an exporter-port twin scan across `runtime*/` roots
  sharing this cell's manifest row, a GPU-claim scan, and an `nvidia-smi`
  free-VRAM report. The GPU-claim scan reads sibling cells' (and their
  `promotion/` projects') orchestrator pid files with the daemon's own
  pid+starttime semantics and **refuses if another cell's live daemon claims
  `--gpu N`**. This cell's own discovery/promotion pair on one GPU is exempt
  — that is the normal finish-time state.
- **tmux session** named from the full cell id, sanitized for tmux (e.g.
  `tcga_luad__kras__uni_v2__clam__s42__preprint-v4`), with three windows:
  `baseline`, `orch`, `agent`.
- **`baseline` window**: the manifest-locked five-fold native baseline
  (outside the 30-attempt budget; it takes a lock, so a duplicate refuses).
- **`orch` window**: the discovery orchestrator, foreground, with
  `AUTOMIL_VISIBLE_GPUS=N` so this cell schedules only on its partition.

`up` is idempotent: it skips whatever is already up (existing session or
windows, registered baseline, live daemon) and re-runs the preflight every
time. Attach with `tmux attach -t <name>` to watch either window.

### 4b. Launch the formal session

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" launch "$CELL"
```

`launch` gates on readiness — the baseline must be registered (per the
controller's own status JSON) and the discovery orchestrator must be alive —
then sends the campaign launcher into the `agent` tmux window, where it execs
the pinned interactive `claude` (this window's pty is why tmux is required).
The launcher re-verifies the locked protocol on this host exactly as before:
pinned CLI version, frozen memory surface, untouched activity settings, this
cell's exporter port being free, running orchestrator, no prior session
evidence. **Never start the session with a bare `claude`.** A
`campaign-launch refusal` in the agent window is fail-closed information:
fix the named precondition, never bypass it.

### 4c. Bind the session — from the operator shell

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" bind "$CELL"
```

Binding no longer happens inside the Claude session. `bind` runs in your
operator shell: it polls the cell's activity journal for the runtime's
`session_open` event, writes the exact two-field
`{"session_id", "started_at"}` attestation from that event, and runs the
controller's `open-agent-session`. The one benign race — the controller
refusing because the first native active-time export has not been scraped
yet — is retried on a 30-second interval **only for that exact refusal**;
any other refusal is printed verbatim and is fatal. On success it prints the
release line:

```
Session is bound. Begin the discovery loop per your policy.
```

Paste exactly that line into the Claude session as its first message. Until
you do, the agent sits idle; and before the bind completes, `automil submit`
refuses outright, so there is no way to burn budget pre-bind.

The binding refuses if the cell is not pristine: any prior experiment, any
extra file in `variants/_policies/`, an edited `plan.md` or `learnings.md`, a
non-zero attempt counter, or a `session_id` already used by another cell.
That is the "one fresh session, no cross-cell memory" rule, enforced.

### 4d. Watch discovery

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" watch "$CELL" --interval 60
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" fleet "$REPO_ROOT/benchmarks/campaigns/preprint_130/runtime"
```

`watch` loops one cell: controller phase, `attempts_charged`, the budget
cell's `consumed_evals`/`eval_budget` and completed count, and the last five
`results.tsv` rows. `fleet` prints one status row (phase, attempts, winner)
per cell root under a runtime root — the cross-cell view this runbook used
to lack.

> Two counters, one budget: `attempts_charged` (controller state, the number
> the 30-attempt freeze audits) and `consumed_evals` (the budget cell,
> billed at launch) are maintained by different components and can disagree
> transiently while a launch is in flight. If they disagree at rest, stop
> and report it — do not hand-reconcile either one.

The agent proposes candidates and submits them with `automil submit`; it
stops at 30 charged attempts.

> `automil rank` shows validation only. **Never pass `--include-held-out`.**

### 4e. End the session

When discovery is exhausted, end the coding-agent session — this is the one
step that happens inside Claude:

```text
/exit
```

Exiting normally lets the SessionEnd hook capture and persist the final
native active-time sample. The controller refuses every later transition
until the bound session has both `SessionEnd` and its durable final
active-time sample; `finish` checks this first and tells you exactly what is
missing. Do not leave Claude open while continuing.

### 4f. Finish the cell

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_operate.py" finish "$CELL" --gpu 0
```

`finish` drives everything after discovery through the audited controllers,
in order, consulting the controller's phase first and skipping transitions
that are already done (it is fully re-entrant — safe to re-run after any
interruption):

1. **Session-end precondition.** If the journal lacks `SessionEnd`, finish
   refuses. If the session's exporter is still serving, the runbook step is
   `/exit` inside Claude (4e). If the exporter is dead — the runtime died
   before its hook — finish runs the operator-attested
   `automil activity close` for you, but only when you supply
   `--attest "runtime died before SessionEnd: <cause>"`; it never invents an
   attestation. See Appendix A.5 for what that close records.
2. Stops the cell's live discovery orchestrator (drained by definition at
   this point).
3. Chains the controller: freeze discovery → materialize promotion → run the
   promotion orchestrator → freeze promotion → select the winner.
   For the promotion orchestrator: if a live daemon already exists (for
   example from a previous finish, or a recovery already in flight), finish
   **adopts** it and only polls; otherwise it **starts one as a supervised
   foreground child** with `AUTOMIL_VISIBLE_GPUS=N`, logging to
   `promotion/automil/orchestrator/operate_supervisor.log`. An explicit
   `--gpu N` is required whenever finish must start that daemon — with the
   variable unset the daemon would schedule on **every** GPU of a shared
   host. It polls until the promotion queue and running set are empty and
   the promotion budget cell shows `consumed_evals == eval_budget`, then
   stops the daemon.
4. Writes the session end attestation (`session_id` from
   `agent_session.json`, `ended_at`, `termination_reason:
   "budget-complete"`, and `usage`) and finalizes the session. Pass
   `--usage-json <file>` with the runtime's real token/cost block to record
   it verbatim; without it, finish records the honest `unavailable` usage
   shape. **Report usage honestly — if a number is unavailable, say
   unavailable. Never write a zero.** The analysis plan treats a coerced
   zero as a data-integrity failure.
5. Prints the final controller status JSON — validation only, as always.

Then set that cell's `Stage` to `W` in the tracker Sheet, with today's date.

### 4g. Two concurrent cells on one host — the contract

Running two cells in parallel is supported exactly when all four hold:

- **Disjoint GPU partitions.** Each cell gets its own `--gpu` value(s);
  `up`/`finish` set `AUTOMIL_VISIBLE_GPUS` from it, and `up` refuses when
  another cell's *live* daemon already claims the requested GPU. A cell's
  own discovery/promotion daemon pair may share its partition.
- **Distinct exporter ports.** Every manifest row has its own deterministic
  port (§3c), so two *different* cells never collide.
- **Twin roots never run concurrently.** The same cell id materialized under
  two runtime roots (e.g. `runtime/` and `runtime-canary/`) shares one
  manifest row and therefore one exporter port. `up` detects the twin and
  refuses while the twin is exporting; never run both at once.
- **One tmux session per cell.** `up` derives the session name from the cell
  id and keeps all three windows for that cell inside it.

---

## 4h. Running cells as SLURM jobs (optional)

Everything above is the interactive path and stays the default. On a SLURM
cluster the same sequence can run unattended, one cell per job, with the
launcher acting as the operator of 4a–4f: see
`benchmarks/campaigns/preprint_130/README.md` §4b and
`benchmarks/scripts/slurm/submit_discovery_cell.sh`. Nothing in the framework
requires it, and nothing in the protocol changes for it.

## 5. Certification — Leo only, once, after all 78

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" freeze-selections
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" certify-all
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_manifest.py" report
```

This is the only point in the entire project where held-out data is opened. It
fails closed unless all 78 cells are frozen and consistent. Do not run these.
A rehearsal set is certified the same way with `--output-root` (campaign
README §6); its held-out values never go into a tracked file.

---

## 6. When something refuses

The controller is deliberately fail-closed. A refusal is information, not an
obstacle — **do not work around it.**

| Message shape | What it means | Do |
|---|---|---|
| `manifest differs...` | Your checkout drifted from the frozen roster | `git pull --ff-only`; do not regenerate the manifest |
| `campaign-launch refusal: ...` | A launch precondition does not hold (CLI version, memory surface, port, prior session evidence, orchestrator) | Fix exactly what it names; never start the session with a bare `claude` |
| `native baseline is already running` | A lock is held | Wait; check for an orphaned process before retrying |
| `open the campaign agent session before the first submit` | Step 4c was skipped | Open the session, then restart the agent |
| `SessionEnd ... required before promotion or winner selection` | Claude is still open or its final native sample was not saved | Exit Claude normally; verify the SessionEnd hook succeeded; do not hand-edit the journal |
| Claude died and cannot re-exit (no SessionEnd recorded) | The runtime was killed before its hook ran | `finish "$CELL" --attest "runtime died before SessionEnd: <cause>"`, which runs `automil activity close` (see 4f and A.5); disclose in `termination_reason` |
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
- **`kappa` is not comparable across arms** (only nnMIL emits it), and
  `sensitivity`/`specificity` are undefined for the 3-class tasks. Do not build
  any cross-arm comparison on those.
- **Do not infer progress from directory listings.** Use `status`. Directories
  exist before they are valid.
- **The aggregate view is `campaign_operate.py fleet <runtime-root>`** — one
  validation-only row per cell. Keep the tracker Sheet current anyway; the
  Sheet is the cross-owner record, `fleet` is your live per-host one.
- **`automil check` reports four must-fix issues on a freshly materialized
  cell** — the `orchestrator/{queue,running,archive,completed}` directories,
  with an unhelpful "run `automil init`". Ignore it and start the orchestrator:
  `orchestrator start` creates those directories itself. Do not run `automil
  init` inside a cell root.
- **`--output-root` must be inside the git repository.** A throwaway root for
  rehearsal has to be something like `benchmarks/campaigns/preprint_130/runtime-canary/`;
  an external path is rejected. A throwaway root is safe to rehearse in: every
  command including `freeze-selections`, `certify-all` and `report` honours
  `--output-root`. A set with a committed `<name>.roster.json` freezes and
  certifies exactly its roster's cells; any other set must hold all 78 grid
  cells and fails closed below that. `report` runs on the full grid only, so a
  rehearsal never produces a publication artifact.
- **A new host needs its own set.** A baseline trained on one GPU type does
  not reproduce on another: an RTX 6000 Ada and a full H100 differ by up to
  0.043 validation AUC per fold, beyond the 0.025 gate. Such a host runs a set
  of its own, with its own roster, its GPU type declared for that set in
  `reproduction_policy.json`, and fresh baselines. Timing anchors in this
  repository are H100-based; re-derive attempt wall-clock there from the set's
  baselines, since the 600-minute attempt timeout bites first on slower cards.

---

## 8. Reference

- Protocol authority: [`benchmarks/campaigns/preprint_130/README.md`](../../benchmarks/campaigns/preprint_130/README.md)
- Active roster (78 cells): [`benchmarks/campaigns/preprint_130/active_roster.json`](../../benchmarks/campaigns/preprint_130/active_roster.json)
- Frozen manifest (130-row superset): [`benchmarks/campaigns/preprint_130/manifest.json`](../../benchmarks/campaigns/preprint_130/manifest.json)
- Pre-registered analysis: [`benchmarks/campaigns/preprint_130/analysis_plan.json`](../../benchmarks/campaigns/preprint_130/analysis_plan.json)
- Per-cell tracker: [campaign tracker Sheet](https://docs.google.com/spreadsheets/d/1e79rsWlc8BOZoi6xFRWQvyi9M1uDuCexlQCsOTE5gOU/edit)
- Overall progress: [`benchmarks/campaigns/preprint_130/PROGRESS.md`](../../benchmarks/campaigns/preprint_130/PROGRESS.md)

---

## Appendix A. Recovery / manual operation (the audit trail)

`campaign_operate.py` runs the commands below in this order, and they remain the
supported way to drive one transition by hand: after an interruption, or to see
exactly what the operator CLI would run. `finish` executes A.5 to A.8 for you.
The protocol behind each step is in the
[campaign README](../../benchmarks/campaigns/preprint_130/README.md), Sections 3
to 5 and "Recovery and audit trail".

### A.1. Check where the cell is

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" status --cell-root "$CELL"
```

Run it between every step. It reports the phase, attempts charged, promoted
candidates and the winner, validation only. The phase order is
`discovery → promotion-ready → promotion → selection-ready → winner-frozen`.
A discovery freeze with zero eligible candidates skips promotion and goes
straight to `selection-ready`; the native baseline then wins by default.

### A.2. Run the native baseline

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" baseline-command --cell-root "$CELL"   # inspect, runs nothing
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" run-baseline    --cell-root "$CELL" --gpu 0
```

Five folds, outside the 30-attempt budget. A lock makes a second invocation
refuse, and an already registered baseline is re-verified and returned.

### A.3. Start metering, launch, and open the agent session

```bash
uv run --project "$REPO_ROOT" automil --project "$CELL" check
uv run --project "$REPO_ROOT" automil --project "$CELL" orchestrator start
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_launch.py" launch --cell-root "$CELL"

# From the operator shell, after the first metric export:
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" open-agent-session \
  --cell-root "$CELL" --agent-session /abs/path/agent_session_start.json
```

Start the orchestrator first so it can scrape the cell's Prometheus endpoint.
Always start the formal session through the launcher, never a bare `claude`.
The session JSON holds the `session_id` from the runtime's `session_open`
journal event and a timezone-aware ISO-8601 `started_at` (`bind` automates this).
`campaign_launch.py preflight` and `launch-command` print the derivation without
starting anything.

### A.4. Run discovery

```bash
uv run --project "$REPO_ROOT" automil --project "$CELL" status
uv run --project "$REPO_ROOT" automil --project "$CELL" rank
```

The orchestrator and the one Claude session from A.3 are already running. The
agent submits candidates with `automil submit` and stops at 30 charged attempts.
`automil rank` shows validation only; never pass `--include-held-out`.

### A.5. Freeze discovery and end the session

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" freeze-discovery --cell-root "$CELL"
```

This requires exactly 30 charged attempts and nothing queued or running. End the
coding-agent session before any promotion or selection step:

```bash
# Inside Claude: exit normally so SessionEnd captures and persists the final sample.
/exit

# Back in the operator shell. Keep agent_session_end.json for step A.8.
```

The controller refuses promotion and zero-candidate winner selection until the
bound discovery session has both `SessionEnd` and its
durable final active-time sample.

**Dead-session recovery.** If the Claude process died without running its
SessionEnd hook (crash, OOM-kill, power loss), close the session from the last
durable sample (`finish --attest "..."` runs exactly this):

```bash
uv run --project "$REPO_ROOT" automil --project "$CELL" activity close \
  --session <session-id> --attest "runtime died before SessionEnd: <cause>"
```

The command refuses while the exporter still serves the session, records
`finalized_by: operator-close` and your attestation in the journal, and unblocks
freeze, promotion and finalize. Disclose the closure in `termination_reason` at
A.8. Never start a replacement session for a cell that has submitted attempts.

If the runtime died before `open-agent-session` completed, nothing has been
charged. Re-materialize that cell's root from the frozen manifest and open a
fresh session. After the first submit this reset is forbidden; use
`activity close`.

### A.6. Promotion on folds 3 and 4

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" materialize-promotion --cell-root "$CELL"
uv run --project "$REPO_ROOT" automil --project "$CELL/promotion" orchestrator start
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" freeze-promotion --cell-root "$CELL"
```

The frozen top 10 are re-run byte-exact on the two held-back validation folds.
No agent runs here. `orchestrator start` stays in the foreground, so keep it in
a tmux window (or let `finish` supervise it) and stop it with
`uv run --project "$REPO_ROOT" automil --project "$CELL/promotion" orchestrator stop`
once the queue drains, before freezing.

### A.7. Select the winner

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" select-winner --cell-root "$CELL"
```

One winner, chosen on the five-fold validation mean, with ties broken in favor
of the baseline. The record is immutable.

### A.8. Finalize the session attestation

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" finalize-agent-session \
  --cell-root "$CELL" --agent-session /abs/path/agent_session_end.json
```

Fill `agent_session_end.template.json` with `session_id`, `ended_at`,
`termination_reason` and the runtime `usage` block. **Report usage honestly: if
a number is unavailable, say unavailable. Never write a zero.** Then set the
cell's `Stage` to `W` in the tracker Sheet, with today's date.

### A.9. Shortcut

```bash
uv run --project "$REPO_ROOT" --package autobench python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" advance --cell-root "$CELL"
```

Performs one legal transition and stops at `winner-frozen`. It requires the same
closed discovery session before promotion or winner selection and does not start
the promotion orchestrator.
