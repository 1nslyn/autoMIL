# Preprint 130-cell campaign operations

This directory contains the immutable manifest and the operator entry points for
the full `automil-preprint-130-v6` campaign. The controller enforces the frozen
protocol independently for every cell:

- native five-fold baseline on folds `0,1,2,3,4`;
- exactly 30 charged discovery attempts on validation folds `0,1,2`, with a
  12-hour agent-active safety cap per cell;
- promotion of at most 10 unique complete candidates on folds `3,4`;
- winner selection by the equal-weight mean of validation folds `0` through `4`;
- one campaign-wide freeze of all 130 validation winners before any held-out
  read;
- paired baseline-and-winner reveal of the already sealed five-fold held-out
  results.

There is no final retraining. Crashes, partial runs, and launch failures
consume discovery attempts — an attempt is billed exactly once, when its spec
is archived. An incomplete promotion candidate is ineligible. The native
baseline wins an exact tie against a searched candidate; searched-candidate
ties use the stable node ID.

## 1. Preflight

Run these commands from the repository root with the campaign workspace synced
through `uv`:

```bash
uv run python benchmarks/scripts/campaign_manifest.py check
uv run python benchmarks/scripts/campaign_manifest.py canary
```

`check` rebuilds the canonical roster and compares it with the byte-locked
manifest. `canary` materializes and audits all 130 roots in a temporary
directory, checks all 10 arm/task regimes, and starts zero GPU processes.
The manifest also locks `analysis_plan.json` before certification. It declares
the task-specific primary metrics, real four-arm tile ranking estimand, separate
TITAN analysis, and the dependency-aware no-p-value reporting rule.

The campaign loads dataset paths from `benchmarks/.env`. Populate that file
before launching training. It is deliberately absent from git worktrees, so
the baseline operator loads it from the main repository and passes its values
to the detached training worktree.

## 2. Materialize the campaign once

The protocol's two content payloads are committed sources in this directory:
`proposal_policy.md` — the exact instruction text every cell session receives
— and `toolset.json` — the machine-readable tool surface the launcher
enforces. Build the publication protocol from them once, pinning the
immutable identities observed in a real runtime session:

```bash
uv run python benchmarks/scripts/campaign_agent_protocol.py build \
  --model-version <immutable model ID from a throwaway session> \
  --runtime-version <exact `claude --version`>
uv run python benchmarks/scripts/campaign_agent_protocol.py verify
```

Take the model ID from a throwaway session, never a formal cell session: the
protocol must be locked before the first formal session exists. Materialization
recomputes both content hashes; placeholder, `unknown`, or content/hash
mismatches are rejected. The locked file therefore archives the resolvable
coding-agent policy before any search starts; it is not an optimization
result. `agent_protocol.template.json` stays as the schema reference.

`agent_protocol.json` is **already built and committed** for this campaign, and
CI re-runs `verify` on every push so it cannot drift from its two sources. Run
`verify` alone to confirm your checkout; you do not need `build`. `build` is
frozen-once — it refuses to overwrite an existing protocol whose content would
differ, and `--runtime-version` must match the CLI on *your* launch host, so
running it on a host with a different `claude --version` is how operators
usually hit that refusal. Rebuilding is deliberate and rare (a changed source
payload, or a re-pinned identity, before anything is materialized): delete
`agent_protocol.json`, run `build`, then `verify`. Never rebuild after
materialization — the runtime copy is what all 130 cells hash-lock.

```bash
uv run python benchmarks/scripts/campaign_manifest.py materialize \
  --agent-protocol benchmarks/campaigns/preprint_130/agent_protocol.json
```

This creates one restart-safe project under
`benchmarks/campaigns/preprint_130/runtime/<cell-id>/` for every manifest row.
Every cell is identified only by `dataset + task + encoder + arm + seed +
protocol_version`. Git commits are operational worktree metadata, not campaign
identity and not a baseline-reuse condition. Re-running materialization
preserves progressed state, plans, learnings, and policy files, and checks the
declared campaign inputs.

`PROTOCOL_VERSION` is bumped manually only when the pinned arm definition,
split/seed policy, or training protocol changes. Documentation, CI, Git history,
and other execution-irrelevant repository changes do not bump it.

Campaign schema v5 is a prelaunch reset: no v4 runtime was materialized, so no
runtime-state migration is defined. Historical result archives must be
explicitly re-attested under the six-field identity before reuse.

Choose one materialized root for the commands below:

```bash
CAMPAIGN_CELL_ROOT="benchmarks/campaigns/preprint_130/runtime/<cell-id>"
uv run python benchmarks/scripts/campaign_stage.py status \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

## 3. Establish the native incumbent

The baseline is outside the 30 agentic attempts. The safe operator runs the
manifest-locked five-fold command in a temporary worktree, keeps validation
evidence public, stores held-out folds in
the sealed archive, and registers the discovery root:

```bash
uv run python benchmarks/scripts/campaign_stage.py run-baseline \
  --cell-root "$CAMPAIGN_CELL_ROOT" --gpu 0
```

Use `baseline-command` to inspect the exact command without running it. If an
equivalent baseline was executed portably elsewhere, import its validation-only
`result.json` plus `certify/fold_<0..4>_result.json` files with
`register-baseline --baseline-archive <path>` instead. The archive must also
carry the deterministic `baseline_attestation.json` emitted by `run-baseline`.
The archive is reusable when its six-field declared identity matches and its
public validation plus sealed evidence covers exactly folds `0,1,2,3,4`.
Git commit, diff, and command provenance do not enter this decision.

## 3b. Verify the incumbent reproduces (loop-start gate)

Discovery imports the registered baseline as its graph root; it never
re-measures it on its own. Before any agent session can open, the cell must
therefore pass the baseline reproduction gate: the manifest's frozen
discovery command (folds 0/1/2) runs once more under loop-parity code
resolution (no worktree PYTHONPATH — `automil` resolves from the installed
environment exactly as it does under the orchestrator daemon), and every
fold's validation primary value must land within the predeclared epsilon of
the registered baseline:

```bash
uv run python benchmarks/scripts/campaign_stage.py run-baseline-reproduction \
  --cell-root "$CAMPAIGN_CELL_ROOT" --gpu 0
```

The tolerance lives in `reproduction_policy.json` next to this file and has
no in-code default: derive it first by running the stage with `--measure` on
a few registered cells (a measurement records the spread but never satisfies
the session gate), then commit the epsilon. A failed verdict blocks
`open-agent-session` and stays recorded; superseding it requires an explicit
`--force`, which keeps the prior verdict in state history. Per-fold
`val_predictions_sha256` agreement is recorded as diagnosis only — most arms
are not bit-deterministic, so hash inequality is expected and never gates.

`run-baseline` also records the executing commit as the baseline's execution
identity, and the launcher preflight refuses to start a session when the
launch HEAD differs from it (or, for baselines registered before identity
recording existed, from the passing reproduction verdict's commit), or when
`src/`, `benchmarks/src/`, or `benchmarks/scripts/` carry tracked
modifications. Moving HEAD after baselines recorded their identity is
recovered by re-running the reproduction gate at the new HEAD — the passing
verdict is the anchor that authorizes the new commit (committing
`reproduction_policy.json` itself moves HEAD, so this path is routine, not
exceptional). Practically: the checkout is frozen while any campaign job is
running — baselines and discovery alike, because discovery resolves
`automil` from the live checkout and a mid-search `git pull` changes
framework code under running attempts. Pull only between jobs; mid-campaign
delivery is limited to files no running worker imports (spooled SLURM
scripts, new standalone scripts), byte-identical to a commit.

## 4. Run discovery

Start the scheduler for the selected cell from the repository root:

```bash
uv run automil --project "$CAMPAIGN_CELL_ROOT" check
uv run automil --project "$CAMPAIGN_CELL_ROOT" orchestrator start
```

Each materialized cell contains the exact Claude native-active-time observer
configuration, including its own dedicated exporter port
(`activity.exporter_port`, assigned deterministically per manifest row), so
any number of cells can meter concurrently on one host without contending
for one endpoint. Start each cell's orchestrator before its Claude session
so it can scrape that cell's localhost Prometheus endpoint. When running
several cells on one host, give each cell orchestrator a disjoint GPU
partition via `AUTOMIL_VISIBLE_GPUS` (for example `AUTOMIL_VISIBLE_GPUS=0,1`
in one tmux session and `2,3` in another) — a malformed value refuses
startup rather than scheduling on every GPU. Start one fresh coding-agent
session per cell with its working directory set to the cell root and no
cross-cell memory. Its startup hook must be the first journal event;
running the binding command from an unrelated shell fails closed.

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
uv run python benchmarks/scripts/campaign_launch.py launch \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

The launcher is the executor of the locked protocol, never a second copy of
it: it verifies the pinned CLI version, the frozen memory surface (repository
`CLAUDE.md` hash, no user-level memory or plugins), the untouched activity
settings, this cell's exporter port being free, a running cell orchestrator,
and the absence of any prior session evidence; then it renders
`proposal_policy_content` byte-exact into the cell root as `CLAUDE.md` and
execs the pinned runtime with its working directory at the cell root. Use
`preflight` or `launch-command` to inspect the same derivation without
starting anything. A `campaign-launch refusal` is fail-closed information —
fix the named precondition, never bypass it. A bare `claude` in the cell root
is not a formal session: it would run an unpinned, unarchived surface.

Web research tools stay enabled (the frozen policy's RESEARCH step delegates
WebSearch and WebFetch subagents), identically for all 130 cells; a cell run
without network access is a protocol deviation to record in that cell's
disclosure. Research and diagnosis are ordinary metered agent-active work.

After the first orchestrator scrape of the native active-time counter, fill
`agent_session.template.json` with the runtime session identifier and a
timezone-aware start time, then bind it before the first proposal:

```bash
uv run python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" \
  open-agent-session --cell-root "$PWD" \
  --agent-session /path/to/agent_session_start.json
```

The controller verifies the hook-recorded and metered session, binds its
immutable digest to the journal, and opens the 12-hour/30-launch budget cell
restart-safely. It fails if the native baseline is not yet registered
(Section 3), if any discovery spec already exists, or if the attempt budget
has been consumed. Every charged proposal must carry a timezone-aware
`submitted_at` no earlier than the controller's `bound_at` minus the declared
120-second clock-skew tolerance
(`PROTOCOL.submit_clock_skew_tolerance_seconds`); beyond that the freeze
fails closed.

Have this session follow the autoMIL experiment loop. The enforced policy
permits source-level train-only policy
implementations under that cell's `automil/variants/_policies/` directory; it
does not reduce the campaign to a fixed hyperparameter menu. Submit and launch
both revalidate the declared cell identity, protocol version, command, budget,
file manifest, and architecture-preserving boundary.

Monitor without exposing held-out values:

```bash
uv run automil --project "$CAMPAIGN_CELL_ROOT" status
uv run automil --project "$CAMPAIGN_CELL_ROOT" rank
uv run python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" status \
  --cell-root "$PWD"
```

Do not freeze early. If the 12-hour agent-active budget exhausts below 30
charged attempts, the freeze fails closed by design: report the cell as
blocked rather than working around it; no path reopens a closed cell.
When the stage ledger reports exactly 30 charged attempts,
freeze the complete unique candidates and select up to 10 by the locked
validation ordering:

```bash
uv run python "$REPO_ROOT/benchmarks/scripts/campaign_stage.py" \
  freeze-discovery --cell-root "$PWD"
```

Immediately after discovery freezes, end this coding-agent session. The
synchronous `SessionEnd` hook closes the exact activity interval. Do not keep
the agent alive through promotion: promotion has no proposals and uses its own
wall-clock controller. Preserve the runtime's end timestamp and usage in a
filled `agent_session_end.template.json`; it will be attested after the winner
freezes.

## 4b. Run discovery cells as SLURM jobs (the HPC option)

Section 4 is the framework's default: an operator on a workstation drives
one cell in an interactive session. On fir the campaign runs the same
sequence unattended, one cell per SLURM job, from a shared tree in project
space, with every team member submitting from their own account. Nothing
here changes the protocol: the launcher is the operator of Section 4,
scripted.

**Shared tree.** `/project/6114359/shared/Pathology/autoMIL/work/autoMIL`
(group `rrg-jma`, directories setgid `2770`, files `0660`), created once by
`benchmarks/scripts/migrate_campaign_tree.sh`. Project space is
inode-limited (500K files per project on fir), so the file-heavy,
rebuildable pieces live in a group-traversable scratch directory named by
the script's `--scratch` option: the Python environment (recorded in
`benchmarks/.env` as `UV_PROJECT_ENVIRONMENT`), the per-attempt git
worktrees (`.automil_worktrees` is a symlink), and the dataset stand-ins
(`guard_roots/<cohort>`, the `AUTOBENCH_*_ROOT` values in `.env`). Rules while any discovery job is queued or
running: no `git pull`, no edits under `src/`, `benchmarks/src/`,
`benchmarks/scripts/`; the repository `CLAUDE.md` stays byte-identical to
the protocol's pinned hash (every launch checks it; a test pins it); no
`CLAUDE.md`, `CLAUDE.local.md` or `.claude/` may appear anywhere on the path
from the runtime root up to `/` (the runtime reads memory from every
ancestor). Exports to `version3` stay Leo-only (`sealed/` is owner-only by
design).

**Per-member setup, once.** `uv` and `claude 2.1.228` on `PATH`
(`DISABLE_AUTOUPDATER=1`); `claude login` on a login node with your Team
seat; no `~/.claude/CLAUDE.md`; an empty `~/.claude/plugins`; membership of
`rrg-jma`. The submit script checks all of it before touching a cell.

**Submit.** From any login node:

```bash
/project/6114359/shared/Pathology/autoMIL/work/autoMIL/benchmarks/scripts/slurm/submit_discovery_cell.sh
```

It classifies every roster cell (`campaign_scan.py`: pending / claimed /
finishable / stranded / blocked / done), takes the first drivable one
(finish-only recoveries first), fits the job to the cell
(`campaign_shape.py`: predicted wall from the cell's baseline elapsed time,
each packed attempt costing twice the baseline's per-fold time and never
under 15 minutes, as measured on the rehearsal cells: the CLAM survival
cell's candidates averaged 52 min against a 25 min per-fold time on
2026-09-06, the TITAN cell's 16 min against 2 min on 2026-09-04; 30
attempts packed 4 per GPU as the frozen cell config allows, 1, 2 or 4 GPUs,
12 h or 24 h wall, 12 cores and 128 GB per GPU, a 4-GPU shape takes the
whole node's memory; the cheapest fitting shape by default, `--prefer fast`
for the shortest wall), submits it, and only then claims the cell with the
new job id. Against the registered 78 cells (2026-09-06) cheap gives 29
cells 1 GPU/12 h, 5 cells 1 GPU/24 h, 20 cells 2 GPU/24 h, 20 cells
4 GPU/24 h (about 2,100 GPU-hours predicted, 3,350 allocated); fast puts 45
cells in the 12 h tier for about 2,150 predicted. Four cells exceed every
shape (HNSC grade Virchow2 CLAM; LUAD KRAS H-optimus-1 nnMIL, Virchow2 CLAM
and Virchow2 nnMIL, each with a scaled five-fold time above 4.5 h) and are
skipped until a longer wall is added to the shape table. Claims are once-only tombstones:
a queued job holds its claim; a dead job's claim is replaced only after a
successful cluster-wide `squeue` shows it gone. `--dry-run` prints the
classification and every cell's shape without submitting; `--cell` picks a
cell; `--max-gpus` caps the shape.

**The job** (`submit_discovery_campaign.sh`) refuses to run a cell whose
claim does not carry its own id, records the plan's remaining allocation
from a throwaway `/status` (the submit script already refused above 85 % of
the weekly window before claiming, so no claim is ever burned on a spent
seat), runs the
reproduction gate on its first GPU, brings the daemon up on all its GPUs,
launches and binds the session in a job-private tmux server, sends the
release line, watches until the orchestrator's own budget-cell census reads
30 charged attempts with drained queues (the ledger's `attempts_charged` is
written by freeze-discovery only), then scrapes the session's own exporter
for token and cost counters (`operator/usage.json`, passed to
`finish --usage-json`), captures `/usage` before and after
(`operator/usage_before.txt`, `usage_after.txt`), ends the
session with `/exit`, runs the finish ladder on the same GPUs, normalizes
the cell's files to group read/write, and submits the next cell as the same
user (`--no-chain` at submission stops after the one cell: rehearsals and
member tests). Failures land in `logs/discovery_cells/FAILED.tsv`; a stranded
cell is reported, never relaunched. (The CLAM survival rehearsal cell stranded
this way on 2026-09-06 at 29 of 30 attempts: its 12 h lane had been shaped
from the undilated per-fold time, and the job ended the session at the
finish-reserve deadline with four candidates still running. Its root is kept
under `logs/discovery_cells/runtime-rehearsal/stranded/`; the cell was
re-materialized and re-run from a fresh baseline.)

**Rehearsal sets.** A run set that must stay out of the final grid lives in
its own cell-root directory beside `runtime/`, built from the same manifest
and protocol (row indices and exporter ports unchanged) with its own committed
roster, `<name>.roster.json` (cohorts, cells census, `cell_ids`). The
committed set is `runtime-rehearsal` (two cells per MIL model on TCGA-LUAD with
the H-optimus-1 encoder). Each materialized root is about 1,000 files, so only
the set's cells are built:

```bash
uv run python benchmarks/scripts/campaign_manifest.py materialize \
    --output-root benchmarks/campaigns/preprint_130/runtime-rehearsal \
    --agent-protocol benchmarks/campaigns/preprint_130/agent_protocol.json \
    --cells @benchmarks/campaigns/preprint_130/runtime-rehearsal.roster.json
sbatch --account=def-jma-ab benchmarks/scripts/slurm/submit_rehearsal_baselines.sh runtime-rehearsal
benchmarks/scripts/slurm/submit_discovery_cell.sh --runtime runtime-rehearsal --no-chain
```

The launchers take `--runtime <name>`; logs go to
`logs/discovery_cells/<name>/`. The shape predictor scales a baseline's time
back when its retry loaded folds from cache (the ledger total then covers
only the fresh folds); a baseline whose retry cached every fold has no timing
and is submitted with `--cell <id> --e5-hours <five-fold hours>`. Nothing from a rehearsal set is mirrored,
frozen into the campaign selections, or certified.

**The nudge.** If the runtime's active time has been flat for 30 minutes
while the queue is drained and attempts remain, the job sends one fixed
line (`DISC_NUDGE_LINE` in `discovery_lib.sh`), at most once an hour and
three times per cell, and records it in the cell's
`operator_events.jsonl`. An empty queue alone never triggers it: the agent
diagnoses and plans between batches with nothing queued.

## 5. Run exact promotion

Materialize exact copies of the frozen candidate overlays. No agent proposes or
edits candidates during promotion:

```bash
uv run python benchmarks/scripts/campaign_stage.py materialize-promotion \
  --cell-root "$CAMPAIGN_CELL_ROOT"
uv run automil --project "$CAMPAIGN_CELL_ROOT/promotion" orchestrator start
```

After every queued promotion job is terminal, freeze eligibility and select the
five-fold validation winner:

```bash
uv run python benchmarks/scripts/campaign_stage.py freeze-promotion \
  --cell-root "$CAMPAIGN_CELL_ROOT"
uv run python benchmarks/scripts/campaign_stage.py select-winner \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

The winner record is immutable. `advance` may perform the next safe transition
through selection, but it intentionally stops at `winner-frozen` and never
reveals held-out data.

After the winner freezes, finalize the already-ended, pre-bound discovery
session with the saved `agent_session_end.template.json`. Exact, estimated, and
unavailable usage are represented explicitly; unavailable token/cost fields
must be null and carry a reason. Finalization verifies that all 30 proposal
timestamps fall inside this session interval and that the journal contains one
matching startup, binding, native active-time sample, and `SessionEnd`. Wait for
Claude to exit and flush its final metric export before finalizing; a premature
attempt fails closed and can be retried. If the runtime died without a durable
`SessionEnd`, close the activity interval from the last durable sample with
`uv run automil activity close --session <session-id> --attest "..."` (it
refuses while the exporter still serves the session) and record the
attestation in the cell's disclosure; the full procedure is in
`docs/tutorials/run_agentic_campaign.md`.

```bash
uv run python benchmarks/scripts/campaign_stage.py finalize-agent-session \
  --cell-root "$CAMPAIGN_CELL_ROOT" \
  --agent-session /path/to/agent_session_end.json
```

Repeat Sections 3–5 for every manifest cell, using a distinct fresh runtime
session identifier each time. Session opening rejects an identifier already
reserved by a sibling cell, and the global freeze rechecks uniqueness across
all 130 attestations. Automatic compaction keeps the same metered session;
`/clear` and resumed sessions are not accepted for the formal one-session cell.
No cell may be certified early.

## 6. Freeze all selections, then certify

After all 130 cells report `winner-frozen`, atomically bind their validation
selections into one campaign artifact:

```bash
uv run python benchmarks/scripts/campaign_manifest.py freeze-selections
uv run python benchmarks/scripts/campaign_manifest.py certify-all
```

Before `selection_freeze.json` exists and contains the exact 130-cell roster,
one `protocol_version`, 130 immutable session attestations, and the complete
failure-inclusive search-process evidence, every per-cell
certification entry point fails closed. `certify-all` is
restart-safe: it verifies and reveals each frozen winner together with its
native baseline, emits paired fold deltas, and writes a hashed
`campaign_certification.json` index. The ordinary `status` output reports only
bundle identity and timestamps, never held-out metric values.

Before `certify-all`, audit each cell's archived agent-session trajectory for
held-out-label retrieval (the cohorts are public; for example GDC
clinical-data downloads) and record the outcome in the campaign disclosure.
Formal sessions run with shell and web research access, so the val-firewall's
anti-accident posture is completed by this audit-trail check, not by an OS
boundary.

Each freeze entry binds the winner kind, candidate, promotion node (when
searched), baseline candidate, and the canonical cell-local path plus SHA-256
of all five sealed winner and baseline fold files. Certification and reporting
re-read those files and recompute their fold metrics and aggregates; a newly
hashed downstream bundle or index therefore cannot relabel a run or replace
its evidence without conflicting with the independently maintained cell state,
sealed-fold hashes, process census, or finalized session attestation.

Generate the complete publication artifact only after certification:

```bash
uv run python benchmarks/scripts/campaign_manifest.py report
```

This command requires all 130 baseline/winner bundles, derives 30 four-arm tile
ranking blocks, reports TITAN separately, and fails without writing a report if
any hash, fold, metric, or cell is missing or inconsistent. It aggregates
survival c-index within folds and never pools raw risks from independently
trained fold models.

## Recovery and audit trail

The authoritative per-cell ledger is `<cell-root>/campaign_state.json`. Every
write is lock-serialized, atomic, revisioned, and content-hashed. Discovery and
promotion artifacts remain in their respective `automil/orchestrator/archive/`
directories; the native baseline is imported into `baseline/archive/`; the
certified winner bundle is recorded by the stage ledger. Re-running a completed
transition is either idempotent or fails closed on declared-identity or artifact
integrity drift. A declared cell spec that becomes unreadable is held at
admission (HOLD) and recovers automatically when the file reads again; it is
never cancelled.

Operate all 130 roots with an external scheduler if desired, but invoke these
same per-cell commands and preserve the one-GPU-per-training-process contract.
Never infer campaign progress from directory counts alone; use the validated
stage ledger and public `status` surface.

## Storage mirror and job hygiene

Registered baseline evidence is mirrored one-way from the runtime (scratch)
into durable project storage by `benchmarks/scripts/campaign_export.py`
(`--cell <cell_id>` after each registration, `--all-registered` as
catch-up), rooted at `AUTOBENCH_EXPORT_ROOT` from `benchmarks/.env`. The
mirror is write-only — nothing in the framework reads it back. A leaf
without `EXPORT_OK` is a partial copy: rerun the exporter; the marker is
written only after destination hashes verify against the cell ledger.
Sealed held-out evidence mirrors to the owner-only `sealed/` tree, opened
by nobody until `automil certify`. During SLURM generation overlap, a
FAIL-file entry reading "native baseline is already running" is benign
flock contention — the next generation re-runs that cell cleanly.

When the campaign runs from the shared project-space tree, the checkout,
the repository `CLAUDE.md`, and the memory path above the runtime root are
frozen for every member for as long as any discovery job is queued or
running; the rules and the per-member checklist are in Section 4b.

The frozen input trees are protected by deep guard-root shims
(`benchmarks/scripts/build_guard_root_shims.py`): each cohort's
`guard_roots/<cohort>/benchmark` is a real directory that byte-copies the
small write-prone subtrees (`dataset_csv/`, `splits/`, `titan/`, `nnmil/`),
per-file-symlinks the heavy read-only `features/`, and turns the legacy
output dirs into empty write sinks — so every campaign-reachable prepare
write (including TITAN's unconditional per-run manifest rewrite) lands in
the shim, never in the frozen source. Rebuild only while NO campaign job is
running (`--i-know-idle`); an unknown top-level source entry fails the
build closed until the write map is re-audited. After any rebuild, audit by
stamping a marker file, running the three prepare entrypoints per cohort,
and asserting `find <frozen source> -newer <marker>` returns nothing.
