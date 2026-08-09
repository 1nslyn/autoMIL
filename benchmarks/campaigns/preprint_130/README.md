# Preprint 130-cell campaign operations

This directory contains the immutable manifest and the operator entry points for
the full `automil-preprint-130-v5` campaign. The controller enforces the frozen
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
