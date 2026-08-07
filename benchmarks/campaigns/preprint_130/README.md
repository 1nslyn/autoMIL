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

There is no final retraining. Crashes and partial runs consume discovery
attempts. An incomplete promotion candidate is ineligible. The native baseline
wins an exact tie against a searched candidate; searched-candidate ties use the
stable node ID.

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

Copy `agent_protocol.template.json`, fill the immutable provider/runtime/model
versions, embed the exact proposal instructions and tool schema, and record the
SHA-256 of each embedded string. Materialization recomputes both hashes;
placeholder, `unknown`, or content/hash mismatches are rejected. The locked
file therefore archives the resolvable coding-agent policy before any search
starts; it is not an optimization result.

```bash
uv run python benchmarks/scripts/campaign_manifest.py materialize \
  --agent-protocol /path/to/agent_protocol.json
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

Campaign schema v4 is a prelaunch reset: no v3 runtime was materialized, so no
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
configuration. Start the orchestrator before Claude so it can scrape Claude's
documented localhost Prometheus endpoint. Only one formal discovery Claude
session may run on a host at once (port 9464); parallelize cells across hosts,
not sessions on one host. Start one fresh coding-agent session with its working
directory set to the cell root and no cross-cell memory. Its startup hook must
be the first journal event; running the binding command from an unrelated shell
fails closed.

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$CAMPAIGN_CELL_ROOT"
claude
```

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
restart-safely. It also fails if any discovery spec already exists or the
attempt budget has been consumed. Every charged proposal must carry a
timezone-aware `submitted_at` at or after the controller's `bound_at`.

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

Do not freeze early. When the stage ledger reports exactly 30 charged attempts,
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
attempt fails closed and can be retried.

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
integrity drift.

Operate all 130 roots with an external scheduler if desired, but invoke these
same per-cell commands and preserve the one-GPU-per-training-process contract.
Never infer campaign progress from directory counts alone; use the validated
stage ledger and public `status` surface.
