# Preprint 130-cell campaign operations

This directory contains the immutable manifest and the operator entry points for
the full `automil-preprint-130-v3` campaign. The controller enforces the frozen
protocol independently for every cell:

- native five-fold baseline on folds `0,1,2,3,4`;
- exactly 60 charged discovery attempts on validation folds `0,1,2`;
- promotion of at most 10 unique complete candidates on folds `3,4`;
- winner selection by the equal-weight mean of validation folds `0` through `4`;
- explicit, winner-only reveal of the already sealed five-fold held-out results.

There is no final retraining. Crashes and partial runs consume discovery
attempts. An incomplete promotion candidate is ineligible. The native baseline
wins an exact tie against a searched candidate; searched-candidate ties use the
stable node ID.

## 1. Preflight

Run these commands from the repository root with the environment used for the
campaign installed in `.venv`:

```bash
.venv/bin/python benchmarks/scripts/campaign_manifest.py check
.venv/bin/python benchmarks/scripts/campaign_manifest.py canary
```

`check` rebuilds the canonical roster and compares it with the byte-locked
manifest. `canary` materializes and audits all 130 roots in a temporary
directory, checks all 10 arm/task regimes, and starts zero GPU processes.

The campaign loads dataset paths from `benchmarks/.env`. Populate that file
before launching training. It is deliberately absent from git worktrees, so
the baseline operator loads it from the main repository and passes its values
to the detached training worktree.

## 2. Materialize the campaign once

```bash
.venv/bin/python benchmarks/scripts/campaign_manifest.py materialize
```

This creates one restart-safe project under
`benchmarks/campaigns/preprint_130/runtime/<cell-id>/` for every manifest row.
All roots are pinned to the same full git commit. Re-running materialization is
an integrity check: it preserves progressed state, plans, learnings, and policy
files, and fails closed if any immutable input differs.

Choose one materialized root for the commands below:

```bash
CAMPAIGN_CELL_ROOT="benchmarks/campaigns/preprint_130/runtime/<cell-id>"
.venv/bin/python benchmarks/scripts/campaign_stage.py status \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

## 3. Establish the native incumbent

The baseline is outside the 60 agentic attempts. The safe operator runs the
manifest-locked five-fold command in a temporary worktree at the campaign's
frozen base commit, keeps validation evidence public, stores held-out folds in
the sealed archive, and registers the discovery root:

```bash
.venv/bin/python benchmarks/scripts/campaign_stage.py run-baseline \
  --cell-root "$CAMPAIGN_CELL_ROOT" --gpu 0
```

Use `baseline-command` to inspect the exact command without running it. If an
equivalent baseline was executed portably elsewhere, import its validation-only
`result.json` plus `certify/fold_<0..4>_result.json` files with
`register-baseline --baseline-archive <path>` instead.

## 4. Run discovery

Start the scheduler for the selected cell:

```bash
.venv/bin/automil --project "$CAMPAIGN_CELL_ROOT" check
.venv/bin/automil --project "$CAMPAIGN_CELL_ROOT" orchestrator start
```

Run one coding-agent session against this project and have it follow the autoMIL
experiment loop. The enforced policy permits source-level train-only policy
implementations under that cell's `automil/variants/_policies/` directory; it
does not reduce the campaign to a fixed hyperparameter menu. Submit and launch
both revalidate the cell identity, frozen base commit, command, budget, file
manifest, and architecture-preserving boundary.

Monitor without exposing held-out values:

```bash
.venv/bin/automil --project "$CAMPAIGN_CELL_ROOT" status
.venv/bin/automil --project "$CAMPAIGN_CELL_ROOT" rank
.venv/bin/python benchmarks/scripts/campaign_stage.py status \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

Do not freeze early. When the stage ledger reports exactly 60 charged attempts,
freeze the complete unique candidates and select up to 10 by the locked
validation ordering:

```bash
.venv/bin/python benchmarks/scripts/campaign_stage.py freeze-discovery \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

## 5. Run exact promotion

Materialize exact copies of the frozen candidate overlays. No agent proposes or
edits candidates during promotion:

```bash
.venv/bin/python benchmarks/scripts/campaign_stage.py materialize-promotion \
  --cell-root "$CAMPAIGN_CELL_ROOT"
.venv/bin/automil --project "$CAMPAIGN_CELL_ROOT/promotion" orchestrator start
```

After every queued promotion job is terminal, freeze eligibility and select the
five-fold validation winner:

```bash
.venv/bin/python benchmarks/scripts/campaign_stage.py freeze-promotion \
  --cell-root "$CAMPAIGN_CELL_ROOT"
.venv/bin/python benchmarks/scripts/campaign_stage.py select-winner \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

The winner record is immutable. `advance` may perform the next safe transition
through selection, but it intentionally stops at `winner-frozen` and never
reveals held-out data.

## 6. Certify explicitly

Only after all search and selection activity for the cell is over, reveal the
frozen winner's existing sealed folds:

```bash
.venv/bin/python benchmarks/scripts/campaign_stage.py certify \
  --cell-root "$CAMPAIGN_CELL_ROOT"
```

Certification verifies the hashes recorded before selection and writes one
winner bundle. The ordinary `status` output reports only bundle identity and
timestamps, never held-out metric values.

## Recovery and audit trail

The authoritative per-cell ledger is `<cell-root>/campaign_state.json`. Every
write is lock-serialized, atomic, revisioned, and content-hashed. Discovery and
promotion artifacts remain in their respective `automil/orchestrator/archive/`
directories; the native baseline is imported into `baseline/archive/`; the
certified winner bundle is recorded by the stage ledger. Re-running a completed
transition is either idempotent or fails closed on identity drift.

Operate all 130 roots with an external scheduler if desired, but invoke these
same per-cell commands and preserve the one-GPU-per-training-process contract.
Never infer campaign progress from directory counts alone; use the validated
stage ledger and public `status` surface.
