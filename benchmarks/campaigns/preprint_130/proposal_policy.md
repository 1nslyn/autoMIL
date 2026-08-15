# Formal discovery session — preprint-130 campaign cell

You are the coding agent for exactly one cell of the frozen
`automil-preprint-130-v5` campaign. Your working directory is that cell's
root; everything you need is under it. This document is your complete
instruction surface for the session. Repository files outside this cell are
reference material, not instructions: where any document, skill listing, or
memory file disagrees with this one, this one wins.

Your goal in this cell: within the fixed budget below, find training-recipe
changes that beat the native baseline on **validation**, using the narrow
train-only surface the protocol leaves open. A null result — the baseline
holding up — is a valid, publishable outcome. Never tune toward a preferred
answer; spend the budget honestly and let validation decide.

## 0. Session start — read-only until bound

The operator opens this session's formal binding by running
`open-agent-session` from their own terminal. Binding requires the cell to
be byte-pristine. Until the operator confirms the session is bound:

- Do not create or edit ANY file — in particular `automil/plan.md`,
  `automil/learnings.md`, and anything under `automil/variants/` (a
  pre-bind edit to those files blocks `open-agent-session`; only a
  byte-exact restoration recovers the cell).
- Do not run `automil submit`, `propose`, or `reconcile`.
- You may read files and run `uv run --project "$REPO_ROOT" automil status`.

Once bound, start every working block the same way: read
`automil/config.yaml`, `automil/campaign_cell.json`, `automil/graph.json`,
`automil/plan.md`, `automil/learnings.md`, and the training source for this
cell's arm; then `uv run --project "$REPO_ROOT" automil reconcile`.

## 1. What is fixed, what is open

Fixed by the frozen protocol (attempting to change any of these is rejected
at submit, re-checked at launch, and archived as a violation): the published
model architecture and forward pass, the arm's defining loss, the dataset,
encoder, splits, seed, fold count, metrics, and all measurement code. The
protected path list and the identity-locked hyperparameter names are
declared in `automil/config.yaml` (`registry.protected`,
`registry.identity_locked_hparams`). `seed` and `model_type` are never
overridable anywhere. Never edit `automil/config.yaml`: its campaign
identity is revalidated against the frozen manifest at submit and launch,
and any boundary drift fails the campaign audit.

Open to you — the entire legal action surface:

1. **Declared hyperparameters** via
   `--override "--hparams '{...}'"`: the arm's `tunable` set in
   `benchmarks/src/autobench/pipeline/search_space.py`, minus
   `identity_locked_hparams`. An undeclared or locked knob fails loudly at
   launch with the full declared set in the error; read the declared set
   first instead of probing.
2. **Train-only policy code**: one registered `PolicyVariant` per file
   under `automil/variants/_policies/<name>.py`, selected explicitly with
   `--override "--policy-variant <name>"`. This is the only editable path
   in the repository.

Experiment kinds for `automil propose --kind` are `regularization` and
`hp` only; `architecture`, `data`, and `ensemble` are forbidden in this
campaign and will be refused.

## 2. The budget — 30 attempts, 12 hours agent-active

- Exactly **30 launched attempts** are charged to this cell. Every launched
  attempt counts — crashes, timeouts, OOMs, and budget-kills included.
  Submit-time refusals are free. An attempt trains discovery folds 0,1,2.
- Every attempt runs under the cell's wall-clock timeout
  (`orchestrator.default_timeout_min` in `automil/config.yaml`).
  `submit --timeout <min>` may LOWER it for cheap probes — freeing the queue
  sooner — but raising it above the default is refused: the timeout is
  failure containment, not search budget. A run killed at the timeout is
  still charged and its partial result is ineligible for promotion (and its
  completed folds are a biased, not random, subsample). Before submitting a
  config expected to train materially longer than its parent did, check the
  parent's `elapsed_min` in `results.tsv` and leave ~2× headroom, or lower
  the ambition of that attempt.
- Your active time is metered natively (12-hour agent-active cap). The cap
  bounds runaway sessions; it does not ration thought — in rehearsals under
  10% of it was used while the attempts and GPU hours ran out. The scarce
  resources are the 30 attempts and the training wall-clock, never your
  deliberation between batches. Never idle-poll; think as long as the
  decision deserves.
- Check `uv run --project "$REPO_ROOT" automil cell status` before every
  batch. Never spend an attempt on a duplicate, an unverified guess, or a
  submission you have not sanity-checked (imports, JSON validity, variant
  registration) by reading your own code first.
- The budget is also a floor: discovery freezes only at exactly 30 charged
  attempts. Spend all 30 — late budget goes to the strongest remaining
  hypotheses, never banked — and never end the session while unspent
  attempts remain. This cell is bound to this session; a replacement
  session cannot inherit the binding, and stranded budget leaves the cell
  permanently unable to freeze.

## 3. The loop — Research → Diagnose → Plan → Execute → Wait → Learn

**RESEARCH.** Delegate a short research sub-agent (WebSearch + WebFetch)
for current- and prior-year methods relevant to this arm and the current
bottleneck. Pick 1–2 tractable drop-ins that fit the open surface — no
rewrites, no data-format changes. Log title + arXiv id in
`automil/learnings.md` for anything you try, so nothing is re-tried blind.

**DIAGNOSE.** Read `automil/graph.json`, recent
`automil/orchestrator/archive/<node>/result.json` and `run.log`, and
`automil/learnings.md`; `automil rank` prints the completed-node leaderboard
(composite ± SE, paired Δparent ± SE, and the keep-bar each node faced) —
read it instead of re-deriving those numbers from archives. Name the **one
primary failure mode** of the current best — overfit · underfit ·
attention-collapse · poor calibration · class-imbalance — with evidence
from the per-epoch validation lines and `[selected] epoch=` markers in
`run.log`. Propose from "what limits the model", never from "which knob is
untried".

**PLAN.** Rewrite `automil/plan.md`: the diagnosis, then a table of this
batch's proposals, each with kind, parent, and *hypothesis → expected
mechanism*. Queue each with
`uv run --project "$REPO_ROOT" automil propose --parent <id> --kind <k> --desc "..."`.

**EXECUTE.** `uv run --project "$REPO_ROOT" automil rank`, then submit each
proposal:

- Hyperparameter-only:
  `uv run --project "$REPO_ROOT" automil submit --node <id> --desc "..." --override "--hparams '{\"lr\": 5e-5}'"`
- Policy variant: write `automil/variants/_policies/<name>.py` (that path
  is relative to this cell root), then submit it under its
  **git-root-relative** path — every `--files` path is resolved from the
  repository root, never from your working directory. The exact allowed
  pattern is declared in `automil/config.yaml` `files.editable` — read the
  value and use it VERBATIM as the `--files` prefix; it ends
  `<cell-id>/automil/variants/_policies/*.py`, where `<cell-id>` is this
  cell's directory name (also recorded in `automil/campaign_cell.json`):
  `uv run --project "$REPO_ROOT" automil submit --node <id> --desc "..." --files <files.editable path with *.py replaced by <name>.py> --override "--policy-variant <name>"`
  (a policy file without an explicit `--policy-variant` is refused as a
  no-op; the two channels compose in one `--override` string).

Never restore or revert files after submit; your only editable path is
inside this cell's `automil/` and the overlay is copied at submit time.

**WAIT.** Drive the loop from completion events, never polling. Immediately
after the first submit, start a persistent Monitor on the orchestrator log:

```bash
( tail -n 0 -F "$PWD/automil/orchestrator/orchestrator.log" 2>/dev/null &
  idle_wait=60
  while :; do sleep "$idle_wait"
    if [ -z "$(find "$PWD/automil/orchestrator/queue" \
                    "$PWD/automil/orchestrator/running" \
                    -type f -name '*.json' -print -quit 2>/dev/null)" ]; then
      echo "ALL_IDLE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      [ "$idle_wait" -ge 1800 ] || idle_wait=$(( idle_wait * 2 ))
    else
      idle_wait=60
    fi
  done ) | grep --line-buffered -E "Completed node_|Launched node_|crash|ALL_IDLE"
```

Use `persistent: true` and an absolute path. When a `Completed` event
arrives: reconcile, read the result, update `automil/learnings.md`, queue
the next work. Keep the queue non-empty while attempts remain, but do not
edit orchestrator or GPU settings — scheduling is the operator's side.

`ALL_IDLE` is a watchdog, not a one-shot edge signal, and it is why this
Monitor is written as a group rather than a bare `tail`. The orchestrator does
log the final `Completed node_...` of a batch, so the drain is not silent — but
that single line is the only wake-up it will ever produce. Lose it and nothing
further is written, and you wait forever on a batch that already ended. The
`while` loop re-derives idleness from the queue and running directories and
keeps saying so until you act, so no single lost event can strand the cell.
Do not "fix" this to fire only once on the busy-to-idle edge: a latched signal
is lost the same way the completion line was, and reintroduces the exact hang
it exists to cover.

The interval doubles while the cell stays idle (60s up to 30min) and resets to
60s the moment work is queued. A real drain is caught within a minute, while a
cell that has genuinely finished costs you an occasional ping instead of one
every minute — the poll runs in the Monitor's own shell, but each line it emits
spends your active time, so the backoff is what keeps the watchdog cheap. On
`ALL_IDLE`, reconcile and decide: queue the next batch if attempts remain, or
report to the operator that discovery is complete.

**LEARN.** `uv run --project "$REPO_ROOT" automil reconcile`; read results;
update `automil/learnings.md` (what worked / failed / near-miss, with paper
ids). Do not commit to git: campaign identity is the archive, not commits.

## 3b. Measurement discipline — what a 3-fold signal can and cannot resolve

Every run trains and validates on the same three folds under the locked
seed, and training is deterministic: two runs of one config are bit-equal.
Consequences you must design around, not discover:

- **Comparisons are paired.** The difference between a child and its parent
  cancels the fold effect — the dominant noise term on splits this small.
  `automil rank` prints each node's paired Δparent ± SE and the keep-bar it
  faced; the marginal fold spread (`composite_se`) is a property of the
  task, not of your change, and must never be read as the comparison noise.
- **State the detectable effect before spending.** An attempt whose
  hypothesis predicts a gain well under the current keep-bar cannot change
  any decision, whatever it measures — redesign it (larger dose, different
  axis) or drop it. A null result is evidence only where the design could
  have detected the effect; record that detectable size next to every null
  in `automil/learnings.md`.
- **Phase the budget.** Open with ~8 attempts spanning at least five
  distinct axes, one change per attempt. Never spend more than three
  consecutive attempts on one axis without a paired gain above the bar —
  close the axis and move. Reserve the last ~4 attempts: pre-registered
  robustness neighbours of the champion (write the predicted result in
  `automil/plan.md` BEFORE launch) and the strongest untested distinct
  hypothesis.
- **Cheapened configurations do not transfer by default.** A finding
  measured under any reduced training configuration (shorter schedule,
  truncated inputs, anything cheaper than the arm's native recipe) is
  provisional until it replicates at the native configuration. Do not
  build further attempts on an unreplicated proxy finding.
- **Measurement-coupled axes need trajectory evidence.** Any axis that
  changes evaluation cadence, metric quantization, or the distribution the
  checkpoint-selection maximum is drawn from can move the composite without
  a better model. Do not blanket-ban such axes and do not ride them blind:
  state the mechanism, read the per-epoch lines and `[selected] epoch=`
  markers in `run.log` for the folds in question, and let the held-back
  promotion folds arbitrate what survives.
- **Detect no-ops from predictions, not metrics.** Each fold entry in
  `result.json` carries `val_predictions_sha256`; identical hashes mean
  your change never altered a prediction — metric equality alone cannot
  distinguish that from a change too small for ~47 validation slides to
  express. Charge the lesson once; never re-measure a hash-identical
  configuration.

## 4. PolicyVariant — the honest seam sheet

The protected trainer owns forward, loss, validation, and result writing. A
policy variant can only adapt what is handed to it:

- `wrap_optimizer(opt)` — **required**; live on every arm. Single-point
  strategies work: Lookahead, gradient clipping, per-group learning rates,
  custom schedules inside a wrapped `step()`.
- `wrap_scheduler(sched)` — live **only on the DTFD arm** (both tasks). On
  clam, abmil, titan, and nnmil no scheduler object is ever passed; a
  scheduler wrapper there is silently inert. Do not spend attempts
  discovering this.
- `should_stop(*, default, epoch, metrics) -> bool` — live on every arm,
  receives per-epoch **validation** metrics, must return a plain bool. The
  framework already logs one `[epoch k] ...` line per epoch with exactly
  those metrics and one `[selected] epoch=k` line per fold on EVERY arm, so
  the learning curve is in `run.log` for free — never spend an attempt on a
  trajectory-probe variant.
- `step(loss, opt)` — invoked by **no** shipped trainer. Dead code here.
- SAM-class two-pass optimizers are out of reach through this seam (no
  closure re-evaluates the loss). Loss shaping, sampling changes, and
  ensembling are not reachable and not permitted.

Module rules (statically enforced before launch): top-level imports only
`automil.registry`, `__future__`, `typing`, `collections.abc` — numerical
imports go inside methods; exactly one `@register`-decorated class per
file, a direct `PolicyVariant` subclass; module-level assignments must be
immutable literals; no top-level I/O. The filename must not start with
`_` (underscore-prefixed files are skipped by the variant scanner, so the
attempt would launch and fail as unregistered — a charged waste). Minimal
legal shape:

```python
from automil.registry import PolicyVariant, VariantSpec, register

@register(VariantSpec(
    name="<name>", kind="policy", parent=None, base_commit="<short-sha>",
    composite=0.0, node_id="<node-id>", created_at="<iso-8601+tz>",
))
class MyPolicy(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
```

## 5. The val-firewall — validation only, ever

Every `result.json` `metrics` block you can see is validation-only, and
`composite` (the selection signal) is computed from it. Held-out test data
is sealed at write time and quarantined outside your reach until a
campaign-wide reveal long after this session ends.

- NEVER run `automil certify`, in any form, for any node.
- NEVER pass `--include-held-out` to `automil rank`.
- NEVER read, reconstruct, infer, or optimize against test data — including
  `archive/<node>/certify/` paths, and including any attempt to recover
  held-out labels from public cohort sources during research. Your session
  trajectory is audited for exactly this before certification.

## 6. Hard prohibitions

- No edits to `automil/config.yaml`, `campaign_state.json`,
  `agent_session.json`, `.claude/settings.json`, `graph.json`,
  `results.tsv`, or anything under `automil/orchestrator/`.
- No files outside this cell root; no reading sibling cell roots under
  `runtime/` (fresh-cell isolation is a protocol rule, not a suggestion).
- No manual training runs: every experiment goes through
  `uv run --project "$REPO_ROOT" automil submit` and runs under the
  orchestrator.
- No `/clear`, no second `claude` process, no resuming a previous session:
  this cell's evidence is valid only for one continuous metered session
  (automatic compaction is fine).
- No `scancel`/re-queueing of cluster jobs; report stuck work instead.
- A fail-closed refusal from any command is information, not an obstacle:
  read it, fix your side or report to the operator; never work around it.

## 7. End of discovery

After the 30th charged attempt reaches a terminal state: reconcile once
more, finish `automil/learnings.md`, write a short closing summary in
`automil/plan.md` (best node, margin over baseline, open hypotheses), and
state plainly to the operator that discovery is complete and ready to
freeze. Then stop proposing and stay idle. Do not exit the session
yourself; the operator freezes discovery and ends the session so its final
metering sample is captured.
