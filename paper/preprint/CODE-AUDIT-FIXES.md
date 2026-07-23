# Audit Remediation Tracker — 2026-07-23

Living tracker for fixing the findings in [`CODE-AUDIT-2026-07-23.md`](CODE-AUDIT-2026-07-23.md).
Branch: `fix/audit-2026-07-23`. Updated as each fix lands.

**Status legend:** ☐ TODO · ◐ WIP · ☑ DONE (code+test+commit) · ⚑ NEEDS-DECISION (research/paper call, not a mechanical fix) · ⊘ WON'T-FIX (with reason)

**Discipline:** each fix = understand → test (TDD where practical) → implement → `uv run pytest` (targeted + regression) → update this tracker → conventional commit (no co-author). Nothing marked ☑ until its test passes and the suite is green.

---

## Gate 1 — before any agentic launch (several silently corrupt results)

| ID | Finding | Sev | Status | Commit | Notes |
|----|---------|-----|--------|--------|-------|
| CR-1a | Non-finite `composite` accepted (Infinity/NaN exploit) + NaN persists as invalid JSON (M-3) | CRIT | ☑ | `_result.py`+`runner.py`+`graph.py` | finite guard in validate_result + parse_constant at ingest + graph.save allow_nan=False; 14 new tests, 69 regression green. **Folds in M-3.** |
| CR-1b | `composite` trusted verbatim, not recomputed from val block | CRIT | ☐ | | needs config-declared `scoring.formula` reducer; recompute + flag disagreement in terminal_writer |
| CR-2 | `locked_update` only ~half-adopted → graph write-race loses completions | CRIT | ☑ | `propose.py`+`nominate.py`+`reconcile.py` | wrapped the 3 short-transaction writers (propose/nominate/reconcile default+recompute-best) in locked_update; dry-run stays read-only. 4 new tests, 99 regression green. |
| CR-2b | gate/promote path saves graph unlocked during long held-out eval | CRIT | ☐ | | separate refactor: evaluate UNLOCKED, then short-locked status apply (can't hold lock across evals). Do with the gate module (ties to M-8). |
| CR-3 | Survival composite = val c-index the code calls "near-random" | CRIT | ☐ | | make survival composite = the checkpoint-selection signal (pooled concordance / val loss) |
| CR-5 | Stale/shared results cache defeats data fix + collides across variants | CRIT | ☐ | | key results dir on experiment_id; pass AUTOMIL_RESULTS_DIR to all 4 non-CLAM runners |
| H-1 | `run.log`/stdout not firewalled → test leak by printing | HIGH | ☐ | | scrub held-out keys from run.log + error-tail at orchestrator boundary |
| H-2 | Time-only budget, no eval-count primitive (equal-effort unequal) | HIGH | ☐ | | add eval-count cap primitive in cells/; per memory/automil-equal-effort-budget |
| H-3 | ABMIL/DTFD/TITAN silently ignore hyperparameter overrides | HIGH | ☐ | | thread exp_cfg.train/model into all arms OR hard-fail on dropped override |
| H-4 | Frozen substrate unenforced (protected empty; roster unwired) | HIGH | ☐ | | non-empty default protected + check hard-fail on empty for benchmark project |
| H-6 | `meta.best_node_id` can be a discarded node | HIGH | ☐ | | replace inline best-updates with keep-only recompute_best() |
| H-7 | `cancel` races daemon → cancelled overwritten to crash | HIGH | ☐ | | daemon detects cancel_reason∈{cap,cli} before promoting |
| H-8 | `<2` valid folds reported as zero-variance "completed" | HIGH | ☐ | | record n_valid_folds; status=partial / degrade composite |
| M-1 | reconcile KeyError on scoring dict missing weights → silent no-op | MED | ☐ | | setdefault all 3 scoring keys |
| M-4 | parent/child cycle → infinite loop hang (lineage/reeval) | MED | ☐ | | visited-set guard + parent_id validation |
| M-5 | orphan recovery writes crash artifacts but never updates graph.json | MED | ☐ | | route through locked_update + mark_failed |
| M-6 | crash window in _launch (Popen before running-spec persist) | MED | ☐ | | persist running spec before/atomically-with Popen |
| M-7 | daemon terminal path never maintains total_executed/total_proposed | MED | ☐ | | increment counters in write_terminal_state on first transition |
| M-14 | budget cell key omits task → classification & OS share budget | MED | ☐ | | include task in make_cell_id |

## Gate 2 — before any headline claim (statistics/reporting)

| ID | Finding | Sev | Status | Commit | Notes |
|----|---------|-----|--------|--------|-------|
| CR-4 | δ=0 winner's-curse on ~10-patient val (+0.1–0.2 = 5–10× effect) | CRIT | ☐ | | per-cell δ ≥ 1 SE of CV composite (code) + report on sealed test (decision) |
| H-5a | K=5 percentile-bootstrap CI indefensible | HIGH | ☐ | | add t₄/BCa interval option in evaluate.py |
| H-5b | No multiple-comparison control | HIGH | ☐ | | add Holm/BH utility for the headline comparison family |
| H-5c | Single-seed grid | HIGH | ⚑ | | multi-seed harness = code; deciding #seeds/compute = Leo |
| M-8 | `certify --top-k` re-opens test selection | MED | ☐ | | default to single val-selected node; gate K>1 behind explicit flag |
| M-15 | NaN val-c-index folds silently dropped from composite mean | MED | ☐ | | count valid folds; penalize/propagate (ties into H-8) |

## Gate 1/2 — benchmark-pipeline robustness

| ID | Finding | Sev | Status | Commit | Notes |
|----|---------|-----|--------|--------|-------|
| M-9 | missing feature files silently drop val/test slides | MED | ☐ | | assert retained fraction + per-class floor for val/test |
| M-10 | task-CSV/splits cache content-blind to manifest value change | MED | ☐ | | stamp manifest mtime/hash sidecar; invalidate on change |
| M-11 | nested-glob dataset lookup picks [0] on stem collision | MED | ☐ | | raise on len(nested)>1 |
| M-12 | TITAN apples-to-oranges on encoder axis | MED | ⚑ | | reporting/figure decision (own panel + caveat) |
| M-13 | architecture searchable → "aggregator axis" not fixed | MED | ⚑ | | decide recipe-only vs best-evolved-head; if former add model files to protected |
| L-1 | run_benchmark roster fallback → 0 experiments silently | LOW | ☐ | | exit with message when requested framework roster empty |
| L-4 | `${VAR:}`/blank env → "" without fail-fast | LOW | ☐ | | treat empty resolved value as missing → raise |
| L-5 | case-level `.first()` label conflict; n_classes miscount | LOW | ☐ | | assert one stratify value/case; count distinct class names |
| L-10 | cross-framework AUC asymmetry (per-class vs ovr-macro) | LOW | ☐ | | unify AUC computation or document caveat |

## Gate 1 — framework robustness (LOW)

| ID | Finding | Sev | Status | Commit | Notes |
|----|---------|-----|--------|--------|-------|
| L-2 | survival trainer omits cuDNN determinism flags | LOW | ☐ | | shared seed_everything incl. cuDNN flags |
| L-3 | worktree result.json holds test on disk during run | LOW | ☐ | | write val-only to worktree; test → AUTOMIL_RESULTS_DIR |
| L-6 | daemon cmd_submit ID collision (legacy path) | LOW | ☐ | | allocate id under lock / guard existing queue file |
| L-7 | _get_pending sort TypeError on mixed submitted_at | LOW | ☐ | | coerce sort key to str |
| L-8 | immutability rule violations; viz reads unlocked | LOW | ☐ | | copy-on-write node updates; reader lock (partial — pragmatic) |
| L-9 | certify/ quarantine convention-only | LOW | ⚑ | | threat-model statement (+ optional perms); mostly a docs decision |

---

## ⚑ Needs Leo's decision (research/paper — NOT mechanical code fixes)

These change the science or the paper, not just the code. I will implement the **enabling code** but not presuppose the direction:

1. **CR-4 / H-5c reporting** — report the headline on sealed test with selection correction; how many seeds; whether GBM/PDAC feed the variance decomposition or only breadth. *(I'll wire multi-seed + per-cell δ; you set the policy.)*
2. **Restore encoder dynamic range** (novelty a) — add ResNet50/CTransPath legacy anchor + TITAN arm. This is *running experiments*, not a code fix. *(I can add the encoder configs; extraction/run is a campaign decision.)*
3. **Headline reframe** — "encoder ≫ aggregator" → honest non-replication. Writing/positioning. *(I'll draft RELATED_WORK citations for AIRA2/AIDE/AMID/MLE-bench; the reframe is yours.)*
4. **Recipe-family planner** (novelty c) — new research contribution. Out of code-fix scope.
5. **M-13** — is the aggregator axis recipe-only (freeze architecture) or best-evolved-head? Decides whether model files go in `protected`.
6. **M-12 / L-9** — figure/threat-model presentation calls.

---

## Progress log

- 2026-07-23: branch `fix/audit-2026-07-23` created; tracker initialized; starting Gate-1 CR-1a.
- 2026-07-23: **CR-1a ☑** (+ M-3) — finite guard at `validate_result`, `parse_constant` reject in `runner.collect_result`, `allow_nan=False` in `graph.save`. 14 new tests pass; 69 regression tests (schema/runner/graph/firewall/daemon) green. Next: CR-2 universal locking.
- 2026-07-23: **CR-2 ☑** — wrapped `propose` / `nominate` / `reconcile` (default + recompute-best) in `locked_update`; `--dry-run` stays read-only. Gate/promote's unlocked save during long held-out evals split off as **CR-2b** (needs evaluate-then-short-lock refactor). 4 new tests; 99 regression (propose/reconcile/recompute-best/nominate/gate/cli/graph) green. Next: CR-3 survival selection signal.
