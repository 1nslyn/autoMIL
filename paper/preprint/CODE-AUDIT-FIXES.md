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
| CR-1b | `composite` trusted verbatim, not recomputed from val block | CRIT | ☑ | `scoring.py`(new)+`graph.py`+`terminal_writer.py` | new `automil.scoring` reducer (`scoring.formula`, default `mean` — reproduces both existing composites exactly); terminal_writer recomputes from the val `metrics` at ingestion, the val-derived value is **authoritative**, disagreement logged at ERROR + recorded in `node.metadata.composite_disagreement`. Opt out with `formula: trust_reported`. 9 new tests (incl. the test-derived-composite exploit); 180 regression green. |
| CR-2 | `locked_update` only ~half-adopted → graph write-race loses completions | CRIT | ☑ | `propose.py`+`nominate.py`+`reconcile.py` | wrapped the 3 short-transaction writers (propose/nominate/reconcile default+recompute-best) in locked_update; dry-run stays read-only. 4 new tests, 99 regression green. |
| CR-2b | gate/promote path saves graph unlocked during long held-out eval | CRIT | ☐ | | separate refactor: evaluate UNLOCKED, then short-locked status apply (can't hold lock across evals). Do with the gate module (ties to M-8). |
| CR-3 | Survival composite = val c-index the code calls "near-random" | CRIT | ☑ | `evaluate.py` + 4 survival trainers + 5 runners + `run_experiment.py` | composite now uses **pooled cross-fold concordance**: each survival trainer exports per-fold val risk records, runners emit a `val_pooled` block, `summary_to_result_json` prefers it (falls back to the fold-mean). Pooling scores concordance once over ~5x the events and stays comparable across recipes AND across cox/nllsurv (a raw val loss would not be). 5 new tests; **494 autobench tests green**. |
| CR-5 | Stale/shared results cache defeats data fix + collides across variants | CRIT | ☑ | 4 runners + `run_experiment.py` | gave abmil/dtfd/nnmil/titan an optional `results_dir` param (CLAM already had it) and forwarded `AUTOMIL_RESULTS_DIR` from the dispatch → each experiment isolated. Standalone runs still fall back to the shared dir. 6 new tests, 218 regression green. |
| H-1 | `run.log`/stdout not firewalled → test leak by printing | HIGH | ☑ | `firewall.py`(new)+`_orchestrator_daemon.py` | new `automil.firewall` redacts held-out lines; daemon scrubs `archive/<node>/run.log` in place at completion (logs a WARNING naming the offending node) and redacts BOTH agent-facing error-tails. Defence-in-depth over per-script gating (which a re-vendored CLAM would silently regress). 6 new tests; 86 regression green. |
| H-2 | Time-only budget, no eval-count primitive (equal-effort unequal) | HIGH | ☐ | | add eval-count cap primitive in cells/; per memory/automil-equal-effort-budget |
| H-3 | ABMIL/DTFD/nnMIL silently discard hyperparameter overrides (only 1 of 4 aggregators was tunable) | HIGH | ☑ | `hparams.py`+`provenance.py`(new)+abmil/dtfd runners | **Mechanism unified, defaults untouched.** Each arm's own config dataclass stays the single source of truth (keeps arm-specific knobs: DTFD `numGroup`/`grad_clip`, ABMIL `M`/`L`; and nnMIL keeps its data-adaptive self-config). Only override *application* is shared: `apply_overrides` / `apply_overrides_to_plan`, with `FIELD_ALIASES` (DTFD's `wd`, nnMIL's `learning_rate`/`num_epochs`) and **loud failure** on an inapplicable knob — replacing the silent discard AND `variant_dispatch`'s wrong-object safety check. `overrides_from_exp_cfg` recovers 'was this explicitly set?' by diffing vs pristine `TrainConfig`, so unset flags can never pull an arm onto the shared schedule. 19 tests incl. a freeze-guard proving no-override == each arm's own defaults; **513 autobench green** → safe mid-campaign. Provenance moved to doc-only `provenance.py` (`provenance_table()` for the methods section). **Remaining: TITAN + nnMIL call sites; Step 3 (return arms to upstream values) is Leo's call — it WOULD invalidate the dispatched grid.** |
| H-4 | Frozen substrate unenforced (protected empty; roster unwired) | HIGH | ☐ | | non-empty default protected + check hard-fail on empty for benchmark project |
| H-6 | `meta.best_node_id` can be a discarded node | HIGH | ☑ | `graph.py`+`terminal_writer.py` | live completion (terminal_writer) + promote now recompute_best() (keep-only) after descendant re-eval; add_executed + both reconcile-recovery branches keep-gate the inline update (preserves D-14). 2 new tests, 68 regression green. |
| H-7 | `cancel` races daemon → cancelled overwritten to crash | HIGH | ☐ | | daemon detects cancel_reason∈{cap,cli} before promoting |
| H-8 | `<2` valid folds reported as zero-variance "completed" | HIGH | ☑ | `run_experiment.py` | summary_to_result_json records `n_valid_folds`/`n_folds` and sets status=partial when <2 valid (quarantined from keep/discard). Also surfaces M-15 (unequal denominators now visible). 4 new tests + regression green. |
| M-1 | reconcile KeyError on scoring dict missing weights → silent no-op | MED | ☑ | `graph.py` | backfill every scoring key in __init__. 1 test + regression green. |
| M-4 | parent/child cycle → infinite loop hang (lineage/reeval) | MED | ☑ | `graph.py` | visited-set guards in lineage + _reevaluate_descendants. 1 test (SIGALRM-guarded) + regression green. |
| M-5 | orphan recovery writes crash artifacts but never updates graph.json | MED | ☐ | | route through locked_update + mark_failed |
| M-6 | crash window in _launch (Popen before running-spec persist) | MED | ☐ | | persist running spec before/atomically-with Popen |
| M-7 | daemon terminal path never maintains total_executed/total_proposed | MED | ☐ | | increment counters in write_terminal_state on first transition |
| M-14 | budget cell key omits task → classification & OS share budget | MED | ☑ | `cells/state.py`+`cells/registry.py`+`cli/submit.py` | `make_cell_id` takes an optional `task` (submit reads `task.name` from config.yaml). `None`/empty reproduces the legacy 3-tuple id exactly, so existing cells keep their ids. 4 new tests; 136 regression green. |

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
| M-11 | nested-glob dataset lookup picks [0] on stem collision | MED | ☑ | `config.py` | raise on len(nested)>1 with the candidate list. 1 test + regression green. |
| M-12 | TITAN apples-to-oranges on encoder axis | MED | ⚑ | | reporting/figure decision (own panel + caveat) |
| M-13 | architecture searchable → "aggregator axis" not fixed | MED | ⚑ | | decide recipe-only vs best-evolved-head; if former add model files to protected |
| L-1 | run_benchmark roster fallback → 0 experiments silently | LOW | ☐ | | exit with message when requested framework roster empty |
| L-4 | `${VAR:}`/blank env → "" without fail-fast | LOW | ☑ | `config.py` | truthiness check → blank env / empty default now raise. 4 tests + regression green. |
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
- 2026-07-23: **CR-5 ☑** — the 4 non-CLAM runners (abmil/dtfd/nnmil/titan) now accept an optional `results_dir` and the dispatch forwards `AUTOMIL_RESULTS_DIR`, isolating each experiment's per-fold `metrics.json` cache (was shared on subdir → resumed stale folds across data-fixes/seeds/variants). 6 new tests; 218 regression (arms/survival/config/instrumentation) green. Next: CR-3 (deferred — deeper multi-trainer change), doing Gate-1 HIGH first.
- 2026-07-23: **H-6 ☑** — best_node recomputed from keep-nodes only. terminal_writer + `promote` call `recompute_best()` after `_reevaluate_descendants`; `add_executed` and both reconcile-recovery branches keep-gate their inline update (D-14 preserved). Reproduced both audit triggers (discard-status child; descendant flipped to discard). 2 new tests; 68 regression green.
- 2026-07-23: **M-1 ☑ + M-4 ☑** — __init__ backfills every scoring key (no more reconcile-crashing KeyError); visited-set guards in lineage + _reevaluate_descendants (no more parent/child-cycle hang). 2 new tests (SIGALRM-guarded); 63 regression green.
- 2026-07-23: **Full-suite regression checkpoint** — `pytest tests/`: 1087 passed, 12 failed. **All 12 now VERIFIED pre-existing, by running them on the base commit `0b2da55` in an isolated worktree** (the earlier claim was asserted before that baseline finished — it has since been confirmed): 3 purity tests fail only on a stale `.pyc` (pass after clearing bytecode; the grep scans `__pycache__`), and the other 9 (clause_08, clause_11, cancel x2, iris end-to-end, phase7 setup gate, setup_dry_run_gate x3) **fail identically on base**. -> **0 regressions**.
- 2026-07-23: **H-8 ☑** — `summary_to_result_json` records `n_valid_folds`/`n_folds` and quarantines (status=partial) any run with <2 finite-primary-val folds, so a degenerate 1-fold run can no longer masquerade as a complete K-fold "completed" result. Surfaces M-15. 4 new tests; 57 regression green.
- 2026-07-23: **M-11 ☑ + L-4 ☑** — `load_dataset_config` raises on an ambiguous bare name (multiple grouped YAMLs) instead of silently taking the first; `_resolve_env_vars` fails fast on a blank env var / empty `${VAR:}` default instead of resolving to "". 5 new tests; 38 regression green.
- 2026-07-23: **H-3 DEFERRED (◐)** — naive `exp_cfg.train` threading would change DTFD/ABMIL baselines (their paper-exact defaults differ from the shared TrainConfig defaults). Needs explicit override-vs-default detection (CLI already parses as `default=None`). Doing it properly, not hastily.
