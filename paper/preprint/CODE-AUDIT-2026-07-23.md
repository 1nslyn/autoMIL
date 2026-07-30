# autoMIL / autobench — Comprehensive Code & Methodology Audit

**Date:** 2026-07-23 · **Scope:** entire pipeline (framework `src/automil/` + benchmark `benchmarks/`) + preprint claims · **Standard:** Nature-level adversarial review (rigor · fairness · novelty · robustness · bugs)

**Method:** 6 parallel adversarial audit streams (3 methodological via Opus reasoning: eval-rigor/leakage, fairness, novelty; 3 engineering: orchestrator daemon, graph/scoring, benchmark pipeline). Each required file:line evidence + concrete failure scenario + CONFIRMED/POSSIBLE. Findings below are the **synthesized, deduplicated, cross-validated** set; severity is the synthesizer's judgment (cross-stream convergence bumps confidence). The two top CRITICALs and the survival-selection contradiction were additionally verified by hand.

---

## Executive verdict

The framework *runs*, and much of the plumbing is genuinely solid (see §"What is solid"). But **the three scientific guarantees the paper sells are currently either unenforced or unsound**:

1. **The val-firewall is not enforced.** The selection scalar (`composite`) is trusted verbatim from an agent-writable file, with no recomputation from the validation block and no finite-value guard — so the agent can (accidentally or adversarially) drive selection with test-derived or `Infinity` values, undetected. Raw training `stdout` (`run.log`) is agent-visible, so the sealed test can leak by simply being printed. **4 of 6 streams converged here.**

2. **The "equal-effort" comparison is systematically unequal on the exact axis it measures.** The budget is time-based with no evaluation-count primitive (cheaper aggregators get more search); 3 of 4 aggregators silently ignore the hyperparameter knobs the search tunes; the frozen substrate is unenforced for the roster; and δ=0 selection over ~10-patient validation folds inflates the reported lift by **5–10× the effect size being resolved**.

3. **The statistics cannot support the headline, and the headline's novelty is largely owned by prior work.** K=5 percentile-bootstrap CIs, no multiple-comparison control, a survival selection signal the code itself calls "near-random," and a headline ("encoder ≫ aggregator") contradicted by the team's own 210-config data and underpowered by design. Meanwhile AIRA2's *Hidden Consistent Evaluation* and AIDE already own the method claims, and a concurrent preprint (AMID) shares the framework framing.

**Bottom line:** as currently scoped, this is a rejection risk at a top venue
on rigor + novelty. **But** every blocking issue has a concrete, mostly-cheap
fix, and there is a defensible result-neutral framing: an auditable
recipe-bias/ranking instrument that reports change, stability, or a null result
without presupposing the Frontiers conclusion. Do not launch the agentic
campaign or make the headline claim until the CRITICALs are closed — several
silently corrupt results.

---

## Cross-validation map (findings multiple independent streams hit — highest confidence)

| Convergent finding | Streams | Verdict |
|---|---|---|
| `composite` trusted verbatim / no finite guard / not recomputed from val | rigor + graph + daemon + fairness (4) + hand-verified | **CRITICAL** |
| Survival composite = val c-index the code calls "near-random" | rigor + benchmark + fairness (3) + hand-verified | **CRITICAL** |
| ABMIL/DTFD/TITAN silently ignore hyperparameter overrides | fairness + benchmark (2) | **HIGH** |
| `run.log` / stdout not firewalled → test leak by printing | rigor + daemon (2) | **HIGH** |
| Frozen substrate unenforced (protected empty; roster unwired) | fairness + novelty (2) | **HIGH** |
| δ=0 winner's-curse on ~10-patient val; headline underpowered | rigor + fairness + novelty (3) | **CRITICAL** |

---

## CRITICAL — fix before trusting any result or launching the agentic campaign

### CR-1 · `composite` is trusted verbatim: val-firewall unenforced + adversarial `Infinity` exploit
**Dimension:** rigor / fairness / security · **CONFIRMED (4 streams + hand-verified)**

`composite` is read straight from the agent-produced `result.json` at every decision site and never recomputed from the validation `metrics` block. Schema defines it as a bare unbounded `{"type":"number"}`, structurally decoupled from `metrics`/`held_out`.
- Evidence: [terminal_writer.py:205](src/automil/terminal_writer.py:205) `composite = result.get("composite", 0.0)`; keep/discard at [:217](src/automil/terminal_writer.py:217); best-node at [:236-237](src/automil/terminal_writer.py:236); [schemas/result.schema.json:9](src/automil/schemas/result.schema.json:9); graph consumers `graph.py:236/330/451`; per-fold aggregation only *averages* the scalar (`cells/reconcile.py:64,88`). No `isfinite/isnan/allow_nan` guard exists anywhere in `src/automil` (grep-confirmed).
- **Failure A (leak, silent):** a `train.py` that computes `composite` from test AUC (bug or copy-paste) produces a schema-valid file; the graph selects on test; `certify` later "reveals" test that already drove selection — no code path detects it. The firewall's central claim is unenforced.
- **Failure B (adversarial):** the agent edits `train.py` and writes `composite: Infinity` (Python `json.dump` emits it; jsonschema accepts it). `_accept(inf,…)` → always keep; `inf > best` → captures `best_node` forever. `NaN` poisons every downstream `>` (all descendants discard).
- **Fix:** at ingestion, recompute `composite` from the declared val `metrics` via a config reducer and reject on disagreement; parse with `json.loads(..., parse_constant=_reject)` and reject non-finite. (PLAN.md §5 "layer 2" — specified, not built.)

### CR-2 · Graph write-race: `locked_update` adopted by only ~half the writers → silent loss of completions
**Dimension:** bug / data-corruption · **CONFIRMED (daemon stream + hand-verified)**

`graph.locked_update()` takes an advisory `flock`, which only excludes *other lock holders*. The daemon writes under the lock on every completion, but several CLI writers do a bare `ExperimentGraph(...) → save()` with no lock.
- Evidence — **unlocked** writers: [propose.py:95-126](src/automil/cli/propose.py:95) (the agent's most frequent action), [nominate.py:41-46](src/automil/cli/nominate.py:41), reconcile's **default** path [reconcile.py:152-159](src/automil/cli/reconcile.py:152), promote/gate. **Locked** (correct): submit.py, cancel.py, dequeue.py, `reconcile --from-archive` (reconcile.py:65), terminal_writer.py.
- **Failure:** agent runs `propose` → reads snapshot v1. Daemon completes node_0030 → writes v2 (composite, keep/discard, best_node). `propose.save()` atomically renames stale-v1+new-proposal over v2 → **node_0030's completion is silently erased**, reverts to `running`, `best_node` rolls back, `rank`/`certify` no longer see it. The designated repair tool (`reconcile`, default path) **shares the defect** and has a wider clobber window.
- **Fix:** wrap every graph read-modify-write in `locked_update`, or make `save()` refuse to run outside a held lock.

### CR-3 · The agentic survival search selects recipes on a signal the code itself calls noise
**Dimension:** rigor · **CONFIRMED (rigor + benchmark streams + hand-verified)**

Within a training run the checkpoint is deliberately **not** selected on val c-index — it uses val loss, with the comment that ~2 events/val fold make c-index "near-random." Yet the autoMIL `composite` (the keep/discard + UCB selection signal for the whole search) **is** that val c-index.
- Evidence: [clam/survival_train.py:162-167](benchmarks/src/autobench/pipeline/clam/survival_train.py:162) (`mode="min"` on val loss; "near-random" comment; mirrored in abmil/dtfd); vs `composite = val_ci` at [run_experiment.py:118](benchmarks/scripts/run_experiment.py:118) and [clam/runner.py:58](benchmarks/src/autobench/pipeline/clam/runner.py:58). Val ≈ 12.5% of cases → ~2 events/fold for CPTAC-GBM/PDAC.
- **Failure:** every OS-cell agentic search optimizes noise; the val-argmax winner is a winner's-curse artifact whose certified test c-index regresses toward 0.5 and will not reproduce on reseed → any claimed agentic survival improvement is unfalsifiable.
- **Fix:** make the survival composite the same quantity used for checkpoint selection (negative val loss / partial-likelihood), or a pooled cross-fold concordance; do not select recipes on a 2-event c-index.

### CR-4 · δ=0 winner's-curse over ~10-patient validation inflates the lift by 5–10× the effect being measured
**Dimension:** rigor / fairness · **CONFIRMED (fairness + rigor + novelty streams)**

Selection keeps every positive fluctuation (`_accept` is strict `>` with δ=0; `best_node` = argmax composite over ~60 candidates), and each per-fold val is itself an early-stopping max-over-epochs on ~10 patients.
- Evidence: [graph.py:47](src/automil/graph.py:47) δ=0; [splits.py:149-155](benchmarks/src/autobench/pipeline/splits.py:149) val carve; ~60 candidates/cell (EXPERIMENT_GRID.md:278).
- **Quantified:** per-fold val AUC SE ≈ 0.15–0.20 on ~4 pos/6 neg; 5-fold-mean composite SE ≈ 0.08–0.12 (correlated folds don't reduce the shared upward bias); max over ~60 correlated candidates at δ=0 inflates the reported val composite by **≈ +0.1 to +0.2** — **5–10× the 0.02–0.03 encoder/aggregator spreads** the paper is trying to resolve, and *heterogeneous* (larger for CPTAC and for whichever arm gets more candidates). On CPTAC the "search improves models" claim may not survive on sealed test at all.
- **Fix:** set δ to a per-cell data-driven margin (≥1 SE of the CV composite); report the headline strictly on sealed test with a selection-corrected estimate; treat GBM/PDAC as generalization-breadth, not variance-decomposition inputs.

### CR-5 · Stale/shared results cache silently defeats the data fix AND collides across search variants
**Dimension:** bug / silent-wrong-result · **CONFIRMED (benchmark stream)**

Four of five runners (nnMIL/ABMIL/DTFD/TITAN) resume a fold from `benchmark_dir/results/<subdir>/fold_i/metrics.json` if it exists, but `subdir = framework/strategy/task/encoder/model[/loss]` — **omitting seed and every hyperparameter** — with no content/mtime/hash key. Only CLAM receives the isolated `AUTOMIL_RESULTS_DIR`.
- Evidence: [nnmil/train.py:55](benchmarks/src/autobench/pipeline/nnmil/train.py:55), abmil/runner.py:84, dtfd/runner.py:100; results_subdir at config.py:116-127; non-CLAM runners called without `results_dir` at run_experiment.py:326-332. Contradicts the isolation invariant asserted at run_experiment.py:260-264.
- **Failure A (data fix defeated):** re-run OS after the censoring fix — even purging `dataset_csv/`+`splits/`, the surviving `results/**/fold_*/metrics.json` resume-loads **pre-fix** c-indices and reports them as corrected.
- **Failure B (live search corruption):** two graph nodes whose variants keep the same `model_type` map to the same results dir → the second silently inherits the first's folds. In an agentic search over recipes, this reports one recipe's result under another's label.
- **Fix:** key the results dir on `experiment_id` (or a content hash of task-CSV + splits) and pass `AUTOMIL_RESULTS_DIR` into all four non-CLAM runners as CLAM already does.

---

## HIGH

- **H-1 · `run.log`/stdout not firewalled → test leak by printing.** *(rigor + daemon)* Seal relocates only `fold_*_result.json`/`results/`; `run.log` (subprocess stdout) stays at the agent-visible archive root ([_orchestrator_daemon.py:980-997](src/automil/backends/_orchestrator_daemon.py:980); terminal_writer.py:278). CLAM's test-AUC prints are gated only by a hand-patch to vendored `lib/CLAM/utils/core_utils.py` behind `AUTOMIL_CERTIFY` (never set) — a `git subtree pull` silently reintroduces the leak. On crash, the last 20 lines of run.log are copied into the agent-facing `result.json["error"]` (daemon:1310-1318). `smmile/runner.py:134` prints test **ungated** (off the paper path today). **Fix:** scrub held-out keys from run.log/error tails at the orchestrator boundary; expose only a filtered view.

- **H-2 · "Equal effort" is unequal: time-based budget, no eval-count primitive.** *(fairness — this is the budget question, now a code defect)* Only cap primitives are `agent_active`/`wall_clock` time ([capconfig.py:17-21](src/automil/cells/capconfig.py:17)); `cells/` has no candidate counter. `agent_active` pauses during GPU runs, so training is "free" and #candidates a cell evaluates ∝ (think-cycles × parallelism × wall-clock), none equalized. `max_concurrent_per_gpu` bin-packs by VRAM → a heavy aggregator (DTFD) gets fewer slots → fewer candidates in the same budget → arm-correlated bias on the measured axis. EXPERIMENT_GRID.md:281-289 independently notes the ~11.7 GPU-h/cell protocol exceeds the 6h cap, so the campaign runs truncated — biting expensive arms first. **Fix:** add an equal-N eval-count cap as the primary fairness primitive; keep time as a safety wall. *(Matches the pre-registration decision in `memory/automil-equal-effort-budget`.)*

- **H-3 · ABMIL/DTFD/TITAN silently ignore the hyperparameter knobs the search tunes.** *(fairness + benchmark)* `--lr/--max_epochs/--patience` and `variant_dispatch` `exp_cfg.train.*`/`model.*` reach CLAM/TITAN but ABMIL/DTFD build their own frozen `ABMILConfig()`/`DTFDConfig()` ([abmil/runner.py:62-63](benchmarks/src/autobench/pipeline/abmil/runner.py:62), dtfd/runner.py:69) and read only `model_type/embed_dim/seed/nll_bins`. Because the fields *exist* on the dataclass, `variant_dispatch` sets them with no "field not found" warning and they never reach training. **Failure:** an equal-effort search that tunes via these knobs optimizes CLAM/nnMIL while ABMIL/DTFD run baseline under the variant's label → "two aggregators improve, two flat" is an interface artifact, and the equal-effort premise is void for half the aggregator axis. **Fix:** thread `exp_cfg.train`/`model` into all four arms, or hard-fail when a set override is dropped.

- **H-4 · Frozen substrate is opt-in, unwired for the roster, inconsistent.** *(fairness + novelty)* `registry.protected` ships empty ([registry/config.py:37](src/automil/registry/config.py:37)); the hard gate works where wired (submit.py:263-274, check.py:389-407) but only 3 of 4 example overlays carry it — `ccrcc` has no `registry:` section (only `files.readonly`, which is a warning). **No roster cohort (LUAD/LGG/GBM/PDAC/HNSC) has an `automil/` overlay at all**, so the freeze is currently unverifiable for the actual campaign. Test is also passed in-process into the worktree (clam/runner.py:119-122). **Fix:** ship a non-empty default `protected` (splits/prepare/evaluate/runner/run_experiment/features) or make `check` hard-fail on empty protected for a benchmark project; wire every roster overlay before launch.

- **H-5 · Statistics cannot support the headline.** *(rigor)* "95% CIs" are a percentile bootstrap over **K=5** fold values ([evaluate.py:69-123](benchmarks/src/autobench/pipeline/evaluate.py:69)) at a **single seed** — the resample mean is bounded by the 5 folds' range, under-covers badly at n=5, ignores the t₄ width and fold correlation. And there is **no multiple-comparison control** anywhere in `benchmarks/` (grep-confirmed): dozens of "A>B" comparisons at α=0.05 → FWER≈1, so several headline "wins" are expected false positives. **Fix:** report `mean ± t₄·s/√5` (or BCa), repeat CV over ≥5 seeds, apply Holm/BH across each headline's pre-declared comparison family.

- **H-6 · `meta.best_node_id` can point to a DISCARDED node.** *(graph — reproduced)* Inline best-updates are status-agnostic strict-`>` and never re-gate on keep/discard ([graph.py:330-332](src/automil/graph.py:330), :408-410, terminal_writer.py:236-238); `_reevaluate_descendants` (which can flip a child to discard) never refreshes `meta.best`. With δ>0 a within-margin child is discarded yet set as best; `status`/viz report a discarded node as winner, `best_composite` inflates the `global_delta` baseline, and `certify --node <best>`/the zero-keep fallback reveal test for a discarded node. **Fix:** replace inline updates with keep-only `recompute_best()` after any status mutation.

- **H-7 · `automil cancel` races the daemon → `cancelled` overwritten to `crash`.** *(daemon)* For local jobs the daemon owns the `Popen`; `cancel` kills the process group directly, then the daemon reaps `-9`, fails to recognize the cancel, synthesizes `status="crash"`, and writes a spurious crash row to `results.tsv`+`completed/` and flips the graph node to crash; `total_proposed` double-adjusts ([cancel.py:114-268](src/automil/cli/cancel.py:114) vs daemon:1176-1345). **Fix:** daemon detects `cancel_reason∈{cap,cli}`/already-cancelled before promoting.

- **H-8 · `<2` valid folds reported as a zero-variance "completed" result.** *(benchmark)* `compute_confidence_intervals` averages non-NaN folds and returns `std=0, ci=mean` when `<2` survive ([evaluate.py:101-108](benchmarks/src/autobench/pipeline/evaluate.py:101)); `status` is unconditionally "completed" (run_experiment.py:139). A 1-valid-fold run masquerades as a 5-fold composite; keep/discard can't tell a robust result from a 1-fold fluke. **Fix:** record `n_valid_folds`; set `status=partial`/degrade composite when `n_valid < n_folds`.

- **H-9 · Novelty is thin as a methods contribution.** *(novelty)* Both load-bearing method claims are owned by prior work predating the preprint: **agentic code-level search = AIDE** (arXiv:2502.13138); **val-firewall/born-sealed = AIRA2 Hidden Consistent Evaluation** (arXiv:2603.26499, which *already* reported the residual gap is "evaluation noise not memorization" — autoMIL's on-record surviving angle). **AMID** (arXiv:2607.10522, CUHK, same month) shares the "autonomous auditable medical-imaging model development" framing. The headline "encoder ≫ aggregator" is owned as a *question* by a Frontiers 2026 paper **and** contradicted by autoMIL's own 210-config data (aggregator spread 3.0 > encoder 2.0) **and** underpowered (3 near-identical PFMs; MDES≈0.10 vs ~0.02 effect; task aliased with cohort). `RELATED_WORK.md` cites none of AIRA/AIRA2/AIDE/MLE-bench/HCE. **Fix:** cite them; reframe (see §Novelty).

---

## MEDIUM

| ID | Finding | Evidence | Stream |
|---|---|---|---|
| M-1 | `recalculate_scores` raises `KeyError` on a scoring dict missing `exploration_weight`/`novelty_weight` → **every `reconcile` silently a no-op** | graph.py:497-498 (backfill only sets `accept_margin`, :177-179) | graph (reproduced) |
| M-2 | Reconcile recovery not topologically ordered + skips descendant re-eval → order-dependent, permanently-wrong keep/discard | graph.py:767-823, 826-883 | graph |
| M-3 | NaN/±Inf `composite` persists as literal `NaN` in graph.json → breaks viz SSE (`JSON.parse`), `jq`, serde readers | graph.py:1079 `allow_nan=True`; viz/server.py:97 | graph (reproduced) |
| M-4 | `lineage()`/`_reevaluate_descendants` infinite-loop on a parent/child cycle (no visited set) → hangs daemon completion | graph.py:254-264, 431-454 | graph (reproduced hang) |
| M-5 | `_recover_orphans` writes crash artifacts but never updates graph.json → recovered nodes stay `running` forever | _orchestrator_daemon.py:649-698 | daemon |
| M-6 | Crash window in `_launch` between `Popen` and running-spec persistence → orphaned GPU child, lost node | daemon:929/993/1034-1047 | daemon |
| M-7 | Daemon terminal path never maintains `total_executed`/`total_proposed` → UCB exploration signal skewed all run | terminal_writer.py:221-238 | daemon |
| M-8 | `certify --top-k` re-opens test-set selection at reporting (prints test for K nodes, only warns) | cli/certify.py:39-106 | rigor |
| M-9 | Missing feature files silently drop slides from **val/test** (only train emptiness guarded) → class-correlated shrink, metric on reduced cohort | abmil/dataset.py:90-101; guards only at abmil/train.py:118 | benchmark |
| M-10 | Task-CSV/splits caches keyed on schema+existence, never content → a manifest **value** change (OS fix) is invisible | prepare.py:240-282, 355-375 | benchmark |
| M-11 | Nested-glob dataset lookup silently picks `nested[0]` on a stem collision (e.g. `templates/foo.yaml` vs `tcga/foo.yaml`) | config.py:239-241 | benchmark |
| M-12 | TITAN (512px slide-embedding, untrainable probe, excluded from search) sits on the "encoder axis" → conflates encoder × field-of-view × method-class | tcga_luad.yaml:64-74; EXPERIMENT_GRID.md:296-297 | fairness |
| M-13 | "Architecture" is a declared search target, so the post-search "aggregator axis" is not a fixed object (spec conflict: PLAN §5 vs the frozen-architecture claim) | PLAN.md §5; model files absent from protected | fairness |
| M-14 | Budget cell key omits `task` → a cohort's classification and OS searches share one budget → whichever runs first starves the other | cells/state.py:99-109 | fairness |
| M-15 | NaN val-c-index folds silently dropped from the composite mean → unequal, unreported denominators in head-to-head | survival_train.py:211; evaluate.py:98-102 | rigor |

---

## LOW

| ID | Finding | Evidence | Stream |
|---|---|---|---|
| L-1 | `run_benchmark.py` roster fallback → **0 experiments run silently** when a requested framework's roster default is empty | run_benchmark.py:139; config.py:151 | benchmark |
| L-2 | Survival trainer omits the cuDNN determinism flags the classification trainer sets | clam/train.py:54-55 vs survival_train.py:38-43 | rigor |
| L-3 | Worktree `result.json` holds test (`held_out`+`summary`) on disk until worktree cleanup | run_experiment.py:145,337-339 | rigor |
| L-4 | `${VAR:}` empty-default and blank env vars resolve to `""` without fail-fast → silently broken relative paths | config.py:27,36-43 | benchmark |
| L-5 | Case-level stratification `.first()` silently resolves conflicting per-slide labels; `n_classes=len(label_map)` miscounts collapsed maps | splits.py:99; config.py:204 | benchmark |
| L-6 | Daemon `cmd_submit` computes IDs from an un-persisted counter → back-to-back submits collide (legacy path) | _orchestrator_daemon.py:2040-2053 | daemon |
| L-7 | `_get_pending` sort can raise `TypeError` on mixed `submitted_at` types → stalls scheduling that tick | daemon:754 | daemon (POSSIBLE) |
| L-8 | Immutability rule violated pervasively (functionally safe today: single-owner + flock; viz reads unlocked) | graph.py promote/reeval/… | graph |
| L-9 | `certify/` quarantine is convention-only (no filesystem perms) — soft boundary vs an adversarial agent | archive/<node>/certify/ | daemon |
| L-10 | Cross-framework AUC asymmetry (CLAM per-class `nanmean` vs nnMIL `roc_auc_score(ovr,macro)`) → comparability caveat on 3-class folds | nnmil/evaluate.py:1-31 | benchmark |

---

## What is SOLID (so this review is balanced)

- **Patient-level split leakage is hard-asserted** — dedup to `case_id`, split cases, expand to slides, per-fold `_assert_no_patient_leakage` ([splits.py:186-203](benchmarks/src/autobench/pipeline/splits.py:186)). *(rigor + benchmark)*
- **`composite` is val-only at both writer sites**; `held_out`+`summary` stripped before any agent surface (terminal_writer.py:180-185); partial/recovery keeps held_out sealed. The *value* is trusted (CR-1), but the *wiring* is val-only.
- **OS censoring fix is sound and reaches training** — per-column non-null reduce with numeric coercion; non-positive/nonnumeric `OS_time` dropped (prepare.py:145-186); nnMIL builds from the filtered `task_df`; nllsurv bins computed from **train** only (no leak).
- **`results.tsv` sole-writer invariant holds** (only `_append_results_tsv`, single-threaded daemon).
- **Cell-budget restart double-count is prevented** (`accrue_active` caps per-tick delta at `idle_grace`); cap-kill vs completion within a tick is ordered safely.
- **`_accept` predicate is correct** (strict `>`, δ clamped ≥0, parent-None handled); **UCB has no div-by-zero/log(0)**.
- **Atomic writes** everywhere (`mkstemp`+`os.replace`); **PID-reuse defenses** (`starttime_ticks`) consistently applied.
- A crashing fold aborts before `result.json` is written — no silent fewer-fold "completed" from a crash (the `<2`-fold issue H-8 is a *degenerate-fold* path, distinct).

---

## Novelty & positioning (strategic)

**Historical framing — superseded by the 2026-07-30 prior-art audit.** This
audit originally proposed priority language around equal-effort code search and
presupposed a de-biased non-replication of the Frontiers result. Neither is now
allowed: AIDE/AIRA/AIRA2 establish direct code-search and controlled-evaluation
precedents, and the completed experiment has not yet established ranking
change, stability, or a directional variance result. Current authority is
[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md): C1 is the narrower auditable
lineage-comparison substrate; C2 is a result-neutral pathology-MIL ranking
audit.

**Additions that would raise the bar:** (a) **restore encoder dynamic range** — add a legacy anchor (ResNet50/CTransPath) + the TITAN slide arm so the axis *can* vary (cheap, feature-extraction only; converts a preordained/wrong-signed result into a measurable one); (b) run the **equal-GPU-hour head-to-head** agentic-vs-Optuna(PathBench)-vs-random-vs-human per cell, with a default-config warm-start control (else it's a known reviewer trap); (c) add a **recipe-family transfer/planner** (cluster winning recipes → cell-metadata→recipe map, evaluate on held-out cells) — the one angle that reads as science, not engineering.

---

## Prioritized remediation roadmap

**Gate 1 — before any agentic launch (cheap, mostly framework-side; several silently corrupt results):**
1. CR-1 finite-guard + recompute `composite` from val at ingestion.
2. CR-2 universal `locked_update` (or lock-guarded `save()`).
3. CR-5 isolate results cache per `experiment_id`; pass `AUTOMIL_RESULTS_DIR` to all runners.
4. CR-3 fix the survival composite (pooled concordance / val-loss).
5. H-3 thread `exp_cfg.train`/`model` into ABMIL/DTFD/TITAN.
6. H-4 ship+enforce a non-empty `protected`; wire roster overlays.
7. H-2 add the eval-count budget primitive (per `memory/automil-equal-effort-budget`).
8. H-1 scrub `run.log`/error tails at the boundary.
9. M-1/M-4 quick robustness: setdefault all scoring keys; add visited-set guards.

**Gate 2 — before any headline claim:**
1. CR-4 + H-5 statistics: multi-seed CV, t/BCa intervals, Holm/BH, per-cell δ ≥ 1 SE; report on **sealed test** with selection correction; demote/strengthen CPTAC.
2. Restore encoder dynamic range (novelty (a)).
3. Reframe the headline as a result-neutral ranking audit; add the missing
   citations and report change, stability, or a null result honestly.

**Gate 3 — to lift novelty:** the equal-GPU-hour AutoML head-to-head + the recipe-family planner.

---

*Provenance: 6 adversarial streams (rigor/leakage, fairness, novelty, orchestrator-daemon, graph/scoring, benchmark-pipeline); graph & daemon findings include harness reproductions; CR-1, CR-2, CR-3 additionally hand-verified by the synthesizer. Companion to `paper/preprint/PRELAUNCH_REVIEW.md`.*

---

# Addendum — Claim-level analysis (2026-07-23)

Written after the per-arm search-space audit (H-3b). The findings above are
code defects; this section maps them onto the **eight figures / claims** in
`EXPERIMENT_GRID.md §4`, and adds claim-level problems that only appear once the
code findings are combined with the experiment plan.

## The load-bearing failure: Fig 2's own caveat is broken

`EXPERIMENT_GRID.md §4` states the headline's escape clause verbatim:

> "this claim is only fair **after** the agentic recipe search equalises recipe
> effort — otherwise a reviewer says the aggregator gap is just under-tuning.
> Fig 2 + Fig 3 must be read together."

So Fig 2 (headline: encoder ≫ aggregator) is defended by Fig 3 (equal-effort
search). **H-3b breaks exactly that defence**: search-space coverage is 100% for
CLAM and 33% for DTFD, so Fig 3 cannot equalise effort. Fig 2 loses its stated
justification and Fig 3 is itself confounded — the paper's headline AND its
framework contribution fail together, and the bias direction is *predictable*
(a ranking flip toward CLAM is the artifact H-3b predicts, not a finding).

## Per-figure status

| Fig | Claim | Blocking problems |
|---|---|---|
| **1** | corrected benchmark is complete | hyperparameter provenance (CLAM 2× upstream lr, ABMIL 3 knobs off); H-5a K=5 bootstrap CI; H-5b no multiple-comparison control. H-8 fixed. |
| **2** | **headline:** encoder ≫ aggregator | O1 (own 210-config data says the reverse); O3 (encoder axis = 3 near-identical PFMs); **H-3b confounds the aggregator axis**; **single-seed mixed-effects cannot separate seed noise from the variance components it reports** |
| **3** | **framework:** equal-effort search lifts / flips rankings | **H-3b (equal effort not achievable)**; H-2 (time budget, not eval-count); CR-4 (δ=0 winner's curse, +0.1–0.2 = 5–10× the target effect); lift must be reported on sealed test, not val; **anchors invalid — see A-1** |
| **4** | TITAN wins slide-level | **TITAN is excluded from the agentic search** (§3.3 "omits the 5 TITAN classification arms") → after the search, tile arms improve and TITAN does not, so this claim can reverse for a reason unrelated to model quality — and if it survives, it survives unfairly. TITAN has only 4 knobs, so "equal effort" is meaningless for a linear probe. GBM/PDAC survival cells sit at ~2 events/fold. |
| **5** | test-quarantine discipline | this figure exists to demonstrate the firewall — but the firewall was **not** intact before this audit (composite trusted verbatim CR-1b; run.log leak H-1). **No pre-fix `graph.json` can be used**; it needs a fresh run on the repaired pipeline. |
| **7** | reproduces published SOTA | direct contradiction: CLAM runs at **2× its upstream learning rate**, so a protocol-parity claim against GOLDMARK is unsound until that is resolved or disclosed. Data also not yet pulled. |

## New claim-level findings

**A-1 · Fig 3's feasibility anchors are off-roster and pre-fix.**
Fig 3 leans on the CCRCC / ovarian-HRD agentic runs as anchors. Neither cohort is
in the post-pivot roster (LUAD/LGG/GBM/PDAC/HNSC), and their `learnings.md` is
dated 2026-06-24 — **before every fix in this audit** (CR-1b composite
derivation, CR-3 survival signal, CR-5 results-cache isolation, H-1 log
firewall). Anchors produced by an unrepaired pipeline cannot support the figure.

**A-2 · The search space is never declared.**
The plan says "architecture-preserving, recipe-only", but there is no
machine-readable per-arm searchable set anywhere. The de-facto search space is
"whatever the transport happens to carry" — accidental, and CLAM-shaped (H-3b).
The methods section cannot state what was searched. Note the target is a
*declared* set, not literally 100% coverage: DTFD's `distill` is deliberately
locked to AFS for a correctness reason, so some knobs should stay unreachable
**by declaration**.

**A-3 · Equal evaluation count does not equalise search difficulty.**
Even with coverage fixed, DTFD's 15-dimensional space and TITAN's 4-dimensional
space do not reach comparable optimisation quality at the same N. "Equal effort"
measured in evaluations equalises *spend*, not *difficulty*. This needs a stated
position (scale budget with dimensionality, or report only anytime curves and
decline the point claim).

**A-4 · The two modification channels contradict each other.**
Route 1 (CLI / variant args): coverage is asymmetric → unfair search.
Route 2 (agent edits config files directly): coverage is complete, but
architecture-preservation is unenforced (`registry.protected` ships empty, no
roster overlay exists) → the "recipe-only" constraint is unverified.
**Neither route currently supports the claim.**

## Decisions this forces (research, not code)

1. Fig 4: include TITAN in the search, or state that it is unsearched and drop
   the head-to-head framing.
2. Fig 3: re-run the anchors on the repaired pipeline, on roster cohorts.
3. Fig 2: multi-seed, or stop reporting variance components.
4. Fig 7 / Fig 1: return CLAM to upstream lr (invalidates the dispatched grid),
   or keep and disclose — but "reproduces SOTA" is unavailable under the latter.
