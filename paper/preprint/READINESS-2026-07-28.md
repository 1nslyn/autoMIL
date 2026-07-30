# Preprint readiness — second-pass analysis (2026-07-28)

_Full re-analysis of the repository against the paper plan, run after the
2026-07-23 audit and its remediation branch (`fix/audit-2026-07-23`, 22 commits).
Method: five independent read-only sweeps (search-space coverage · frozen-substrate
enforcement · budget+gate+statistics · figure traceability · cluster state) plus
direct verification of every load-bearing claim in the main thread._

**Companion docs:** [`CODE-AUDIT-2026-07-23.md`](CODE-AUDIT-2026-07-23.md) (the
39-finding audit) · [`CODE-AUDIT-FIXES.md`](CODE-AUDIT-FIXES.md) (tracker) ·
[`PRELAUNCH_REVIEW.md`](PRELAUNCH_REVIEW.md) (O1/O2/O3) ·
[`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) (the plan) · [`PLAN.md`](PLAN.md) (the claims).

> **Point-in-time status warning.** The implementation findings below record the
> 2026-07-28 audit state and are retained for their reasoning. They are not the
> current completion ledger. Use [`CODE-AUDIT-FIXES.md`](CODE-AUDIT-FIXES.md)
> for live fix status and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) for current
> claim maturity.

---

> ## Scope — settled 2026-07-28 by Leo
>
> **Contribution hierarchy (reframed after the 2026-07-30 prior-art audit).**
> The primary contribution is the autoMIL auditable research-operations
> framework (C1), including its matched-evaluation and sealed-certification
> contract. The agentic campaign is indispensable evidence for C1 and supplies
> the result-neutral pathology-MIL ranking audit (C2). Fig. 3 is the planned main
> empirical result, with Fig. 5 and the gate as its rigor backbone; Figs. 1/4 are
> the benchmark substrate. See [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). C3 is
> deliberately unassigned; C4/C5 remain candidates.
>
> **The "encoder ≫ aggregator" claim is DROPPED.** It came from the Frontiers
> precedent, not from us; it is contradicted by our own baseline; and neither axis
> has a designed dynamic range. Fig 2 survives only as a descriptive variance
> decomposition with no "≫" assertion, or not at all.
>
> **This voids:** A-0 (Fig 2 deferring to Fig 3), O1 (self-contradiction), §1.8
> AXIS-RANGE, decision 8 (legacy encoder anchor — **no new experiments**), M-12.
>
> **This promotes:** CR-4 / H-5a / H-5b / multi-seed. The empirical claim
> includes per-cell lift selected on ~10 validation patients at δ=0.0 — the
> winner's-curse case — as well as cross-method reranking. Interval estimation
> is no longer optional.
>
> Ordering below in §3 is superseded by §3′.

## 0. Headline

The code fixes on this branch are real and hold up. What this pass establishes is
that **the remaining gap is not in the training pipeline — it is in the layer
between results and claims**, and it is larger than the tracker records.

Three things are true simultaneously, and together they set the critical path:

1. **No figure in the paper plan has producing code that reads real results.**
   Not one. Figures 1–5 exist only as `make_mock_figures.py` (fabricated `rng`
   values), Fig 7's input file is not in the repo, Fig 8 has no source asset, and
   Tables 1–2 are hardcoded transcriptions. The only code that ever turned real
   fold metrics into figures is `tasks/baseline_summary/scripts/` — **gitignored**,
   and structurally broken for this roster (see §2.3).
2. **Dataset identity is not recorded in any results artifact.** Not in
   `summary.json`, not in `aggregated/*.csv`, not in `results.tsv`, not in
   `graph.json`. It exists only as a filesystem path. Every planned figure is
   cross-cohort.
3. **The static grid's fold-level cache is blind to seed and to every
   hyperparameter** (§1.1). Any re-run — a corrected learning rate, a second seed —
   silently returns the old numbers. This traps three of the six decisions that
   were already pending.

Two prior conclusions are **corrected** below (§1.2, §1.3). Both were mine.

---

## 1. Corrections and new critical findings

### 1.1 CR-5b — the static grid resumes on a seed- and hyperparameter-blind key ⚑ NEW, CRITICAL

CR-5 (commit `e695304`) fixed this for the **orchestrated** path only. The
**static grid** — which produces Figs 1, 2, 4, 7 — is unfixed.

Verified chain:

- `submit_benchmark.sh`, `submit_titan.sh`, `submit_survival_benchmark.sh` never
  set `AUTOMIL_RESULTS_DIR`. The only writer is
  [`_orchestrator_daemon.py:855`](../../src/automil/backends/_orchestrator_daemon.py).
- Unset ⇒ every runner falls back to
  `benchmark_dir/results/{framework}/{strategy}/{task}/{encoder}/{model}[/{loss}]`
  ([`clam/runner.py:93`](../../benchmarks/src/autobench/pipeline/clam/runner.py),
  `abmil/runner.py:79`, dtfd/nnmil/titan identically).
  `ExperimentConfig.results_subdir` ([`config.py:116`](../../benchmarks/src/autobench/pipeline/config.py))
  contains **no seed, no lr, no epoch count**.
- Every trainer short-circuits when `metrics.json` exists: `clam/train.py:127`,
  `nnmil/train.py:55`, `titan/train.py:100`, `abmil/runner.py:97`,
  `dtfd/runner.py:113`, `clam/survival_train.py:128`, `titan/survival_train.py:141`.
- `run_benchmark.py:101` exposes `--seed` (default 42) → `TrainConfig.seed` (`:159`)
  and also drives split generation (`:196`).

**Consequences.**

| Action | What actually happens |
|---|---|
| `--seed 43` after a seed-42 grid | returns **seed 42's numbers verbatim**. A multi-seed variance study reports **zero variance**, silently. |
| Re-run after changing CLAM's `lr` | returns the **old 2e-4 numbers**. |
| Re-run after changing ABMIL's optimizer | same. |
| Adding an encoder (e.g. ResNet50 anchor) | **safe** — encoder is in the path. |

Split caches are not seed-keyed either, so `--seed` would not resample folds in
any case: "multi-seed" here means training-init variance only.

**Mitigation:** purge the results tree before any re-run; longer term put `seed`
plus a config hash into `results_subdir`, or set `AUTOMIL_RESULTS_DIR` from the
sbatch wrappers. This must be settled **before** decisions 1, 2 or 5 are acted on.

### 1.2 H-3b corrected — nnMIL is 0/11, not 4/11; CLAM's 100% is a circular measurement

Re-measured from source. The qualitative finding ("the search space is
CLAM-shaped") is correct and **understated**. Two numbers in the tracker are wrong:

| Arm | Tracker | Verified | Note |
|---|--:|--:|---|
| CLAM | 12/12 (100%) | **12/15 (80%)** | denominator was the transport itself — circular. `bag_loss`, `inst_loss`, `no_inst_cluster` are hardcoded at [`clam/train.py:76-79`](../../benchmarks/src/autobench/pipeline/clam/train.py) although all three are live upstream. |
| TITAN | 3/4 | 3/4 | correct (5/6 on the effective surface incl. the two knobs it reads off `exp_cfg.train`). |
| ABMIL | 5/8 | 5/8 | correct. Unreachable: `M`, `L`, `dropout`. |
| nnMIL | 4/11 (36%) | **0/11 (0%)** | see below. |
| DTFD | 5/15 (33%) | 5/15 | correct (5/13 = 38% counting live knobs only; `total_instance` is dead code, `distill` is a lock assertion). |

**nnMIL is zero.** `prepare_nnmil_experiment` declares `hparam_overrides` at
[`nnmil/prepare.py:53`](../../benchmarks/src/autobench/pipeline/nnmil/prepare.py)
and forwards it at `:181`, but **no production caller passes it** — verified:
`grep -rn hparam_overrides benchmarks/` returns exactly those two lines. Commit
`101ab35` ("wire TITAN and nnMIL into the override mechanism") wired TITAN and
added nnMIL's parameter without connecting it. Two further defects would survive
wiring: the plan cache early-returns on `(strategy, task, encoder, survival_loss)`
with no hyperparameter component (`prepare.py:79-80`, `:21-37`), so a second
experiment on the same combo inherits the first's hyperparameters; and
`early_stopping` is in `OVERRIDABLE` but is not a plan key, so it would raise.

**Two narrowings the earlier audit missed.**

- **`--override` cannot carry an arm-specific knob at all.** `run_experiment.py`
  uses `parse_args()` (`:93`), not `parse_known_args()`, so
  `--override "--numGroup 8"` is `SystemExit(2)` — it crashes the run rather than
  tuning DTFD. And there is **no `--weight_decay` and no `--early_stopping` CLI
  flag**; those two are reachable only through a registered variant's `CLAM_ARGS`.
  An agent driving the loop with `submit --override` alone gets **4 knobs on CLAM,
  3 on ABMIL/DTFD, 2 on TITAN, 0 on nnMIL**.
- **Reachable knobs have punched-out values.** `overrides_from_exp_cfg` reports a
  knob as overridden only when it differs from a pristine `TrainConfig()`
  ([`hparams.py:62,69`](../../benchmarks/src/autobench/pipeline/hparams.py)). So
  DTFD can never be asked for `lr=2e-4` or `weight_decay=1e-5`, and TITAN can
  never be asked for `lr=2e-4`, `weight_decay=1e-5` or `patience=20`. The
  docstring calls this "a harmless false negative"; it is harmless only where an
  arm's default equals the shared default, i.e. only for ABMIL.

**The proposed fix needs revising.** Step 2 of the plan in `HANDOFF.md`
("widen `OVERRIDABLE` to include `dropout`, `optimizer`, `weighted_sample`,
`stop_epoch`") is partly wrong: `dropout` lives on `ModelConfig`, and
`overrides_from_exp_cfg` reads only `exp_cfg.train` (`hparams.py:64`); and
forwarding `optimizer`/`weighted_sample`/`stop_epoch` trips the fail-loud
`ValueError` on ABMIL/DTFD/TITAN, which do not declare them. Any widening needs a
**per-arm allowlist**, not a global tuple.

### 1.3 The structured variant channel is a hyperparameter menu ⚑ NEW, HIGH

[`PLAN.md`](PLAN.md) §3 states the paper's differentiator: *"ours is an LLM agent
modifying training-recipe **code** directly … not config/hyperparameter menu
search — a different execution regime."* The 2026-07-30 prior-art audit removed
the unsupported word “stronger”: AIDE/AIRA already establish code-space search,
so the remaining distinction must be demonstrated at the auditable
lineage-comparison substrate.

In the benchmark consumer that is only true of the **unguarded** channel.
`apply_model_variant_to_exp_cfg`
([`variant_dispatch.py:182-213`](../../benchmarks/src/autobench/pipeline/variant_dispatch.py))
looks up the registered variant class and then uses it **solely** to read
`CLAM_ARGS`, assigning ≤14 scalar fields onto `ModelConfig`/`TrainConfig`; an
unrecognised field is dropped to a `logger.warning`. The `ModelVariant.forward()`
the agent writes is **never instantiated or called** by any autobench code.
Loss variants and policy variants are hard-refused with an explicit exception
([`lifecycle/apply.py:63-82`](../../src/automil/cli/lifecycle/apply.py)).

So the two channels are:

| Channel | What it can change | Guarded? |
|---|---|---|
| Registered variant (`CLAM_ARGS`) | ≤14 CLAM-shaped scalars | yes — and it *is* a menu |
| Free-mode source editing (instructed by the agent skill) | anything, incl. `splits.py`, the composite writer, fold/seed dataclasses | **no** (§1.4) |

This is finding A-4 stated at its sharpest: **the paper's edge and the paper's
rigor claim are in direct tension.** The more the substrate is enforced, the more
the search collapses toward the menu the paper claims to beat. The resolution is
not to pick one — it is to build the missing third thing: a *guarded code-level
channel* (`registry.mode: architecture-preserving` + a populated `protected`
list), so recipe/architecture edits are free while substrate files are
hard-rejected. H-4 done properly is what makes the differentiator claim both true
and safe.

### 1.4 H-4 confirmed and understated — the live project is the unprotected one

Commit `0b2da55` ("Add protected benchmark registry rules") added `protected`
globs to **`clwd`, `ovarian_hrd`, `placeholder`** — three template/example
projects whose `files.editable` entries (`train.py`, `models/`) match nothing in
this repo layout. It did **not** touch
`benchmarks/experiments/ccrcc/automil/config.yaml`, the only project with a real
`run.command`, `program.md`, `learnings.md` and hooks. ccrcc has no `registry:`
block at all, so `protected == ()`.

Because the gate is written `if reg_cfg.protected and _matches_scope(...)`
([`submit.py:263`](../../src/automil/cli/submit.py)), an empty tuple
short-circuits it to a **no-op** — and with it `check.py:389` and
`revert-baseline`. What remains on ccrcc is `files.readonly`, which prints
`Warning: … submitting anyway`.

Worse, on ccrcc the split entry point is affirmatively **editable**:
`run_experiment.py` (controls split generation inputs, the composite formula, and
the `metrics`/`held_out` partition) and `pipeline/config.py` (holds
`TrainConfig.seed` and `n_folds`) are both in `files.editable`, and the
no-`--files` branch of `submit` auto-detects them **with no `readonly` check at
all** (`submit.py:201-241`).

Three further gaps worth recording:

- **The shared `benchmark_dir` is outside git and outside every gate.** Split
  CSVs, task CSVs and converted `.pt` features live under `${data_root}/benchmark`.
  Every control in the system is git-path-based. `prepare.py:307-380` checks fold
  *count* and slide-ID coverage — not fold *membership*, and no hash.
- **The one hash that exists is never verified.** `submit.py:313` writes
  `overlay_manifest[f] = "sha256:…"` into the queue spec; no consumer checks it
  (`backends/local.py:167` writes empty strings; `port_variant.py:189` reads only
  `.keys()`). Verifying it in the daemon is a cheap, high-value integrity gain.
- **CR-1b's recompute catches an inconsistent scalar, not a tampered substrate.**
  `scoring.recompute_composite` takes the mean of whatever is in `metrics`, so a
  rewritten writer emitting `{"val_auc": <test_auc>}` recomputes to exactly the
  reported value and passes. That is a real limitation to state, not a defect.

`PLAN.md:339-340` ("not yet applied to any config") is now **stale** — correct it
before the preprint goes out.

### 1.5 No roster cohort has an agentic overlay ⚑ NEW, HIGH

`benchmarks/experiments/` contains **ccrcc, clwd, ovarian_hrd, placeholder** —
and nothing for TCGA-LUAD, TCGA-LGG, CPTAC-GBM, CPTAC-PDAC or TCGA-HNSC. The
agentic layer, which supplies the main empirical result, is not configured for a single
preprint cohort. The one live overlay (ccrcc) is off-roster, which is also why
Fig 3's anchors are off-roster (finding A-1).

### 1.6 ABMIL runs CLAM's schedule, by design ⚑ NEW, HIGH

[`abmil/config.py`](../../benchmarks/src/autobench/pipeline/abmil/config.py)'s own
docstring: *"paper-exact hidden dims (Ilse et al., 2018) **plus the shared
benchmark training schedule (matches `TrainConfig` defaults)**."* Values are
identical to `TrainConfig`: lr 2e-4, weight_decay 1e-5, max_epochs 200,
patience 20. Upstream ABMIL (`lib/AttentionDeepMIL/main.py:18`) is lr 5e-4,
reg 1e-4, epochs 20.

So the benchmark's implicit default recipe **is CLAM's**, and one of the four
aggregators inherited it wholesale while DTFD and nnMIL kept their own. Combined
with §1.2, the aggregator axis is CLAM-shaped in **both** the default regime and
the search regime. `provenance.py` records ABMIL's deviation but not that it
deviates *toward CLAM*; that direction is what matters for the headline.

Note also that `TrainConfig.lr = 2e-4` reaches CLAM directly
(`clam/train.py:83`), and ABMIL holds its own literal copy — so decisions 1 and 2
are **independent**: changing `TrainConfig.lr` moves CLAM only.

### 1.7 The evidence behind O1 is mis-described and confounded ⚑ NEW, MEDIUM

`tasks/baseline_summary/README.md` and `REPORT.md` describe `simple_mil` as
"mean-pool MIL". It is not:
`lib/nnMIL/network_architecture/models/simple_mil.py` is **gated attention**
(`V`/`U`/`w` + softmax over instances — the Ilse gated form) plus a random
256-channel subsampling regulariser.

So O1's contrast — `clam_mb` 0.637 vs `simple_mil` 0.607, "3.0 points" — is
gated-attention-plus-clustering-loss vs gated-attention: a *within-family*
architectural gap, measured across **two different recipe provenances** (CLAM on
the shared 2e-4 schedule, nnMIL on its own self-configured 3e-4/100-epoch plan).
Architecture and recipe are fully confounded in the single number the docs use to
contradict the headline. The baseline also ranks by **test** AUC, while the new
pipeline selects on val — not a defect, but not the same quantity either.

### 1.8 Neither axis has a designed dynamic range ⚑ NEW, MEDIUM

`PRELAUNCH_REVIEW` O1/O3 criticise the encoder axis as three same-generation ViT
foundation models. The same criticism applies to the aggregator axis, and has not
been made: `clam_mb` (gated attention + multi-branch + instance clustering),
`simple_mil` (gated attention + channel subsampling), `abmil` (**non-gated**
attention — `abmil/model.py:222` maps the roster key to the non-gated class),
`dtfd_mil` (two-tier pseudo-bag). Three of four are Ilse-lineage attention
pooling; DTFD is the only architectural outlier.

**Whichever direction "X ≫ Y" lands, it is attributable to roster choice.** The
remedy is therefore not only "restore encoder range" (O1 option a) — it is to
*declare and justify the intended dynamic range on both axes* in the methods.

### 1.9 Smaller defects found in passing

| ID | Finding | Where |
|---|---|---|
| TSV-1 | `results.tsv` locks its metric columns from the **first row**; later rows are aligned to that header and unknown keys are silently dropped. A campaign mixing classification (`val_auc`/`val_bacc`) and survival (`val_c_index`) loses `val_c_index` for every row after the first. | `_orchestrator_daemon.py:1772-1796` |
| CFG-1 | The `gate:` block in the config template (`auto_nominate`, `K`, `p_threshold`, `bootstrap_reps`) is **never read by any code** — `grep -rn 'get("gate")' src` is empty. All values come from `automil gate register-manifest` CLI defaults. The block is decorative. | `templates/config.yaml.j2:203-207` vs `cli/gate.py:78,84,90` |
| CFG-2 | The template places `default_vram_estimate_gb` / `max_concurrent_per_gpu` under `cap:`, but the daemon reads them from `orchestrator:`. A project following the template silently gets the hardcoded default. `max_concurrent_per_gpu` is exactly the arm-correlated bias knob H-2 names. | `config.yaml.j2:190-191` vs `_orchestrator_daemon.py:439-444` |
| CAP-1 | The **launch path has no cap check**. `_launch()` / `_get_pending()` never consult cell status, so specs already queued when a cell flips to `REFUSING_NEW` still launch. Only `automil submit` gates. Legacy nodes without `metadata.cell_id` are "immune to cap enforcement" by the daemon's own comment. | `_orchestrator_daemon.py:742-754, 917, 1054-1069` |
| CELL-1 | **`cell_id` never appears in `graph.py`.** The graph has no notion of cell membership, so "how many evaluations did cell X get" is unanswerable from the graph — which blocks eval-count reporting even as a *secondary* metric, and blocks Fig 3's per-cell keying. | `src/automil/graph.py` |
| TAB-1 | Tables 1 and 2 are **hardcoded literals** (`make_dataset_table.py:86-98`, `:152-158`), not read from YAMLs or manifests. `EXPERIMENT_GRID.md` §2.1 calls Table 2 "verified by running `generate_all_experiments`"; the script never calls it. | `paper/preprint/figures/make_dataset_table.py` |

---

## 2. Claim-by-claim readiness

### 2.1 Historical claim readiness, reconciled to the current spine

| Claim | Status | What blocks it |
|---|---|---|
| **Retired Fig. 2 direction — encoder ≫ aggregator** | **dropped; not a contribution** | Contradicted by the baseline and unsupported by the roster's axis design. A descriptive decomposition may remain, but no direction is frozen. |
| **C2 / Fig. 3 — equal-effort agentic search** | **not supportable today** | (i) coverage asymmetry 80% CLAM / 0% nnMIL (§1.2); (ii) budget is time-only, no eval-count primitive, and cells are not linked to the graph (§1.9 CELL-1); (iii) no roster cohort has an agentic overlay (§1.5); (iv) the substrate freeze is unenforced on the one live project (§1.4); (v) anchors are off-roster and pre-fix (A-1); (vi) selection is δ=0.0 on ~10 val patients (O2). |

### 2.2 Figures

| # | Figure | Producing code | Data | Verdict |
|---|---|---|---|---|
| 1 | Leaderboard heatmap | mock only (`rng`) | partial — no `dataset` column anywhere | **write from scratch** |
| 2 | Encoder-vs-aggregator variance | none (fractions hardcoded) | no — single seed, per-fold values absent from every roll-up | **blocked** (needs multi-seed decision) |
| 3 | Recipe-search effect | none | no — cell identity not on any result row; test sealed per-node with no bulk export | **blocked** (needs H-3b, H-4, overlays) |
| 4 | Survival c-index | mock only | partial — `test_c_index_*` exists in `aggregated/*.csv`, but the only aggregator has no `c_index` key and no `survival_loss` path level | **write from scratch** |
| 5 | Search trajectory | none static (live SSE dashboard only) | yes — `graph.json` carries composite/status/parent/created_at | **small**: one export script; needs a fresh post-fix run (A-6) |
| 6 | Competitive table | none | yes (`PLAN.md` §3) | **small**: render; one row self-flagged unverified |
| 7 | Protocol parity | none | **no** — `goldmark_exact/COMPARISON.csv` not in repo, on purgeable scratch; code unmerged (`origin/feat/goldmark-parity`, `d42f0b4`) | **blocked**; claim also contested by CLAM's lr (A-7) |
| 8 | Pipeline schematic | none | n/a | **nothing exists** — no `.svg/.drawio/.puml/.mmd` tracked |
| T1/T2 | Dataset stats, grid breakdown | exists but hardcoded | not recomputable | **rewrite to read real sources** |

### 2.3 The one real results→figures pipeline is gitignored and wrong for this roster

`tasks/baseline_summary/scripts/` is the only code that has ever produced figures
from real fold metrics. `tasks/` is gitignored, and it has five independent
breakages against the current roster:

- `ROOT = Path(".../TCGA")` + `ROOT.glob("TCGA-*")` ⇒ **CPTAC-GBM and CPTAC-PDAC
  are structurally invisible** (2 of 5 cohorts).
- `for fw in ("clam", "nnmil")` ⇒ `abmil`, `dtfd`, `titan` never walked.
- `KEEP_AND_RENAME` keys on `("nnmil","ab_mil")` / `("nnmil","dtfd_mil")` — both
  wrong: the key is `abmil`, and both are their own frameworks. Can never match.
- `METRIC_KEYS` has no `c_index` ⇒ survival never aggregated (100 of 165 experiments).
- The walk has no `<survival_loss>` level ⇒ cox and nllsurv variants collide.

### 2.4 What is genuinely solid

Worth stating, because it is the part that does not need re-litigating:

- **The generalization gate is rigorous.** Pre-registered, git-committed manifest
  that refuses overwrite; one-sided paired Wilcoxon + BCa bootstrap CI on the
  median delta + Bonferroni over `K_effective`; three conjunctive conditions
  (`p ≤ α/K`, `ci_low > 0`, every cell wins). Cap-exhausted cells are excluded and
  drive an explicit `inconclusive` branch rather than a silent pass.
  (`src/automil/gate/`)
- **The val-firewall is real** at the level it claims: born-sealed test block,
  quarantine to `certify.json`, log redaction (H-1), composite recomputation from
  the val block (CR-1b). Its residual limit is stated in §1.4.
- **Data integrity was independently verified** (PRELAUNCH §4): label
  distributions, patient-level splits with a leakage assertion, 1:1 feature
  coverage across all three encoders in all five cohorts, HNSC grade recomputed
  with 0 mismatches.
- **The multi-class path is genuine**, not merely configured.
- **Upstream fidelity is now documented field-by-field** (`provenance.py`):
  DTFD faithful, nnMIL faithful, CLAM off by one knob, ABMIL off by three.

---

## 3′. Critical path — ordered by what blocks Fig 3 (settled scope)

1. **CR-5b** (§1.1) — prerequisite for every re-run; today `--seed 43` silently
   returns seed 42's numbers.
2. **H-3b + A-2** (§1.2) — per-arm allowlist, connect nnMIL (0/11 today), an
   override channel that does not `SystemExit(2)` on an arm-specific name, and a
   **declared** per-arm searchable set. Without this a Fig-3 ranking flip toward
   CLAM is a channel-width artifact, not a finding.
3. **H-4 + VARIANT-MENU + HASH-0** (§1.3, §1.4) — the guarded code-level channel.
   This is both the rigor backbone and the only thing supporting the paper's
   "code-level search, not menu search" differentiator.
4. **CELL-1 + H-2 + CAP-1** (§1.9) — "equal effort" must be measurable;
   `cell_id` does not currently reach `graph.py` at all.
5. **NO-OVERLAY** (§1.5) — five roster cohorts have no agentic overlay.
6. **Statistics** — per-cell δ (CR-4), t₄/BCa (H-5a), Holm/BH (H-5b), multi-seed.
   Promoted from optional: the empirical analysis includes per-cell lift on ~10 val patients.
7. **FIG-0 + DATA-ID** (§0) — the results→claims layer.
8. **Upstream defaults** (decision 2, settled) + the grid re-run.
9. Remaining MED/LOW backlog.

## 3. Critical path (superseded by §3′ — kept for the reasoning)

Ordered by what unblocks the most, not by severity.

1. **CR-5b** — put `seed` + a config hash into the results path (or set
   `AUTOMIL_RESULTS_DIR` from the sbatch wrappers). ~1 file, but it gates every
   re-run and the multi-seed decision. **Do this before any re-run is dispatched.**
2. **The results→figures layer** — record `dataset` in `summary.json`; write one
   cross-cohort collector that walks all five frameworks, both task types, and the
   `survival_loss` level; add a bulk `certify --export`. This is what turns a
   finished grid into a paper, and today none of it exists in git.
3. **H-3b + A-2** — per-arm allowlist (not a global tuple), an opaque
   `hparam_overrides` channel that actually reaches nnMIL, `parse_known_args` or a
   dedicated override path, and a **declared** per-arm searchable set.
4. **H-4** — populate `registry.protected` on the *live* project, add
   `architecture-preserving` mode, verify the overlay manifest hash in the daemon.
   This is also what makes §1.3's differentiator claim defensible.
5. **H-2 + CELL-1** — eval-count budget primitive, and record `cell_id` on graph
   nodes so per-cell effort is reportable at all.
6. Then the statistics work: per-cell δ (CR-4), t₄/BCa intervals (H-5a), a
   correction family for the grid (H-5b).

---

## 4. Decisions

**Settled 2026-07-28:**

- **#7 Scope — the agentic campaign is required to validate the framework.**
  Full C1/C2 study (Fig. 3 + Fig. 5), not a static benchmark with a feasibility
  demo. C1 remains the primary system contribution.
- **#8 O1 — the "encoder ≫ aggregator" claim is dropped**, and **no new
  experiments** are added (no legacy encoder anchor). This voids O1, A-0, §1.8 and
  M-12 rather than resolving them.
- **#1 + #2 Hyperparameter provenance — every arm returns to its own published
  upstream defaults**, after CR-5b lands. CLAM lr 2e-4 → 1e-4; ABMIL → 5e-4 /
  1e-4 / 20 epochs. Requires purging and re-running the dispatched grid
  (≈40 GPU-h, training only). Restores the Fig 7 SOTA-parity claim as available.
- **#5 Seeds — promoted from optional to required** by the scope decision: the
  empirical analysis includes per-cell lift selected on ~10 validation patients at δ=0.0.
  Blocked on CR-5b.

**Still open:**

- **#3** Fig 4: include TITAN in the agentic search, or drop the head-to-head
  framing (A-5). TITAN has 4 knobs, so "equal effort" is ill-defined for it.
- **#4** Fig 3: re-run the CCRCC / ovarian-HRD anchors (off-roster **and**
  pre-fix), or replace them with roster cells once the overlays exist (A-1).
- **#6** A-3: equal eval-count ≠ equal search difficulty (DTFD 13-dim vs TITAN
  4-dim). Needs a stated position — scale budget with dimensionality, or report
  anytime curves only.
- **#9** Keep "code-level search, not menu search" (⇒ H-4 is mandatory, which is
  the plan) or soften to a declared search space.
- **#10** Figure ownership: who writes the results→figures layer, and does the
  system schematic get drawn in-repo.

_Original framing of the ten decisions, kept for the reasoning:_

| # | Decision | Consequence |
|---|---|---|
| 1 | **CLAM back to upstream `lr=1e-4`?** (currently 2e-4) | invalidates the dispatched grid → re-run (≈40 GPU-h, training only). Gates the Fig 7 SOTA-parity claim (A-7). |
| 2 | **ABMIL back to upstream optimizer?** (lr 5e-4, reg 1e-4, epochs 20) | same. Note upstream ABMIL is a toy MNIST-bags experiment, so deviating is defensible **if stated** — but §1.6 shows it deviates *toward CLAM*, which is the harder thing to defend. |
| 3 | Fig 4: include TITAN in the agentic search, or drop the head-to-head framing (A-5) | as-is, tile arms improve after the search and TITAN does not. |
| 4 | Fig 3: re-run the CCRCC / ovarian-HRD anchors (off-roster **and** pre-fix) (A-1) | compute + coordination. |
| 5 | Seeds: multi-seed, or stop reporting variance components (A-8 / H-5c) | **blocked by CR-5b** — today a second seed silently returns the first seed's numbers. |
| 6 | A-3: equal eval-count ≠ equal search difficulty (DTFD 13-dim vs TITAN 4-dim) | needs a stated position. |
| **7** | **RESOLVED 2026-07-28 — Scope.** The campaign remains required to validate C1 and establish empirical C2; it does not replace C1 as the primary contribution. | Recorded in [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md). |
| **8** | **RESOLVED 2026-07-28 — O1 position.** Drop the directional claim; do not add a legacy encoder merely to revive it. | Fig. 2 is descriptive only; no extra experiment is required for that retired claim. |
| **9** | **new — the differentiator claim** (§1.3). Keep "code-level search, not menu search" and build the guarded code-level channel, or soften the claim to a declared search space? | Keeping it makes H-4 mandatory, not optional. |
| **10** | **new — figure ownership.** Who writes the results→figures layer, and does the paper's schematic (Fig 8, intended as the paper's Fig 1) get drawn in-repo? | ~nothing exists today; it is on the critical path regardless of decision 7. |

**Cost note for 1 & 2:** re-running costs only *training*. Features, `.pt`
conversions, splits and any TITAN extraction survive — ≈40 GPU-h, ~½–1 day on
4×H100. Much cheaper now than after the agentic layer, whose C2 analysis includes
both cross-method rank/stability and within-method lift. **But it is only safe
after CR-5b is fixed or the results tree is purged.**

---

## 5. Not verified this pass

- **Cluster state.** SSH to `fir` failed at authentication (control master expired;
  both agent keys rejected, only `keyboard-interactive` offered — an MFA prompt
  that cannot be answered non-interactively). Queue state, the cluster's checkout
  commit, per-dataset completion counts, failures and `sacct` GPU-hours are all
  **unknown as of this document**. Rebuild the master from an interactive terminal
  and re-run the status check.
- Whether CPTAC-PDAC's underlying continuous infiltration score is recoverable
  (PLAN.md §4 open item — still unchecked).

---

## 6. Positions taken on the reporting findings (M-13, L-9)

> **Superseded decision notice — 2026-07-30.** The M-13 position below is
> retained as audit history but no longer governs the preprint. The current
> authority is [`PLAN.md`](PLAN.md) §5 and `CODE-AUDIT-FIXES.md` METHOD-1:
> C2 uses method-identity-preserving search with both scalar configuration and
> executable train-only recipe programs. Defining inference/forward/core-loss
> mechanisms are frozen per published method. Identity-breaking descendants are
> archived separately and excluded from the cross-method leaderboard.

Two findings were carried as "needs a decision" but did not actually need a new
one — the decision existed and had not been written down where it binds.

### M-13 — historical position: the aggregator axis is a LINEAGE, not a fixed architecture

`PLAN.md` §5 (resolved 2026-07-15) already places *"Architecture / aggregator
internals / training hyperparams / loss"* in the **searchable** column. That is
the de-biasing objective; narrowing the search to recipe-only would remove the
thing the paper claims. So the position is **best-evolved-head**, and M-13's
alternative — freeze the architecture and add the model files to
`registry.protected` — is rejected as inconsistent with the settled scope.

**The consequence has to be stated in the paper, not just in the code.** If
architecture evolves, then "CLAM-MB" in a results table does not mean CLAM-MB. It
means *the best variant descended from CLAM-MB under an equal-effort search*.
Printing the bare model name would be a straightforward misattribution: a reader
would take it as a statement about a published method when it is a statement
about a lineage.

Concretely, for the write-up:

- Figures 1 and 4 (the **default-recipe** leaderboard) may use bare arm names —
  those really are the published architectures at their own defaults.
- Figure 3 (the **searched** result) must label arms as lineages, and the methods
  section must say what an arm label denotes.
- `search_space.py` is what makes this auditable: it declares, per arm, exactly
  which knobs the search could move and which were locked and why.

### L-9 — the `certify/` quarantine is a convention, and the paper should say so

`archive/<node>/certify/` holds the sealed test block. What actually enforces the
firewall, in descending order of strength:

1. **The orchestrator never copies `certify/` into a worktree**
   (`runner.apply_overlay` skips the whole subtree). This is real: the agent's
   filesystem view does not contain it.
2. **`terminal_writer` strips `held_out` and `summary`** from every agent-facing
   artifact, and **`firewall.py` redacts `run.log` and both error tails** (H-1) —
   so test values do not leak through the channels an agent does read.
3. **The composite is recomputed from the val block** (CR-1b), so a writer that
   reported a test-derived scalar is caught.
4. **Directory convention.** `certify/` is not permission-protected. An agent
   running as the same user, with shell access outside its worktree, could read
   it.

Layer 4 is the honest limit and it should be disclosed rather than implied away.
The threat model this defends against is **an agent that follows its instructions
and the tooling it is given** — the plausible failure is inadvertent leakage
(printing test metrics into a log, folding test into the reported composite), not
an adversary deliberately reading a path it was told not to. Every such
inadvertent route is closed by layers 1–3.

The structural fix that would close layer 4 — withhold the test split from the
worktree entirely and have the orchestrator evaluate the frozen model
out-of-process — is `PLAN.md` §5's layer 3, deferred to Phase 2. State the
current boundary in the methods section; do not claim the stronger one.
