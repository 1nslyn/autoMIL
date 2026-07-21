---
title: AutoMIL Proposal
category: references
project: automil
org: wanglab
doc_type: Proposal
source: https://app.notion.com/p/3515155f4162804885dbea8c3b6e9e53
source_type: notion
summary: >
  Authoritative autoMIL proposal reframing the project as an equal-effort,
  auditable agent-driven training-recipe search and bias-correction framework
  for pathology MIL.
created: 2026-04-29T04:40:52Z
updated: 2026-06-26T12:00:00Z
lifecycle: stable
lifecycle_changed: 2026-06-26
---

> **Status note (added 2026-07-01):** This is an old reference, not exactly
> right relative to the confirmed pivot in
> [`../preprint/PLAN.md`](../preprint/PLAN.md). Kept here verbatim for provenance and because
> several sub-parts (freeze-before-test discipline, reviewer-attack
> rebuttals in §10, statistical-analysis plan in §6.5) may still be useful.
> It predates the confirmed pivot and does not mention slide-level
> PFMs, regression, or the PathBench-MIL/Frontiers comparison.

# AutoMIL Proposal

Rewritten 2026-04-29 after critical review. This section is the authoritative
proposal. The previous broader draft is retained below as Appendix A for
provenance.

## TL;DR

Current MIL evaluations can be biased because each architecture is evaluated
under training recipes that may be unevenly tuned or accidentally favorable
to one model. autoMIL should be framed as an auditable equal-effort recipe
search framework: for every task/encoder/model cell, it lets an agent search
for a strong validation-selected recipe under the same budget, freezes that
recipe before final testing, and reports how much leaderboard ranking changes
after recipe bias is controlled.

One-line summary: autoMIL makes MIL comparisons fairer by giving every model
the same audited recipe-optimization effort before comparing final test
performance.

## 1. Core Thesis

Current pathology MIL leaderboards often compare architectures under
inconsistent training recipes. This means published rankings can partially
reflect optimizer, loss, sampling, regularization, early stopping,
thresholding, and ensembling choices rather than architecture quality.

autoMIL should be framed as an agent-driven recipe discovery and auditing
framework for MIL. It lets coding agents propose and implement training-policy
changes, runs each candidate in isolated git worktrees, stores every code
overlay and result, freezes the winning recipe before final testing, and
reports how much leaderboard variance comes from recipe choice rather than
architecture.

The first paper should not claim "maximum performance." That is not
provable. The defensible claim is:

> Under a pre-registered equal-effort protocol, agent-driven recipe search
> reveals and corrects substantial training-recipe bias in MIL benchmarking.
> The resulting leaderboard is more reproducible, better audited, and more
> informative about architecture quality than published-default comparisons.

This is the strongest and most publishable framing.

## 2. Recommended Paper Framing

### Title Direction

autoMIL: Auditing and Correcting Training-Recipe Bias in Pathology Multiple
Instance Learning

Alternative titles:
- Training Recipes, Not Just Architectures: Agent-Driven Auditing of MIL
  Benchmarks
- autoMIL: Equal-Effort Agentic Recipe Optimization for Computational
  Pathology MIL
- How Much of MIL Performance Is Recipe Bias? An Agentic Benchmarking Study

### One-Sentence Pitch

autoMIL quantifies how much MIL architecture rankings change when every
method receives an equal-effort, architecture-preserving recipe search, then
releases the frozen recipes, experiment traces, and corrected leaderboard.

### Why This Is Attractive

- It targets a real pain point: MIL papers are hard to compare because
  training details differ across architectures, encoders, datasets, and
  codebases.
- It is timely: pathology foundation models make patch features strong, so
  the remaining performance differences often come from aggregation and
  recipe.
- It has clear artifacts: framework, frozen recipes, corrected leaderboard,
  experiment graph traces, and code overlays.
- It differentiates from pure AutoML by allowing code-level recipe invention
  while preserving architecture identity.
- It produces scientific insight, not only better scores: variance
  decomposition of architecture vs encoder vs task vs recipe.

### What To Avoid Claiming

- Do not claim autoMIL finds the "best achievable" recipe.
- Do not claim per-cell recipes are automatically a field standard unless
  transfer and stability are shown.
- Do not claim full code-level model improvement in the first paper.
- Do not use test-set metrics inside the search loop.

## 3. Competitive Landscape and Positioning

| Work | Why It Matters | autoMIL Positioning |
|---|---|---|
| PathBench-MIL | Direct competitor. It already provides MIL benchmarking, preprocessing, feature extraction, aggregation, Optuna optimization, and visualization. | autoMIL must not be "another MIL AutoML." Position it as agentic code-level recipe discovery with full experiment traceability, evaluated against PathBench-MIL/Optuna at equal compute. |
| nnMIL | Strong generalizable MIL framework across large pathology scale. Shows careful recipes can beat complex methods. | Use as motivation and baseline. autoMIL asks a different question: how much do rankings change when each architecture receives equal recipe-search effort? |
| Patho-Bench | Public benchmark with many canonical pathology tasks and splits. | Prefer Patho-Bench tasks/splits where possible. This reduces "private benchmark" criticism. |
| MLE-STAR / AI Scientist / AlphaEvolve | Establish that LLM agents can improve ML/code via targeted search and feedback. | Use as general agentic-search precedent, not as the main novelty. autoMIL's novelty is the pathology MIL protocol and artifacts. |
| Optuna / random search / human recipes | Necessary baselines. | autoMIL must beat or complement these under equal budget. If it ties Optuna, the auditability and code-level recipe space must still be valuable. |

Key differentiation sentence:

> Unlike PathBench-MIL, which optimizes over a configured pipeline/search
> menu, autoMIL lets an agent modify recipe code under a restricted
> architecture-preservation contract, stores every code overlay, and audits
> whether resulting leaderboard changes are due to recipe, architecture,
> encoder, or dataset.

## 4. Primary Research Questions

### RQ1: Recipe Bias

How much of apparent MIL architecture performance is explained by training
recipe choice?

Evidence:
- Ranking flips between published/default recipes and equal-effort optimized
  recipes.
- Variance decomposition across model, encoder, task, and recipe.
- Confidence-bounded rankings where overlapping confidence intervals are
  tied, not forced into a ranking.

### RQ2: Agentic Search Value

Does agent-driven code-level recipe search outperform menu-based AutoML and
random mutation at equal compute?

Evidence:
- autoMIL vs PathBench-MIL/Optuna under matched GPU-hour or wall-clock
  budget.
- autoMIL vs random recipe mutation from the same typed policy space.
- autoMIL vs human-curated strong recipe.
- Ablation: agent memory on/off, literature retrieval on/off, code-level
  edits vs fixed hyperparameter menu.

### RQ3: Stability and Transfer

Are discovered recipe families stable and interpretable, or are they
per-cell overfits?

Evidence:
- Frozen recipe performance on held-out test only after search completes.
- Transfer from source cells to unseen target cells.
- Repeated searches with different LLM seeds on a subset of cells.
- Recipe-family analysis: e.g., SAM helps small binary tasks, thresholding
  helps imbalanced binary tasks, multi-init helps high-variance tasks.

### RQ4: Auditability

Can the field inspect exactly what changed?

Evidence:
- Every recipe is a committed artifact: YAML policy + code patch + manifest.
- Every candidate has base commit, overlay hash, logs, result JSON, and
  status.
- Architecture-preservation validator output is released.

## 5. Scope: First Paper vs Follow-Up

### First Paper: Recipe-Bias Audit and Corrected Leaderboard

This is the recommended primary paper.

**Allowed changes:**
- Optimizer and wrappers: AdamW, SAM, Lookahead, gradient clipping.
- LR schedule and early stopping.
- Existing dropout rate and weight decay.
- Loss terms outside the model forward pass: label smoothing, focal
  variants, class-balanced loss, calibration penalties.
- Sampling and batch construction.
- Multi-init or snapshot ensembling.
- Validation-only threshold optimization.
- Feature/bag augmentation, if it does not change architecture.

**Forbidden changes:**
- Modifying `forward()`.
- Adding/removing model layers.
- Changing attention, pooling, heads, transformer blocks, or aggregator
  logic.
- Adding learned positional encodings.
- Changing input/output signatures.
- Any change that makes the result hard to attribute to the parent
  architecture.

### Follow-Up Paper: Full Code-Level Model Improvement

This should be deferred. It becomes credible only after the recipe-only
protocol is clean.

Possible later claim: given a frozen recipe-search baseline, full code-level
agentic search adds measurable gain beyond recipe optimization for selected
parent models. Do not make F2 load-bearing in the first manuscript.

## 6. Experimental Design

### 6.1 Benchmark Grid

Use public, canonical tasks wherever possible. Prefer Patho-Bench
tasks/splits when raw image access and features are available.

Recommended full grid:
- Tasks / cohorts (6 representative tasks): CCRCC high-grade, CLWD subtype,
  TCGA-LUAD EGFR or KRAS, TCGA-BRCA PIK3CA, TCGA-GBM PTEN, TCGA-COAD MSI or
  BRAF
- Encoders: H-optimus-1, UNI2, Virchow2
- MIL architectures: AB-MIL, DSMIL, TransMIL, CLAM-MB, DTFD-MIL
- Cells: 6 tasks × 3 encoders × 5 models = 90 cells

Use "task" rather than "dataset" as the unit. If one cohort contributes two
labels, those are two tasks and should be counted separately.

### 6.2 Methods Compared

For each cell, compare:
1. Published/default recipe reimplementation under identical splits and
   feature files.
2. Human strong recipe based on nnMIL/CLAM best practices.
3. PathBench-MIL/Optuna or equivalent menu-based AutoML at equal compute.
4. Random mutation from the same typed recipe policy space.
5. autoMIL agent-driven recipe search.

Original-paper reported numbers should be cited only as context, not used as
the primary baseline, because those papers usually differ in splits,
encoders, preprocessing, and evaluation.

### 6.3 Search and Evaluation Protocol

Central rule: **the agent may optimize only on training/validation data.
Test metrics are hidden until the recipe is frozen.**

Recommended protocol per cell:

1. **Data split** — use public train/test split when available; carve
   validation from training data; for small datasets, use nested
   cross-validation (inner folds for search, outer folds for final
   reporting).
2. **Discovery stage** — budget 60 candidates per cell for the first full
   paper; each candidate uses a cheap proxy (1 seed × 3 inner folds, or 1
   official train/val split depending on dataset size); score candidates by
   validation composite only; store all failed/crashed/discarded candidates.
3. **Promotion stage** — rerun top 10 candidates with 3 seeds on inner
   validation; select top 3 by mean validation composite and stability;
   freeze exactly one recipe before touching test.
4. **Final evaluation** — evaluate frozen recipes on outer test or official
   test; use 5 seeds × 5 folds where feasible; no further edits after test
   results are seen.
5. **Extended-search ablation** — on 5 representative cells, run 5× budget;
   report whether gain saturates and whether the same recipe family emerges.

This structure solves the most dangerous reviewer attack: repeated adaptive
test-set overfitting.

### 6.4 Metrics

**Primary:**
- Binary tasks: AUROC, balanced accuracy, AUPRC, composite.
- Multiclass tasks: macro AUROC, macro F1, balanced accuracy, composite.
- Composite should be pre-registered per task family. Avoid changing
  composite after seeing results.

**Secondary:**
- Calibration: ECE / Brier score.
- Runtime: GPU-hours and wall-clock.
- Stability: standard deviation across seeds.
- Failure rate: crash/OOM/NaN rate for each search method.

### 6.5 Statistical Analysis

Do not treat 5 folds × 5 seeds as 25 independent samples without
qualification. Use hierarchical modeling or aggregate to cell-level paired
differences.

Recommended reporting:
- Per-cell bootstrap confidence intervals.
- Cell-level paired tests across the 90 cells.
- Mixed-effects model: `performance ~ architecture + encoder + task + recipe
  + (1 | cohort)`
- Variance decomposition: percent variance explained by recipe vs
  architecture vs encoder vs task.
- Ranking flip rate with CI-bounded ties.

Headline numbers should be:
- Percent of pairwise architecture rankings that flip after recipe
  optimization.
- Mean improvement of autoMIL over default, Optuna, random, and human
  recipe.
- Fraction of cells where recipe explains more variance than architecture.
- Transfer performance of recipe families on unseen cells.

## 7. Concrete Deliverables

**Public Artifacts:**
- autoMIL framework — CLI, orchestrator, graph, worktree overlay runner,
  visualization.
- Recipe Set v1.0 — one frozen recipe per cell (YAML policy + code patch +
  base commit + manifest).
- Corrected Leaderboard v1.0 — default vs human strong recipe vs Optuna vs
  random vs autoMIL, CI-bounded ties.
- Experiment Trace Archive — all candidates, not just winners.
- Architecture-Preservation Audit — automated validator result for every
  candidate, human-readable diff summary for each frozen recipe.

**Internal Engineering Needed Before Full Study (blockers, not polish):**
- Remove hardcoded CCRCC recipe knobs from shared benchmark files.
- Make all recipe knobs config-driven or recipe-driven.
- Add `automil apply <node_id>` / `automil export-recipe <node_id>`.
- Commit winning overlays as versioned artifacts, not only gitignored
  archives.
- Add an architecture-preservation validator.
- Add split-leakage checks and fold-isolation checks.
- Ensure `graph.json` cannot mark known-invalid leakage nodes as current
  best.

The current CCRCC state proves the framework can discover improvements, but
it also proves the publication pipeline needs stronger controls before
scaling.

## 8. Pilot Study

Run a smaller pilot before committing to the full grid.

**Pilot Grid:**
- Tasks: CCRCC high-grade, CLWD subtype, TCGA-LUAD EGFR
- Encoders: H-optimus-1, UNI2
- Models: CLAM-MB, AB-MIL, TransMIL
- Cells: 3 × 2 × 3 = 18
- Methods: default, human strong recipe, Optuna, random, autoMIL

**Pilot Success Criteria** — proceed to full paper only if at least two are
true:
1. autoMIL beats Optuna/random/human recipe on mean validation-selected test
   performance across the 18 cells.
2. At least 25% of confidence-bounded architecture pairwise rankings change
   under equal-effort recipes.
3. Recipe explains at least as much variance as architecture in a
   mixed-effects analysis.
4. At least one recipe family transfers to unseen cells better than default.

If none are true, pivot to a negative/audit paper: recipe search matters on
some cells, but architecture/encoder dominate after strong human recipes.
The artifact is still valuable as a reproducibility and audit framework, but
the headline changes.

## 9. Current Evidence and How To Present It

**Existing Evidence:**
- Ovarian HRD example: 189 experiments, composite improved from about 0.814
  to 0.851.
- CCRCC high-grade: trusted baseline about 0.7443; trusted best is
  `node_0176` about 0.8074 after hoptimus1 + LS 0.08 + dropout 0.42 +
  patience 25 + N=3 + thresholding + WeightedRandomSampler + SAM rho=0.05.
- CCRCC also produced an invalid `node_0194` result around 0.8914 due to
  cross-fold leakage in Lookahead. This must be explicitly excluded and used
  as motivation for stronger leakage checks.

**How To Use This Evidence:**
- Use current results as feasibility evidence only: "autoMIL can find
  meaningful recipe improvements." "Naive autonomous loops can also discover
  invalid leakage artifacts, so the paper contributes an audited protocol."
- Do not use the CCRCC result as the main scientific conclusion until it is
  rerun under the final locked validation/test protocol.

## 10. Reviewer Attacks and Required Answers

**Attack 1: "This is just Optuna with Claude."**
Required answer: compare to Optuna/PathBench-MIL under equal compute; show
examples where the agent generated code-level recipe changes outside a fixed
hyperparameter menu; include random mutation and human-recipe baselines;
report cases where Optuna ties or wins honestly.

**Attack 2: "You overfit the test set."**
Required answer: test metrics are invisible during search; recipes are
frozen before final evaluation; all candidate selection uses validation
only; include nested CV or official train/val/test splits.

**Attack 3: "Per-cell recipe optimization is not a standard."**
Required answer: the contribution is not a universal recipe; it is a
protocol for equal-effort recipe auditing; add transfer experiments to
discover recipe families; release recipes as artifacts, not as universal
rules.

**Attack 4: "Architecture identity is not preserved."**
Required answer: F1 forbids `forward()` edits and module graph changes; the
validator checks model graph, parameter names/shapes, and input/output
signature; recipe patches are separated from model code where possible.

**Attack 5: "The benchmark is too small/private."**
Required answer: use Patho-Bench tasks/splits where possible; include at
least one external validation task; release all split files, recipes, and
manifests.

**Attack 6: "The LLM is non-reproducible."**
Required answer: reproducibility is at the protocol/artifact level; frozen
recipes are deterministic artifacts; repeat search on a subset with multiple
agent seeds and report variation.

**Attack 7: "The compute budget favors complex methods unfairly."**
Required answer: equalize by GPU-hours or wall-clock, not only candidate
count; report performance-vs-compute curves; include budget sensitivity
analysis.

## 11. More Promising Pivot: Recipe Planner, Not Per-Cell Brute Force

The highest-upside version is not 90 independent recipe searches. It is a
self-configuring recipe planner for pathology MIL.

**Planner Goal:** given metadata about a new MIL task (sample size; class
imbalance; binary vs multiclass; number of slides per case; encoder type and
feature dimension; model family; validation instability; bag size
distribution), predict a strong starting recipe: optimizer; LR schedule;
sampling policy; regularization; ensembling level; threshold/calibration
policy; early stopping policy.

**Why This Is Stronger:**
- Closer to nnU-Net's real contribution: not just search, but learned
  self-configuration.
- More useful to the community than per-cell recipes.
- Reduces "you brute-forced the benchmark" criticism.
- Produces interpretable scientific rules about when recipe components work.

**How To Add It Without Derailing F1:** use the 90-cell search traces as
training data for a recipe-family analysis — cluster winning recipes; fit a
simple meta-model from cell metadata to recipe family; evaluate planner
recommendations on held-out cells; compare planner vs full autoMIL search vs
default. This can become the strongest discussion section or the second
paper.

## 12. Final Recommended Roadmap

**Phase 0: Clean the Framework and Benchmark Harness**
- Make benchmark training fully config/recipe-driven.
- Remove hardcoded CCRCC choices from shared files.
- Add recipe export/apply.
- Add architecture-preservation and leakage validators.
- Fix graph invalidation so known leakage artifacts cannot remain best.

**Phase 1: Pilot**
- Run 18-cell pilot.
- Include default, human recipe, Optuna, random, and autoMIL.
- Use validation-only search and frozen-test evaluation.
- Decide whether full F1 is worth scaling.

**Phase 2: Full F1**
- Run 90-cell grid if pilot passes.
- Produce corrected leaderboard, variance decomposition, and recipe
  artifacts.
- Submit as benchmark/audit paper.

**Phase 3: Planner / Transfer**
- Learn recipe families from search traces.
- Evaluate on held-out cells.
- This is the path toward an nnU-Net-style contribution.

**Phase 4: F2 Full Code-Level Search**
- Only after F1 is accepted or strongly validated.
- Compare full code-level variants against parent + frozen F1 recipe.
- Treat identity preservation as a central methodological contribution.

## 13. Bottom Line

The proposal is promising if the first paper is narrowed to recipe-bias
auditing and corrected MIL benchmarking. It is weaker if it tries to
simultaneously claim agentic AutoML, leaderboard standardization, full
code-level discovery, and new improved model variants.

The highest-probability publishable contribution is:

> autoMIL shows that MIL architecture rankings are materially affected by
> training recipe choice, provides an equal-effort agentic protocol to audit
> and correct that bias, and releases the frozen recipes and experiment
> traces.

The highest-upside extension is:

> autoMIL learns a self-configuring recipe planner that predicts strong MIL
> training policies for new pathology tasks.

Focus the manuscript around those two ideas.
