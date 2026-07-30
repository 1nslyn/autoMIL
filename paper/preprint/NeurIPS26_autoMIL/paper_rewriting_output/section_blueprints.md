# Section Blueprints

## Whole-paper contract

- **Working title:** `autoMIL: Auditable Source-Level Research for Controlled
  Model Comparison`
- **Paper identity:** framework-first public NeurIPS-format preprint.
- **Controlling question:** Can competing published methods be given a matched,
  attributable opportunity for autonomous source-level research without
  changing which method is being measured, and does doing
  so alter a pathology-MIL ranking?
- **Contribution order:** C1 autoMIL system; C2 controlled pathology-MIL ranking
  audit.
- **Current evidence state:** C1 mechanisms and focused invariant tests are
  available. C2 design, static roster, and validation-only baseline screen are
  available. Corrected baselines, matched campaign, seeds, and certified
  ranking outcomes are not.
- **Page target:** at most nine content pages before references.

## Abstract

**Function.** Five-sentence contribution contract: ranking confound; missing
repository-level measurement contract; autoMIL; current evidence and audit
design; payoff and boundary.

**Required anchors.** Confirmed motivation, CL1-CL10, citation candidates C003,
C009, C015, C029, C083, and E10/E13/E17.

**Prohibited.** Any claim that rankings changed or remained stable; “first,”
“fully autonomous,” “fair,” or “secure”; a feature list.

## 1. Introduction

### 1.1 Rankings hide research effort

**Move.** Start from a concrete measurement problem: a leaderboard number is
jointly produced by a method and the recipe/debugging effort allocated to it.
Use tunability and HPO-aware benchmarking to establish that rankings can depend
on optimization opportunity.

**Evidence/citations.** C083-C086; confirmed motivation.

### 1.2 Coding agents enlarge the confound

**Move.** Credit autonomous ML systems for code editing, tree search,
repository tasks, budgets, and hidden evaluation. Then isolate what those
systems do not automatically provide for cross-method measurement.

**Evidence/citations.** C001-C026; SOTA gap map.

### 1.3 autoMIL and contributions

**Move.** State C1 as the operations/evidence instrument and C2 as its
pathology-MIL use. Preview the current implementation evidence and explicitly
state that final ranking outcomes are not yet reported.

**Evidence.** Confirmed contribution; E01-E18.

**Figure.** F1 lifecycle on page one.

## 2. From Agentic Search to a Controlled Comparison

### 2.1 Research-opportunity estimand

**Move.** Define repository \(R\), published method class \(a\), cell \(c\), node \(v\),
parent \(p(v)\), overlay \(\Delta_v\), launched-attempt budget \(B_c\),
validation score \(s_v\), and held-out result \(h_v\). Define native/no-search
ranking, identity-preserving matched-search ranking, within-method lift, and
rank agreement without claiming values. Define the admissible set by frozen
method-defining inference/forward/core-loss mechanisms plus scalar and
executable train-only recipe changes.

**Evidence.** E01-E09, CL1-CL4, CL10.

### 2.2 What matched opportunity does and does not mean

**Move.** Explain why crashes consume attempts, why usable outcomes are reported
separately, and why this does not equalize compute or search difficulty.

**Evidence.** E05-E06; source comments in `state.py` and orchestrator.

### 2.3 Closest prior and residual gap

**Move.** Use a compact comparison table with explicit axes and an estimand
column. State that individual mechanisms are precedented; the residual is the
method-aware measurement use.

**Table.** T1.

## 3. autoMIL

### 3.1 Research layer: method identity and persistent state

**Move.** Describe parent-addressed nodes, overlays, validation-only
keep/discard, and stored trajectories. Keep search policy separate from the
substrate.

**Evidence.** `graph.py`, trajectory package, CL1.

### 3.2 Execution layer: reconstruction and failure semantics

**Move.** Explain pinned commits, detached worktrees, digest-checked overlays,
path safety, launch accounting, recovery, and backend abstraction.

**Evidence.** E01-E06. Limit backend claims using E26.

### 3.3 Evidence layer: permissions and born-sealed results

**Move.** Explain consumer-declared protected paths, split result writes,
exclusion of `certify/` from descendants, and deliberate final reveal.

**Evidence.** E04, E07-E09.

### 3.4 Threat model and consumer boundary

**Move.** State that the framework mediates normal experiment paths and is not
an OS security boundary. Explain the generic framework versus consumer-owned
command, task, result, and permissions.

**Evidence.** E11-E12; source map.

## 4. Autobench: A Pathology-MIL Ranking Audit

### 4.1 Cohorts, tasks, and model regimes

**Move.** Introduce five cohorts, classification and overall-survival axes,
three patch encoders, four tile aggregators, and TITAN as a separate slide-level
regime.

**Evidence/citations.** E13, E20; C027-C050, C073-C082.

**Table.** T2.

### 4.2 Native/no-search and matched-search regimes

**Move.** Define the native recipe baseline, source provenance, matched
launched-attempt opportunity, and required cross-arm/within-arm outputs. State
that the native baseline is not a controlled architecture-only comparison.
Specify that matched search includes scalar configuration and programmatic
train-only policies, while defining inference operators, forward branches, and
core loss/training mechanisms are invariant. Identity-breaking candidates are
archived separately and excluded from the cross-method leaderboard.

**Evidence.** CL10; E16, E21-E25.

**Figure.** F2 protocol schematic.

### 4.3 Metrics, selection, and inference

**Move.** Keep classification and survival metrics separate. Explain
validation-only selection and one-time certification. Predeclare cohort-aware
summaries, seeds, uncertainty, and rank agreement; reject cell independence.

**Evidence.** E07-E09, E24-E25; C047-C048, C069-C070.

### 4.4 Reproducibility and resource ledger

**Move.** Specify what must be released: source overlays, configs, upstream and
benchmark-added recipe components, attempts, usable results, failures, GPU/time
and LLM costs, selected nodes, and certified outputs.

**Evidence.** Confirmed contribution and C2 promotion requirements.

## 5. Validation and Current Evidence

### 5.1 Does the framework enforce its contract?

**Move.** Report the focused 51-test run by invariant family. Treat this as
implementation validation, not proof of scientific generality.

**Evidence.** E10-E12.

**Table.** T3.

### 5.2 Is the static study substrate usable?

**Move.** Report the external cluster audit: complete canonical roster within a
larger 195-run tree, fold integrity, non-finite-primary check, and the config
provenance defect. State source level and local-artifact limitation.

**Evidence.** E13-E16.

**Table.** T4.

### 5.3 Are the no-search baselines obvious strawmen?

**Move.** Report the validation-only screen: 151 clear, 19 near, 4 at/below
chance among 174 gradeable runs; concentration in GBM survival. Interpret only
as a triage diagnostic.

**Evidence.** E17-E18.

### 5.4 What remains unestablished

**Move.** State that the available materials do not establish corrected native
rankings, matched-search lift, rank change/stability, seed robustness, or
transfer. Explain exactly which artifacts unlock each statement.

**Evidence.** E21-E27.

## 6. Discussion and Limitations

### 6.1 Why the systems contribution matters

**Move.** Explain that the value is the estimand enabled by the integrated
contract, not novelty of its individual parts.

### 6.2 Internal validity

**Move.** Discuss baseline provenance, method-identity enforcement and the
programmatic-recipe seam, single seeds, validation overfitting, shared cohorts,
and certification discipline.

### 6.3 External validity and security boundary

**Move.** Limit claims beyond pathology, consumers, model roster, and tested
backends; state data and same-user threat boundaries.

### 6.4 Outcome-neutral scientific value

**Move.** Explain why either stable or changed ranks are informative, without
pre-committing the result.

## 7. Conclusion

**Function.** Re-state the established C1 contract, the C2 question, the current
evidence boundary, and the artifact-level path to a final answer in two compact
paragraphs.

## Appendix

- Full source/result contract and state transitions.
- Dataset/task provenance and attrition.
- Per-arm native versus benchmark-added recipe table.
- Per-arm method-identity invariants, guarded programmatic-recipe contract,
  candidate classifications, and file-permission declarations.
- Reproduction commands and environment.
- Full test inventory and known collection failures.
- Statistical analysis plan.
- NeurIPS checklist.
