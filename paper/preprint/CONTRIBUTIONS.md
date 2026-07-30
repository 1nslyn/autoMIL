# Preprint contribution register

_Status: active and authoritative for the preprint's contribution hierarchy and
claim maturity. `EXPERIMENT_GRID.md` remains authoritative for experiment
counts. Updated 2026-07-30 after Leo confirmed the prior-art-driven reframing
and the method-identity-preserving, programmatic-recipe search boundary._

This register distinguishes three phrases that older planning documents used
interchangeably:

- **core contribution** — what the paper adds;
- **main empirical result** — the strongest experiment validating the
  contribution;
- **headline figure** — how that result is presented.

Fig. 3 may be the main empirical figure without replacing autoMIL as the
paper's primary contribution.

## Confirmed contribution structure

### C1 — autoMIL auditable research-operations framework

**Type:** system / framework contribution · **Role:** primary contribution

**Contribution statement.** We introduce autoMIL, an auditable
research-operations substrate for applying coding agents to existing,
multi-file ML repositories and for comparing the resulting model and training
lineages under a controlled protocol. Each candidate is represented as a
parent-addressed source overlay, executed in an isolated worktree under explicit
evaluation and resource budgets, and governed by declared editable/protected
source surfaces. Validation drives search while held-out results are born sealed
and revealed only through a separate certification path.

This contribution includes the matched-evaluation and leakage-control protocol;
that protocol is a design and enforcement contract of the framework, not a
separate claim that equal budgets, hidden tests, provenance, or sandboxing were
individually invented here.

**Evidence already available.**

- git-worktree isolation with per-node source overlays and provenance;
- persistent parent-child experiment tree, archived results, and learnings;
- validation-only keep/discard with a quarantined held-out test block;
- budget cells, local/SLURM/Ray backends, recovery, and multi-runtime assets;
- file-level editable/protected boundaries, per-cell identity, evaluation-count
  caps, overlay manifests, and a separate certification path;
- framework-purity tests separating generic `src/automil/` mechanisms from the
  autobench consumer;
- end-to-end consumers in pathology MIL and sklearn-iris.

**Evidence still required before the strongest paper claim.**

- close the known end-to-end gate, cell-identity, and budget-enforcement defects;
- demonstrate the corrected production path, not only unit-level mechanisms;
- verify born-sealed non-interference with adversarial leakage tests rather than
  only checking artifact placement;
- verify matched launched-attempt accounting, report usable
  completed/partial-result counts separately, and enforce each roster arm's
  method-identity contract across both configuration and source-edit channels;
- establish consistent failure and result semantics across real Local, SLURM,
  and Ray executions, or narrow the backend-portability claim;
- report the exact consumer contract and the adaptation required by each
  demonstration project, including at least one additional non-toy consumer
  before making a strong domain-agnostic claim;
- compare the concrete substrate mechanisms against AIDE, AIRA/AIRA2,
  MLAgentBench, AI Scientist-v2, RE-Bench, MLRC-Bench, and AMID.

**Closest-prior and claim boundary.** autoMIL does **not** introduce a new
code-space search algorithm, experiment-tree policy, autonomous scientist,
hidden-test principle, resource-budget concept, or sandboxing mechanism. AIDE
already formulates ML engineering as code-space tree search; AIRA/AIRA2 combine
candidate graphs, memory, bounded and isolated execution, and
validation/final-evaluation separation; MLAgentBench, AI Scientist-v2,
RE-Bench, MLRC-Bench, and AMID establish further agentic experimentation and
auditable-evaluation precedents. The potentially distinguishing system claim is
the use of parent-addressed multi-file overlays, framework-mediated
non-interference, and matched-attempt accounting to audit competing methods
inside existing repositories.

autoMIL does not infer an arbitrary project's search space or make an arbitrary
training script compatible without a consumer-provided command, result
contract, and permissions. Equal effort means an equal declared cap on launched
evaluation attempts: crashed, partial, and budget-killed runs consume the cap,
while usable completed/partial-result counts are reported separately.
Wall-clock, GPU use, LLM tokens/cost, failures, and invalid nodes must also be
reported. Equal attempt opportunity does not mean different methods have
search spaces of equal difficulty. Test sealing constrains the normal
framework-mediated path; it is not an OS-level security boundary against an
adversarial same-user process. Do not use categorical priority language such as
“first autonomous ML framework.”

### C2 — controlled measurement of recipe bias in pathology MIL

**Type:** planned empirical study / benchmark analysis · **Role:** primary
empirical contribution and validation of C1

**Result-neutral contribution statement.** Applying autoMIL to a controlled
pathology MIL benchmark, we measure how equal-effort source-level recipe
optimization affects cross-method performance, ranks, within-method lift, and
rank stability relative to a frozen, declared no-search baseline with per-arm
provenance. The searched intervention space is broader than a hyperparameter
table: it includes consumer-declared, programmatic train-only policies while
preserving each published method's defining mechanisms.

**Required evidence.**

- frozen no-search cross-arm leaderboard, with published/native components and
  benchmark-added protocol components identified arm by arm;
- matched equal-effort searched cross-method leaderboard on sealed test,
  containing only candidates that pass the predeclared method-identity contract;
- within-method lift, rank shift, anytime curves, and resource accounting;
- classification and survival reported separately, with cohort-aware inference;
- seed sensitivity for the frozen baseline and selected final recipes;
- TITAN labelled as a distinct slide-level regime unless it receives a matched
  search protocol;
- a per-arm method-identity contract that freezes the inference operator family
  and every defining training mechanism while permitting declared scalar
  configuration and programmatic train-only recipe edits;
- a runnable programmatic-recipe channel, candidate-level classification
  (`config-only`, `programmatic-recipe`, `identity-breaking`, `invalid`), and
  evidence that both accepted edit classes are represented in the archived
  campaign. Identity-breaking candidates are excluded from the C2 leaderboard.

**Current status.** Planned empirical question, not an established finding. The
corrected baseline, agentic campaign, and sealed-test analysis have not yet
supplied the final answer. The paper must report ranking change, stability, or a
null result honestly rather than presupposing a flip. M-13's scientific policy
is resolved, but its method-identity enforcement and programmatic-recipe
channel are not implemented; BASE-1/BASE-2 also remain open. SEED-1 currently
defers multi-seed execution, so the strongest robustness claim is blocked unless
that decision changes. If it does not, C2 must be narrowed and the missing
seed-sensitivity evidence disclosed.

**Claim boundary.** Do not treat heterogeneous task metrics as directly
exchangeable or the 165 cells as independent samples. Only identity-preserving
candidates may enter the searched cross-method leaderboard. Architecture- or
mechanism-changing descendants may be archived as secondary evolved lineages,
but they are not results for the published method and may not appear under its
bare name.

### C3 — deliberately unassigned

No third contribution is claimed before the corrected campaign and final
analysis reveal a distinct field-level result. This prevents framework
components from being promoted merely to preserve consecutive numbering. The
historical C4/C5 candidate labels below are retained because Leo explicitly
reserved them as possible fourth and fifth contributions; neither is promoted
by this gap.

## Candidate contributions — not yet promoted

### C4 candidate — open corpus of autonomous ML research trajectories

**Potential statement.** Release a structured corpus of source diffs,
experiment-tree lineage, agent trajectories, failures, validation outcomes, and
sealed final evaluations from a real multi-cohort autonomous research campaign.

**Promotion gate.** C4 becomes a contribution only if the release contains real
non-mock campaign artifacts with stable identifiers, a documented schema,
secret/privacy redaction, provenance and licenses, sufficient coverage across
lineages/tasks/cohorts, and a reproducible loading/analysis path. The corrected
C1/C2 campaign must be complete; the release must cover every completed roster
cell and artifact class except explicitly documented technical/privacy
exclusions, and report capture completeness, dropped-event counts, and missing
data mechanisms. Otherwise the traces are supporting artifacts for C1.

### C5 candidate — transferable research knowledge across held-out cells

**Potential statement.** Show that modifications discovered in one search cell
generalize to pre-registered held-out cells, demonstrating transferable research
knowledge rather than isolated per-cell tuning.

**Promotion gate.** C5 requires the rebuilt Stage-B gate, exact matched
held-out baselines, real artifact reads with no score fallback, pre-declared
transfer hypotheses, multiple target cells, seed-aware uncertainty, and
positive sealed-test evidence. Target cells must remain isolated from search,
candidate selection, and hypothesis formation; registration must precede any
target-artifact read. Candidate and matched parent must use the same command,
split, seeds, and budget on each target and fail closed on any mismatch. The
promotion rule must predeclare an effect threshold, uncertainty criterion, and
multiplicity family. A null result remains scientifically useful but must be
reported as a limitation or boundary, not promoted as this contribution.

## Contribution freeze

No additional contribution is added before the corrected campaign and final
analysis are complete. Scheduler/backend support, the dashboard, hardware
autodetection, individual firewall components, static cohort coverage, TITAN
integration, and protocol-parity reproduction are features or validation
evidence unless the completed experiments establish a separate field-level
claim.

The current paper spine is therefore:

```text
C1 autoMIL auditable research-operations framework
  -> C2 pathology-MIL equal-research-effort ranking audit

C3 remains deliberately unassigned.

C4 trajectory corpus and C5 cross-cell transfer remain candidates.
```
