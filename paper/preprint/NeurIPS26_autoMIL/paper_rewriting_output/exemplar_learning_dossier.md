# Exemplar Learning Dossier

## Exemplar Inventory

| Exemplar | Status and type | Narrative spine | Transferable lesson | Claim warning |
|---|---|---|---|---|
| AIDE | 2025 arXiv; autonomous ML engineering system | engineering problem → code-space formulation → tree search → evaluation → limitations | Give a minimal formal abstraction, then concrete operators; balance method and evaluation | Code-space search and experiment trees are not autoMIL novelty |
| AIRA | 2025 arXiv; controlled research-agent study | identify confounds → factor policy/operators/environment → controlled evaluation → generalization gap | Isolate system factors and organize results by scientific question | Do not claim all search, operators, and infrastructure as core innovations |
| MLAgentBench | ICML 2024; benchmark and baseline agent | research question → task contract → environment/actions → agent → outcome/process/efficiency | Define initial files and evaluator before the agent; measure both endpoint and process | Its task boundary differs from a persistent multi-file repository |
| MLRC-Bench | NeurIPS 2025; research-agent benchmark | measurement-validity gap → objective contract → protocol → scaffold/model/reliability/cost | Prove the evaluation signal is trustworthy before ranking systems | autoMIL is not only a benchmark; C1 must remain visible |
| RE-Bench | 2024 report; AI R&D evaluation suite | benchmark desiderata → human calibration → anytime evaluation → behavior/failures → limits | Use explicit desiderata, matched budgets, process curves, and failure behavior | Its report-length structure is unsuitable for nine pages |
| AMID | 2026 preliminary report; medical-imaging research agent | domain constraints → artifact contract → workflow → verification mechanisms → cases/limits | Define the final artifact contract, lifecycle, and verification control layer early | Do not inherit broad “first” or fully autonomous language |

## Structural Patterns

1. **Complete the story on page one.** Present the failure mode, gap, narrow
   contribution, and validation plan; use a lifecycle diagram rather than a
   feature poster.
2. **Define the object and contract before the agent.** State repository input,
   source-level intervention, parent-linked state, selection signal, resource
   accounting, and final reconstructable artifact.
3. **Separate three system layers.**

   - Research layer: proposal, parent choice, and persistent knowledge.
   - Execution layer: overlay reconstruction, worktree isolation, scheduling,
     failure and recovery.
   - Evidence layer: result contract, validation firewall, lineage, and
     certification.

4. **Use only necessary formalism.** A small state definition, acceptance rule,
   and lifecycle figure are enough. Do not mathematically decorate the CLI or
   filesystem.
5. **Organize experiments by research question.** Candidate questions are:

   - RQ1: Does autoMIL execute source-level experiments reproducibly and with
     the promised isolation/accounting?
   - RQ2: Under matched declared research effort, how do within-method
     performance and cross-method rankings change?
   - RQ3: What reliability, resource, leakage, and portability costs accompany
     those conclusions?

6. **Cover four evidence dimensions.** Endpoint quality, search process,
   systems reliability, and computational cost.
7. **Keep failure analysis in the main paper.** Invalid edits, validation
   overfitting, infrastructure failures, orphan recovery, or misleading memory
   are scientific evidence about the system.

## Rhetorical Patterns

- Move rapidly from broad automated discovery language to the narrower,
  testable object: source-level, traceable research iteration in an existing
  repository.
- Use controlled axes and concrete comparison conditions.
- Pair each contribution with evidence:

  - “introduces” requires an artifact and comparison;
  - “enforces” requires invariants or adversarial tests;
  - “supports” requires a real project/backend/runtime demonstration.

- Answer leakage, budget fairness, randomness, failure semantics, human
  intervention, and contamination before the reviewer asks.
- Compare closest work along precise capability axes: source-level
  intervention, repository scope, isolation, lineage, persistent knowledge,
  validation/test separation, resource orchestration, and runtime portability.
- Distinguish “implemented” from “scientifically established.”
- Avoid unqualified “first,” “fully autonomous,” “domain-agnostic,”
  “general,” “robust,” and “end-to-end scientific discovery.”

## Language Patterns

Preferred verbs are concrete and falsifiable:

| Function | Pattern |
|---|---|
| Define gap | “Existing agents can propose code changes, but do not by themselves provide …” |
| Narrow contribution | “We study the systems layer required to make these iterations …” |
| Describe mechanism | “autoMIL records / isolates / enforces / reconstructs …” |
| State control | “Holding the declared evaluation budget fixed, we compare …” |
| Report evidence | “Across predeclared cells, we measure …” |
| Bound conclusion | “This evaluation establishes X under Y; it does not establish Z.” |
| State limitation | “The current study does not isolate …” |

Use one principal claim per sentence. Keep numbers, conditions, and comparison
objects next to the claim. Use experiment, node, parent, candidate, validation,
and held-out consistently.

## Nine-page allocation

| Section | Pages | Main responsibility |
|---|---:|---|
| Title and abstract | 0.45 | problem, narrow contribution, mechanism, evidence type, boundary |
| 1. Introduction | 0.85 | ranking confound, system gap, C1/C2, first-page lifecycle figure |
| 2. Related Work and Claim Boundary | 0.55 | AutoML, agents, evaluation systems, pathology MIL; capability table |
| 3. Problem Setting and Contracts | 0.60 | repository, source diff, node, budget, result, validation/certification |
| 4. autoMIL | 1.80 | three layers, overlay/worktree, tree, recovery, firewall |
| 5. Experimental Design | 0.85 | RQs, cohorts/arms, baselines, matched budget, seeds, metrics |
| 6. Results | 2.45 | framework evidence, within-arm lift, cross-arm ranking, reliability/cost |
| 7. Analysis, Threats, and Limitations | 1.10 | failures, dependence, leakage, lineage, external validity |
| 8. Conclusion | 0.35 | established contribution and boundary |
| **Total** | **9.00** | |
