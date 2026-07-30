# Research Dossier — NeurIPS 2026 / autoMIL

## Venue Requirements

### Version and track

- The NeurIPS 2026 abstract and paper deadlines were May 4 and May 6, 2026
  (AoE). As of July 30, 2026, this artifact can be a NeurIPS-format preprint,
  but it cannot be described as a new NeurIPS 2026 submission unless an
  OpenReview submission already existed.
- The public manuscript should use `\usepackage[preprint]{neurips_2026}`. It
  must not use `final`, claim acceptance, or say “under review at NeurIPS.”
- Main, Evaluations & Datasets (E&D), and Position Paper are separate tracks.
  Their rhetorical conventions are useful even though this deliverable is a
  public preprint.

### Pages and files

- An anonymous submission permits nine content pages including all figures and
  tables. References, technical appendices, and the mandatory checklist do not
  count toward that limit. The preprint should preserve this discipline.
- The PDF should be US Letter, use the unmodified official 2026 style, embed
  fonts, and place the checklist after references and any appendix.
- Essential evidence for a core claim must remain in the main paper; the
  appendix is optional reading.

### Checklist and reproducibility

- The 2026 checklist is mandatory in the official format and requests
  justifications for claims, limitations, code/data, experiment details,
  statistical significance, compute, assets, ethics, and LLM use.
- Because an LLM agent is part of the method, the paper must report the model
  and runtime, prompts/search policy, permissions, budget, human intervention,
  failures, and selection protocol.
- A public evaluation-framework paper should provide executable code,
  documentation, and exact artifact reconstruction instructions even when the
  artifact is presented as a preprint.

## Review Criteria

NeurIPS evaluates quality, clarity, significance, and originality. For this
paper, these translate into:

- **Quality:** claims must be established by audited artifacts, not feature
  existence or mock figures. Budget matching, seeds, method-identity
  enforcement, a real programmatic-recipe surface, dependency-aware inference,
  and held-out isolation must be explicit.
- **Clarity:** define the repository, intervention, parent, node, budget,
  validation signal, and held-out certification contract before describing the
  agent implementation.
- **Significance:** explain why published model rankings are scientifically
  incomplete when methods receive unequal research effort and why an auditable
  substrate changes what can be measured.
- **Originality:** distinguish the system contract from AIDE, AIRA/AIRA2,
  MLAgentBench, AI Scientist-v2, RE-Bench, MLRC-Bench, and AMID. Do not claim
  code editing, tree search, budgets, hidden tests, or sandboxes as individually
  new.

E&D is the closest rhetorical fit when the evaluation itself is treated as the
scientific object. It explicitly welcomes evaluation tools, methodology,
auditing, stress testing, and informative null results. Main/General remains
plausible only if the framework-first claim is supported by convincing
cross-project, end-to-end, and systems evidence.

## Accepted Paper Patterns

Strong recent agent-system and benchmark papers repeatedly use the following
structure:

1. State one concrete failure of current measurement.
2. Define the task, artifact, evaluator, resource, and isolation contract.
3. Decompose the system into independently inspectable layers.
4. Organize experiments around research questions rather than software
   modules.
5. Measure endpoint quality, search process, reliability, and resource cost.
6. Include failure analysis in the main paper.
7. Connect every numbered contribution to a specific artifact or experiment.

The most transferable narrative for autoMIL is:

> ranking confound → auditable contract → controlled method-identity protocol →
> experiment design and resources → no-search versus matched search →
> leakage/statistical audit → limitations

This is stronger than a feature-led sequence centered on the dashboard,
scheduler, or CLI.

## Constraints for This Paper

1. The bundled `neurips_2026.tex` is still the untouched explanatory template;
   the manuscript source does not yet exist.
2. The authoritative hierarchy is C1 (framework) followed by C2 (controlled
   pathology-MIL ranking audit). C3 is deliberately unassigned. C4 and C5 are
   conditional candidates.
3. Current evidence does not establish the final C2 result. The corrected
   baseline campaign, matched autonomous search, seed analysis, and sealed-test
   analysis are incomplete.
4. The method-identity policy is settled, but its per-arm enforcement and
   executable programmatic-recipe channel remain open; baseline recipe
   provenance, GBM survival, single-seed robustness, and cross-cell dependence
   also remain active claim constraints.
5. Test sealing should be described as framework-mediated non-interference,
   not an OS-level security boundary.
6. Backend portability must be limited to the execution modes actually
   validated on real infrastructure.
7. Medical-data reporting must cover TCGA/CPTAC access and licenses,
   patient-level splits, privacy/ethics status, survival attrition, and task
   aliasing.
8. Compute reporting must separate per-run and campaign totals, including GPU,
   CPU, memory, wall-clock, LLM/API use, failed experiments, and excluded
   exploratory work.
9. A nine-page main paper should reserve space for the claim boundary,
   framework contract/threat model, matched-evaluation protocol, primary
   results, statistics/resources, and limitations. CLI inventories, full
   configurations, long traces, and backend details belong in the appendix.
