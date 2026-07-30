# Style Profile

## Intended paper identity

- **Format:** public NeurIPS 2026-style preprint.
- **Primary identity:** framework paper whose scientific value is demonstrated
  by a controlled evaluation audit.
- **Contribution order:** C1 autoMIL framework; C2 pathology-MIL
  equal-research-effort ranking audit.
- **Tone:** precise, evidence-led, modest about novelty, explicit about failure
  modes and incomplete evidence.

## Narrative spine

1. Published model rankings combine method quality with unequal research and
   recipe effort.
2. Coding agents make source-level research scalable, but ordinary agent loops
   do not themselves make cross-method comparisons attributable, budgeted, or
   leakage controlled.
3. autoMIL supplies the operations and evidence contract: parent-addressed
   overlays, isolated reconstruction, declared edit surfaces, budget
   accounting, validation-only selection, and separate certification.
4. Autobench instantiates that contract to audit pathology-MIL rankings under
   native/no-search and matched-search regimes.
5. The paper reports the answer only after baseline, lineage, seed, dependence,
   and sealing requirements are satisfied, including an honest null result.

## Section behavior

- **Abstract:** one problem, one system contribution, one empirical question,
  one evidence boundary. No subsystem inventory.
- **Introduction:** explain the ranking confound before naming autoMIL; state C1
  and C2 with their dependency.
- **Related work:** compare capability axes and explicitly state what is not
  novel.
- **Problem setting:** define repository, lineage, node, intervention,
  evaluation attempt, completed evaluation, validation signal, and sealed
  certification.
- **System:** use the three-layer research/execution/evidence decomposition and
  one lifecycle figure.
- **Experiments:** organize around reliability of the contract, within-method
  lift and cross-method rankings, and costs/threats.
- **Results:** distinguish no-search leaderboard, matched-search leaderboard,
  within-method lift, rank shift/stability, anytime behavior, failures, and
  resource accounting.
- **Limitations:** state seed, cohort dependence, method-identity enforcement and programmatic-recipe channel, backend,
  threat-model, data, and generality boundaries directly.

## Claim discipline

- Use “we introduce,” “records,” “reconstructs,” “enforces through the
  framework,” “we evaluate whether,” and “under the declared protocol.”
- Reserve “we find” or “we show” for audited outputs.
- Do not use “first,” “fully autonomous,” “domain-agnostic,” “universal,”
  “robust,” “secure,” or “fair” without a local definition and evidence.
- Do not call equal attempts equal compute, equal difficulty, or equal
  successful research.
- Do not equate test sealing with protection against an adversarial same-user
  process.
- Never present the 165 cells as independent samples or pool classification and
  survival metrics as directly exchangeable.
- Keep TITAN visibly separate if it does not receive the same search regime.

## Visual grammar

- Figure 1: end-to-end lifecycle and the three layers, including the
  validation/certification boundary.
- Table 1: closest-prior capability comparison with precise definitions.
- Table 2: benchmark cohorts, tasks, arms, encoders, roster, and regimes.
- Main result: paired no-search versus matched-search cross-arm rankings, with
  within-arm lift and uncertainty adjacent.
- Secondary result: rank stability/shift versus declared budget, only if
  preregistered checkpoints and certification permit it.
- Analysis: dependency-aware cohort summaries, failure taxonomy, resource
  ledger, and one representative tree/trajectory.
- All figures must be regenerated from audited machine-readable artifacts.

## Language and formatting

- English, compact NeurIPS prose, one principal claim per sentence.
- Introduce domain terms for a general ML reviewer; avoid pathology-specific
  abbreviations before definition.
- Put numbers, conditions, and comparison objects in the same sentence.
- Use consistent terms: lineage, arm, cell, experiment, node, parent,
  candidate, validation, held-out, certification.
- Prefer short paragraphs and descriptive section headings over marketing
  headings.
- Main paper targets nine content pages; references, appendix, and checklist
  follow.
