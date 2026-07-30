# Reviewer-Aware Audit

## 1. Reviewer Value Map

| Reviewer criterion | What reviewers/editors want | Our manuscript evidence | Current weakness | Revision action |
|---|---|---|---|---|
| Novelty | A precise residual relative to the closest systems, without claiming familiar components as inventions. | Introduction; Table 1; Sec. 6; verified SOTA gap map. | The residual is an integrated measurement use, so it becomes compelling only when C2 demonstrates the estimand. | Preserve the explicit prior-art concessions and complete the cross-method audit. |
| Significance | A question whose answer changes how readers interpret a real benchmark. | Native versus matched-search ranking estimand in Sec. 2 and Fig. 2. | The question is important but still unanswered in certified results. | Produce both cross-arm rankings, rank agreement/displacement, and explanatory lift/cost traces. |
| Technical soundness | Operational definitions, honest assumptions, and enforcement paths that match the claims. | Attempt semantics, worktree reconstruction, result splitting, threat model, 51 selected tests. | CLAM silently ignores a declared optimizer; method-identity enforcement, the programmatic-recipe seam, and backend equivalence remain unresolved. | Fix/test optimizer fidelity, implement the settled method-identity contract plus guarded `RecipePolicy`, and validate real backend executions. |
| Evidence sufficiency | Enough direct evidence to support both numbered contributions and reject obvious alternative explanations. | C1 focused tests; static roster audit; validation-only viability screen. | No corrected native ranking, matched campaign, multi-seed ledger, or sealed paired outcome. | Treat the current version as protocol preprint; do not submit the empirical claim before these artifacts exist. |
| Clarity | A contribution hierarchy and figures that a reviewer can follow in one pass. | Two explicit contribution bullets; two protocol figures; evidence-status table. | Affiliations are absent, and bibliography URLs are visually loose. | Add verified affiliations; optionally shorten bibliography URLs to DOI/proceedings records. |
| Venue fit | NeurIPS-relevant systems/evaluation framing, required checklist, page discipline, and reproducibility. | NeurIPS 2026 preprint style; 8 main-text pages; filled checklist; agent-evaluation and benchmark framing. | Several checklist answers are honestly “No”; as an empirical submission, the paper is incomplete. | Close the C2, artifact-release, compute, license, and statistical gates before submission. |

## 2. Reviewer Objection Register

| Likely objection | Where triggered | Severity | What the reviewer may say | Preemptive fix | Status |
|---|---|---|---|---|---|
| C2 has no result | Abstract; Contribution 2; Sec. 5.4 | CRITICAL | “The paper’s significance rests on a ranking audit that has not been run under the corrected protocol.” | Keep the preprint result-neutral; complete corrected native and matched-search certification before conference submission. | OPEN |
| Integrated system looks incremental | Intro; Table 1; Sec. 6 | MAJOR | “This is worktrees, a graph, budgets, and hidden evaluation assembled from existing systems.” | Retain the non-novelty concessions and demonstrate the new cross-method estimand with C2 evidence. | PARTLY MITIGATED |
| Baseline/search fidelity is broken | Sec. 5.2 | MAJOR | “Historical configs are unreliable, and a searchable CLAM optimizer is silently ignored.” | Fix OPT-1 with a regression test; reconstruct native recipes from runtime provenance and rerun affected canonical cells. | OPEN |
| Published-method boundaries are unenforced | Sec. 4.2; Sec. 6 | MAJOR | “The agent can turn one arm into another through flags or loss weights, while the advertised structured policy channel cannot run.” | Implement the preregistered method invariants, symmetric protected cores, guarded programmatic `RecipePolicy`, and four-way candidate classification before search. | PARTLY MITIGATED — policy settled, code open |
| Statistical precision is invalid | Sec. 4.3; Sec. 6 | MAJOR | “Single-seed results and 165 dependent cells cannot support the proposed comparison.” | Add repeated seeds and patient/cohort-aware paired uncertainty; prohibit cell-independent pooled tests. | OPEN |
| Reproducibility is incomplete | Appendix B; checklist | MAJOR | “The external cluster artifacts, full environment, compute ledger, and clean full test gate are unavailable.” | Release versioned artifacts and audit reader; repair collection; report compute, failures, runtime, and LLM cost. | OPEN |
| Generality is overstated | Sec. 3.4; Sec. 6 | MAJOR | “One synthetic consumer does not establish domain generality or backend portability.” | Keep mechanism-level wording or add a non-toy second consumer and real backend equivalence runs. | PARTLY MITIGATED |
| Author metadata is incomplete | Title page | MAJOR | “A public preprint should identify affiliations.” | Add affiliations only after author verification. | OPEN |
| Figures are too dense | Figs. 1-2 | MINOR | “The workflow is hard to parse at print scale.” | Three-layer Figure 1 and separated Figure 2 outputs were visually inspected after recompilation. | RESOLVED |

## 3. Editorial Fit Map

- **Venue fit:** The paper fits NeurIPS systems/evaluation interests through
  coding-agent research infrastructure, benchmark methodology, and
  computational pathology. It uses the official preprint style, remains within
  the nine-page main-text target, includes references, appendix, broader impact,
  and the official checklist. The empirical maturity, not formatting, is the
  current mismatch.
- **Editor-facing value:** The manuscript identifies a measurable confound in
  model leaderboards and supplies a repository-level operations/evidence
  contract for studying it. An editor can defend sending the completed version
  to review because the cross-method ranking response is a concrete,
  outcome-neutral evaluation question rather than another agent leaderboard.
- **Desk-reject risks:**
  - final C2 evidence absent — **OPEN and blocking for submission**;
  - several checklist reproducibility/compute/license answers are “No” —
    **OPEN**;
  - author affiliations absent — **OPEN**;
  - required checklist, preprint mode, page budget, bibliography, and PDF
    compilation — **RESOLVED**;
  - overlap with closest prior work hidden or overstated — **RESOLVED in text**;
  - fabricated result or mock figure — **RESOLVED; none used**.
