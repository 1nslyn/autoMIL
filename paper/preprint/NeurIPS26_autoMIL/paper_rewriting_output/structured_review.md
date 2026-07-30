# Structured Peer Review

- **Manuscript:** `final_paper/main.tex`
- **Review scene:** public NeurIPS-format preprint
- **Overall judgment:** coherent and unusually candid framework/protocol paper;
  not yet a competitive NeurIPS empirical submission because C2 is deliberately
  unfinished.

## Methods & Reproducibility Reviewer

| ID | Severity | Finding | Evidence | Revision command | Status |
|---|---|---|---|---|---|
| MET-1 | CRITICAL | The primary C2 comparison has no corrected native ranking, matched-search ranking, or sealed paired certification. The paper cannot support a ranking-change/stability conclusion in its current state. | Table 4; `evidence_bank.md` E21-E25 | Keep the manuscript explicitly result-neutral; complete the canonical provenance join, matched campaign, and one-time certification before submission. | OPEN; correctly disclosed |
| MET-2 | MAJOR | Baseline execution fidelity is not yet controlled: historical configs misstate recipes, and CLAM survival silently ignores the declared optimizer. | Validation Sec. 5.2; E16; OPT-1 audit entry | Repair and test the optimizer path, regenerate native baselines from runtime provenance, and release the reconciliation ledger. | OPEN |
| MET-3 | MAJOR | The method-identity policy is now explicit, but file-level locks do not yet enforce it and the guarded programmatic-recipe channel cannot run. Loss/flag changes can still collapse one arm into another. | Sec. 4.2; E23 | Implement per-arm defining-mechanism invariants, symmetric protected cores, the consumer-side `RecipePolicy` hook, and candidate classification before search. | PARTLY MITIGATED — policy settled, code open |
| MET-4 | MAJOR | Single-seed training and shared cohorts/patients preclude cell-independent significance claims. | Sec. 4.3 and 6; E24-E25 | Add repeated seeds and patient/cohort-aware paired uncertainty; do not pool 165 cells as independent replicates. | OPEN |
| MET-5 | MAJOR | C1 evidence is a selected 51-test gate; the full local workspace did not collect cleanly, and production backend equivalence is untested. | Table 3; Appendix B; E10, E26 | Keep the selected-test qualifier, repair the dependency/collection gate, and add real comparable Local/SLURM/Ray executions before a portability claim. | OPEN; wording mitigated |

### Methods score

- Method clarity: 4/5
- Assumptions and threat model: 4/5
- Current reproducibility: 2/5
- Experimental completeness: 1/5

## Contribution & Novelty Reviewer

| ID | Severity | Finding | Evidence | Revision command | Status |
|---|---|---|---|---|---|
| CON-1 | MAJOR | Individual components overlap AIDE, AIRA2, MLRC-Bench, and adjacent systems; without C2 results, the integrated measurement contribution may still look like systems assembly. | Intro; Table 1; `sota_gap_map.md` | Preserve the explicit non-novelty concessions and make the completed cross-method audit the principal evidence that the integration enables a new estimand. | PARTLY MITIGATED |
| CON-2 | CRITICAL | C2 is the significance engine of the paper but is currently a protocol contribution only. Stable or changed ranks could both matter, yet neither has been measured under the corrected contract. | Abstract, contribution bullet 2, Table 4 | Do not present this version as the final NeurIPS submission; promote C2 only after certified cross-arm rankings and explanatory lift/cost traces exist. | OPEN |
| CON-3 | MAJOR | Generality beyond pathology is supported only by code separation and one synthetic consumer. | Sec. 3.4 and 6; E11-E12 | Add a non-toy second consumer or retain the present mechanism-level wording. | OPEN; wording mitigated |
| CON-4 | MINOR | The value proposition could be confused with conventional HPO-aware benchmarking. | Sec. 2; citations 1-2 | Make the guarded executable train-only recipe surface concrete while preserving published-method identity; retain failure accounting and certification distinctions. | PARTLY MITIGATED — channel not implemented |

### Contribution score

- Claim clarity: 5/5
- Differentiation from closest work: 4/5
- Current evidence-to-claim fit: 4/5
- Submission-level significance today: 2/5

## Structure & Clarity Reviewer

| ID | Severity | Finding | Evidence | Revision command | Status |
|---|---|---|---|---|---|
| CLR-1 | MINOR | Figure 1 originally compressed six states into an unreadable horizontal chain. | Visual PDF audit | Use a three-layer serpentine flow and remove overlapping edge labels. | RESOLVED |
| CLR-2 | MINOR | The paper contains many explicit caveats; a reader could mistake evidence discipline for a lack of a positive contribution. | Abstract, Sec. 5-6 | Keep C1 established mechanisms and the C2 scientific payoff prominent before the blocker table. | RESOLVED |
| CLR-3 | MINOR | Long bibliography URLs create loose line spacing but no clipping. | PDF pages 8-10 | Prefer DOI/proceedings metadata and suppress redundant URLs in the camera-ready bibliography if desired. | OPTIONAL |
| CLR-4 | MAJOR | The public preprint author block has names and correspondence but no affiliations. | Page 1 | Add verified affiliations before public release; do not infer them from email domains. | OPEN |

### Clarity score

- Narrative: 5/5
- Contribution hierarchy: 5/5
- Figure/table legibility: 4/5
- Venue presentation: 4/5

## Editor Synthesis

All three views agree that the manuscript has a clear spine: C1 is the
framework, C2 is a cross-arm ranking audit, and within-method lift is
explanatory. They also agree that the paper is not ready to support C2 as an
empirical contribution. The strongest current qualities are precise scope,
closest-prior concessions, and unusually visible negative audit evidence.

Revision priority:

1. fix baseline and search-fidelity defects, including CLAM optimizer handling;
2. implement the settled method-identity invariants and programmatic-recipe
   seam, then regenerate the canonical native ranking;
3. execute matched search with failure/resource ledgers and repeated seeds;
4. certify once and analyze dependence-aware cross-arm rank response;
5. add affiliations, public artifact paths, asset licenses, and compute details.

**Recommendation today:** major revision as a conference submission; suitable
as a transparent protocol/system preprint if labelled exactly as the current
abstract and Table 4 label it.
