# Source Map

## Authority order

1. Executed experiment artifacts and current source code are authoritative for
   what actually ran and what the framework currently enforces.
2. `CODE-AUDIT-FIXES.md`, `CONTRIBUTIONS.md`, and the newest dated status blocks
   are authoritative for open defects, contribution maturity, and decisions.
3. `EXPERIMENT_GRID.md` is authoritative for experiment counts only where its
   newest cluster-state block is not contradicted by a later audited log.
4. `PLAN.md` is authoritative for the intended protocol, not proof that an
   intended experiment or mechanism has completed.
5. Mock figures are layout prototypes only and cannot support a result claim.
6. Official NeurIPS pages and the bundled 2026 template are authoritative for
   format and review expectations.
7. External papers must be verified through a primary paper, proceedings page,
   publisher page, or official repository before citation.

## Claim-to-source map

| Claim family | Primary sources | Secondary sources | Current status |
|---|---|---|---|
| C1: parent-addressed source overlays and isolated reconstruction | `src/automil/runner.py`, `src/automil/graph.py`, runner/graph tests | `CONTRIBUTIONS.md`, `PLAN.md` | Implemented; exact boundaries still need line-level evidence extraction |
| C1/C2 boundary: declared file permissions and consumer-owned method identity | `src/automil/registry/`, registry tests; planned consumer `RecipePolicy` seam | `PLAN.md` Sections 5-6, `CODE-AUDIT-FIXES.md` METHOD-1/ARCH/VAR-1/BOUND-1 | File-level protection is implemented. The scientific method-identity rule is settled; per-arm invariants, symmetric core protection, programmatic-recipe execution, and candidate classification remain unimplemented. |
| C1: validation-only selection and born-sealed held-out certification | `src/automil/schemas/_result.py`, `runtime_helpers.py`, `cli/certify.py`, born-sealed tests | `CONTRIBUTIONS.md`, `PLAN.md` | Implemented as a framework-mediated contract; not an OS security boundary |
| C1: matched launched-attempt accounting | `src/automil/cells/`, orchestrator ingestion, cap tests | `CONTRIBUTIONS.md`, `CODE-AUDIT-FIXES.md` | The cap increments at launch, so crashes, partials, and budget-killed runs count; usable completed/partial-result counts are secondary reported evidence |
| C1: local/SLURM/Ray portability | backend implementations and backend tests | `.planning/PROJECT.md` | Interface and test evidence exists; strongest real-cluster portability claim is blocked |
| C2: benchmark design and roster | dataset YAMLs, run configs, roster scripts, executed artifacts | `PLAN.md`, newest `EXPERIMENT_GRID.md` block | Static roster present; audited reruns and agentic campaign remain incomplete |
| C2: baseline quality screen | validation artifacts and screening script/output | `CODE-AUDIT-FIXES.md` SCREEN-1/2 | Interim validation-only diagnostic; not a headline result |
| C2: recipe bias, rank shift, and rank stability | corrected baselines, matched agentic campaign, sealed certification outputs | `CONTRIBUTIONS.md` C2 protocol | Not yet established; result wording must remain neutral |
| C4: trajectory corpus | trajectory archives, schema, redaction logs, release package | `CONTRIBUTIONS.md` C4 gate | Candidate only |
| C5: transferable research knowledge | preregistered gate manifests and sealed matched transfer results | `CONTRIBUTIONS.md` C5 gate | Candidate only |
| Venue and checklist compliance | official NeurIPS 2026 pages, `neurips_2026.sty`, `checklist.tex` | PaperSpine scene dossier | Verified scene constraints |

## Known contradictions and resolution rules

| Conflict | Resolution |
|---|---|
| `EXPERIMENT_GRID.md` contains both an older “TITAN unavailable” account and a newer block reporting all five TITAN cohorts. | Use the newest dated cluster-state block, then verify against on-disk artifacts before stating counts. |
| Baseline rerun counts appear as both `75/90` and `81/93`. | Do not trust either narrative count. Regenerate the canonical 165-cell roster, join it to stored result identities, and report only the resulting canonical count; the paper must exclude off-roster heads. |
| Many `config.json` files record a nominal recipe that differs from what ran. | Use `benchmarks/src/autobench/pipeline/provenance.py` and executable runner logic, not stale nominal fields. |
| Planning documents describe completed mechanisms more strongly than production evidence warrants. | Describe implementation and validation separately; narrow claims when only unit/integration tests exist. |
| Mock figures resemble final result figures. | Never cite, reproduce, or numerically summarize them as evidence. Rebuild every result figure from audited artifacts. |

## Drafting firewall

- No final cross-arm ranking, rank elasticity, agent lift, transfer, or
  generalization claim is permitted before corrected baselines and matched
  agentic certification are complete.
- Interim counts may appear only as implementation/status facts, clearly
  labeled as such.
- Every quantitative statement in the manuscript must point to a script,
  machine-readable artifact, or audited table.
- The abstract and conclusion must remain result-neutral until the final
  evidence matrix is populated.
