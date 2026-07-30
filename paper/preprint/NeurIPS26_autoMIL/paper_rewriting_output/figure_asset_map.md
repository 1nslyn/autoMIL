# Figure and Table Asset Map

| Asset ID | Planned manuscript asset | Source / generation path | Status | Evidence rule |
|---|---|---|---|---|
| F1 | autoMIL lifecycle: research, execution, and evidence layers with the validation/certification boundary | New TikZ figure in `final_paper/main.tex`, derived from E01-E09 | ready to create | Mechanism-only; every arrow maps to code |
| F2 | Controlled ranking-audit design: native/no-search versus matched-search, cross-arm ranking plus within-method lift | New TikZ figure in `final_paper/main.tex`, derived from confirmed motivation and CL10 | ready to create | Protocol figure; must not imply an observed effect |
| T1 | Closest-prior capability and estimand comparison | Hand-built LaTeX table from citation bank C001-C030 | ready to create | Avoid binary checkmarks unless axis definitions are explicit |
| T2 | Five-cohort pathology roster | Hand-built LaTeX table from E20 | provisional | Label counts as audited/planned and note manifest regeneration requirement |
| T3 | Current framework validation | Hand-built LaTeX table from E10 and mechanism anchors | ready to create | State “selected tests,” never “full suite” |
| T4 | Campaign evidence status and blockers | Hand-built LaTeX table from E13-E25 | ready to create | Separates reported static evidence from missing final outcomes |
| F3 | Native/no-search versus matched-search cross-arm leaderboard | `paper/preprint/figures/make_figures.py` after canonical artifacts exist | blocked | Do not use `figures/mock/fig1_leaderboard_heatmap.png` |
| F4 | Within-lineage lift and rank stability versus budget | audited result reader after matched campaign | blocked | Requires preregistered checkpoints, seeds, and one-time certification |
| F5 | Anytime curves, failure taxonomy, and resource ledger | trajectory/result artifacts after campaign | blocked | Must include attempts, usable results, failures, wall-clock, GPU, and LLM cost |
| F6 | Transfer/generalization gate | gate artifacts after preregistered Stage-B evaluation | blocked candidate | C5 only if its promotion gate is met |
| X1 | Historical mock result figures | `paper/preprint/figures/mock/` | prohibited | Layout reference only; no pixel or number may enter the manuscript |
