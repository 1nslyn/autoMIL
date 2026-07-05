# Phase 2 — Full journal submission plan (provisional)

_Background/shared context: [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Active preprint plan: [`../preprint/PLAN.md`](../preprint/PLAN.md). Compiled
2026-07-01._

**This doc is intentionally thin, and the full experiment plan is undecided.**
The team hasn't discussed Phase 2's experiment design yet — cohorts, model
roster, task types, grid size, and analysis are all open. Phase 1 (the
preprint) has the team's attention right now. What's below is context and
candidate framing carried over as starting points, **not decisions**.

## Context (not a committed experiment plan)

A couple of things carry over from Phase 1 and the shared background. They
bound the space Phase 2 *could* occupy — but the actual experiment design
remains undecided (see Open / pending):

- **Scope ceiling: the full cohort inventory.** All 16 TCGA + 10 CPTAC
  cohorts are already extracted on shared cluster storage (see shared
  background's Data scale section) — so the data ceiling exists — but which
  cohorts Phase 2 actually uses, and whether "all" means literally all 26 or
  some other cut, is undecided.
- **Core claim carries over.** The auto pipeline (autoMIL's agent-driven
  recipe search) stays the contribution in both phases — Phase 2 isn't a
  different paper, it's the same claim at larger scale. *How* that scale and
  its evaluation are designed is undecided.
- **Model roster: undecided.** Phase 1 is fixed at 4 models (clam_mb,
  simple_mil, ab_mil, dtfd_mil). Phase 2 would plausibly widen it — TransMIL,
  possibly DSMIL, completing the citation-ranking set the preprint scoped
  down — but nothing here is settled.

## Candidate framing carried forward from the old Notion proposal

The pre-pivot Notion doc ("AutoMIL Proposal," archived at
[`../references/automil-proposal-2026-04-29.md`](../references/automil-proposal-2026-04-29.md))
is not exactly right for where things stand now, but its
ambition level and rigor look like a closer match for Phase 2's larger scope
than for the fast preprint. Worth revisiting once Phase 1 ships, not before —
none of this is decided:

- **Recipe-bias-audit thesis**: quantify how much MIL architecture rankings
  are explained by training-recipe choice, using an equal-effort,
  architecture-preserving agentic recipe search per cell, frozen before final
  test evaluation.
- **90-cell grid**: 6 tasks × 3 encoders × 5 models (its proposed set:
  AB-MIL, DSMIL, TransMIL, CLAM-MB, DTFD-MIL — notably the same 4 candidates
  the citation ranking landed on, plus CLAM-MB).
- **Statistical rigor**: mixed-effects model
  (`performance ~ architecture + encoder + task + recipe + (1 | cohort)`),
  variance decomposition, CI-bounded ranking-flip rates — see its §6.5.
- **Reviewer-attack rebuttals** (its §10) — "this is just Optuna with
  Claude," "you overfit the test set," "per-cell recipes aren't a standard,"
  etc. — these arguments don't depend on the exact grid size and are likely
  reusable regardless of how Phase 2 is finally scoped.
- **Recipe-planner follow-on** (its §11) — a self-configuring, nnU-Net-style
  meta-model that predicts a strong starting recipe from task metadata,
  trained on the search traces from the full grid. Framed there as "the
  strongest discussion section or the second paper."

None of this should drive engineering work yet. It's here so the option
isn't lost, and so Phase 1's design choices (which models, which task types,
which comparison axes) don't accidentally foreclose it.

## Open / pending

**The entire experiment plan is undecided.** None of Phase 2's design —
cohorts, model roster, task types, grid size, statistical analysis, or scope
of "all" — has been discussed or agreed. Revisit after the preprint ships and
work out the plan then, the same way
[`../preprint/PLAN.md`](../preprint/PLAN.md) was built from the confirmed pivot.
