# Phase 2 — Full journal submission plan (provisional)

_Background/shared context: [`../shared/BACKGROUND.md`](../shared/BACKGROUND.md).
Active preprint plan: [`../preprint/PLAN.md`](../preprint/PLAN.md). Compiled
2026-07-01._

**This doc is intentionally thin.** The formal journal is meant to cover all
datasets — but the specifics of "all" haven't been worked out yet, because the
team's energy is on Phase 1 (the preprint) right now. What's here is the known
scope ceiling plus the most plausible candidate framing, not a committed plan.

## What's confirmed

- **Scope ceiling: the full cohort inventory.** All 16 TCGA + 10 CPTAC
  cohorts already extracted on shared cluster storage (see shared
  background's Data scale section) — vs. the 5 datasets Phase 1 uses. Whether
  "all 26" literally means all of them, or "all" relative to some other
  cut, is undecided.
- **Core claim carries over.** The auto pipeline (autoMIL's agent-driven
  recipe search) is the contribution in both phases — Phase 2 isn't a
  different paper, it's the same claim at full scale with a fuller
  evaluation.
- **Model roster likely grows too**, though not yet settled the
  way the dataset/model scope was for Phase 1. If Phase 1 lands on 4 or 5
  models (clam_mb, simple_mil, ab_mil, dtfd_mil, possibly ds_mil), Phase 2 is
  the natural place to also add TransMIL — completing all four models from
  the citation-ranking PDF, not just the preprint-scoped subset.

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

Everything, realistically — this phase hasn't been settled yet. Revisit after
the preprint ships and update this doc then, the same way
[`../preprint/PLAN.md`](../preprint/PLAN.md) was built from the confirmed pivot.
