# Confirmed Controlling Motivation

Confirmed by Leo on 2026-07-30: **Option A — rankings confound method quality
with research effort.**

## Controlling statement

Published machine-learning leaderboards often compare methods after unequal
amounts of human recipe engineering. Coding agents make source-level research
scalable, but do not by themselves make that research effort attributable,
budgeted, reproducible, or safely separated from final evaluation. We introduce
autoMIL as an auditable research-operations substrate for applying matched,
declared source-level research opportunity to competing published methods in
existing repositories. We then use it to ask whether pathology-MIL rankings
remain stable under that controlled protocol.

## Contribution dependency

> autoMIL is the measurement instrument; the pathology-MIL ranking audit is the
> scientific use of that instrument.

- **C1 remains the first contribution:** the autoMIL framework and its
  research/execution/evidence contract.
- **C2 remains the second and main empirical contribution:** the controlled
  pathology-MIL audit, including the no-search cross-arm leaderboard, the
  matched-search cross-arm leaderboard, within-method lift, and rank
  stability or change.
- Within-arm lift is explanatory evidence for C2; it does not replace cross-arm
  benchmarking.

## Required precision

- “Matched research opportunity” means an equal declared cap on launched
  evaluation attempts. Crashed, partial, and budget-killed runs consume the
  cap; usable completed/partial-result counts, compute, time, failures, and LLM
  cost are reported separately.
- Equal attempt opportunity does not imply equal search-space difficulty,
  equal wall-clock compute, or equal successful research.
- C2 admits both scalar configuration and executable train-only recipe programs,
  but only inside a predeclared method-identity contract. Defining
  inference/forward/core-loss mechanisms are frozen per published method;
  identity-breaking descendants are excluded from its leaderboard.
- The search surface must therefore be demonstrably broader than a finite
  hyperparameter table: programmatic sampling, adaptive schedules,
  optimizer/gradient policy, additive regularization, and stopping are
  admissible without changing the method being attributed.
- The empirical conclusion remains result-neutral until corrected baselines,
  method-identity enforcement, the runnable programmatic-recipe channel, the
  matched campaign, seeds, and sealed certification are complete.
- A stable ranking is scientifically informative; the paper must not
  presuppose a ranking reversal.
- The manuscript is a public NeurIPS 2026-format preprint and must use the
  `preprint` style rather than claim an active or accepted NeurIPS submission.
