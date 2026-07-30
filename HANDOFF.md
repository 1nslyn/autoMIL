# Handoff — audit remediation branch

**Branch:** `fix/audit-2026-07-23` (40 commits on top of `0b2da55`)
**Updated:** 2026-07-29 · **Status:** scope settled; the agentic layer's blockers are cleared

---

## TL;DR

A six-stream adversarial audit (2026-07-23) found 39 issues; a second full pass
against the paper plan (2026-07-28) found 14 more, corrected two of the first
pass's own conclusions, and forced a scope decision. **Everything on the critical
path is now fixed and green.** What remains is a short list of research decisions
and a mid-priority backlog, both enumerated below.

**Read these four docs before touching anything:**

| Doc | What it is |
|---|---|
| [`paper/preprint/CONTRIBUTIONS.md`](paper/preprint/CONTRIBUTIONS.md) | **authoritative contribution hierarchy and claim maturity — read first** |
| [`paper/preprint/READINESS-2026-07-28.md`](paper/preprint/READINESS-2026-07-28.md) | the second-pass analysis — **start here**; it opens with the settled scope |
| [`paper/preprint/CODE-AUDIT-FIXES.md`](paper/preprint/CODE-AUDIT-FIXES.md) | **living tracker** — every finding with status ☐/◐/☑/⚑/⊘, commit, notes |
| [`paper/preprint/CODE-AUDIT-2026-07-23.md`](paper/preprint/CODE-AUDIT-2026-07-23.md) | the original 39-finding audit + the claim-level addendum |
| [`paper/preprint/PRELAUNCH_REVIEW.md`](paper/preprint/PRELAUNCH_REVIEW.md) | the pre-existing review; O1/O2/O3 referenced throughout |

---

## The scope decision (settled by Leo, 2026-07-28)

This is the frame everything else hangs from, so it goes first.

**Contribution hierarchy (reframed after the 2026-07-30 prior-art audit).** C1
is the autoMIL auditable research-operations framework, including its
matched-evaluation and sealed-certification contract. C2 is the result-neutral
pathology-MIL ranking audit produced by the agentic campaign. Fig. 3 is the
planned main empirical result, with Fig. 5 and the pre-registered gate as its
rigor backbone; Figs. 1 and 4 are the benchmark substrate. C3 is deliberately
unassigned; the historically reserved C4/C5 remain candidates. See
[`CONTRIBUTIONS.md`](paper/preprint/CONTRIBUTIONS.md).

**The "encoder ≫ aggregator" claim is dropped.** It came from the Frontiers
precedent rather than from us, it is contradicted by our own 210-config baseline,
and neither axis has a designed dynamic range. **No new experiments** are added to
rescue it — no legacy encoder anchor. Fig 2 survives only as a descriptive
variance decomposition with no "≫" assertion, or not at all.

This **voids** four findings rather than resolving them: A-0, O1, AXIS-RANGE,
M-12. It **promotes** four others from optional to required — CR-4, H-5a, H-5b and
multi-seed — because the empirical analysis includes per-cell lift selected on
~10 validation patients alongside cross-lineage reranking.

---

## What is fixed

Grouped by what it protects. Every row is code + test + commit.

**The selection signal.**

| ID | Fix | Commit |
|---|---|---|
| CR-1a | non-finite `composite` rejected at 3 layers (blocks an `Infinity` best-node exploit) | `bf7ef14` |
| CR-1b | composite **derived from val metrics**, not trusted from the agent-written file | `731cea2` |
| CR-3 | survival selection uses **pooled cross-fold concordance** (was a ~2-event/fold signal the code itself calls "near-random") | `ff7a55e` |
| H-6 | `best_node` recomputed from keep-nodes only | `e5d0d56` |
| H-8 | `n_valid_folds` recorded; degenerate CV quarantined as `partial` | `bb6a276` |
| M-15 | valid-fold denominators surfaced rather than silently averaged over | `e9b19d7` |

**The val-firewall.**

| ID | Fix | Commit |
|---|---|---|
| H-1 | `run.log` + error tails redacted (no longer depends on a vendored hand-patch) | `73908f6` |
| M-8 | `certify --top-k` K>1 requires `--unseal-multiple`; comparing K test blocks *is* selection on test | `aaa2c3e` |

**The results cache — two silent-wrong-number bugs.**

| ID | Fix | Commit |
|---|---|---|
| CR-5 | per-fold cache isolated per experiment on the orchestrated path | `e695304` |
| CR-5b | **the static grid too**: seed is now a path segment and the rest of the config is fingerprinted; `--seed 43` used to return seed 42's numbers verbatim | `75e7e78` |
| CFG-3 | `run_benchmark.py` declared a CLI literal for every training knob and passed them in unconditionally, so the dataclass defaults were **dead on the static-grid path** | `9be9ef9` |

**The search space — the equal-effort precondition.**

| ID | Fix | Commit |
|---|---|---|
| H-3 | uniform hyperparameter override across all 5 arms (3 of 4 aggregators silently discarded tuning) | `b50873a` `7f65718` `101ab35` `91d741a` |
| H-3b + A-2 | **declared per-arm search space** + an opaque `hparam_overrides` channel; coverage was CLAM 12/15 vs **nnMIL 0/11** | `46725e7` |
| — | CLAM and ABMIL returned to their **upstream** defaults | `d86bdce` |

**The frozen substrate.**

| ID | Fix | Commit |
|---|---|---|
| H-4 + H-4b | `registry.protected` populated on the **live** project (`0b2da55` patched only the three templates); `architecture-preserving` now requires a non-empty list | `be01096` |
| HASH-0 | the overlay manifest sha256 is verified before any file lands | `be01096` |
| NO-OVERLAY | agentic overlays created for all five roster cohorts | `be01096` |
| — | end-to-end proof that H-3b and H-4 **compose** | `4ad90ad` |

**Equal effort, measurable.**

| ID | Fix | Commit |
|---|---|---|
| H-2 | **eval-count budget** as an orthogonal second axis (time stays as the safety wall) | `8642fdd` |
| CELL-1 | `cell_id` reaches `graph.py`, so per-cell effort is answerable at all | `8642fdd` |
| CAP-1 | the launch path consults the cap; queued specs no longer slip through a closed cell | `8642fdd` |

**Results → claims.**

| ID | Fix | Commit |
|---|---|---|
| FIG-0 | a real-data cross-cohort collector + Figs 1/4 that read it (nothing in git read real results before) | `0838194` |
| DATA-ID | dataset identity recorded in every results artifact | `be01096` |
| TSV-1 | `results.tsv` no longer locks its metric columns on the first row | `e0cd416` |

**Statistics.**

| ID | Fix | Commit |
|---|---|---|
| H-5a | Student-t cross-fold intervals (the K=5 percentile bootstrap resamples five numbers) | `e9b19d7` |
| H-5b | Holm–Bonferroni and Benjamini–Hochberg adjustment for the grid-wide family | `a9890c0` |

**Correctness and hygiene.** CR-2 `2506489` · M-1/M-4 `da9805a` · M-11/L-4
`2a09890` · M-14 `550c0c7` · CFG-1/CFG-2 `e0cd416` · purity scan `b93f3e2`.

---

## Verification — what "green" means

```bash
uv run pytest benchmarks/tests/ -q
```
→ **738 passed, 1 skipped**

```bash
uv run pytest tests/ -q --ignore=tests/skills --ignore=tests/acceptance --deselect tests/test_cli_cancel_resubmit.py::test_cancel_local_direct_kill --deselect tests/test_cli_cancel_resubmit.py::test_cancel_no_starttime_ticks
```
→ **1234 passed, 54 skipped**

The deselect list is now **two tests**, down from three. The framework-purity test
used to be deselected as "pre-existing"; it turned out to be structurally unable
to pass — it grepped `__pycache__`, and a `.pyc` embeds its module's string
literals, so every *allowlisted* source line reappeared as an unmatchable
`Binary file … matches` row. pytest compiles the modules during collection, so it
failed on every run. Fixed in `b93f3e2`; a deselected purity test was also not
catching the real violations it exists for.

The two remaining deselects and `tests/skills` (4 failures) are genuinely
platform-specific and **verified failing identically on base `0b2da55`**: macOS
process-kill semantics, absent `nvidia-smi`, and 90-second subprocess gates. This
project is Linux-only.

**→ 0 regressions from this branch.**

---

## Open work

**Two agents were mid-flight when this was written** — CR-4 (per-cell keep margin
derived from CV noise) and the daemon robustness group (CR-2b, H-7, M-5, M-6,
M-7). Check `git log` before assuming either is unlanded.

Remaining after those: M-9, M-10, M-13, L-1, L-2, L-3, L-5, L-6, L-7, L-8, L-9,
L-10. All MED/LOW; none blocks a claim. See the tracker.

---

## Decisions still open — Leo only

| # | Decision | Consequence |
|---|---|---|
| 3 | Fig 4: include TITAN in the agentic search, or drop the head-to-head framing (A-5) | as-is, tile arms improve after the search and TITAN does not, so the claim can reverse for a non-model reason. TITAN has ~5 declared knobs, so "equal effort" is ill-defined for it |
| 4 | Fig 3's anchors: re-run CCRCC / ovarian-HRD, or replace them with roster cells now that the overlays exist (A-1) | they are off-roster **and** predate every fix here |
| 5 | Seeds: how many, and does Fig 2 keep a variance decomposition at all (A-8 / H-5c) | **unblocked** — CR-5b means a second seed now actually retrains. Promoted to required by the scope decision |
| 6 | A-3: equal eval-count ≠ equal search difficulty (DTFD 13 declared knobs vs TITAN 5) | needs a stated position: scale budget with dimensionality, or report anytime curves only |
| 10 | Who writes the remaining figures, and does the system schematic (intended paper Fig 1) get drawn in-repo | Figs 1 and 4 exist; 2, 3, 5, 6, 7, 8 do not |

**Operational consequence of the landed work:** the dispatched static grid must be
**purged and re-run**. CLAM and ABMIL train at different hyperparameters now, and
CR-5b's fingerprint guard will refuse to resume the old folds — it raises naming
the changed field and prints the `rm -rf`. That is the intended behaviour: a
re-run fails loudly rather than quietly reporting stale numbers. Cost is training
only (~40 GPU-h); features, `.pt` conversions, splits and any TITAN extraction all
survive.

**Also note:** a newly initialised project now gets `max_concurrent_per_gpu` and
`default_vram_estimate_gb` from the hardware healthcheck rather than the static
1 / 0.5 (CFG-2). `init` always computed those values; it wrote them into a block
nothing read.

---

## Not verified

**Cluster state is unknown.** SSH to `fir` failed at authentication (control
master expired; both agent keys rejected, only `keyboard-interactive` offered — an
MFA prompt that cannot be answered non-interactively). Queue state, the cluster's
checkout commit, per-dataset completion counts and `sacct` GPU-hours have not been
checked since this branch began. Rebuild the master from an interactive terminal:

```bash
ssh -fN -M -S ~/.ssh/cm/fir-master.sock -o ControlPersist=12h -o ServerAliveInterval=60 yinshuol@login3.fir.alliancecan.ca
```

Also still unchecked: whether CPTAC-PDAC's underlying continuous infiltration
score is recoverable (`PLAN.md` §4 open item).

---

## Conventions

- One finding (or one tight group) per commit; conventional-commit subject with
  the finding ID; the body explains the failure mode, not just the change.
- Every fix ships a test that **reproduces the defect**. Nothing is marked ☑
  without both the test and a targeted regression run.
- Update `CODE-AUDIT-FIXES.md` in the same commit as the fix.
- No co-author trailer (disabled globally in `~/.claude/settings.json`).

## Gotchas

- `tasks/` is **gitignored** — audit docs live in `paper/preprint/` so they travel
  with the branch. The baseline-report correction (BASE-DOC) is therefore local
  only; the finding itself is recorded in the tracker.
- `benchmarks/lib/` holds 6 vendored upstreams (CLAM, DTFD-MIL, AttentionDeepMIL,
  nnMIL, SMMILe, TRIDENT). They are ground truth for any "does this match
  upstream?" question — and `test_provenance.py` now parses their argparse
  defaults with AST rather than trusting a transcription.
- `hparams.apply_overrides` **raises** on a knob outside the arm's declared search
  space, with the lock's reason when there is one. When wiring a new arm, declare
  its knobs in `search_space.py` first; the honesty test fails if the declaration
  does not match the arm's real fields.
- `run_experiment.py` and `pipeline/config.py` are **protected** now. That is
  affordable only because H-3b gave every arm a real hyperparameter channel — the
  two fixes are load-bearing for each other (`4ad90ad` proves it).
- Full `tests/` run ≈ 95 s; `benchmarks/tests/` ≈ 30 s; `tests/skills/` alone is
  ≈ 10 min of subprocess gates.
