# Handoff — audit remediation branch

**Branch:** `fix/audit-2026-07-23` (21 commits on top of `0b2da55`)
**Date:** 2026-07-23 · **Status:** code fixes landed and green; several decisions pending

---

## TL;DR

A six-stream adversarial audit of the whole pipeline found 39 issues. **All 5
CRITICAL code defects are fixed, tested, and committed.** Along the way a
**new CRITICAL** surfaced that is not yet fixed (**H-3b**) and which currently
blocks the paper's headline. Nothing here is a blocker for *running* the static
grid; it is a blocker for the **agentic layer** and for two of the paper's
figures.

**Read these three docs before touching anything:**

| Doc | What it is |
|---|---|
| [`paper/preprint/CODE-AUDIT-2026-07-23.md`](paper/preprint/CODE-AUDIT-2026-07-23.md) | the full 39-finding audit + a **claim-level addendum** mapping findings onto the paper's 8 figures |
| [`paper/preprint/CODE-AUDIT-FIXES.md`](paper/preprint/CODE-AUDIT-FIXES.md) | **living tracker** — every finding with status ☐/◐/☑/⚑, commit, and notes |
| [`paper/preprint/PRELAUNCH_REVIEW.md`](paper/preprint/PRELAUNCH_REVIEW.md) | the earlier (pre-existing) review; O1/O2/O3 referenced throughout |

---

## What is fixed (all tested, all committed)

| ID | Fix | Commit |
|---|---|---|
| CR-1a | non-finite `composite` rejected at 3 layers (blocks an `Infinity` best-node exploit) | `bf7ef14` |
| CR-1b | **composite derived from val metrics**, not trusted from the agent-written file | `731cea2` |
| CR-2 | `propose`/`nominate`/`reconcile` now write `graph.json` under the lock | `2506489` |
| CR-3 | **survival selection uses pooled cross-fold concordance** (was a ~2-event/fold signal the code itself calls "near-random") | `ff7a55e` |
| CR-5 | per-fold results cache isolated per experiment (was defeating the OS fix + colliding across variants) | `e695304` |
| H-1 | `run.log` + error tails redacted (firewall no longer depends on a vendored hand-patch) | `73908f6` |
| H-3 | **uniform hyperparameter override across all 5 arms** (3 of 4 aggregators previously discarded tuning silently) | `b50873a` `7f65718` `101ab35` `91d741a` |
| H-6 | `best_node` recomputed from keep-nodes only | `e5d0d56` |
| H-8 | `n_valid_folds` recorded; degenerate CV quarantined as `partial` | `bb6a276` |
| M-1, M-4 | scoring-key backfill; parent/child cycle guards | `da9805a` |
| M-11, L-4 | ambiguous dataset name + blank env now fail fast | `2a09890` |
| M-14 | task included in budget-cell identity | `550c0c7` |

### Verification status — what "green" means here

```bash
uv run pytest benchmarks/tests/ -q          # 518 passed
uv run pytest tests/ -q --ignore=tests/skills --ignore=tests/acceptance \
  --deselect tests/test_cli_cancel_resubmit.py::test_cancel_local_direct_kill \
  --deselect tests/test_cli_cancel_resubmit.py::test_cancel_no_starttime_ticks \
  --deselect tests/test_framework_purity.py::test_framework_purity_no_autobench_refs
                                            # 1082 passed
```

The three deselected + two skipped dirs are **pre-existing failures, verified by
running them on base `0b2da55` in an isolated worktree**:
- 3 purity tests fail only on a stale `.pyc` (the grep scans `__pycache__`) —
  they pass after `find src -name __pycache__ -type d -exec rm -rf {} +`
- the other 9 (clause_08, clause_11, cancel ×2, iris end-to-end, phase7 setup
  gate, setup_dry_run_gate ×3) **fail identically on base** — macOS process-kill
  semantics, absent nvidia-smi, 90 s subprocess gates. This project is Linux-only.

**→ 0 regressions from this branch.**

---

## The blocker: H-3b (NEW CRITICAL, not fixed)

Search-space coverage is **CLAM-shaped**. Measured per arm:

| Arm | tunable knobs | reachable | coverage |
|---|--:|--:|--:|
| CLAM | 12 | 12 | **100%** |
| TITAN | 4 | 3 | 75% |
| ABMIL | 8 | 5 | 62% |
| nnMIL | 11 | 4 | **36%** |
| DTFD | 15 | 5 | **33%** |

Root cause: the transport is `ModelConfig` + `TrainConfig`, which were designed
around CLAM (`bag_weight`, `B`, `model_size` are CLAM concepts). CLAM's whole
surface is natively in the channel; nobody else's is. DTFD cannot receive
`numGroup`, `total_instance`, `mDim`, `numLayer_Res`, `droprate`, `droprate_2`,
`grad_clip`, `lr_decay_ratio`, `lr_decay_step` — i.e. **its own paper's
contributions**. nnMIL cannot receive `warmup_epochs`, `dropout`, `batch_size`,
`batch_sampler`, `hidden_dim`, `max_seq_length`.

**Why it blocks the paper (finding A-0):** `EXPERIMENT_GRID.md §4` defends the
headline (Fig 2, encoder ≫ aggregator) by pointing at Fig 3's equal-effort
search — *"Fig 2 + Fig 3 must be read together"*. H-3b breaks exactly that
defence. A ranking flip toward CLAM in Fig 3 is **the artifact H-3b predicts**,
not a finding. Headline and framework contribution fail together.

### The fix (designed, not implemented)

1. An **opaque key/value override channel** — e.g. `hparam_overrides: {numGroup: 8,
   grad_clip: 1.0}` in the spec/config — threaded to `apply_overrides` /
   `apply_overrides_to_plan`. **Those already accept arbitrary field names**
   (verified: `droprate` / `droprate_2` are individually settable); they just have
   nothing feeding them.
2. Widen `hparams.OVERRIDABLE` beyond the 5 canonical names so `exp_cfg` fields an
   arm recognises (`dropout`, `optimizer`, `weighted_sample`, `stop_epoch`) are
   forwarded too.
3. **Declare a per-arm searchable set** (finding A-2). The target is a *declared*
   set, **not** literal 100% — DTFD's `distill` is deliberately locked to `AFS`
   for a correctness reason documented in its config.

---

## Recommended order for the next session

1. **H-3b + A-2** — the override channel and the declared search space. Unblocks A-0.
2. **H-4** — enforce the frozen substrate (`registry.protected` ships empty; no
   roster overlay exists). Resolves A-4: today the agent can reach full coverage
   only by editing config files, which is the same hole that lets it edit
   `splits.py`.
3. **H-2** — eval-count budget primitive (`cells/` is time-only today). See the
   memory note `automil-equal-effort-budget`: this was settled on rigor grounds —
   eval-count is the comparison axis, agent-worktime is a reported secondary.
4. Then: CR-2b, M-5/6/7, M-8/9/10, and the Gate-2 statistics helpers
   (t₄/BCa intervals, Holm/BH correction, per-cell δ).

---

## Decisions pending — Leo only, do not guess

| # | Decision | Consequence |
|---|---|---|
| 1 | **CLAM back to upstream `lr=1e-4`?** (currently 2e-4, i.e. 2×, no recorded rationale) | invalidates the dispatched static grid → re-run. Also gates the Fig 7 "reproduces published SOTA" claim (A-7). |
| 2 | **ABMIL back to upstream optimizer?** (all three differ: lr 5e-4→2e-4, reg 1e-4→1e-5, epochs 20→200) | same. Note upstream ABMIL is a toy MNIST-bags experiment, so deviating is defensible **if stated**. |
| 3 | Fig 4: include TITAN in the agentic search, or drop the head-to-head framing (A-5) | as-is, tile arms improve after the search and TITAN does not — the claim can reverse for a non-model reason |
| 4 | Fig 3: re-run the CCRCC / ovarian-HRD anchors — they are off-roster **and** predate every fix here (A-1) | compute + coordination |
| 5 | Seeds: multi-seed, or stop reporting variance components (A-8 / H-5c) | Fig 2's decomposition is single-seed and cannot separate seed noise |
| 6 | A-3: equal eval-count does not equalise search *difficulty* (DTFD 15-dim vs TITAN 4-dim) | needs a stated position: scale budget with dimensionality, or report anytime curves only |

**Cost note for 1 & 2:** re-running the static grid costs only *training*.
Features, `.pt` conversions, splits and the TITAN `conch_v15` extraction all
survive — ≈40 GPU-h total, ~½–1 day on 4×H100 per the grid doc. It is much
cheaper now than after the agentic layer, whose headline metric is
lift-over-baseline.

**Verified upstream comparison** (checked field-by-field against every package
under `benchmarks/lib/`, see `provenance.py`): DTFD **faithful**; nnMIL
**faithful** (3e-4/1e-4 + 100 epochs are nnMIL's own trainer defaults, not a
benchmark deviation); CLAM off by one knob; ABMIL off by three; TITAN n/a.
SMMILe is vendored but unreachable from `--framework`, so it is off the paper path.

---

## Conventions used here

- One finding (or one tight group) per commit; conventional-commit subject with
  the finding ID; body explains the failure mode, not just the change.
- Every fix ships a test that **reproduces the defect**, plus a targeted
  regression run. Nothing marked ☑ in the tracker without both.
- Update `CODE-AUDIT-FIXES.md` in the same commit as the fix.
- No co-author trailer (disabled globally in `~/.claude/settings.json`).

## Gotchas worth knowing

- `tasks/` is **gitignored** — audit docs live in `paper/preprint/` so they travel
  with the branch.
- The purity test greps `__pycache__`; clear bytecode before trusting it.
- `uv run pytest tests/` takes ~18 min; the deselect list above runs in ~90 s.
- `benchmarks/lib/` holds 6 vendored upstreams (CLAM, DTFD-MIL,
  AttentionDeepMIL, nnMIL, SMMILe, TRIDENT) — they are the ground truth for any
  "does this match upstream?" question.
- `hparams.apply_overrides` **raises** on an inapplicable knob by design. When
  wiring a new arm, route only the knobs that arm actually owns — this is how the
  TITAN `max_epochs` mixed-provenance case was caught during wiring.
