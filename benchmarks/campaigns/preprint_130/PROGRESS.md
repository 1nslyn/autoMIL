# Preprint 130-cell campaign — overall progress

**Per-cell detail lives in the Google Sheet, not here.**

> **[autoMIL — 130-cell campaign tracker](https://docs.google.com/spreadsheets/d/1e79rsWlc8BOZoi6xFRWQvyi9M1uDuCexlQCsOTE5gOU/edit)**
> One row per cell, 130 rows, filterable by `Owner`. That is where you update
> your own cells as you run them.

This file is the **overall** view: the gates that block everything, the rolling
totals, and the weekly snapshot. Keep it short. If you find yourself adding a
per-cell row here, it belongs in the Sheet.

- Runbook: [`docs/tutorials/run_agentic_campaign.md`](../../../docs/tutorials/run_agentic_campaign.md)
- Protocol authority: [`README.md`](README.md)
- Frozen roster: [`manifest.json`](manifest.json) — 130 cells, lock in `manifest.json.sha256`

## Stage codes (used in the Sheet)

A cell moves left to right; the one legal skip is a zero-eligible-candidate
freeze, which jumps from `F` past `P` (baseline wins by default):

| Code | Stage | Done when |
|---|---|---|
| `.` | not started | — |
| `M` | materialized | the cell has its own isolated `automil/` root |
| `B` | baseline done | native-recipe five-fold result archived |
| `D` | discovery running | agent session open, ≤30 attempts charged, ≤12h agent-active |
| `F` | discovery frozen | 30 attempts charged, top-10 frozen |
| `P` | promotion done | frozen top-10 evaluated on folds 3/4 |
| `W` | winner selected | one winner frozen on five-fold validation mean |
| `C` | certified | held-out unsealed once, winner paired with its native baseline |
| `X` | blocked | put the reason in `Blocker / notes`; includes 12h exhaustion below 30 attempts (freeze fails closed) |

> **No test metric goes in the Sheet, ever.** Certification writes results into
> the campaign archive. If you are about to paste a test number into a
> spreadsheet cell, stop — that is the val-firewall you are stepping over.

---

## Gates — nothing downstream starts until these are green

### Gate 0 — baseline regeneration (owner: Leo)

| Item | Status | Updated | Notes |
|---|---|---|---|
| 70 reusable cells migrated + SHA-256 verified | ✅ done | 2026-08-06 | nnMIL 30, DTFD 30, TITAN 10 |
| 60 CLAM/ABMIL reruns | 🔄 queued | 2026-08-06 | `fir` job `53319143`, 4×H100, submitted Aug 5 22:47 |
| 130/130 exact manifest coverage audit | ⬜ pending | — | blocked on the 60 reruns |

### Gate 1 — canary, all 10 arm/task regimes (owner: Leo)

One real-GPU end-to-end cell per regime. **No formal cell launches until all ten
pass.** Check archive, firewall, budget counts, timing and reproducibility each time.

| Arm | Classification | Survival |
|---|:--:|:--:|
| clam | ⬜ | ⬜ |
| nnmil | ⬜ | ⬜ |
| abmil | ⬜ | ⬜ |
| dtfd | ⬜ | ⬜ |
| titan | ⬜ | ⬜ |

### Gate 2 — launch prerequisites (owner: Leo)

| Item | Status | Notes |
|---|---|---|
| `agent_protocol.json` generated + hash-verified | ⬜ | still a template; blocks materialization |
| Per-cell agent launcher (one fresh session, locked tool surface) | ⬜ | |
| Allocation request (~8,000 GPU-h) | ⬜ | re-derive from canary timings first |

---

## Rolling totals

`130` is the only acceptable final number in every row — the analysis plan
fails closed on an incomplete campaign.

| Stage | Done | Target |
|---|--:|--:|
| Materialized | 0 | 130 |
| Baseline archived | 0 | 130 |
| Discovery frozen | 0 | 130 |
| Promotion done | 0 | 130 |
| Winner selected | 0 | 130 |
| Certified | 0 | 130 |

## By owner

26 cells each. Counts come from the Sheet — filter by `Owner` and count `Stage`.

| Owner | Dataset | Done | Running | Blocked | Not started |
|---|---|--:|--:|--:|--:|
| Leo | tcga_luad | 0 | 0 | 0 | 26 |
| Yeonwoo | tcga_lgg | 0 | 0 | 0 | 26 |
| Ryan | tcga_hnsc | 0 | 0 | 0 | 26 |
| Keishi | cptac_gbm | 0 | 0 | 0 | 26 |
| Terry | cptac_pdac | 0 | 0 | 0 | 26 |

## Weekly snapshot

One line per week. Keeps a history the Sheet does not.

| Week of | Certified | Note |
|---|--:|---|
| 2026-08-06 | 0 | Campaign not launched. Baseline reruns queued; canary and agent protocol outstanding. |

## Open issues

| Issue | Owner | Status |
|---|---|---|
| Authorship: 3 authors listed, 5 running cohorts; `CITATION.cff` and the manuscript disagree on corresponding author | all | open — blocks public release (P-AUTH-1) |
| Campaign Sheet sharing must match the repository's intended visibility | Leo | open — verify before public release (P-SHARE-1) |
