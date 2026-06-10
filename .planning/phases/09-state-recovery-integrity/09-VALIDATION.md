---
phase: 9
slug: state-recovery-integrity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-10
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 09-RESEARCH.md §Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing, 48+ tests) |
| **Config file** | `pyproject.toml` (pytest section) |
| **Quick run command** | `uv run pytest tests/test_result_schema_validation.py tests/test_submit_cell_identity.py tests/test_terminal_writer.py -v` |
| **Full suite command** | `uv run pytest tests/ -v` |
| **Estimated runtime** | ~quick <30s · full ~2-3 min |

---

## Sampling Rate

- **After every task commit:** Run the quick command above
- **After every plan wave:** Run the full suite command
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds (quick command)

---

## Per-Requirement Verification Map

Task IDs are assigned at planning; this draft maps each REC requirement to its
test(s) so the planner can attach `<automated>` verify blocks. "Wave 0" = test
file does not exist yet and must be created before/with the implementing task.

| Requirement | Behavior | Test Type | Automated Command | File Exists |
|-------------|----------|-----------|-------------------|-------------|
| REC-01 | SIGTERM flush writes to `AUTOMIL_RESULTS_DIR`, not cwd | unit | `uv run pytest tests/test_sigterm_flush.py -v` | ❌ W0 |
| REC-01 | Fold aggregation tried before synthesis in `_collect_or_synthesize_result` | unit | `uv run pytest tests/test_collect_or_synthesize.py -v` | ❌ W0 |
| REC-01 | Main-PID-first kill: SIGTERM to pid, then SIGKILL to pgid after grace | unit (mock os.kill) | `uv run pytest tests/test_handle_timeout.py -v` | ❌ W0 |
| REC-01 | Kill with N completed folds → composite = mean of N folds, not 0.0 | integration | `uv run pytest tests/test_partial_fold_recovery.py -v` | ❌ W0 |
| REC-01 | Partial result is `status=partial`, excluded from keep/discard & best_node | unit | `uv run pytest tests/test_partial_fold_recovery.py -v` | ❌ W0 |
| REC-02 | Normal completion writes all four artifacts via terminal_writer | unit | `uv run pytest tests/test_terminal_writer.py -v` | ❌ W0 |
| REC-02 | Cap-kill completion writes all four artifacts | unit | `uv run pytest tests/test_terminal_writer.py::test_cap_kill_writes_all_four -v` | ❌ W0 |
| REC-02 | `automil rank` and `results.tsv` agree after completion | integration | `uv run pytest tests/test_terminal_writer_consistency.py -v` | ❌ W0 |
| REC-02 | `reconcile --from-archive` refreshes existing node composite | unit | `uv run pytest tests/test_reconcile_from_archive.py -v` | ❌ W0 |
| REC-03 | `partial` status validates against updated schema | unit | `uv run pytest tests/test_result_schema_validation.py -v` | ✅ extend |
| REC-03 | `termination_reason` field validates | unit | `uv run pytest tests/test_result_schema_validation.py -v` | ✅ extend |
| REC-03 | `crashed` canonicalized to `crash` in `_crashed_payload` | unit | `uv run pytest tests/test_aggregate_folds.py -v` | ❌ W0 |
| REC-03 | `oom`/`timeout` synthesis produces canonical status + termination_reason | unit | `uv run pytest tests/test_collect_or_synthesize.py -v` | ❌ W0 |
| REC-04 | `make_cell_id(dataset, encoder, mil_model)` deterministic | unit | `uv run pytest tests/test_submit_cell_identity.py -v` | ✅ extend |
| REC-04 | Re-parenting joins same cell (not a fresh budget) | unit | `uv run pytest tests/test_submit_cell_identity.py::test_reparent_joins_same_cell -v` | ❌ W0 |
| REC-04 | `--mil-model` missing with no config fallback → ClickException | unit | `uv run pytest tests/test_submit_cell_identity.py -v` | ❌ W0 |
| REC-04 | `mil_model` normalization collapses whitespace, lowercases | unit | `uv run pytest tests/test_mil_model_normalization.py -v` | ❌ W0 |
| REC-04 | `automil cells migrate` merges elapsed budget without double-count | unit | `uv run pytest tests/cells/test_migrate.py -v` | ❌ W0 |
| REC-04 | `read_cell` compat shim: legacy `parent_id`-keyed cells load without TypeError | unit | `uv run pytest tests/cells/test_migrate.py::test_legacy_cell_loads -v` | ❌ W0 |

*Status legend: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

New test files (must exist before/with their implementing task):

- [ ] `tests/test_sigterm_flush.py` — REC-01 flush write target
- [ ] `tests/test_collect_or_synthesize.py` — REC-01 fold-first synthesis + REC-03 status canonicalization
- [ ] `tests/test_handle_timeout.py` — REC-01 D-04 main-PID-first signaling
- [ ] `tests/test_partial_fold_recovery.py` — REC-01 end-to-end kill simulation + quarantine
- [ ] `tests/test_terminal_writer.py` — REC-02 all four artifacts (normal + cap-kill)
- [ ] `tests/test_terminal_writer_consistency.py` — REC-02 rank/TSV agreement
- [ ] `tests/test_reconcile_from_archive.py` — REC-02 D-11 opt-in refresh
- [ ] `tests/test_aggregate_folds.py` — REC-03 `crashed` canonicalization
- [ ] `tests/test_mil_model_normalization.py` — REC-04 normalization
- [ ] `tests/cells/test_migrate.py` — REC-04 budget-merge migration + legacy-cell compat shim

Extend existing:

- [ ] `tests/test_result_schema_validation.py` — add `partial` status + `termination_reason` cases
- [ ] `tests/test_submit_cell_identity.py` — add re-parent test + `--mil-model` resolution + normalization

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live cell-key migration preserves real TCGA-LUAD/CCRCC elapsed budget | REC-04 | Requires Leo's workstation with live `automil/cells/*.json` runtime state | Run `automil cells migrate --dry-run` against a live overlay; confirm merged budget totals match pre-migration sums per `(dataset, encoder, mil_model)` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
