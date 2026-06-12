---
phase: 14-housekeeping-tech-debt
plan: "02"
subsystem: test-infrastructure
tags: [dbt-03, clause-11, em-dash-cleanup, anchor-stability, acceptance-test]
dependency_graph:
  requires: [14-01]
  provides: [dbt-03-complete, clause-11-fixed, full-suite-green]
  affects: [tests/test_framework_purity.py, tests/acceptance/test_phase8_acceptance.py, src/automil/backends/_orchestrator_daemon.py]
tech_stack:
  added: []
  patterns: [stable-text-anchoring, archive-first-precedence]
key_files:
  created: []
  modified:
    - src/automil/backends/_orchestrator_daemon.py
    - tests/test_framework_purity.py
    - tests/acceptance/test_phase8_acceptance.py
decisions:
  - "DBT-03: switched clause_07 from brittle line-number assertion to stable-text matching on file name + content anchor substring -- decoupled from future line-number drift permanently"
  - "clause_11: archive-first precedence -- v1.0 acceptance gate reads .planning/milestones/v1.0-REQUIREMENTS.md when present, falls back to live REQUIREMENTS.md only when archive absent"
metrics:
  duration: "~8 minutes"
  completed: "2026-06-12"
  tasks_completed: 2
  files_modified: 3
---

# Phase 14 Plan 02: DBT-03 Em-Dash Cleanup + clause_11 Precedence Fix Summary

**One-liner:** ASCII em-dash substitution + deferral-comment removal in daemon + stable-text anchor update across purity + acceptance tests, then clause_11 archive-first precedence flip closes the milestone to 1058 passed, 0 failed.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | DBT-03: em-dash cleanup + anchor updates | b8c4902 | _orchestrator_daemon.py, test_framework_purity.py, test_phase8_acceptance.py |
| 2 | clause_11: req_path precedence flip to archive-first | 8539b25 | test_phase8_acceptance.py |

## What Was Built

### Task 1: DBT-03 — Em-Dash Cleanup + Stable-Text Anchoring

**`src/automil/backends/_orchestrator_daemon.py`:**
- Converted two em-dashes (U+2014) to ASCII `--` in the env-whitelist comment block (original lines 58 and 63, now 56 and 61 after line removal)
- Removed the 2-line deferral comment `"but that shift would re-number daemon lines and re-break the framework-purity allowlist / acceptance anchors. Left here until Phase 14 / DBT-03 anchor cleanup."` from the IN-01 note block
- The IN-01 note is retained (4 lines, still accurate); only the Phase-14 deferral tail was trimmed
- Anchor line `Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)` shifted from line 62 to line 60

**`tests/test_framework_purity.py`:**
- Updated `_ALLOWLIST` key from `src/automil/backends/_orchestrator_daemon.py:62` to `:60`
- Updated comment to note the shift context (DBT-03 / 14-02 removal of 2-line deferral block)

**`tests/acceptance/test_phase8_acceptance.py` (clause_07):**
- Replaced brittle hardcoded `assert "src/automil/backends/_orchestrator_daemon.py:62" in test_src` with two stable-text assertions:
  - `assert "_orchestrator_daemon.py:" in test_src` (file is still tracked in allowlist)
  - `assert "Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)" in test_src` (content anchor)
- DBT-03 is the last time this anchor needs touching -- decoupled from line-number drift

### Task 2: clause_11 — Archive-First Precedence Fix

**`tests/acceptance/test_phase8_acceptance.py` (clause_11):**
- Old logic: read `.planning/REQUIREMENTS.md` first; fall back to archive only when absent
- Problem: `.planning/REQUIREMENTS.md` exists (v1.1 content, zero DEC rows) so the fallback never fired; assertion loop failed on missing DEC-01..07 Complete rows
- New logic: prefer `.planning/milestones/v1.0-REQUIREMENTS.md` when it exists (the authoritative v1.0 record with all 7 DEC-NN Complete rows); fall back to live path only when archive is absent
- The `req_path` variable remains the final resolved name as required by the downstream `req_path.read_text()` call

## Verification Results

### Post-Task-1:
```
tests/test_framework_purity.py::test_framework_purity_no_autobench_refs PASSED
tests/test_framework_purity.py::test_allowlist_anchors_still_present PASSED
tests/test_framework_purity.py::test_purity_test_does_not_execute_consumer_code PASSED
tests/acceptance/test_phase8_acceptance.py::test_d208_clause_07_framework_purity_grep_gate PASSED
4 passed in 0.75s
```

### Post-Task-2 (full D-208 acceptance suite):
```
11 passed in 42.12s (all clauses 01-11 green)
```

### Full framework suite gate:
```
1058 passed, 53 skipped in 224.36s (0:03:44)
0 failed -- milestone-close target achieved
```

## Deviations from Plan

None — plan executed exactly as written.

The em-dash count in the file at large (36 remaining throughout the body) was flagged during the adjacent-region check. The plan's constraint specifies em-dashes "adjacent to the anchor comment region" (lines 51-65); the anchor region is clean. The remaining 36 em-dashes are in unrelated docstrings and comments throughout the 1800-line file and are out of scope for DBT-03.

## Known Stubs

None.

## Threat Flags

None — all changes are comment-text substitutions and test-logic fixes. No new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- `src/automil/backends/_orchestrator_daemon.py` exists and anchor region has no em-dashes
- `tests/test_framework_purity.py` exists with updated `:60` key
- `tests/acceptance/test_phase8_acceptance.py` exists with stable-text clause_07 + archive-first clause_11
- Commit b8c4902 present: `git log --oneline | grep b8c4902` -- confirmed
- Commit 8539b25 present: `git log --oneline | grep 8539b25` -- confirmed
- Full suite: 1058 passed, 0 failed -- confirmed
