---
phase: 14
slug: housekeeping-tech-debt
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-12
---

# Phase 14 — Validation Strategy

> Derived from 14-RESEARCH.md §Validation Architecture. Final v1.1 phase — goal is a 100%-green `tests/` suite at milestone close.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml` (project root) |
| **Quick run** | `uv run pytest tests/test_graph.py tests/test_tick_cells.py tests/acceptance/test_phase8_acceptance.py tests/test_framework_purity.py -q` |
| **Framework suite** | `uv run pytest tests/ -q` (benchmark tree run SEPARATELY — rootdir collision) |
| **Estimated runtime** | quick <30s; full ~4min |

## Sampling Rate
- After every task commit: quick command
- After every wave: framework suite (`tests/` only)
- Before `/gsd-verify-work`: framework suite **100% green (0 failed)** — this is the milestone-close gate

## Per-Requirement Verification Map

| Req | Behavior | Type | Command | File |
|-----|----------|------|---------|------|
| DBT-01 | Legacy flat-key graph.json (no `metrics` dict, `schema_version` absent/1) loads through `ExperimentGraph.__init__` → fully-populated `node["metrics"]` (4 consumer keys spread), no KeyError, `schema_version`→2 | unit | `pytest tests/test_graph.py -k legacy_schema_round_trip` | ❌ W0 (new fixture+test) |
| DBT-01 | Already-D-200 graph (has `metrics`/`schema_version`=2) is NOT re-migrated (idempotent; no corruption) | unit | `pytest tests/test_graph.py -k legacy_idempotent` | ❌ W0 |
| DBT-01 | Flat keys retained after spreading (back-compat); migration spreads ONLY val_auc/val_bacc/test_auc/test_bacc | unit | same round-trip test asserts both | ❌ W0 |
| DBT-02 | The 3 named tick_cells tests pass (verify-and-guard — already GREEN via Phase 9 CR-01 33b5383) | unit | `pytest tests/test_tick_cells.py::test_tick_cells_active_to_refusing_new ::test_tick_cells_terminating_fires_cancel_with_cap_reason ::test_tick_cells_finalized_when_running_empty` | ✅ exists (PASSING) |
| DBT-03 | No em-dash (U+2014) chars remain neighboring the daemon allowlist/acceptance anchor in `_orchestrator_daemon.py` | unit/grep | `pytest tests/test_framework_purity.py` (+ grep guard) | ✅ exists (needs anchor update) |
| DBT-03 | Allowlist/acceptance anchor still matches its content after the em-dash removal shifts lines (prefer stable-TEXT anchoring over hardcoded `:62`) | unit | `pytest tests/test_framework_purity.py` + `tests/acceptance/test_phase8_acceptance.py` (clause_07) | ✅ exists (update `_ALLOWLIST` key + clause_07) |
| clause_11 | `test_d208_clause_11_state_roadmap_complete` passes (req_path precedence flipped to prefer the v1.0 archive) | acceptance | `pytest tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` | ✅ exists (currently FAILING → PASS) |
| clause_11 | No other D-208 clause broken by the precedence flip | acceptance | `pytest tests/acceptance/test_phase8_acceptance.py` | ✅ exists |
| all | full framework suite 100% green | regression | `pytest tests/ -q` → 0 failed (target 1055 passed) | ✅ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements
- [ ] `tests/test_graph.py` — DBT-01 (new `test_legacy_schema_round_trip` + `test_legacy_idempotent`; legacy graph.json fixture built in REAL legacy shape, loaded through the real `_load`)
- (DBT-02 tests already exist + pass; DBT-03 + clause_11 tests already exist — updated alongside the fix, not pre-stubbed)

## Critical Anti-Theater Constraint (DBT-01, from RESEARCH)
The legacy graph.json fixture MUST be built in the LEGACY shape (flat `val_auc`/`val_bacc`/`test_auc`/`test_bacc`, NO `metrics` dict, `schema_version` absent or 1) and loaded through the real `ExperimentGraph.__init__`/`_load`. Assert the MIGRATED result (populated `node["metrics"]`, no KeyError). Do NOT hand-construct the post-migration `metrics` dict and assert it equals itself — that gives false-green on a broken migration.

## Manual-Only Verifications
*None — all four items are CI-testable (DBT-01 with a fabricated legacy fixture; DBT-02 already green; DBT-03 via grep/anchor test; clause_11 via the acceptance suite).*

## Validation Sign-Off
- [ ] All tasks have `<automated>` verify or Wave 0 deps
- [ ] DBT-01 test loads a REAL legacy-shaped graph.json through `_load` (anti-theater)
- [ ] DBT-03 line shift reflected in BOTH `test_framework_purity.py` `_ALLOWLIST` and clause_07 (or both switched to stable-text anchoring)
- [ ] Full `tests/` suite is 100% green (0 failed) — milestone-close gate
- [ ] `nyquist_compliant: true` set when complete

**Approval:** pending
