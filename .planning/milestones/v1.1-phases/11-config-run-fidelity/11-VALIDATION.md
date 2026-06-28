---
phase: 11
slug: config-run-fidelity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 11 — Validation Strategy

> Per-phase validation contract. (Research skipped — contained fixes; anchors in 11-CONTEXT.md.)

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest tests/test_cfg_run_fidelity.py benchmarks/tests/test_run_experiment_config.py -v` |
| **Framework suite** | `uv run pytest tests/ -q` |
| **Benchmark suite** | `uv run pytest benchmarks/tests/ -q` (run SEPARATELY — combined run with tests/ triggers a pytest rootdir collision, see lessons 2026-06-11) |
| **Estimated runtime** | quick <30s |

## Sampling Rate
- After every task commit: quick command
- After every wave: framework + benchmark suites (separately)
- Before `/gsd-verify-work`: both suites green

## Per-Requirement Verification Map

| Req | Behavior | Type | Command | File |
|-----|----------|------|---------|------|
| CFG-01 | run_experiment without `--lr/--seed/--max_epochs/--patience/--stop_epoch/--n_folds` uses TrainConfig snapshot defaults (NOT 1e-4/42/200/...) | unit | `uv run pytest benchmarks/tests/test_run_experiment_config.py -v` | ❌ W0 |
| CFG-01 | explicit `--lr 5e-4` IS honored (override still works) | unit | same | ❌ W0 |
| CFG-02 | `submit` without `--timeout` omits `timeout_min` from the queue spec | unit | `uv run pytest tests/test_cfg_run_fidelity.py -v` | ❌ W0 |
| CFG-02 | `submit --timeout 90` writes `timeout_min: 90` | unit | same | ❌ W0 |
| CFG-02 | `--max-time` still WINS when both `--max-time` and `--timeout` given (D-195 sentinel now `is not None`) | unit | same | ❌ W0 — the regression-risk interaction |
| CFG-03 | `submit --override "--seed 42 --lr 1e-4"` writes the override into the queue spec | unit | same | ❌ W0 |
| CFG-03 | daemon appends spec override args AFTER the base `run.command` (suffix-append, base authoritative) | unit | same | ❌ W0 |
| all | existing submit + run_experiment tests stay GREEN | regression | both suites | ✅ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements
- [ ] `tests/test_cfg_run_fidelity.py` — CFG-02 (timeout omit/honor + max-time interaction) + CFG-03 (override write + daemon append)
- [ ] `benchmarks/tests/test_run_experiment_config.py` — CFG-01 (None defaults honor snapshot; explicit flag still overrides)

## Manual-Only Verifications
*None — all CFG behaviors have automated verification (no external data needed).*

## Validation Sign-Off
- [ ] All tasks have `<automated>` verify or Wave 0 deps
- [ ] No watch-mode flags
- [ ] CFG-02 `--max-time` interaction test present (the sentinel-change regression guard)
- [ ] `nyquist_compliant: true` set when complete

**Approval:** pending
