---
phase: 11-config-run-fidelity
plan: "02"
subsystem: benchmarks/scripts
tags: [cfg-01, argparse, trainconfig, config-fidelity, bug-fix]
dependency_graph:
  requires: [11-01]
  provides: [CFG-01-fix]
  affects: [benchmarks/scripts/run_experiment.py]
tech_stack:
  added: []
  patterns: [conditional-kwargs-filter, argparse-none-sentinel]
key_files:
  created: []
  modified:
    - benchmarks/scripts/run_experiment.py
decisions:
  - "CFG-01 / D-01: six training-override argparse flags changed to default=None; conditional dict-comprehension filter passes only non-None values into TrainConfig; ExperimentConfig n_folds guarded similarly"
  - "Downstream prepare_all and prepare_nnmil_experiment call sites updated to use exp_cfg.n_folds (resolved value) rather than args.n_folds which is now None by default"
metrics:
  duration: "~4 minutes"
  completed: "2026-06-11T19:42:05Z"
  tasks_total: 1
  tasks_completed: 1
  files_changed: 1
---

# Phase 11 Plan 02: CFG-01 argparse None-defaults Summary

**One-liner:** Changed six argparse training-override defaults from hard-coded values to None, then applied a dict-comprehension filter at TrainConfig/ExperimentConfig construction so dataclass defaults (lr=2e-4, n_folds=10, etc.) are honored whenever flags are not explicitly supplied on the CLI.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Change parse_args() defaults to None + conditional TrainConfig/ExperimentConfig construction (CFG-01, D-01) | 277f83d | benchmarks/scripts/run_experiment.py |

## What Was Done

**Task 1 — CFG-01 fix (3 coordinated changes):**

**Step A — parse_args() defaults:**
Changed `--max_epochs`, `--lr`, `--seed`, `--n_folds`, `--patience`, `--stop_epoch` from their hard-coded values (`default=200`, `default=1e-4`, `default=42`, `default=5`, `default=20`, `default=50`) to `default=None`. `--gpu` was already `default=None` and was not touched.

**Step B — conditional TrainConfig construction:**
Replaced the unconditional `TrainConfig(max_epochs=args.max_epochs, lr=args.lr, ...)` with a dict-comprehension filter:
```python
_train_overrides = {k: v for k, v in {
    "max_epochs": args.max_epochs,
    "lr": args.lr,
    "seed": args.seed,
    "patience": args.patience,
    "stop_epoch": args.stop_epoch,
}.items() if v is not None}
train_cfg = TrainConfig(**_train_overrides)
```
This means `TrainConfig.lr = 2e-4` (the dataclass default) is used when `--lr` is absent; `TrainConfig.lr = 5e-4` is used when `--lr 5e-4` is supplied.

**Step C — conditional ExperimentConfig n_folds:**
Wrapped `n_folds` in a guard:
```python
_exp_kwargs = {}
if args.n_folds is not None:
    _exp_kwargs["n_folds"] = args.n_folds
exp_cfg = ExperimentConfig(..., **_exp_kwargs)
```
So `ExperimentConfig.n_folds = 10` (dataclass default) is used when `--n_folds` is absent.

**Step D — downstream None propagation fix:**
Two call sites (`prepare_all` and `prepare_nnmil_experiment`) previously passed `n_splits=args.n_folds`, which is now `None` by default. Both were changed to `n_splits=exp_cfg.n_folds` (the resolved value after ExperimentConfig construction).

## Verification

```
benchmarks/tests/test_run_experiment_config.py::TestCFG01NoneDefaults::test_no_lr_flag_uses_trainconfig_default     PASSED
benchmarks/tests/test_run_experiment_config.py::TestCFG01NoneDefaults::test_no_n_folds_flag_uses_experimentconfig_default PASSED
benchmarks/tests/test_run_experiment_config.py::TestCFG01ExplicitFlagsHonored::test_explicit_lr_flag_is_honored      PASSED
benchmarks/tests/test_run_experiment_config.py::TestCFG01ExplicitFlagsHonored::test_explicit_n_folds_flag_is_honored PASSED

4 passed in 3.74s

Full benchmark suite: 292 passed, 1 skipped, 0 failures
```

## Deviations from Plan

None — plan executed exactly as written. All three steps (A, B, C) and the downstream fix (D) matched the plan specification.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced. Pure argparse/dataclass wiring change contained within benchmarks/scripts/run_experiment.py.

## Self-Check: PASSED

- `benchmarks/scripts/run_experiment.py` exists and has been modified (commit 277f83d)
- `default=None` appears on all 6 training-override flags (lines 63-68)
- `args.n_folds` only appears in the conditional guard (not in prepare call sites)
- `exp_cfg.n_folds` used at both prepare call sites (lines 222, 247)
- All 4 CFG-01 tests GREEN; 292 benchmark tests pass
