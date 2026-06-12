---
phase: 11-config-run-fidelity
verified: 2026-06-11T19:58:54Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
---

# Phase 11: Config & Run Fidelity — Verification Report

**Phase Goal:** Config file and snapshot values drive experiment runs without being silently overridden by argparse or CLI defaults; per-node run-command overrides are expressible without editing snapshotted code.
**Verified:** 2026-06-11T19:58:54Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CFG-01: `parse_args` defaults all 6 training-override flags to `None`; `main()` only passes non-None values into `TrainConfig`; downstream `n_folds` refs use resolved `exp_cfg.n_folds`, not `args.n_folds` | VERIFIED | `run_experiment.py:63-68` (all 6 flags `default=None`), `L157-163` (dict-comprehension filter), `L166-169` (n_folds guard), `L222,247` (`exp_cfg.n_folds` at both `prepare_all` / `prepare_nnmil` call sites) |
| 2 | CFG-02: `--timeout` defaults `None`; `timeout_min` omitted from spec when unset; D-03 sentinel is `timeout is not None` (not `!= 150`); `--max-time` still wins when both flags given | VERIFIED | `submit.py:29` (`default=None, type=int`), `L65` (`if timeout is not None:` sentinel), `L455-456` (conditional write), `L59-70` (max-time interaction block preserved) |
| 3 | CFG-03: `--override` writes `run_command_override` into queue spec; daemon appends `shlex.split(override_str)` as a list after `shlex.split(self.run_command)` with no `shell=True` | VERIFIED | `submit.py:42-45` (`--override` option), `L458-459` (conditional spec write), `_orchestrator_daemon.py:906-908` (list concat append), `L909` (`subprocess.Popen` with no `shell=True`), `L924` (`spec.get("timeout_min", self.default_timeout)` daemon fallback) |

**Score:** 3/3 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `benchmarks/scripts/run_experiment.py` | 6 training-override flags default=None; conditional TrainConfig construction | VERIFIED | Lines 62-69 (parse_args), 155-169 (main), 222, 247 (downstream uses resolved value) |
| `src/automil/cli/submit.py` | `--timeout` default=None; conditional timeout_min write; D-03 sentinel; `--override` option | VERIFIED | Lines 29, 42-45, 65, 455-459 |
| `src/automil/backends/_orchestrator_daemon.py` | `spec.get("run_command_override")` append + `spec.get("timeout_min", self.default_timeout)` fallback | VERIFIED | Lines 906-908 (override append), 924 (timeout fallback) |
| `tests/test_cfg_run_fidelity.py` | 5 tests: CFG-02 timeout omit, explicit, max-time interaction; CFG-03 spec write, daemon append | VERIFIED | 5/5 PASSED (live run confirmed) |
| `benchmarks/tests/test_run_experiment_config.py` | 4 tests: CFG-01 None-default lr, n_folds; explicit override lr, n_folds | VERIFIED | 4/4 PASSED (live run confirmed) |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `submit.py --override` | `queue spec["run_command_override"]` | `if override is not None: spec["run_command_override"] = override` | WIRED | `submit.py:458-459` |
| `queue spec["run_command_override"]` | `_orchestrator_daemon.py` Popen cmd | `spec.get("run_command_override")` → `cmd = cmd + shlex.split(override_str)` | WIRED | `_orchestrator_daemon.py:906-908` |
| `submit.py --timeout` | `queue spec["timeout_min"]` | `if timeout is not None: spec["timeout_min"] = timeout` | WIRED (conditional) | `submit.py:455-456`; absent when None |
| `queue spec missing timeout_min` | `daemon self.default_timeout` | `spec.get("timeout_min", self.default_timeout)` | WIRED | `_orchestrator_daemon.py:924` |
| `parse_args --lr/n_folds/etc` | `TrainConfig(**_train_overrides)` | dict-comprehension filtering None values | WIRED | `run_experiment.py:157-164` |
| `args.n_folds` | `exp_cfg.n_folds` at downstream calls | `if args.n_folds is not None: _exp_kwargs["n_folds"] = args.n_folds` then `exp_cfg.n_folds` | WIRED | `run_experiment.py:166-169, 222, 247` |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CFG-02: 5 tests for timeout omit, explicit, max-time sentinel, override spec, daemon append | `uv run pytest tests/test_cfg_run_fidelity.py -v` | 5 passed in 0.86s | PASS |
| CFG-01: 4 tests for None-default lr, n_folds and explicit override | `uv run pytest benchmarks/tests/test_run_experiment_config.py -v` | 4 passed in 4.07s | PASS |

---

## D-03 Sentinel Verification (Critical)

The highest-risk edit in Phase 11 was changing the `--max-time` interaction sentinel from `timeout != 150` to `timeout is not None`. Verified at `submit.py:65`:

```python
if timeout is not None:  # caller passed --timeout explicitly
```

The test `test_max_time_wins_over_explicit_timeout` (passing) exercises `--max-time 120 --timeout 99` and asserts `spec["timeout_min"] == 2` (ceil-div of 120s). This confirms the sentinel change preserved the D-195 `--max-time` wins path.

---

## CFG-03 Daemon Append — Real Launch Site Verification

The SUMMARY claim that "daemon appends override at launch site" was verified directly at the real Popen call site in `_orchestrator_daemon.py`:

- **L899-902**: `cmd = shlex.split(self.run_command)` (or `sys.executable` branch)
- **L906-908**: `override_str = spec.get("run_command_override"); if override_str: cmd = cmd + shlex.split(override_str)`
- **L909**: `process = subprocess.Popen(cmd, ...)` — no `shell=True` keyword (confirmed: `shell=True` appears only once in the file, in a comment at L904, not in executable code)
- **L924**: `timeout_min = spec.get("timeout_min", self.default_timeout)` — confirms daemon fallback to `self.default_timeout` (loaded from `orchestrator.default_timeout_min` at L436)

This is list concatenation, not string concatenation. No shell injection vector.

---

## Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| CFG-01 | 11 | argparse defaults None; dataclass defaults honored | SATISFIED | `run_experiment.py:62-69, 155-169` + 4 passing tests |
| CFG-02 | 11 | `submit --timeout` defaults None; timeout_min omitted when unset; D-03 sentinel fix | SATISFIED | `submit.py:29, 65, 455-456` + 3 passing tests |
| CFG-03 | 11 | `--override` writes to spec; daemon suffix-appends as list | SATISFIED | `submit.py:42-45, 458-459` + `_orchestrator_daemon.py:906-908` + 2 passing tests |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No TBD/FIXME/XXX markers found in any of the 5 Phase 11 modified files |

Scanned: `benchmarks/scripts/run_experiment.py`, `src/automil/cli/submit.py`, `src/automil/backends/_orchestrator_daemon.py`, `tests/test_cfg_run_fidelity.py`, `benchmarks/tests/test_run_experiment_config.py`. Zero debt markers found.

---

## Commits Verified

| Commit | Description | Files |
|--------|-------------|-------|
| `277f83d` | CFG-01: argparse None-defaults + conditional TrainConfig construction | `benchmarks/scripts/run_experiment.py` (+25/-18) |
| `82a1d9b` | CFG-02: None-default --timeout, D-03 sentinel, --override option | `src/automil/cli/submit.py` (+16/-5) |
| `1d3e1b5` | CFG-03: daemon suffix-append + GREEN Popen-mock test | `_orchestrator_daemon.py` (+6), `test_cfg_run_fidelity.py` (+79/-31) |

All three commits confirmed present via `git show --stat`.

---

## Human Verification Required

None. All three success criteria are verifiable programmatically and confirmed by live test runs.

---

## Gaps Summary

No gaps. All three CFG success criteria are fully implemented, wired, and covered by passing tests.

---

_Verified: 2026-06-11T19:58:54Z_
_Verifier: Claude (gsd-verifier)_
