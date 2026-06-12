---
phase: 11-config-run-fidelity
plan: 01
subsystem: testing
tags: [pytest, red-stubs, tdd, argparse, click, cfr-01, cfg-02, cfg-03]

requires:
  - phase: 10-variant-application-integrity
    provides: variant dispatch and apply_overlay infrastructure (stable baseline for Phase 11)

provides:
  - RED test stubs for CFG-02 (submit --timeout omit / max-time sentinel interaction)
  - RED test stubs for CFG-03 (--override spec write + daemon suffix-append)
  - RED test stubs for CFG-01 (argparse defaults masking TrainConfig/ExperimentConfig defaults)
  - Nyquist compliance: all 3 CFG behaviors pre-registered before production code is touched

affects: [11-02, 11-03]

tech-stack:
  added: []
  patterns:
    - "RED-stub-before-fix: Wave 0 creates failing tests before production code changes (Nyquist compliance)"
    - "sys.argv-patch: calling run_experiment.parse_args() via sys.argv patch (script not a module)"
    - "importlib.util.spec_from_file_location: loading scripts/ as importable modules in tests"

key-files:
  created:
    - tests/test_cfg_run_fidelity.py
    - benchmarks/tests/test_run_experiment_config.py
  modified: []

key-decisions:
  - "pytest.fail() used for test_daemon_appends_override_to_run_command to keep it RED with clear message rather than a flaky ImportError"
  - "sys.argv patching chosen over monkeypatching parse_args internals — keeps test isolated from argparse internals"
  - "run_experiment module loaded with unique name run_experiment_cfg01 to avoid sys.modules collision with other test files"

patterns-established:
  - "Wave-0 RED stubs committed before any production fix — makes regression protection pre-existing"
  - "Both test trees run separately (tests/ vs benchmarks/tests/) to avoid pytest rootdir collision (established lesson 2026-06-11)"

requirements-completed:
  - CFG-01
  - CFG-02
  - CFG-03

duration: 12min
completed: 2026-06-11
---

# Phase 11 Plan 01: Config & Run Fidelity RED Stubs

**Wave-0 Nyquist stubs: 5 CFG-02/03 tests in tests/ + 4 CFG-01 tests in benchmarks/tests/ pre-register all CFG behaviors before any production code is modified**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-11T19:24:00Z
- **Completed:** 2026-06-11T19:36:00Z
- **Tasks:** 2/2
- **Files modified:** 2 created

## Accomplishments

- Created `tests/test_cfg_run_fidelity.py` with 5 tests covering CFG-02 (timeout_min omit, explicit timeout honor, D-03 max-time sentinel regression guard) and CFG-03 (--override spec write, daemon suffix-append)
- Created `benchmarks/tests/test_run_experiment_config.py` with 4 tests covering CFG-01 (args.lr None-default, args.n_folds None-default, explicit lr honored, explicit n_folds honored)
- All RED stubs fail with AssertionError (not ImportError or SyntaxError) — correct failure mode for Wave 0
- Zero existing tests regressed in either suite (framework: 1018 pass / 53 skip; benchmark: 290 pass / 1 skip, pre-existing acceptance failure unaffected)

## Task Commits

1. **Task 1: Create tests/test_cfg_run_fidelity.py** - `1c0bb76` (test)
2. **Task 2: Create benchmarks/tests/test_run_experiment_config.py** - `d8e5b98` (test)

## Files Created/Modified

- `tests/test_cfg_run_fidelity.py` — 5 tests: 3 RED (timeout-omit, --override spec, daemon append), 2 GREEN (explicit timeout, max-time sentinel guard)
- `benchmarks/tests/test_run_experiment_config.py` — 4 tests: 2 RED (lr None, n_folds None), 2 GREEN (explicit lr, explicit n_folds)

## Decisions Made

- `pytest.fail()` explicit call used in `test_daemon_appends_override_to_run_command` rather than a structural assertion that could raise `ImportError` or produce an ambiguous failure. This ensures the RED state is unambiguous and the failure message describes exactly what Plan 11-03 must implement.
- `sys.argv` patching chosen for calling `parse_args()` in the benchmark tests because `run_experiment.parse_args()` takes no arguments and reads `sys.argv[1:]` directly. Patching `sys.argv` is the cleanest isolation strategy without modifying the production script.
- Module loaded under the unique name `run_experiment_cfg01` (not `run_experiment`) to prevent `sys.modules` collisions if other test files in the benchmark suite ever import similarly named scripts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wrong class name for daemon import in test**
- **Found during:** Task 1 (test_daemon_appends_override_to_run_command) — first run
- **Issue:** Test imported `OrchestratorDaemon` but the actual class in `_orchestrator_daemon.py` is `ExperimentOrchestrator`. This caused an `ImportError` instead of the intended `AssertionError` RED failure.
- **Fix:** Corrected import to `ExperimentOrchestrator`; restructured test to produce `pytest.fail()` RED signal rather than relying on an import-time error.
- **Files modified:** tests/test_cfg_run_fidelity.py
- **Verification:** Re-run confirmed AssertionError/Failed (not ImportError)
- **Committed in:** 1c0bb76 (Task 1 commit, final version)

**2. [Rule 1 - Bug] Wrong path calculation in benchmark test loader**
- **Found during:** Task 2 (test_run_experiment_config.py) — first run produced 4 SKIPPED
- **Issue:** `Path(__file__).resolve().parents[2]` resolved to the workspace root (`autoMIL/`), not `benchmarks/`. The correct index for `benchmarks/` from `benchmarks/tests/` is `parents[1]`.
- **Fix:** Changed `parents[2]` to `parents[1]` in `_load_run_experiment()`.
- **Files modified:** benchmarks/tests/test_run_experiment_config.py
- **Verification:** Re-run produced 2 RED + 2 GREEN as expected
- **Committed in:** d8e5b98 (Task 2 commit, final version)

**3. [Rule 1 - Bug] parse_args() reads sys.argv not a passed list**
- **Found during:** Task 2 — first iteration design assumed `parse_args(cli_args)` was callable with a list
- **Issue:** `run_experiment.parse_args()` takes no parameters; it calls `p.parse_args()` internally which reads `sys.argv[1:]`. Initial design was incompatible.
- **Fix:** Introduced `_parse(mod, cli_args)` helper that patches `sys.argv` for the call duration then restores it.
- **Files modified:** benchmarks/tests/test_run_experiment_config.py
- **Verification:** All 4 tests collect and run with correct RED/GREEN state
- **Committed in:** d8e5b98 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (Rule 1 - Bug, all in test files only)
**Impact on plan:** All fixes were to the newly created test files, not production code. No scope creep. The fixes ensure RED failures are AssertionError (not ImportError/SKIPPED), which is the correct Wave-0 state.

## Issues Encountered

- Pre-existing acceptance test `test_d208_clause_11_state_roadmap_complete` was already failing before this plan (verified via git stash). It tracks STATE.md/ROADMAP.md completeness for Phase 11 and will pass once the phase completes. Not caused by this plan.

## Known Stubs

The following are intentional test stubs (not production stubs):

| File | Test | Stub type | Resolves in |
|------|------|-----------|-------------|
| tests/test_cfg_run_fidelity.py | test_submit_without_timeout_omits_timeout_min | RED — assert "timeout_min" not in spec | Plan 11-03 |
| tests/test_cfg_run_fidelity.py | test_submit_override_written_to_spec | RED — --override option does not exist | Plan 11-03 |
| tests/test_cfg_run_fidelity.py | test_daemon_appends_override_to_run_command | RED — pytest.fail() explicit stub | Plan 11-03 |
| benchmarks/tests/test_run_experiment_config.py | test_no_lr_flag_uses_trainconfig_default | RED — args.lr==1e-4 not None | Plan 11-02 |
| benchmarks/tests/test_run_experiment_config.py | test_no_n_folds_flag_uses_experimentconfig_default | RED — args.n_folds==5 not None | Plan 11-02 |

These stubs are intentional (Wave-0 Nyquist compliance). The plan goal is achieved: all CFG behaviors are pre-registered as failing tests before any production code is modified.

## Next Phase Readiness

- Plan 11-02 can now implement CFG-01 fix in `benchmarks/scripts/run_experiment.py` and verify `test_no_lr_flag_uses_trainconfig_default` + `test_no_n_folds_flag_uses_experimentconfig_default` go GREEN
- Plan 11-03 can implement CFG-02 + CFG-03 fixes in `src/automil/cli/submit.py` and daemon, verifying all 3 remaining RED stubs go GREEN
- The D-03 sentinel regression guard (`test_max_time_wins_over_explicit_timeout`) provides explicit named coverage for the highest-risk edit in this phase

---
*Phase: 11-config-run-fidelity*
*Completed: 2026-06-11*
