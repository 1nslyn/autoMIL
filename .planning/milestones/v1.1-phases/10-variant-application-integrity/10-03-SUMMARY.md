---
phase: 10-variant-application-integrity
plan: "03"
subsystem: variant-application
tags: [APL-02, variant-dispatch, clam-args, applied_variant_json, registry, framework-purity]

dependency_graph:
  requires:
    - phase: 10-01
      provides: variant_dispatch.py stub + test_variant_dispatch_clam.py RED scaffolding
    - phase: 10-02
      provides: applied_variant.json written by automil apply + propagated into worktree
  provides:
    - apply_model_variant_to_exp_cfg: full implementation reading applied_variant.json PRIMARY path
    - CLAM_ARGS optional class attribute convention on ModelVariant ABC
    - run_experiment.py call site wired (Pitfall 7 closed)
    - test_no_config_yaml_still_dispatches: APL-02 A1-closure proof
  affects:
    - 10-04 (iris dispatch — can now reference variant_dispatch pattern)
    - future CLAM variant modules (must declare CLAM_ARGS dict)

tech-stack:
  added: []
  patterns:
    - "applied_variant.json as PRIMARY read path (not config.yaml) for worktree-propagated variant selection"
    - "Three-path priority chain: applied_variant.json → config.yaml (deprecated fallback) → AUTOMIL_VARIANT_MODEL env"
    - "MODEL_VARIANTS keyed on (parent_name, variant_name) tuple — must match _state.py key shape"
    - "getattr(cls, 'CLAM_ARGS', {}) optional attribute read — backward-compatible, no ABC change"
    - "T-10-05 path traversal guard: Path(variant_name).name == variant_name before any filesystem op"
    - "Consumer-agnostic CLAM_ARGS docstring in ModelVariant ABC preserves D-206 purity gate"

key-files:
  created: []
  modified:
    - benchmarks/src/autobench/pipeline/variant_dispatch.py
    - benchmarks/tests/test_variant_dispatch_clam.py
    - benchmarks/scripts/run_experiment.py
    - src/automil/registry/variants/model.py

key-decisions:
  - "applied_variant.json is the PRIMARY read path (not config.yaml) because config.yaml is gitignored and absent from worktrees; applied_variant.json IS propagated into the worktree by apply_overlay from archive/<node_id>/"
  - "test_no_config_yaml_still_dispatches exercises the primary path with NO config.yaml present — the APL-02 A1-closure proof"
  - "CLAM_ARGS docstring in model.py uses consumer-agnostic language (no literal 'autobench' word) to pass the purity gate grep check"
  - "apply_model_variant_to_exp_cfg call in run_experiment.py uses local import to avoid top-level import ordering issues"
  - "5 existing tests migrated from _write_automil_config (config.yaml) to _write_applied_variant_json — tests now exercise the real production dispatch route"

patterns-established:
  - "APL-02 dispatch pattern: applied_variant.json → scan_variants → MODEL_VARIANTS lookup → getattr(CLAM_ARGS) → setattr on model/train config"
  - "Purity gate compliance: ModelVariant ABC documents consumer conventions without naming consumer packages"

requirements-completed:
  - APL-02

duration: ~10min
completed: 2026-06-11
---

# Phase 10 Plan 03: APL-02 CLAM Variant Dispatch Layer Summary

**apply_model_variant_to_exp_cfg reads applied_variant.json (not config.yaml) and patches ExperimentConfig.model/train via CLAM_ARGS before _make_clam_args runs — APL-02 A1-closure proven by test_no_config_yaml_still_dispatches with config.yaml deliberately absent.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-06-11T09:58Z
- **Completed:** 2026-06-11T10:08Z
- **Tasks:** 2 (TDD: RED + GREEN for each)
- **Files modified:** 4

## Accomplishments

- Replaced the `NotImplementedError` stub in `variant_dispatch.py` with the full `apply_model_variant_to_exp_cfg` implementation featuring a three-path priority chain (applied_variant.json → config.yaml fallback → env var last resort)
- Added `test_no_config_yaml_still_dispatches` as the 6th test and migrated all 5 existing tests to write `applied_variant.json` instead of `config.yaml` — all 6 tests GREEN, APL-02 A1-closure proved
- Wired `apply_model_variant_to_exp_cfg(exp_cfg, Path("automil"))` into `run_experiment.py` after `exp_cfg` construction (Pitfall 7 closure: real entry path now calls dispatch)
- Added consumer-agnostic `CLAM_ARGS` optional attribute documentation to `ModelVariant` ABC; framework purity gate (D-206) PASSED: zero literal "autobench" references in `src/automil/`

## Task Commits

1. **RED: Update tests to applied_variant.json primary path + add 6th test** - `417132c` (test)
2. **GREEN: Implement variant_dispatch.py + CLAM_ARGS doc on ModelVariant** - `b7b1444` (feat)
3. **Wire call site in run_experiment.py** - `8ab993f` (feat)

## Files Created/Modified

- `benchmarks/src/autobench/pipeline/variant_dispatch.py` — Full implementation of `apply_model_variant_to_exp_cfg` with three-path priority read, path traversal guard, scan_variants call, MODEL_VARIANTS lookup, and CLAM_ARGS patching
- `benchmarks/tests/test_variant_dispatch_clam.py` — Added `_write_applied_variant_json` helper; migrated 5 tests to primary path; added `test_no_config_yaml_still_dispatches` (APL-02 A1-closure proof)
- `benchmarks/scripts/run_experiment.py` — Added `apply_model_variant_to_exp_cfg(exp_cfg, Path("automil"))` call after `exp_cfg` construction
- `src/automil/registry/variants/model.py` — Added `CLAM_ARGS: ClassVar[dict]` optional attribute documentation on `ModelVariant` ABC (consumer-agnostic, purity-safe)

## Decisions Made

1. **applied_variant.json as PRIMARY path:** `config.yaml` is gitignored and never in worktrees. `applied_variant.json` is written to `archive/<node_id>/` by `automil apply` (plan 10-02) and propagated into the worktree by `apply_overlay`. Any dispatch relying on `config.yaml` would silently no-op in real orchestrated runs.

2. **Consumer-agnostic CLAM_ARGS docstring:** The purity gate (`test_framework_purity.py`) does a literal grep for "autobench" anywhere in `src/automil/`. Rewriting the docstring to use generic "consumer-side dispatch layer" language instead of "autobench" preserves D-206 without needing an allowlist addition.

3. **All 5 existing tests migrated to applied_variant.json:** The tests originally wrote `config.yaml` (which would have made them test the deprecated fallback path, not the production path). Migrating them ensures the full test suite exercises the correct dispatch route.

4. **Local import in run_experiment.py:** `from autobench.pipeline.variant_dispatch import apply_model_variant_to_exp_cfg` is placed inside `main()` as a local import (after the `sys.path` worktree-fix block at module level) to avoid import ordering issues.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test file wrote config.yaml instead of applied_variant.json**
- **Found during:** Task 1 (GREEN phase pre-check)
- **Issue:** All 5 existing tests used `_write_automil_config` which wrote `config.yaml`. The plan specifies applied_variant.json as the PRIMARY read path. Leaving tests on config.yaml would mean they tested the deprecated fallback, not the production route, and `test_no_config_yaml_still_dispatches` would fail because it expects config.yaml to be absent.
- **Fix:** Added `_write_applied_variant_json` helper; migrated all 5 tests to use it. Kept `_write_automil_config` as deprecated (for potential future backward-compat fallback test).
- **Files modified:** `benchmarks/tests/test_variant_dispatch_clam.py`
- **Committed in:** `417132c` (RED commit)

**2. [Rule 1 - Bug] CLAM_ARGS docstring triggered purity gate grep**
- **Found during:** Task 1 (GREEN phase verification)
- **Issue:** Initial CLAM_ARGS docstring in `model.py` mentioned "autobench" and `benchmarks/src/autobench/pipeline/variant_dispatch.py` — causing `test_framework_purity_no_autobench_refs` to FAIL.
- **Fix:** Rewrote docstring to use consumer-agnostic language ("consumer-side dispatch layer", "argument-passing seam") without naming the consumer package.
- **Files modified:** `src/automil/registry/variants/model.py`
- **Committed in:** `b7b1444` (GREEN feat commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — test correctness + purity gate)
**Impact on plan:** Both fixes necessary for correctness; no scope creep.

## Known Stubs

None — `apply_model_variant_to_exp_cfg` is fully implemented. The workstation real-run test (`benchmarks/tests/test_apl02_real_run.py`) remains SKIPPED by design (requires `AUTOBENCH_CCRCC_ROOT`).

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. The `apply_model_variant_to_exp_cfg` function adds a new file access pattern (`automil/applied_variant.json`) but this is already covered by the threat register:

| Flag | File | Description |
|------|------|-------------|
| T-10-05 (mitigated) | variant_dispatch.py | Path traversal guard implemented: `Path(variant_name).name == variant_name` check before any filesystem op; raises ValueError on failure |
| T-10-06 (accepted) | variant_dispatch.py | CLAM_ARGS setattr guard: `hasattr(exp_cfg.model, field)` check prevents arbitrary attribute injection |

## Self-Check: PASSED

Files confirmed:
- FOUND: benchmarks/src/autobench/pipeline/variant_dispatch.py
- FOUND: benchmarks/tests/test_variant_dispatch_clam.py
- FOUND: benchmarks/scripts/run_experiment.py
- FOUND: src/automil/registry/variants/model.py

Commits confirmed:
- FOUND: 417132c (RED test update)
- FOUND: b7b1444 (GREEN implementation)
- FOUND: 8ab993f (call site wire-up)

Test verification:
- benchmarks/tests/test_variant_dispatch_clam.py: 6 PASSED (incl. test_no_config_yaml_still_dispatches)
- tests/test_framework_purity.py: 3 PASSED (D-206 purity gate)
- benchmarks/tests/ (all, -k not workstation): 288 PASSED, 0 regressions

Grep checks:
- grep -r "autobench" src/automil/registry/variants/model.py → 0 matches
- grep -n "apply_model_variant_to_exp_cfg" benchmarks/scripts/run_experiment.py → line 179, 180

## Next Phase Readiness

- APL-02 is closed: variant dispatch layer fully implemented and wired
- Plan 10-04 (iris dispatch, APL-01) can proceed independently
- Any new CLAM variant module must declare `CLAM_ARGS: ClassVar[dict]` to flow fields through the seam
- The workstation composite-delta verification (D-04) remains for human verification with `AUTOBENCH_CCRCC_ROOT` set

---
*Phase: 10-variant-application-integrity*
*Completed: 2026-06-11*
