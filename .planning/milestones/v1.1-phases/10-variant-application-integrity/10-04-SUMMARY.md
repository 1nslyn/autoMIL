---
phase: "10"
plan: "04"
subsystem: "examples/sklearn-iris consumer dispatch"
tags: [APL-01, A1-closure, iris, variant-dispatch, importlib, applied_variant_json]
dependency_graph:
  requires: [10-02, 10-03]
  provides: [APL-01, A1-runtime-reachability-closure]
  affects: [examples/sklearn-iris/train.py, tests/test_apl01_iris_dispatch.py]
tech_stack:
  added: []
  patterns: [stdlib importlib dispatch, applied_variant.json read, variant_dispatched result marker]
key_files:
  created: []
  modified:
    - examples/sklearn-iris/train.py
    - tests/test_apl01_iris_dispatch.py
decisions:
  - "Read applied_variant.json first (A1 worktree path), then config.yaml model.variant, then AUTOMIL_VARIANT_MODEL env fallback — matches the overlay propagation mechanism from plan 10-02"
  - "Use raw importlib.util.spec_from_file_location for variant dispatch — no automil.* imports (consumer-decoupled Pitfall 4)"
  - "variant_dispatched key added to result.json payload so tests can assert dispatch fired without monkey-patching the module"
  - "T-10-08: Path(variant_name).name == variant_name guard before constructing _variants_dir"
  - "Checkpoint:human-verify auto-approved (autonomous run) — workstation CLAM delta check remains deferred/workstation-gated as documented in 10-VALIDATION.md"
metrics:
  duration: "10 minutes"
  completed: "2026-06-11T14:18:49Z"
  tasks_completed: 1
  tasks_total: 2
  files_modified: 2
---

# Phase 10 Plan 04: APL-01 Iris Dispatch Summary

**One-liner:** Iris train.py now dispatches to `classifier_v0.make_classifier` via applied_variant.json → config.yaml → env-var priority chain, using raw importlib (no automil imports), closing the A1 inert-variant trap.

## What Was Built

### Task 1: Add applied_variant.json dispatch to iris train.py (APL-01, D-01/D-02)

**Files modified:**
- `examples/sklearn-iris/train.py` — variant dispatch block + `variant_dispatched` result marker
- `tests/test_apl01_iris_dispatch.py` — replaced `pytest.fail` stub in test 3 with real A1 closure test

**Variant resolution priority (read order):**
1. `automil/applied_variant.json` (A1 fix — written by `apply.py` into archive, propagated into worktree by `apply_overlay`)
2. `automil/config.yaml` `model.variant` key (existing config path)
3. `os.environ.get("AUTOMIL_VARIANT_MODEL")` (runtime env fallback)

**Dispatch mechanism:** Raw `importlib.util.spec_from_file_location` — same pattern as `scanner.py:46–67`, but inlined to preserve the no-automil-imports consumer-decoupled contract documented at `train.py:7`. Scans `automil/variants/<variant_name>/` for non-private `.py` files, loads the first (sorted), calls `_mod.make_classifier(seed=seed)`.

**Path-traversal guard:** `Path(variant_name).name != variant_name` raises `ValueError` — mitigates T-10-08.

**Observability marker:** `variant_dispatched` key added to `result.json` payload (only present when variant fires; `None`/absent for baseline path). Both `_write_result` and the SIGTERM handler propagate it.

**Test 3 A1 closure:** `test_iris_applied_variant_reaches_worktree_at_runtime` upgraded from a hard `pytest.fail` stub to a real test: writes only `applied_variant.json` (no `config.yaml`), runs iris, asserts `variant_dispatched == "classifier_v0"` — definitively proves variant reaches the consumer at runtime even when `config.yaml` is gitignored and absent from the worktree.

### Task 2: checkpoint:human-verify (AUTO-APPROVED)

Per autonomous directive: the workstation CLAM composite-delta verification (`AUTOBENCH_CCRCC_ROOT` required) is deferred/workstation-gated — already captured as a `human_needed` item in the Phase 10 validation table and in 10-CONTEXT.md §Deferred. The checkpoint is auto-approved; plan is complete.

## Test Results

```
tests/test_apl01_iris_dispatch.py::test_iris_baseline_no_variant         PASSED
tests/test_apl01_iris_dispatch.py::test_iris_dispatches_classifier_v0_when_variant_set  PASSED
tests/test_apl01_iris_dispatch.py::test_iris_applied_variant_reaches_worktree_at_runtime  PASSED
```

Full suite (1302 passed, 53 skipped, 0 new failures — 1 pre-existing Phase 8 acceptance test unrelated to this plan).

## Verification Checks

- `grep -n "applied_variant.json" examples/sklearn-iris/train.py` → line 68, 72 (dispatch read present)
- `grep -r "autobench" src/automil/` → zero code imports (two template comments only — framework-pure)
- `grep -n "_classify_variant_route" src/automil/cli/lifecycle/apply.py` → present and before `_atomic_write_text` (APL-03 ordering, from plan 10-02)

## Deviations from Plan

None — plan executed exactly as written. The dispatch block matches the code pattern from 10-RESEARCH.md §APL-01 Deep-Dive (lines 421–441). The `variant_dispatched` result marker and the test 3 real implementation were specified in the plan.

## Known Stubs

None — all variant dispatch paths are fully wired. The baseline path (no variant set) is intentional and tested.

## Threat Flags

None — the T-10-08 path-traversal mitigation is implemented. T-10-09 (arbitrary code via variant module) and T-10-10 (env var disclosure) are accepted per the threat model (variants are git-committed, variant name is not a secret).

## Self-Check: PASSED

- `examples/sklearn-iris/train.py` — exists, modified
- `tests/test_apl01_iris_dispatch.py` — exists, modified
- Commit `4723b59` — present in git log
- 3/3 APL-01 tests GREEN
