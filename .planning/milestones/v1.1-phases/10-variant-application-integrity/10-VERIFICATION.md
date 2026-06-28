---
phase: 10-variant-application-integrity
verified: 2026-06-11T14:25:37Z
status: human_needed
score: 3/3 must-haves verified
overrides_applied: 0
human_verification:
  - test: "Run a real CLAM experiment on a workstation with autobench data. Apply a registered
      model variant via `automil apply <node_id>`, then submit and run through the orchestrator.
      Confirm result.json composite differs from the un-applied baseline composite."
    expected: "The variant-applied run produces a composite score that differs (ideally improves)
      from the baseline run composite, proving CLAM_ARGS propagated all the way through
      _make_clam_args to the actual training loop."
    why_human: "APL-02 real-data composite-delta requires a GPU workstation with dataset
      paths configured in benchmarks/.env. The unit tests (test_variant_dispatch_clam.py)
      prove the translation layer correctly patches ExperimentConfig, but they cannot
      confirm the patched args produce a meaningfully different training outcome on real
      data. Marked @pytest.mark.workstation — skipped in CI."
---

# Phase 10: Variant Application Integrity Verification Report

**Phase Goal:** A registered variant is never silently inert — it applies to the actual live
model through existing open seams; loop-opening variants fail loudly.

**Verified:** 2026-06-11T14:25:37Z
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | APL-01: `examples/sklearn-iris/train.py` reads `applied_variant.json` and importlib-loads `classifier_v0`; A1 closure proven by `test_iris_applied_variant_reaches_worktree_at_runtime` (config.yaml-absent path) | VERIFIED | `train.py:52-70` reads `applied_variant.json` first (priority 1 of 3); `train.py:77-92` importlib-loads `_py_files[0]` from `automil/variants/<name>/`; test at `tests/test_apl01_iris_dispatch.py:133` writes only `applied_variant.json` (no config.yaml) and asserts `result["variant_dispatched"] == "classifier_v0"` — PASSES |
| 2 | APL-02: `variant_dispatch.py::apply_model_variant_to_exp_cfg` reads `applied_variant.json` (primary), patches `exp_cfg` via `CLAM_ARGS` → `_make_clam_args`; wired into `run_experiment.py`; no autobench refs in `src/automil/` | VERIFIED | `variant_dispatch.py:80-99` reads `applied_variant.json` as primary path; `variant_dispatch.py:168-186` sets `exp_cfg.model.*` / `exp_cfg.train.*` from `CLAM_ARGS`; `run_experiment.py:174-180` imports and calls `apply_model_variant_to_exp_cfg` before training; `grep -rn "autobench" src/automil/ --include="*.py"` → 0 results |
| 3 | APL-03: `_classify_variant_route` raises "requires loop opening — deferred (ISSUE-007 / RTA)" for registered `LossVariant`, BEFORE state mutation, with no false-positive on string selectors | VERIFIED | `apply.py:23-68` implements `_classify_variant_route`; raises `click.ClickException` at `apply.py:50-56` (loss) and `apply.py:62-68` (policy) only when `loss_name in LOSS_VARIANTS` / `policy_name in POLICY_VARIANTS`; string selectors bypass the registry check entirely (not in `LOSS_VARIANTS`); guard fires at `apply.py:168` — **before** `raw_yaml` load (line 170), backup (line 189), and writes (lines 191, 206) |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `examples/sklearn-iris/train.py` | Reads `applied_variant.json`, importlib-loads variant | VERIFIED | Lines 52-92: three-path priority chain with `applied_variant.json` at priority 1; importlib dispatch at lines 77-92 |
| `benchmarks/src/autobench/pipeline/variant_dispatch.py` | Reads `applied_variant.json` (primary), patches `exp_cfg` via `CLAM_ARGS` | VERIFIED | 195 lines; full three-path read chain (lines 79-134); CLAM_ARGS patch loop (lines 168-186) |
| `src/automil/cli/lifecycle/apply.py` | `_classify_variant_route` raises before mutation for loop-opening variants | VERIFIED | `_classify_variant_route` defined lines 23-68; called line 168 before any config mutation |
| `tests/test_apl01_iris_dispatch.py` | `test_iris_applied_variant_reaches_worktree_at_runtime` exercises no-config.yaml path | VERIFIED | Test at line 133; writes only `applied_variant.json`; asserts `variant_dispatched == "classifier_v0"` |
| `benchmarks/tests/test_variant_dispatch_clam.py` | Unit tests for APL-02 translation layer | VERIFIED | Tests call `apply_model_variant_to_exp_cfg` directly; 7 test functions; all pass |
| `tests/test_apl03_loud_fail.py` | Tests for APL-03 loud-fail behavior | VERIFIED | File exists; all tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `automil apply` (apply.py:204-209) | worktree `automil/applied_variant.json` | `apply_overlay` in runner | WIRED | `apply.py:204-209` writes `archive/<node_id>/applied_variant.json`; overlay propagation confirmed by test design at `test_apl01_iris_dispatch.py:157-163` |
| `train.py` | `applied_variant.json` | stdlib `json.loads` / `Path.read_text` | WIRED | `train.py:54-63`: reads `Path("automil/applied_variant.json")` before config.yaml |
| `train.py` | `classifier_v0` module | `importlib.util.spec_from_file_location` | WIRED | `train.py:80-91`: resolves `.py` files from `automil/variants/<name>/`, loads module, calls `make_classifier` |
| `run_experiment.py` | `apply_model_variant_to_exp_cfg` | import at call site | WIRED | `run_experiment.py:174` (APL-02 comment) + lines 179-180: inline import and call before training |
| `_classify_variant_route` | state mutation in `apply()` | call order in `apply()` body | WIRED | Guard at line 168; `raw_yaml` load at line 170; backup+write at lines 189-191 — guard is unconditionally first |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `train.py` | `variant_name` | `applied_variant.json` → `json.loads` → `model.variant` | Yes — written by `automil apply` from graph node's `variant_spec` | FLOWING |
| `train.py` | `clf` | importlib-loaded `make_classifier(seed)` from variant module | Yes — real classifier instantiation from variant code | FLOWING |
| `variant_dispatch.py` | `variant_cls` | `scan_variants` populates `MODEL_VARIANTS`; looked up by `(parent_name, variant_name)` | Yes — scans on-disk `.py` files | FLOWING |
| `variant_dispatch.py` | `exp_cfg.model.*` | `variant_cls.CLAM_ARGS` dict | Yes — class-level dict from committed variant code | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 13 APL unit tests pass | `uv run pytest tests/test_apl01_iris_dispatch.py tests/test_apl03_loud_fail.py benchmarks/tests/test_variant_dispatch_clam.py -q --import-mode=importlib` | `13 passed, 17 warnings` | PASS |
| Framework purity: zero autobench refs in `src/automil/` | `grep -rn "autobench" src/automil/ --include="*.py"` | 0 matches | PASS |
| APL-03 guard fires before mutation | line order check in `apply.py` | `_classify_variant_route` call at line 168; first mutation at line 170 | PASS |

### Probe Execution

Step 7c: SKIPPED — no `scripts/*/tests/probe-*.sh` declared or found for Phase 10.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| APL-01 | 10-04-PLAN.md | Iris train.py dispatches variant from `applied_variant.json` (never inert) | SATISFIED | `train.py:52-92`; `test_apl01_iris_dispatch.py:133` A1-closure test passes |
| APL-02 | 10-02-PLAN.md, 10-03-PLAN.md | autobench CLAM consumer applies variant via `CLAM_ARGS` seam; wired into `run_experiment.py`; framework stays generic | SATISFIED (CI portion) | `variant_dispatch.py:45-195`; `run_experiment.py:179-180`; 0 autobench refs in `src/automil/`; real-data composite-delta requires workstation |
| APL-03 | 10-01-PLAN.md | Loop-opening variants detected and reported loudly before state mutation | SATISFIED | `apply.py:23-68` + line 168; raises before line 170 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | No TBD/FIXME/XXX markers found in modified files; no stub returns on code paths exercised by these requirements |

### Human Verification Required

#### 1. APL-02 Real-Data Composite-Delta (workstation-gated)

**Test:** On a workstation with `benchmarks/.env` configured and GPU available:
1. Run a baseline CLAM experiment on an autobench dataset (e.g. ovarian) via `automil submit`.
2. After it completes, `automil apply <node_id>` with a node whose `variant_spec` contains a `CLAM_ARGS`-bearing model variant.
3. Submit a new experiment from that node.
4. Compare `result.json["composite"]` of the variant run vs. the baseline run.

**Expected:** The variant-applied run produces a composite score that differs from the baseline, confirming CLAM_ARGS fields (e.g. `model_type`, `dropout`, `bag_weight`) propagated through `apply_model_variant_to_exp_cfg` → `_make_clam_args` → actual CLAM training.

**Why human:** Requires GPU workstation with dataset paths in `benchmarks/.env`. The unit tests prove the translation layer correctly patches `ExperimentConfig` fields, but only a real training run can confirm the patched args drive a meaningfully different model outcome. Test is `@pytest.mark.workstation` — skipped in CI.

---

## Gaps Summary

No automated gaps found. All three APL success criteria are verified by code inspection and passing tests:

- **APL-01**: `train.py` reads `applied_variant.json` first (no config.yaml required); importlib dispatch is fully wired; A1-closure test proves the no-config.yaml path green.
- **APL-02**: `variant_dispatch.py` is a complete, substantive implementation (195 lines); wired into `run_experiment.py:179-180`; framework stays autobench-free; 7 unit tests pass. Real-data composite-delta is the sole remaining human check.
- **APL-03**: `_classify_variant_route` raises before any state mutation for registered LossVariant/PolicyVariant callables; string selectors do not trigger the guard; pre-mutation ordering proven by line-number inspection.

The single `human_needed` item (APL-02 workstation composite-delta) is architecturally sound — the translation layer is complete and tested; only the end-to-end training outcome on real data requires a human with GPU access to confirm.

---

_Verified: 2026-06-11T14:25:37Z_
_Verifier: Claude (gsd-verifier)_
