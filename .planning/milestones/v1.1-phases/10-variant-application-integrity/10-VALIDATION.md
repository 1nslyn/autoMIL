---
phase: 10
slug: variant-application-integrity
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-11
---

# Phase 10 — Validation Strategy

> Per-phase validation contract. Derived from 10-RESEARCH.md §Validation Architecture.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | `pyproject.toml` (workspace root) |
| **Quick run command** | `uv run pytest tests/test_apl01_iris_dispatch.py tests/test_apl03_loud_fail.py -v` |
| **Full suite command** | `uv run pytest tests/ benchmarks/tests/ -v` |
| **Estimated runtime** | quick <30s · full ~3 min |

## Sampling Rate
- After every task commit: quick command
- After every plan wave: full suite
- Before `/gsd-verify-work`: full suite green
- Max feedback latency: ~30s

## Per-Requirement Verification Map

| Req | Behavior | Type | Command | File |
|-----|----------|------|---------|------|
| APL-01 | iris `train.py` instantiates `classifier_v0` when selected | integration | `uv run pytest tests/test_apl01_iris_dispatch.py -v` | ❌ W0 |
| APL-01 | iris baseline still works when no variant set | unit | same | ❌ W0 |
| APL-01 | applied variant reaches worktree `train.py` at RUNTIME (not just config written) | integration | same | ❌ W0 — closes the "still inert" risk |
| APL-02 | variant patches ModelConfig from registry | unit | `uv run pytest benchmarks/tests/test_variant_dispatch_clam.py -v` | ❌ W0 |
| APL-02 | variant fields flow through `_make_clam_args` into args namespace (spy/stub `clam_train`) | unit | same | ❌ W0 — CI-gated, no data |
| APL-02 | real CLAM composite differs from baseline | workstation-only | `AUTOBENCH_CCRCC_ROOT=... uv run pytest benchmarks/tests/test_apl02_real_run.py -v` | ❌ W0 — `@pytest.mark.workstation` |
| APL-03 | registered `LossVariant` raises loud error at apply time | unit | `uv run pytest tests/test_apl03_loud_fail.py -v` | ❌ W0 |
| APL-03 | string-selector / model variants do NOT raise | unit | same | ❌ W0 (regression guard) |
| APL-03 | error raised BEFORE `config.yaml`/state mutated (ordering) | unit | same | ❌ W0 |
| APL-01 | existing `test_lifecycle_apply.py` (14 tests) still pass | regression | `uv run pytest tests/test_lifecycle_apply.py -v` | ✅ exists |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements
- [ ] `tests/test_apl01_iris_dispatch.py` — APL-01 iris end-to-end dispatch + runtime-reaches-worktree
- [ ] `tests/test_apl03_loud_fail.py` — APL-03 loud fail + ordering
- [ ] `benchmarks/tests/test_variant_dispatch_clam.py` — APL-02 arg-threading via stub
- [ ] `benchmarks/tests/test_apl02_real_run.py` — APL-02 real-data (workstation-marked)
- [ ] `benchmarks/src/autobench/pipeline/variant_dispatch.py` — APL-02 translation layer module
- [ ] `CLAM_ARGS` convention on `ModelVariant` — document (or optional `@classmethod` hook)

## Manual-Only Verifications

| Behavior | Req | Why Manual | Instructions |
|----------|-----|------------|--------------|
| Real CLAM variant run composite differs from baseline by > noise | APL-02 | Needs `AUTOBENCH_CCRCC_ROOT` workstation data | Run `benchmarks/tests/test_apl02_real_run.py` on workstation; confirm applied-variant composite ≠ baseline |

## Validation Sign-Off
- [ ] All tasks have `<automated>` verify or Wave 0 deps
- [ ] No 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] `nyquist_compliant: true` set when complete

**Approval:** pending
