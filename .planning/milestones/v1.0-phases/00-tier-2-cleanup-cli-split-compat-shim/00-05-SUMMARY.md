---
phase: 00-tier-2-cleanup-cli-split-compat-shim
plan: 05
subsystem: orchestrator
tags: [security, env-whitelist, subprocess, cln-02, threat-model]

# Dependency graph
requires:
  - phase: 00-tier-2-cleanup-cli-split-compat-shim/01
    provides: cli/check.py (so this plan can extend it with whitelist/passthrough visibility)
  - phase: 00-tier-2-cleanup-cli-split-compat-shim/02
    provides: python-dotenv-backed _load_dotenv (so the new whitelist works on top of a sane env-loading layer)
  - phase: 00-tier-2-cleanup-cli-split-compat-shim/03
    provides: NVIDIA_SMI_PATH constant + check.py print line (the file this plan extends)
provides:
  - Explicit subprocess env whitelist replacing `{**os.environ, ...}` at orchestrator._launch
  - `_build_subprocess_env` instance method centralising env construction
  - System-minimal hardcoded whitelist (literal + prefix-glob), config-driven `env.passthrough`, orchestrator-injected fixed keys, per-spec `spec.env` with `_SPEC_ENV_BLOCKED`
  - `automil check` per-key passthrough OK/MISSING report + system-whitelist visibility
  - ccrcc/automil/config.yaml declaring `env.passthrough` with all AUTOBENCH_*_ROOT vars
  - 12 new tests covering system literals/globs, secret non-leak, passthrough hit/miss, WARN-on-missing, blocked-key invariant
affects:
  - phase-08 (DEC-01 will remove the orchestrator-side AUTOBENCH_ROOT injection; ccrcc passthrough is already wired)
  - phase-08 (DEC-05 will introduce env.required as a fail-fast sibling to env.passthrough)
  - phase-01+ (any new env var needed by an experiment now requires an explicit passthrough entry — no silent inheritance)

# Tech tracking
tech-stack:
  added: []  # No new runtime deps; all stdlib (os, frozenset, tuple) + existing yaml.
  patterns:
    - "Subprocess env via explicit whitelist: literal + prefix-glob over os.environ + literal-name passthrough from config"
    - "Module-level constants (frozenset / tuple) for whitelist; instance state (_env_passthrough) for per-config opt-ins"
    - "Operator visibility via `automil check` — surface what experiment subprocesses will receive before they run"
    - "_SPEC_ENV_BLOCKED for keys the orchestrator owns (CUDA_VISIBLE_DEVICES, AUTOMIL_GPU) — per-spec env CANNOT override"

key-files:
  created:
    - tests/test_orchestrator_env_whitelist.py
  modified:
    - src/automil/orchestrator.py
    - src/automil/cli/check.py
    - benchmarks/experiments/ccrcc/automil/config.yaml

key-decisions:
  - "Whitelist literal {PATH, HOME, USER, SHELL, LANG, TZ, TMPDIR, LD_LIBRARY_PATH, PYTHONPATH} + prefix-globs {LC_*, CUDA_*, NVIDIA_*, AUTOMIL_*} verbatim from D-04"
  - "config.yaml: env.passthrough is literal-names-only — no globs at the config layer (operator cannot widen the surface from config; D-04 / T-00-08)"
  - "Missing passthrough vars WARN at orchestrator construction, never block scheduling (D-04 / T-00-10)"
  - "AUTOBENCH_ROOT injection at orchestrator level retained in Phase 0 — Phase 8/DEC-01 owns removal (D-05)"
  - "_SPEC_ENV_BLOCKED = {AUTOMIL_GPU, CUDA_VISIBLE_DEVICES} — per-spec env CANNOT override (T-00-09 GPU-mask spoofing mitigation)"
  - "automil check now reports system whitelist + per-key passthrough OK/MISSING (D-06)"

patterns-established:
  - "Pattern: explicit subprocess env construction via dedicated method; never `{**os.environ, ...}` again"
  - "Pattern: consumer config declares its own env needs via env.passthrough; framework provides the system-minimal floor"

requirements-completed: [CLN-01, CLN-02]

# Metrics
duration: ~25min
completed: 2026-05-01
---

# Phase 00 Plan 05: Subprocess env whitelist (CLN-02) Summary

**System-minimal env whitelist + literal-name passthrough replaces `{**os.environ, ...}` at `_launch`, closing the operator-secret exfiltration vector while keeping ccrcc's AUTOBENCH_*_ROOT references wired through a consumer-declared passthrough list.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-05-01T13:50:00Z (approx — agent start)
- **Completed:** 2026-05-01T14:15:00Z
- **Tasks:** 1 (TDD: RED → GREEN, two commits)
- **Files modified:** 3 + 1 created (4 total)

## Accomplishments

- Replaced the HIGH-severity `env = {**os.environ, ...}` leak at `orchestrator.py:_launch` with a centralised `_build_subprocess_env` helper that assembles env from an explicit whitelist + literal-name passthrough.
- Wired `env.passthrough` config loading at orchestrator construction with a WARN-on-missing log per declared key (never blocks scheduling).
- Extended `automil check` to surface the system whitelist (literal + prefix-glob) and per-key passthrough OK/MISSING — operator can audit what every experiment subprocess will receive before submitting.
- Updated ccrcc/automil/config.yaml to declare `env.passthrough` with `AUTOBENCH_ROOT` + every `AUTOBENCH_*_ROOT` referenced in `benchmarks/datasets/*.yaml`, so when Phase 8/DEC-01 removes the orchestrator-side injection the consumer config is already correct.
- Added 12 tests covering system-literal pass-through, prefix-glob match, secret non-leakage (`OPENAI_API_KEY`, `WANDB_API_KEY`, `GITHUB_TOKEN`, `AWS_SECRET_ACCESS_KEY`), passthrough hit/miss, WARN-on-missing at construction, orchestrator-injected precedence, AUTOBENCH_ROOT phase-0 retention, spec.env override + blocked-key invariant, config without env: section, and PYTHONPATH precedence.

## Task Commits

Each task was committed atomically (TDD — RED then GREEN):

1. **Task 1 RED: failing tests for subprocess env whitelist** - `fa986c8` (test)
2. **Task 1 GREEN: replace os.environ leak with explicit env whitelist** - `b1e9462` (fix)

(SUMMARY commit will be the third.)

## Files Created/Modified

- **`src/automil/orchestrator.py`** (modified)
  - Added module-level `_SYSTEM_ENV_WHITELIST_LITERAL` (frozenset of 9 names) and `_SYSTEM_ENV_WHITELIST_PREFIX` (tuple of 4 prefixes) constants, plus `_SPEC_ENV_BLOCKED` for keys the orchestrator owns.
  - Added `env.passthrough` loading in `__init__` — reads `config.env.passthrough`, validates list-of-strings, warns at construction for any declared var missing from `os.environ`, exposes `self._env_passthrough` for the helper.
  - Added `_build_subprocess_env(*, gpu_id, node_id, archive, spec, pythonpath, worktree_benchmarks) -> dict[str, str]` instance method — assembles env in four ordered layers (system whitelist → config passthrough → orchestrator-injected → per-spec env, with `_SPEC_ENV_BLOCKED` veto on the last).
  - Replaced the inline 13-line env block in `_launch` with a single helper call. AUTOBENCH_ROOT injection retained at line 487 per D-05.
- **`src/automil/cli/check.py`** (modified)
  - Hardened existing config loader (`config: dict = {}` initialised before the if-branch so subsequent reads always have a binding).
  - Imported `_SYSTEM_ENV_WHITELIST_LITERAL` and `_SYSTEM_ENV_WHITELIST_PREFIX` from `automil.orchestrator` and printed both lists.
  - Iterated the loaded `env.passthrough` and printed `<key>: passthrough OK` or `passthrough MISSING` per declared var; rejects non-list values with a WARN; reports `(none declared)` when empty.
- **`benchmarks/experiments/ccrcc/automil/config.yaml`** (modified)
  - Added top-level `env.passthrough` block listing `AUTOBENCH_ROOT`, `AUTOBENCH_CCRCC_ROOT`, `AUTOBENCH_CLWD_ROOT`, `AUTOBENCH_HANCOCK_ROOT`, `AUTOBENCH_OVARIAN_ROOT`, `AUTOBENCH_PLACEHOLDER_ROOT`, `AUTOBENCH_TCGA_LUAD_ROOT` — exactly the var names referenced in `benchmarks/datasets/*.yaml` `data_root: "${...}"` entries.
- **`tests/test_orchestrator_env_whitelist.py`** (created, 170 lines)
  - 12 tests, all green. See "Test cases" below.

## Test Cases

All 12 in `tests/test_orchestrator_env_whitelist.py`, fixture-shared `orch` builds an orchestrator with `env.passthrough: [MY_CUSTOM_VAR, OPTIONAL_MISSING_VAR]` and four pre-set fake secrets:

| # | Test | Asserts |
|---|------|---------|
| 1 | `test_system_literals_pass` | PATH, HOME, USER, SHELL, LANG, TZ, TMPDIR, LD_LIBRARY_PATH all carry their os.environ value |
| 2 | `test_system_prefix_globs_pass` | LC_ALL, LC_CTYPE, CUDA_HOME, CUDA_PATH, NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES, AUTOMIL_FOO all flow through |
| 3 | `test_secrets_do_not_leak` | OPENAI_API_KEY, WANDB_API_KEY, GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY are NOT in the subprocess env |
| 4 | `test_passthrough_present_passes` | MY_CUSTOM_VAR (in passthrough list, present in env) flows through |
| 5 | `test_passthrough_missing_does_not_block` | OPTIONAL_MISSING_VAR is absent; MY_CUSTOM_VAR still made it |
| 6 | `test_passthrough_missing_warns_at_construction` | A WARN log mentioning the missing key is emitted at orchestrator construction (D-04) |
| 7 | `test_orchestrator_injected_vars_always_set` | CUDA_VISIBLE_DEVICES=str(gpu_id), AUTOMIL_GPU=0, AUTOMIL_NODE_ID, AUTOMIL_RESULTS_DIR, AUTOMIL_DESC always present |
| 8 | `test_autobench_root_still_injected_phase0` | env["AUTOBENCH_ROOT"] = str(worktree_benchmarks.resolve()) (D-05 retention) |
| 9 | `test_spec_env_overrides` | spec.env["MY_CUSTOM_VAR"]="spec-wins" wins over the passthrough value |
| 10 | `test_spec_env_cannot_override_blocked_keys` | spec.env attempting to override AUTOMIL_GPU / CUDA_VISIBLE_DEVICES has no effect (T-00-09 mitigation) |
| 11 | `test_config_without_env_section` | Orchestrator constructs cleanly when config.yaml has no `env:` section; `_env_passthrough == []` |
| 12 | `test_pythonpath_overrides_whitelist_value` | Orchestrator-injected PYTHONPATH wins over the whitelisted os.environ value |

Full test suite count: 88 baseline + 12 new = 100 tests, all passing in 4.17s.

## AUTOBENCH_*_ROOT vars discovered in benchmarks/datasets/*.yaml

Grep `${AUTOBENCH_*_ROOT}` across `benchmarks/datasets/*.yaml` yielded:

- `${AUTOBENCH_CCRCC_ROOT}` (ccrcc.yaml)
- `${AUTOBENCH_CLWD_ROOT}` (clwd.yaml)
- `${AUTOBENCH_HANCOCK_ROOT}` (hancock.yaml)
- `${AUTOBENCH_OVARIAN_ROOT}` (ovarian.yaml)
- `${AUTOBENCH_PLACEHOLDER_ROOT}` (templates/placeholder.yaml)
- `${AUTOBENCH_TCGA_LUAD_ROOT}` (tcga_luad.yaml)
- `${AUTOBENCH_TCGA_{CODE}_ROOT}` (templates/tcga_template.yaml — template placeholder; not real var, omitted)

All real vars are in ccrcc's `env.passthrough` plus `AUTOBENCH_ROOT` itself.

## Decisions Made

None new — followed plan and locked decisions D-04 (whitelist shape), D-05 (Phase 0 retains orchestrator-side AUTOBENCH_ROOT), D-06 (`automil check` surfaces resolved whitelist + passthrough) verbatim.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Hardened `cli/check.py` `config` binding**
- **Found during:** Task 1 (Step 6 — appending passthrough report to check.py)
- **Issue:** Existing `cli/check.py` only assigned `config = yaml.safe_load(...)` inside the `else` branch (when `config_path.exists()`). The new passthrough report needs to read `config.env.passthrough` after that branch — referencing `config` when the file is missing would raise `NameError`.
- **Fix:** Initialised `config: dict = {}` before the if-branch and changed the assignment to `config = yaml.safe_load(...) or {}` so a binding always exists. The new code then reads `(config or {}).get("env") or {}` defensively.
- **Files modified:** `src/automil/cli/check.py`
- **Verification:** Smoke-tested `automil check` against a fresh skeleton with and without `env:` section — output renders correctly in both cases.
- **Committed in:** `b1e9462` (Task 1 GREEN commit)

**2. [Rule 2 - Missing critical] Added 12th test for PYTHONPATH precedence**
- **Found during:** Task 1 (Step 1 — TDD test scaffold)
- **Issue:** Plan listed 11 test cases (the 10 enumerated in `<behavior>` plus the explicit "construction-time WARN" case in Step 1 scaffold). Realised PYTHONPATH is a corner case: it's both in the literal whitelist AND orchestrator-injected, so layering must be tested explicitly.
- **Fix:** Added `test_pythonpath_overrides_whitelist_value` — sets `PYTHONPATH=/some/parent/path` in `os.environ`, calls `_build_subprocess_env(pythonpath="/tmp/wt/benchmarks/src")`, asserts the orchestrator-injected value wins.
- **Files modified:** `tests/test_orchestrator_env_whitelist.py`
- **Verification:** Test passes, confirms layering invariant for the one whitelist key that's also orchestrator-injected.
- **Committed in:** `fa986c8` (Task 1 RED commit)

---

**Total deviations:** 2 auto-fixed (1 latent bug in check.py guard; 1 missing-critical test for PYTHONPATH layering edge case)
**Impact on plan:** Both fixes within scope — first prevents `NameError` on bare repos with no config, second hardens the layering invariant for the one shared key. No scope creep; no architectural changes.

## Issues Encountered

- **Worktree branch behind merge-base:** the worktree was checked out to `137aa70` (an ancestor of the expected base `1907990`), so the planning artefacts and Wave 1+2 changes were absent. Resolved per the `<worktree_branch_check>` step — `git reset --hard 1907990` restored the expected state. No data loss.

## TDD Gate Compliance

- **RED gate:** `fa986c8 test(00-05): add failing tests for subprocess env whitelist (CLN-02)` — 12 failing tests asserting the missing helper.
- **GREEN gate:** `b1e9462 fix(orchestrator): replace os.environ leak with explicit env whitelist (CLN-02)` — implementation; all 12 tests + the 88 baseline tests pass.
- **REFACTOR gate:** none required — implementation is clean, no follow-up commit.

## User Setup Required

None. The system whitelist + ccrcc passthrough are wired in code; operator just needs to keep using `benchmarks/.env` as before. New consumer projects must add their own `env.passthrough` list to opt project-specific vars in.

## Next Phase Readiness

- CLN-02 (HIGH-severity) closed. Wave 3 of Phase 0 done.
- Phase 8 / DEC-01 can delete the orchestrator-side `AUTOBENCH_ROOT` injection at line 487 in `_build_subprocess_env`; ccrcc and any other consumer that copies its config will continue working via `env.passthrough`.
- Phase 8 / DEC-05 (`env.required`) can sit alongside `env.passthrough` — the field-name distinction is intentional (D-06).
- No new blockers introduced.

## Self-Check

Verifying claims:

- File `src/automil/orchestrator.py`: FOUND (modified)
- File `src/automil/cli/check.py`: FOUND (modified)
- File `benchmarks/experiments/ccrcc/automil/config.yaml`: FOUND (modified)
- File `tests/test_orchestrator_env_whitelist.py`: FOUND (created, 170 lines)
- Commit `fa986c8` (RED): FOUND in `git log`
- Commit `b1e9462` (GREEN): FOUND in `git log`
- `grep -E '\\{\\*\\*os\\.environ' src/automil/orchestrator.py` → only in comments/docstrings, no live leak code
- `_build_subprocess_env` defined in orchestrator.py: FOUND
- `passthrough` references in cli/check.py: FOUND
- ccrcc config has `env.passthrough` with 7 entries: FOUND
- All 100 tests pass: VERIFIED (`uv run pytest tests/ -v` → 100 passed in 4.17s)

## Self-Check: PASSED

---
*Phase: 00-tier-2-cleanup-cli-split-compat-shim*
*Plan: 05*
*Completed: 2026-05-01*
