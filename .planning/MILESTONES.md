# Milestones

## v1.1 Bug Fixing (Shipped: 2026-06-12)

**Phases completed:** 6 phases (9–14), 23 plans. **20/20 requirements satisfied.** Full framework test suite green (**1058 passed, 0 failed, 53 skipped**). Milestone audit PASSED (6/6 cross-phase integration seams wired). Branch: `milestone/v1.1-bug-fixing`.

**Key accomplishments:**

- **State & recovery integrity (Phase 9, REC):** single-writer terminal state (`terminal_writer.py` — graph → completed → archive result.json → results.tsv in fixed order); partial-fold recovery on SIGTERM/timeout (interrupted runs report aggregated composite, not `0.0`); canonical result-status vocabulary + `result.schema.json` (partial/termination_reason); budget cells keyed by `(dataset, encoder, mil_model)` so re-parenting doesn't reopen a 6h budget.
- **Variant application integrity (Phase 10, APL):** registered variants actually apply to the live model through existing open seams (`active_variant.json` → submit copies to node archive → worktree overlay → iris `train.py` importlib dispatch; CLAM_ARGS translation for autobench); loop-opening variants fail loudly (`requires loop opening — deferred (ISSUE-007/RTA)`) instead of silently no-op'ing.
- **CLI lifecycle & operability (Phase 13, OPS):** `automil cancel` now direct-kills daemon-launched LOCAL jobs by reading pid/pgid from the on-disk running spec (the empty-in-memory-map no-op fixed), PID-reuse-guarded by starttime; new `automil dequeue`; submit transitions existing pending proposals to running; group-level `--project PATH`; `viz start` port config fallback.
- **Config/run fidelity + scheduling (Phases 11–12, CFG/SCH):** config/snapshot values drive runs (None argparse defaults stop masking them) + per-node `submit --override`; GPU `scheduling_policy` knob (best_fit | least_loaded | round_robin); opt-in editable-install overlay guard + `automil check` warning (D-199 honored).
- **Housekeeping (Phase 14, DBT):** pre-D-200 `graph.json` legacy round-trip (migrate-on-read, schema_version-gated, idempotent); the 3 tick_cells tests verified-and-guarded (resolved by Phase 9 CR-01); daemon allowlist anchor switched to stable-text matching (ends line-number brittleness); D-208 `clause_11` fixed → **full suite reaches 100% green**.
- **Quality discipline:** the code-review hard gate caught + closed ship-stoppers in Phases 9/10/12/13 (e.g. cancel's daemon-race + total_proposed drift; OPS-01 PID-reuse on the kill path; variant-overlay propagation). Cross-phase integration audit verified 6/6 seams wired end-to-end (producer/consumer contracts, counter math, command-launch composition).

## v1.0 F2-readiness framework refactor (Shipped: 2026-05-08)

**Phases completed:** 9 phases, 92 plans, 54 tasks

**Key accomplishments:**

- Split the 725-line src/automil/cli.py monolith into a 12-file src/automil/cli/ package using per-command-group, fine organisation; all 62 tests green; user-facing CLI byte-identical.
- `_load_dotenv` now delegates to `python-dotenv`'s `dotenv_values`, fixing silent value-corruption on quoted strings, `export` prefixes, and inline `# comments` that the legacy `partition("=")` parser mishandled.
- 1. [Rule 3 — Blocking] Plan literal code vs. plan verification regex were inconsistent
- Two-section deprecation-shim module shipped empty-but-documented for Phase 1/2/3 future relocations, plus 4 new pytest tests covering importability and shape — zero behavioural change, zero new dependencies.
- System-minimal env whitelist + literal-name passthrough replaces `{
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- 1. [Rule 1 - Bug] test_lifecycle_skeleton.py stub tests blocked GREEN suite
- 1. [Rule 1 - Bug] git rev-parse --verify does not check commit existence
- One-liner:
- One-liner:
- One-liner:
- Before (16 lines):
- One-liner:
- 1. [Rule 2 - Missing Critical Functionality] Added `_kill_experiment()` to ExperimentOrchestrator
- One-liner:
- One-liner:
- 1. [Rule 2 - Missing functionality] MockSLURMBackend
- One-liner:
- Positive-case parametrize — 9 cases across 7 leak classes:
- 1. [Rule 1 - Bug] submit.py auto-detect picks up AGENTS.md as changed file
- 1. [Rule 1 - Bug] `python -m automil.cli` fails — automil.cli is a package
- Cell frozen dataclass + CellStatus str-Enum + atomic JSON IO via tempfile+os.replace — the foundational cells.state module that every Phase 4 cap layer imports
- SIGTERM handler (register_sigterm_flush) with sys.exit(0) flush contract and aggregate_folds pure function bootstrapping the cells package
- One-liner:
- Canonical D-119 aggregate_folds implementation + D-123 reconcile_budget_kill stub with metadata.budget_killed=True tagging
- One-liner:
- 1. [Rule 2 - Missing Logic] fold_count merged into existing training: section
- One-liner:
- One-liner:
- One-liner:
- One-liner:
- GateManifest frozen dataclass (D-137):
- `src/automil/gate/nominate.py`
- One-liner:
- One-liner:
- `src/automil/gate/promote.py`
- `src/automil/cli/gate.py`
- 1. [Rule 1 - Bug] Test T-7 pass path failed with K=2 (Wilcoxon mathematically blocked)
- HTTP API (viz/server.py):
- Pitfall-6 anti-acceptance gate test (D-149): 9-assertion end-to-end held-out isolation verifier + AST-based framework purity guards for gate/ — Phase 5 goal-backward verifier is green
- `tests/gate/test_calibration_pilot_smoke.py`
- pyproject.toml:
- One-liner:
- config.yaml.j2 gains a top-level `backend:` block with TODO_FILL_IN sentinels for required SLURM directives; `automil check` gains `_validate_slurm_directives` (raises `SlurmDirectivesIncompleteError` on TODO/missing keys) and `_validate_ray_backend` (advisory-only Ray reachability check)
- 1. [Rule 1 - Bug] Module-level `import submitit` prevented importing pure helpers without extras
- One-liner:
- Before/After summary (8+ running_dir reference sites):
- `_atomic_write_lines(path: Path, lines: list[str]) -> None`
- One-liner:
- 1. [Rule 1 — Design simplification] Worktree path via Runner convention, not running JSON
- One-liner:
- Generated:
- 1. [Rule 3 - Worktree mismatch] Committed to main repo instead of worktree
- src/automil/cli/submit.py
- Three backend stub additions (Task 1)
- Parametrised test_healthcheck_returns_health_report extends test_contract.py to lock all 4 BCK-01 backends against the D-189 HealthReport shape and NotImplementedError message contract
- LocalBackend.healthcheck() wired into automil init with empirical VRAM quantile_95 from results.tsv vram_gb column, --no-healthcheck CI bypass, and cap/hardware sections stamped in config.yaml.j2
- 1. [Rule 1 - Bug] Consolidated multiline bash command to single line
- [Rule 1 - Adaptation] _required_h2_sections() uses 2 sections instead of 7
- One-liner:
- 1. [Rule 2 - Missing Critical Functionality] TODO-substring check uses YAML-value-level assertion, not raw text
- One-liner:
- Generated:
- 1. [Rule 3 - Blocking] jsonschema not installed in virtual environment
- One-liner:
- Task 1: app.js metric reader (1 line changed)
- One-liner:
- 1. [Rule 1 - Bug] Comment text contained AUTOBENCH_ token triggering purity grep gate
- One-liner:
- 1. [Rule 1 - Bug] --non-interactive flag does not exist
- One-liner:
- Generated:

---
