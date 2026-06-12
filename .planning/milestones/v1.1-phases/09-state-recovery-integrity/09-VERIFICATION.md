---
phase: 09-state-recovery-integrity
verified: 2026-06-11T06:15:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
---

# Phase 9: State & Recovery Integrity — Verification Report

**Phase Goal:** The framework's recorded truth (terminal state, status, budget identity) is correct, single-sourced, and survives mid-run interruption without data loss.
**Verified:** 2026-06-11T06:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | An experiment killed mid-run by SIGTERM/timeout with completed folds on disk reports those folds' aggregated composite in result.json — NOT composite=0.0 | VERIFIED | `runtime_helpers.py` L58-68: reads `AUTOMIL_RESULTS_DIR`, calls `aggregate_folds(target, n)`, writes to archive. `_orchestrator_daemon.py` L1367-1385: `_collect_or_synthesize_result` probes `fold_*_result.json` before synthesizing. `_handle_timeout` L1489: `os.kill(pid, SIGTERM)` main PID first. |
| 2 | Normal completion AND cap-kill completion both write graph.json, completed/<node>.json, archive result.json, and results.tsv through a SINGLE writer; rank and dashboard never disagree | VERIFIED | `terminal_writer.py` L59-207: `write_terminal_state` writes all four in fixed order (graph → completed → archive → TSV). `_handle_completion` L1199-1212 and `_handle_cap_killed_completion` L1288-1298 both delegate to `write_terminal_state`. `reconcile_budget_kill` has NO `write_text` call (grep confirmed clean). |
| 3 | result.json payloads written by framework recovery paths (partial, timeout, oom, crashed) all validate against result.schema.json | VERIFIED | `result.schema.json` enum = `["completed", "crash", "budget_killed", "cancelled", "partial"]`. `termination_reason` optional property added (D-05/D-07). `reconcile.py:90` `_crashed_payload` emits `"status": "crash"` (D-06 — was `"crashed"`). `terminal_writer.py:26-30` `_STATUS_CANON` canonicalizes `"crashed"/"oom"/"timeout"` → `"crash"`. |
| 4 | automil propose and automil submit require a --mil-model field; budget cell key is (dataset, encoder, mil_model); re-parenting does not open a fresh 6h budget for the same MIL model | VERIFIED | `cells/state.py:99-109` `make_cell_id(dataset, encoder, mil_model)`. `cells/state.py:112-125` `normalize_mil_model`. `cells/state.py:187-188` `read_cell` compat shim for legacy `parent_id` cells. `cli/submit.py:38-41,365-384`: `--mil-model` option + D-12 three-step resolution chain + `ClickException` if unresolved. `cli/propose.py:88-124`: `--mil-model` stored in node metadata. `cells/migrate.py:28-100`: mode-aware budget merge for D-15. |

**Score: 4/4 truths verified**

---

### Context Decisions Spot-Check

| Decision | Claim | Status | Evidence |
|----------|-------|--------|----------|
| D-01: partial quarantined from best_node/keep-discard | `best_node()` returns `None` for partial; `_reevaluate_descendants` skips partial children | VERIFIED | `graph.py:155-156`: `if node.get("status") == "partial": return None`. `graph.py:352-353`: `if child.get("status") == "partial": continue`. `terminal_writer.py:134-135`: `graph_status = "partial"` for partial results. |
| D-10: terminal_writer is sole results.tsv writer | `reconcile_budget_kill` does NOT call `write_text` for result.json | VERIFIED | `grep -n "write_text" reconcile.py` returns zero matches. `reconcile_budget_kill` calls `node_archive.mkdir()` then returns payload — comment at L142 explicitly documents D-10. |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/automil/terminal_writer.py` | Sole writer of all four terminal artifacts | VERIFIED | 207 lines; `write_terminal_state` with fixed 6-step write order; `_STATUS_CANON`; `_atomic_write_json`; `_canonicalize` |
| `src/automil/cells/state.py` | `make_cell_id(dataset, encoder, mil_model)`, `normalize_mil_model`, `Cell.mil_model` field, `read_cell` compat shim | VERIFIED | All four present and substantive at L99-190 |
| `src/automil/schemas/result.schema.json` | `partial` in status enum + optional `termination_reason` property | VERIFIED | Enum is `["completed","crash","budget_killed","cancelled","partial"]`; `termination_reason` property with documented rationale |
| `src/automil/cells/reconcile.py` | `_crashed_payload` emits `status="crash"`, no `write_text` in `reconcile_budget_kill` | VERIFIED | L90: `"status": "crash"`. `grep write_text` returns no matches. `_STATUS_CANON` constant at L12. |
| `src/automil/cells/migrate.py` | `migrate_cells(cells_dir, mil_model, dry_run)` with mode-aware merge | VERIFIED | 101 lines; mode-aware merge (agent_active: sum; wall_clock: min started_at); atomic write-before-delete |
| `src/automil/cli/cells.py` | `automil cells migrate` subcommand | VERIFIED | `@cells.command("migrate")` with `--mil-model` (required) and `--dry-run` |
| `src/automil/cli/__init__.py` | `cells` subcommand registered | VERIFIED | L34: `from automil.cli import cells  # noqa: E402,F401  (REC-04 / D-15)` |
| `src/automil/runtime_helpers.py` | SIGTERM flush writes to `AUTOMIL_RESULTS_DIR` | VERIFIED | L58-68: reads env var, validates `is_absolute()`, falls back to `Path.cwd()`, writes to `target`; adds `termination_reason="sigterm"` |
| `src/automil/backends/_orchestrator_daemon.py` | `_collect_or_synthesize_result` fold-first + `_handle_timeout` main-PID-first + both handlers delegate to terminal_writer | VERIFIED | D-03 fold probe at L1367-1385; D-04 main-PID SIGTERM at L1489; `write_terminal_state` calls at L1199 and L1288 |
| `src/automil/cli/submit.py` | `--mil-model` option + D-12 three-step resolution chain | VERIFIED | `@click.option("--mil-model", ...)` at L38; resolution chain at L365-384 with `ClickException` |
| `src/automil/cli/propose.py` | `--mil-model` stored in node metadata | VERIFIED | L88 option; L121-124 stores `normalize_mil_model(mil_model)` in `gnode.metadata` |
| `src/automil/cli/reconcile.py` | `--from-archive NODE_OR_ALL` opt-in refresh; skips running nodes | VERIFIED | `--from-archive` at L30-35; Pitfall 3 guard at L76-77; `locked_update` at L65 |
| `src/automil/graph.py` | D-01 partial quarantine in `best_node` and `_reevaluate_descendants` | VERIFIED | L155-156 `best_node`; L352-353 `_reevaluate_descendants` |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_orchestrator_daemon._handle_completion` | `terminal_writer.write_terminal_state` | `from automil.terminal_writer import write_terminal_state` | WIRED | L1199-1212 — direct writes removed; all four artifacts delegated |
| `_orchestrator_daemon._handle_cap_killed_completion` | `terminal_writer.write_terminal_state` | `from automil.terminal_writer import write_terminal_state` | WIRED | L1288-1298 — direct graph dict mutation removed; delegated |
| `runtime_helpers._handler` | `AUTOMIL_RESULTS_DIR` env var | `os.environ.get("AUTOMIL_RESULTS_DIR")` | WIRED | L58-65: reads, validates `is_absolute()`, uses as write target |
| `_collect_or_synthesize_result` | `aggregate_folds` (fold-first) | `from automil.cells.reconcile import aggregate_folds` | WIRED | L1368-1373 — probes before log-heuristic synthesis |
| `_handle_timeout` | main PID SIGTERM + configurable grace | `os.kill(pid, SIGTERM)` then `time.sleep(grace)` then `os.killpg` | WIRED | L1483-1497 — `timeout_grace_seconds` from config (default 10) |
| `cli/submit.py` | `normalize_mil_model + get_or_create_cell` | `from automil.cells.state import normalize_mil_model` at L378 | WIRED | D-12 resolution chain at L365-384; `get_or_create_cell(..., mil_model=_mil_model_norm)` at L384 |
| `cells/migrate.py` | `make_cell_id + normalize_mil_model + read_cell + write_cell` | `from automil.cells.state import ...` at L17-23 | WIRED | All four state imports present; used substantively in `migrate_cells` |
| `cells/reconcile.reconcile_budget_kill` | terminal_writer (sole archive writer) | write_text REMOVED | WIRED | `grep write_text reconcile.py` returns no results; `mkdir` kept |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| REC-01 partial fold recovery (5 tests) | `uv run pytest tests/test_partial_fold_recovery.py tests/test_collect_or_synthesize.py tests/test_sigterm_flush.py tests/test_handle_timeout.py -q` | All pass (13 tests) | PASS |
| REC-02 single terminal writer (4 tests) | `uv run pytest tests/test_terminal_writer.py tests/test_terminal_writer_consistency.py tests/test_reconcile_from_archive.py -q` | All pass (7 tests) | PASS |
| REC-03 schema validation (5 tests) | `uv run pytest tests/test_result_schema_validation.py tests/test_crashed_canonicalization.py -q` | All pass (8 tests) | PASS |
| REC-04 mil_model cell identity (3 test files) | `uv run pytest tests/test_submit_cell_identity.py tests/cells/test_migrate.py tests/test_mil_model_normalization.py -q` | All pass (14 tests) | PASS |
| All 45 Phase 9 targeted tests | Combined targeted run | 45 passed, 0 failed | PASS |

---

### Requirements Coverage

| Requirement | Phase | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| REC-01 | Phase 9 | SIGTERM/timeout kills aggregate completed folds | SATISFIED | `runtime_helpers.py` D-02 fix; `_orchestrator_daemon.py` D-03 + D-04 fixes; 13 tests GREEN |
| REC-02 | Phase 9 | Single terminal-state writer for all four artifacts | SATISFIED | `terminal_writer.py` `write_terminal_state`; both completion handlers delegate; 7 tests GREEN |
| REC-03 | Phase 9 | Canonical status vocabulary; recovery payloads validate against schema | SATISFIED | `result.schema.json` updated enum; `_crashed_payload` canonicalized; `_STATUS_CANON` in terminal_writer + reconcile; 8 tests GREEN |
| REC-04 | Phase 9 | Budget cells keyed by (dataset, encoder, mil_model); --mil-model required | SATISFIED | `make_cell_id` + `normalize_mil_model` in state.py; `--mil-model` on submit + propose; `automil cells migrate` command; 14 tests GREEN |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No TBD/FIXME/XXX markers, no stubs, no empty returns in Phase 9 source files | — | — |

One note on `reconcile_budget_kill` docstring: the function-level docstring (L109-137) still mentions "Writes archive/<node_id>/result.json" in its description block and a numbered item "3. Writes archive/<node_id>/result.json". This is stale documentation — the write was removed per D-10 and the code confirms no `write_text` call is present. This is a WARNING-level documentation inconsistency only; the actual behavior is correct. The inline comment at L142 correctly documents the D-10 invariant.

| `src/automil/cells/reconcile.py` | 109-137 | Docstring says `reconcile_budget_kill` writes `result.json` to disk — contradicts the D-10 removal | INFO | No behavioral impact; code is correct; only documentation is stale |

---

### Human Verification Required

None — all checks are mechanically verifiable.

---

### Pre-existing Test Failure (out of scope)

`tests/acceptance/test_phase8_acceptance.py::test_d208_clause_11_state_roadmap_complete` — FAILS because it asserts the live `REQUIREMENTS.md` still contains v1.0 `DEC-01..DEC-07` rows. When the `/gsd-new-milestone` workflow rotated `REQUIREMENTS.md` to v1.1 content (with `REC-*` requirement IDs), this test became stale. Its archive fallback only triggers when the live file is absent, not when content changed. This is **pre-existing milestone-rotation debt**, predates Phase 9 (present in the baseline before Plan 01 executed), and is NOT a Phase 9 regression.

---

### Gaps Summary

No gaps. All four ROADMAP success criteria (REC-01 through REC-04) are fully implemented, wired, and green in the test suite. The stale docstring note in `reconcile_budget_kill` is documentation debt only and does not affect behavior.

---

_Verified: 2026-06-11T06:15:00Z_
_Verifier: Claude (gsd-verifier)_
