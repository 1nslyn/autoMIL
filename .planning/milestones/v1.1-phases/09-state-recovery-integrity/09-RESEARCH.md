# Phase 9: State & Recovery Integrity - Research

**Researched:** 2026-06-10
**Domain:** Python signal handling, JSON-schema validation, atomic file I/O, cell budget identity
**Confidence:** HIGH — all claims grounded in direct source reading at file:line

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**A. Partial-fold recovery semantics (REC-01)**
- D-01: Partial results quarantined. Mean over completed folds, `status=partial`. Excluded from keep/discard and `best_node` selection; visible in rank/dashboard.
- D-02: `register_sigterm_flush()` aggregates and writes to `AUTOMIL_RESULTS_DIR` (not `Path.cwd()`).
- D-03: `_collect_or_synthesize_result()` MUST try archive fold-aggregation before synthesizing.
- D-04: Timeout becomes main-PID-first: SIGTERM main PID → wait `orchestrator.timeout_grace_seconds` (default 10s) → SIGKILL process group.

**B. Status vocabulary (REC-03)**
- D-05: `status` tight enum = `[completed, crash, budget_killed, cancelled, partial]` + free-form `termination_reason` field.
- D-06: Canonicalize `crashed` → `crash` everywhere.
- D-07: `result.schema.json` updated to allow `partial` + optional `termination_reason`.
- D-08: Partial rows written to `results.tsv` (visibility, not comparability).

**C. Single terminal-state writer (REC-02)**
- D-09: Standalone `terminal_writer` module; fixed write order: graph node → `completed/<node>.json` → archive `result.json` → `results.tsv`. Both `_handle_completion` and `_handle_cap_killed_completion` call it.
- D-10: `terminal_writer` is the sole `results.tsv` writer; updates graph through locked API.
- D-11: `automil reconcile --from-archive [<node>|all]` opt-in refresh of existing nodes. Default reconcile stays missing-node-only.

**D. Budget-cell identity (REC-04)**
- D-12: `--mil-model` required-with-inference: flag → `run.mil_model` config → error.
- D-13: Cell key = `sha256(dataset|encoder|mil_model)` — graph parent lineage separate from budget identity.
- D-14: `mil_model` free-form, normalized: strip + lowercase + collapse whitespace.
- D-15: Back-fill helper (`automil cells migrate` or reconcile step) re-derives `mil_model` and merges elapsed budget from old parent-keyed cells.

### Claude's Discretion
- Grace-window default (10s) — planner may tune.
- Exact `termination_reason` value set beyond `timeout`/`oom`/`sigterm`.
- Module/function naming (`terminal_writer`); whether back-fill is `cells migrate` subcommand or reconcile extension — budget-merge semantics in D-15 must hold.

### Deferred Ideas (OUT OF SCOPE)
- `termination_reason` → viz dashboard rendering (post-v1).
- Opening the closed CLAM training loop (ISSUE-007 / RTA-01/02).
- `graph.json` legacy schema round-trip (DBT-01, Phase 14).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REC-01 | SIGTERM/timeout-killed runs aggregate completed folds into `result.json` instead of `composite=0.0` | D-02: fix `runtime_helpers` write target; D-03: try archive aggregation first; D-04: main-PID-first kill; fold atomicity confirmed safe |
| REC-02 | Single terminal-state writer for all four artifacts | D-09/10: `terminal_writer` module; precise split between `_handle_completion` (L1207/1212) and `_handle_cap_killed_completion` (L1296/1307) documented |
| REC-03 | Canonical status vocabulary; recovery payloads validate against `result.schema.json` | D-05/06/07: schema enum + `termination_reason`; canonicalization in `_crashed_payload` and synthesis path; schema validator already wired at ingestion |
| REC-04 | Budget cells keyed by `(dataset, encoder, mil_model)`; re-parenting does not open fresh budget | D-12/13/14/15: `make_cell_id` signature change; all callers identified; back-fill contract designed |
</phase_requirements>

---

## Summary

Phase 9 fixes four tightly related defects in how the framework records and recovers terminal state. All implementation targets are known from the CONTEXT.md code seam map; this research confirms them against current source, surfaces implementation risks, and answers the five open technical questions raised in the objective.

**The core finding** is that all four decisions are sound and implementable without architectural surgery. The main risks are: (1) the fold-result write in `autobench` is not atomic (plain `write_text`) — a kill mid-fold-write can leave a partial JSON file that `aggregate_folds` will skip via its malformed-file guard (safe but lossy); (2) the `_handle_cap_killed_completion` path does direct `gnode[...] =` dict mutation instead of going through `locked_update` — the new `terminal_writer` must use the locked API consistently; (3) the `Cell` dataclass has a `parent_id` field in its constructor and persisted JSON — the migration helper must handle `read_cell` deserialization for old cells that do not have a `mil_model` field; and (4) the SLURM and Ray backends never call `_handle_timeout` — main-PID-first signaling is a local-backend concern only and does not need a backend-abstraction seam.

**Primary recommendation:** Implement the four changes as four focused waves in a single phase execution, with the `terminal_writer` module as the keystone that unlocks both REC-01 and REC-02 path consolidation.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SIGTERM flush + partial aggregation | Training-process layer (`runtime_helpers.py`) | Orchestrator daemon (`_collect_or_synthesize_result`) | The training script owns its own graceful exit; the daemon is a fallback path |
| Timeout signal delivery | Orchestrator daemon (`_handle_timeout`) | — (local backend only) | `_handle_timeout` is called from `_check_running` which only iterates `self.running` (local Popen processes); SLURM/Ray have their own termination mechanisms |
| Terminal state persistence | Orchestrator daemon (`terminal_writer` — new) | — | Single writer invariant; no other tier writes `results.tsv` |
| Status canonicalization | Schema layer (`automil.schemas`) | `cells/reconcile.py` (`_crashed_payload`) | Canonicalize at the source; schema validates at ingestion |
| Budget-cell identity | CLI submit/propose layer | `cells/state.py` + `cells/registry.py` | Cell key derived at submit time; stored in `automil/cells/<id>.json` |
| Cell migration | New `cells migrate` subcommand or `reconcile` extension | `cells/state.py` (read/write) | Budget-merge is a one-time operator action; not part of normal execution path |

---

## Standard Stack

This phase adds no new dependencies. All tools are already in the installed environment.

### Core (already present)
| Library | Purpose | Location |
|---------|---------|---------|
| `jsonschema` (Draft202012Validator) | Schema validation at ingestion | `src/automil/schemas/_result.py` |
| `signal`, `os` stdlib | SIGTERM delivery, `os.killpg` | `_orchestrator_daemon.py`, `runtime_helpers.py` |
| `tempfile` + `os.replace` / `os.rename` | Atomic file writes | `graph.save()`, `write_cell()` — pattern to follow |
| `fcntl.flock` | Graph file locking | `locked_update` context manager, `graph.py:50-63` |
| `hashlib.sha256` | Deterministic cell ID | `cells/state.py:106` |

### No new packages required
The implementation is pure internal refactoring + schema update. No new `pip` dependencies.

---

## Package Legitimacy Audit

No external packages are added in this phase. Section not applicable.

---

## Architecture Patterns

### System Architecture Diagram

```
Training process (worktree)
  register_sigterm_flush() → on SIGTERM:
    aggregate_folds(AUTOMIL_RESULTS_DIR, n) → writes AUTOMIL_RESULTS_DIR/result.json
    sys.exit(0)

Orchestrator daemon (_check_running loop)
  [local backend only]
  exp.process.poll() → timeout?
    _handle_timeout(exp_id):
      os.kill(main_pid, SIGTERM)          ← D-04 change
      sleep(grace_seconds)
      os.killpg(pgid, SIGKILL) if alive
      self._timed_out[exp_id] = True
      _handle_completion(exp_id, rc=-9)

  _handle_completion / _handle_cap_killed_completion
    ↓
  terminal_writer(node_id, result, graph, paths)   ← D-09 new module
    1. graph.locked_update(node_id, ...)   → graph.json  (atomic via tempfile+rename)
    2. write completed/<node>.json         (atomic via tempfile+rename)
    3. archive result.json                 (atomic via tempfile+rename)
    4. _append_results_tsv(...)            (append-only, no locking needed)

Schema validation
  validate_result(payload)                 ← already wired at L1349
  result.schema.json                       ← add "partial" enum + termination_reason
  _crashed_payload() → "crashed" → canonicalize → "crash"  ← D-06

Cell key migration
  make_cell_id(dataset, encoder, mil_model)   ← D-13 replaces parent_id
  Cell.parent_id field kept for lineage display; cell_id recomputed
  automil cells migrate                       ← back-fill helper
```

### Recommended Project Structure (new files only)

```
src/automil/
├── terminal_writer.py       # D-09: standalone four-artifact writer
├── cells/
│   └── migrate.py           # D-15: back-fill helper (budget merge)
└── cli/
    └── cells.py             # D-15: `automil cells migrate` subcommand
                             # (or extend cli/reconcile.py for --from-archive)
```

---

## Detailed Findings by Decision Area

### Area A: Partial-fold recovery (REC-01)

#### D-02: `register_sigterm_flush` write target

**Current code** (`src/automil/runtime_helpers.py:53-54`):
```python
payload = aggregate_folds(Path.cwd(), n)
(Path.cwd() / "result.json").write_text(json.dumps(payload, indent=2))
```

**Bug confirmed:** The orchestrator sets `AUTOMIL_RESULTS_DIR` to `archive/<node_id>/` and the CLAM runner writes `fold_<i>_result.json` there. But the flush handler reads from and writes to `Path.cwd()` — the worktree directory, not the archive. Folds are never found, so `aggregate_folds(Path.cwd(), n)` returns a `_crashed_payload` with `composite=0.0`. [VERIFIED: source read]

**Fix pattern:**
```python
def _handler(signum, frame):
    from automil.cells.reconcile import aggregate_folds
    n = int(os.environ.get(fold_count_env, "5"))
    results_dir_env = os.environ.get("AUTOMIL_RESULTS_DIR")
    target = Path(results_dir_env) if results_dir_env else Path.cwd()
    payload = aggregate_folds(target, n)
    # Add termination_reason for D-05
    payload["termination_reason"] = "sigterm"
    (target / "result.json").write_text(json.dumps(payload, indent=2))
    sys.exit(0)
```

The `status` returned by `aggregate_folds` for 1..K-1 folds is already `"partial"` (`reconcile.py:73`) — no further canonicalization needed at the write site once the schema accepts `partial`.

#### D-03: `_collect_or_synthesize_result` fold-aggregation first

**Current code** (`_orchestrator_daemon.py:L1322-1386`):
The function calls `self.runner.collect_result(wt_path, archive)` which reads `archive/result.json` if it exists (written by the flush handler) or from the worktree. When that returns `None`, it synthesizes from log heuristics. It does NOT scan for `fold_<i>_result.json` files.

**Gap:** If the flush handler wrote to `Path.cwd()` (the worktree), `collect_result` may find it. But after D-02 is fixed, the flush handler writes to `AUTOMIL_RESULTS_DIR` (the archive). If the process was SIGKILLed before the flush handler could run (e.g. the grace window elapsed), `archive/result.json` will not exist but `fold_<i>_result.json` files may. The D-03 fix inserts a `try aggregate_folds(archive, expected)` probe before the synthesis fallback.

**Fix insertion point** (`_orchestrator_daemon.py:L1366` — the `if result is None:` branch):
```python
if result is None:
    # D-03: try fold aggregation before synthesizing
    fold_files = list(archive.glob("fold_*_result.json"))
    if fold_files:
        from automil.cells.reconcile import aggregate_folds
        expected = self._read_fold_count_for_node(node_id)
        result = aggregate_folds(archive, expected)
        result["termination_reason"] = "timeout" if self._timed_out.get(node_id) else "sigkill"
        (archive / "result.json").write_text(json.dumps(result, indent=2))
    else:
        # original synthesis path ...
```

#### D-04: Main-PID-first timeout signaling (local backend only)

**Current code** (`_orchestrator_daemon.py:L1434-1456`):
```python
os.killpg(os.getpgid(pid), signal.SIGTERM)   # ← whole group SIGTERM
time.sleep(5)
if exp.process.poll() is None:
    os.killpg(os.getpgid(pid), signal.SIGKILL)
```

**Confirmed:** `_handle_timeout` is called ONLY from `_check_running` (L1063), which iterates `self.running` — the in-memory dict of local `subprocess.Popen` processes. SLURM jobs are managed by submitit; Ray jobs by ray actors. Neither calls `_handle_timeout`. **This is local-backend-only. No backend-abstraction seam needed.** [VERIFIED: source read]

**PyTorch DataLoader fragility (confirmed from ISSUE-009 triage):** Sending SIGTERM to the whole process group hits DataLoader worker processes before the main process's handler runs. Workers can be mid-write to shared memory or the fold file at that moment. Sending SIGTERM to the main PID first lets the Python signal handler run to completion (it calls `sys.exit(0)` which flushes Python buffers), then SIGKILL of the group terminates workers after the partial write is safe.

**Implementation note:** The configurable grace window must be read from the orchestrator config each call (the orchestrator already hot-reloads config each tick via `_reload_orchestrator_config`). Default 10s per D-04.

```python
def _handle_timeout(self, exp_id: str):
    exp = self.running[exp_id]
    pid = exp.process.pid
    grace = int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10))
    logger.warning("Timeout for %s, SIGTERMing main PID %d (grace=%ds)", exp_id, pid, grace)
    try:
        os.kill(pid, signal.SIGTERM)          # main PID only — flush handler runs
    except ProcessLookupError:
        pass
    time.sleep(grace)
    if exp.process.poll() is None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)   # whole group
        except ProcessLookupError:
            pass
    self._timed_out[exp_id] = True
    self._handle_completion(exp_id, returncode=-9)
```

**SLURM/Ray backends:** No change needed. SLURM uses `--signal=B:TERM@30` (`slurm.py:L193`) which sends TERM to the batch script 30 seconds before wall-time expiry — the training script's `register_sigterm_flush` handler fires naturally. Ray uses `ray.cancel(force=True)` (`ray.py:L341`). Both paths are already correct for their execution model; they bypass `_handle_timeout` entirely.

#### Per-fold atomicity risk

**Current write** (`autobench/pipeline/clam/runner.py:L70`):
```python
fold_path.write_text(_json.dumps(payload, indent=2))
```

**Risk assessment:** `Path.write_text` is NOT atomic — it opens the file, truncates, writes, closes. A SIGKILL mid-write leaves a truncated JSON file. `aggregate_folds` handles this: the malformed file is skipped with `logger.warning` (`reconcile.py:L55-57`) and the fold is excluded from aggregation. The result is that the fold is silently lost, not that aggregation crashes.

**Verdict:** Acceptable for the current phase. The `aggregate_folds` guard ("Malformed fold files are skipped... NOT silently used as zeros") already defends against this. Making fold writes atomic is a quality improvement for a future phase. The planner should note this limitation in the wave-0 docs.

**Framework-side note:** `_write_fold_result_json` lives in `benchmarks/src/autobench/` (consumer code), not in `src/automil/`. Changing it to use `tempfile+rename` is a legitimate improvement but crosses the framework/consumer boundary. Recommend deferring or flagging as a consumer-advisory note.

---

### Area B: Status vocabulary (REC-03)

#### Current vocabulary audit

| Status value | Source | In schema enum? |
|-------------|--------|----------------|
| `"completed"` | `aggregate_folds`, synthesis, training script | YES |
| `"partial"` | `aggregate_folds` (L73) | NO — **gap** |
| `"crashed"` | `_crashed_payload` (L84) | NO — **gap** |
| `"crash"` | synthesis fallback (L1373), schema validation failure (L1358) | YES |
| `"oom"` | synthesis fallback (L1368-1369) | NO — **gap** |
| `"timeout"` | synthesis fallback (L1370-1371) | NO — **gap** |
| `"budget_killed"` | schema enum | YES (but never emitted by orchestrator paths — only valid for training-script payloads) |
| `"cancelled"` | schema enum | YES |

**Current schema** (`src/automil/schemas/result.schema.json:L14-17`):
```json
"enum": ["completed", "crash", "budget_killed", "cancelled"]
```

**D-05/06/07 implementation:**
1. Schema enum becomes `["completed", "crash", "budget_killed", "cancelled", "partial"]`
2. Add optional `termination_reason` string property (no enum constraint — free-form per D-05)
3. `_crashed_payload` in `reconcile.py:L83-92` emits `"crashed"` → change to `"crash"` (D-06)
4. Synthesis paths already emit `"crash"` — confirmed correct
5. `"oom"` and `"timeout"` synthesis statuses: these are written to `archive/result.json` by the synthesis path (L1381). They should move to `termination_reason` with `status="crash"` for OOM and `status="partial"` (if folds exist) or `status="crash"` (if no folds) for timeout. This is the canonicalization step.

**Backward compatibility:** `result.schema.json` has `"additionalProperties": true` — adding `partial` to the enum and a new optional `termination_reason` property is backward-compatible. Existing `result.json` files with `status="completed"` or `status="crash"` continue to validate. Old files with `status="oom"` or `status="timeout"` or `status="crashed"` will fail validation (they always did) — but these are synthesis-path writes in the archive, not training-script writes. No schema version bump is needed for this phase; the plan should document this reasoning. [VERIFIED: schema + source read]

**The `reconcile.py` path for existing executed nodes** (`graph.reconcile()` L611): when reconciling from `completed/<node>.json`, the code checks `if node and node["type"] == "executed": continue` — existing executed nodes are skipped entirely. So corrected schema validation only applies to newly ingested results.

---

### Area C: Single terminal-state writer (REC-02)

#### Current write split — confirmed

| Artifact | `_handle_completion` | `_handle_cap_killed_completion` |
|----------|---------------------|---------------------------------|
| `graph.json` | NO (missing) | YES via `gnode[...] =` + `self.graph.save()` (L1289-1296, L1299-1307) |
| `completed/<node>.json` | YES (L1207-1209) | NO (missing) |
| archive `result.json` | via `reconcile_budget_kill` → L138 | YES |
| `results.tsv` | YES via `_append_results_tsv` (L1212) | NO (missing) |

**`_handle_completion` additionally does NOT update `graph.json`** — it writes the `completed/<node>.json` notification file which `graph.reconcile()` later reads to update the graph. This means between completion and the next `automil reconcile` run, graph.json is stale.

**`_handle_cap_killed_completion` direct dict mutation** (L1288-1295):
```python
gnode["type"] = "executed"
gnode["status"] = "keep"
gnode["composite"] = payload["composite"]
...
self.graph.save()
```
This bypasses the locked API. It works because the daemon holds the graph in memory, but it is not consistent with the `locked_update` pattern and makes the code hard to test in isolation.

#### `graph.save()` atomicity

**Confirmed** (`graph.py:L975-989`): uses `tempfile.mkstemp` + `os.rename` — atomic POSIX rename. Safe. [VERIFIED: source read]

#### `locked_update` context manager

**Confirmed** (`graph.py:L50-63`): uses `fcntl.flock` (exclusive) on a `.lock` file. The pattern is: acquire lock → load fresh graph → yield → `graph.save()` → release. The terminal_writer should call `graph.save()` inside `locked_update` context to prevent races if multiple daemon workers ever write concurrently (currently not the case for local backend but defensively correct).

#### `_append_results_tsv` write safety

**Confirmed** (`_orchestrator_daemon.py:L1525-1575`): opens in `"a"` (append) mode, no locking. Comment says "no locking needed" — this is safe for the local-backend single-daemon model (only one writer). The terminal_writer preserves this.

**D-08: partial rows in TSV**: currently the `status` column in TSV is written as-is from `result.get("status", "completed")` (L1538). After canonicalization, `"partial"` will be a valid status value. No TSV format change needed — the status column already accepts any string. [VERIFIED: source read]

#### D-11: `reconcile --from-archive` opt-in

**Current `reconcile` CLI** (`cli/reconcile.py:L29-79`): calls `graph.reconcile(...)` which skips existing executed nodes at `graph.py:L611` (`if node and node["type"] == "executed": continue`). Default behavior is preserved.

**New `--from-archive` flag** must: for each archive node dir with a `result.json`, look up the graph node regardless of type, and overwrite its `composite`/`status`/`metrics` from the archive result. The `archive_path` scan at `graph.py:L729-786` already does the recovery logic for missing nodes — the new flag lifts the `if node_id_r not in self.nodes` guard to also process existing nodes when explicitly requested.

**Risk:** Calling `graph.save()` inside the `--from-archive` path must go through the same `locked_update` guard, since an operator could run `reconcile` while the daemon is live.

---

### Area D: Budget-cell identity (REC-04)

#### Current `make_cell_id` signature

**Confirmed** (`cells/state.py:L97-106`):
```python
def make_cell_id(dataset: str, encoder: str, parent_id: str) -> str:
    return hashlib.sha256(f"{dataset}|{encoder}|{parent_id}".encode("utf-8")).hexdigest()[:16]
```

**`Cell` dataclass** (`state.py:L31-94`): has `parent_id: str` field. This field stores the graph node_id of the cell-root (D-108 docstring). After REC-04, this field's semantics change — it will store `mil_model` (not `parent_id`). Two options: (a) rename the field to `mil_model` in the dataclass and schema; (b) keep `parent_id` as the field name but populate it with `mil_model`. Option (a) is cleaner and consistent with D-13 but requires migrating all existing `cells/*.json` files (the `read_cell` deserialization uses `Cell(**data)`). Option (b) is a silent lie in the schema. **Recommendation: option (a) — rename field to `mil_model` and update `read_cell` to handle both keys for backward compatibility.** Flag this as a planner decision.

#### All callers of `make_cell_id` / `get_or_create_cell`

| Caller | Location | What passes as `parent_id` today |
|--------|---------|----------------------------------|
| `submit.py` | L361-371 | `parent if parent else "root"` |
| `registry.py` | L59 | `parent_id` arg to `get_or_create_cell` |
| `registry.py` | `get_or_create_cell` signature | passes through from callers |

**No other callers found** in `src/automil/`. [VERIFIED: grep of source tree]

After D-13, `submit.py` passes `mil_model` (resolved per D-12). `get_or_create_cell` signature changes `parent_id` → `mil_model`. The `Cell` dataclass field changes accordingly.

#### D-12: `--mil-model` resolution chain

**`submit.py` changes needed:**
1. Add `@click.option("--mil-model", default=None, ...)` to the options list (currently at L23-37)
2. Resolution in the body after option parsing: `mil_model = mil_model_flag or config.get("run", {}).get("mil_model") or raise ClickException`
3. Normalize: `mil_model_normalized = " ".join(mil_model.strip().lower().split())`
4. Pass `mil_model_normalized` to `get_or_create_cell` instead of `_parent_for_cell`

**`propose.py` changes needed:** Add `--mil-model` option (L80-87). The cell is not created at propose time (propose only writes a graph node, not a queue spec), BUT the node metadata should carry `mil_model` so submit can inherit it when targeting that proposal. Alternatively, `propose` stores `mil_model` in the graph node's metadata and `submit` reads it as a fallback. This is cleaner than requiring `--mil-model` to be re-specified at submit. **Planner decision: recommend storing in graph node metadata at propose time, used as fallback at submit time.**

#### D-15: Back-fill migration

**Per-cell JSON location:** `automil/cells/<cell_id>.json` — each file is a serialized `Cell` dataclass. The cell_id is `sha256(dataset|encoder|parent_id)[:16]` for old cells.

**Migration algorithm:**
1. Scan all `automil/cells/*.json` (each has `dataset`, `encoder`, `parent_id`, and budget fields).
2. For each cell, find executed graph nodes whose lineage starts from `parent_id` (or whose `metadata.cell_id` equals the old cell_id).
3. Look up those nodes' `spec.json` in `archive/<node_id>/spec.json` to find `run.mil_model` (if stored) or prompt the operator.
4. Compute new `cell_id = make_cell_id(dataset, encoder, mil_model_normalized)`.
5. If new cell file already exists, merge `consumed_active_seconds` (sum) and keep `started_at` of the earlier cell. If new cell does not exist, write it with the old cell's budget fields and accrued time.
6. Remove old cell file.

**Risk: double-counting.** If two old parent-keyed cells for the same `mil_model` are merged, their `consumed_active_seconds` must sum, not overwrite. `wall_clock` mode cells use `now - started_at`; for these, keep the earliest `started_at` and do not sum (the elapsed time is continuous). `agent_active` mode cells use `consumed_active_seconds` accumulator; these must sum. The helper must be mode-aware.

**Operator path (Leo's TCGA-LUAD/CCRCC budget):** The helper should print a dry-run summary first. Since `run.mil_model` is not in existing specs (it doesn't exist yet), the operator may need to supply it via CLI (`automil cells migrate --mil-model clam_sb`). Recommend a single `--mil-model <value>` override that applies to all cells in the overlay, since Leo's experiments are all CLAM-SB variants.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Atomic file write | Custom write-then-rename | `tempfile.mkstemp` + `os.replace` — already used in `graph.save()` and `write_cell()` |
| Graph file locking | Ad-hoc lock file | `locked_update` context manager (`graph.py:L44-63`) — `fcntl.flock` exclusive |
| JSON schema validation | Custom field checks | `jsonschema.Draft202012Validator` — already wired in `automil.schemas` |
| Status normalization | `if/elif` chain at every write site | Single `_canonicalize_status(payload)` function called before any `result.json` write |

---

## Common Pitfalls

### Pitfall 1: `_handle_cap_killed_completion` direct dict mutation
**What goes wrong:** After introducing `terminal_writer`, if `_handle_cap_killed_completion` still does `gnode["type"] = "executed"` + `self.graph.save()` in addition to calling `terminal_writer`, graph.json gets written twice (second write is redundant but correct) or there's a race if `terminal_writer` also calls `save()`.
**How to avoid:** Fully replace the direct mutation in `_handle_cap_killed_completion` with a call to `terminal_writer`. Remove the `gnode[...] =` assignments and `self.graph.save()` from `_handle_cap_killed_completion` (L1286-1307).

### Pitfall 2: `_handle_completion` missing graph.json write
**What goes wrong:** `_handle_completion` today writes `completed/<node>.json` but NOT `graph.json`. If `terminal_writer` only adds the graph write, the old `completed/<node>.json` write also needs to move into `terminal_writer` so neither call site writes anything directly.
**How to avoid:** `terminal_writer` owns all four artifacts. Both `_handle_completion` and `_handle_cap_killed_completion` become thin callers with no direct file writes.

### Pitfall 3: `reconcile --from-archive` overwriting live running nodes
**What goes wrong:** If `automil reconcile --from-archive all` is run while a node is in `running` state, it could overwrite the running node's graph entry with stale archive data.
**How to avoid:** `--from-archive` should skip nodes whose current graph status is `running`. Add a guard: `if existing_node.get("status") == "running": skip`.

### Pitfall 4: Cell `parent_id` field deserialization after rename
**What goes wrong:** Old `cells/*.json` files have `"parent_id": "node_0042"`. If `Cell` dataclass field is renamed to `mil_model`, `Cell(**data)` raises `TypeError: unexpected keyword argument 'parent_id'`.
**How to avoid:** In `read_cell` (`state.py:L154-162`), add a compatibility shim: `if "parent_id" in data and "mil_model" not in data: data["mil_model"] = data.pop("parent_id")`. This must ship before the migration helper runs.

### Pitfall 5: `consumed_active_seconds` double-count in budget merge
**What goes wrong:** Merging two `agent_active` cells by summing their `consumed_active_seconds` is correct. But a `wall_clock` cell's "consumed" is `now - started_at` — summing two `started_at` values and taking the minimum loses precision.
**How to avoid:** Migration helper is mode-aware: `agent_active` → sum `consumed_active_seconds`; `wall_clock` → keep oldest `started_at`, discard the newer cell's started_at.

### Pitfall 6: `_append_results_tsv` called twice if `terminal_writer` is not the only path
**What goes wrong:** If `_handle_completion` still calls `self._append_results_tsv()` after also calling `terminal_writer` (which calls it internally), the TSV gets a duplicate row.
**How to avoid:** Remove `_append_results_tsv` call from `_handle_completion` (L1212) — it moves into `terminal_writer`. The method stays but is only called from `terminal_writer`.

---

## Code Examples

### Existing atomic write pattern to follow
```python
# Source: src/automil/graph.py:L975-989
def save(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")
        os.rename(tmp_path, str(self.path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
```

### Existing cell write pattern to follow
```python
# Source: src/automil/cells/state.py:L129-151
tmp_fd, tmp_path = tempfile.mkstemp(dir=str(cells_dir), suffix=".tmp")
try:
    with os.fdopen(tmp_fd, "w") as f:
        f.write(payload)
    os.replace(tmp_path, str(path))
except Exception:
    try: os.unlink(tmp_path)
    except OSError: pass
    raise
```

### Schema validation pattern (already wired)
```python
# Source: src/automil/backends/_orchestrator_daemon.py:L1346-1364
try:
    from automil.schemas import validate_result, ValidationError
    validate_result(result)
except ValidationError as exc:
    logger.warning("result.json schema validation failed for %s: %s", node_id, exc.message)
    result = {"status": "crash", "composite": 0.0, "metrics": {},
              "error": f"result.json failed schema validation: {exc.message}"}
```

### `locked_update` usage pattern
```python
# Source: src/automil/graph.py:L44-63
from automil.graph import locked_update
with locked_update(graph_path) as g:
    node = g.get_node(node_id)
    node["type"] = "executed"
    node["status"] = keep_discard
    # g.save() called automatically on context exit
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `crashed` status from `_crashed_payload` | Keep as `crash` per schema enum | Canonicalization fixes schema validation gap |
| `oom`/`timeout` as top-level status values | Move to `termination_reason`; top-level `status` stays enum | Cleaner separation of machine-readable enum vs human-readable reason |
| Cell keyed by graph parent | Cell keyed by MIL model | Re-parenting no longer fragments budget |

---

## Runtime State Inventory

> This is a migration-affecting phase (cell key change). All five categories answered explicitly.

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | `automil/cells/<cell_id>.json` — Leo has live TCGA-LUAD and CCRCC cells keyed by parent_id | Data migration via `automil cells migrate` (D-15); merge elapsed budget |
| Live service config | Orchestrator daemon reads `orchestrator.timeout_grace_seconds` from config.yaml — this key does not exist yet in any deployed config.yaml | No migration needed; the key is optional with a default of 10s |
| OS-registered state | None — cells are flat JSON files, not OS registrations | None |
| Secrets/env vars | `AUTOMIL_RESULTS_DIR`, `AUTOMIL_FOLD_COUNT` — read-only env vars injected by orchestrator; no rename | None |
| Build artifacts | None — no compiled artifacts affected by this change | None |

---

## Open Questions (RESOLVED)

> All four resolved via inline recommendations below; all four are implemented by the Phase 9 plans (09-02, 09-05, 09-06, 09-03 respectively). Q2 (propose→submit `--mil-model` inheritance) was confirmed by Leo during plan-check (2026-06-10) and recorded as an approved extension to CONTEXT.md D-12.

1. **Cell dataclass field rename (`parent_id` → `mil_model`)**
   - What we know: renaming the field requires a `read_cell` compatibility shim and the migration helper.
   - What's unclear: should the `Cell` dataclass also keep a `parent_id` field for display purposes (the "cell-root experiment" concept in D-108), or is that information derivable from the graph?
   - **RESOLVED:** drop `parent_id` from `Cell`; it was only used for keying (now replaced by `mil_model`). Graph lineage is in `graph.json`, not the cell. Keep the `read_cell` shim for backward-compat deserialization.

2. **`propose` + `submit` `--mil-model` inheritance**
   - What we know: `propose` creates a graph node but no queue spec; `submit` creates the queue spec and cell.
   - What's unclear: should `--mil-model` at propose time be stored in the graph node metadata so `submit` can inherit it, or must the operator re-specify at submit?
   - **RESOLVED:** store `mil_model` in graph node metadata at propose time. `submit` reads it as a fallback before erroring. This avoids redundant flag specification in the common propose-then-submit workflow.

3. **`terminal_writer` module location**
   - What we know: D-09 says "standalone module, not a daemon-private method."
   - What's unclear: `src/automil/terminal_writer.py` (top-level) vs `src/automil/backends/terminal_writer.py`.
   - **RESOLVED:** `src/automil/terminal_writer.py` — it has no backend dependency; it takes a `graph`, `paths`, and `result` dict. Placing it in `backends/` would imply backend coupling.

4. **Schema version bump**
   - What we know: adding `partial` enum value and `termination_reason` is backward-compatible per `additionalProperties: true`.
   - What's unclear: whether a schema version field (e.g. `"schema_version": 2`) is warranted.
   - **RESOLVED:** do not add a schema version in this phase. The schema change is additive-only; existing consumers can still validate. Schema versioning is a broader concern tied to DBT-01 (Phase 14). Document the reasoning in a code comment.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing, 48+ tests) |
| Config file | `pyproject.toml` (pytest section) |
| Quick run command | `uv run pytest tests/test_result_schema_validation.py tests/test_submit_cell_identity.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REC-01 | SIGTERM flush writes to AUTOMIL_RESULTS_DIR, not cwd | unit | `uv run pytest tests/test_sigterm_flush.py -v` | Wave 0 |
| REC-01 | Fold aggregation before synthesis in `_collect_or_synthesize_result` | unit | `uv run pytest tests/test_collect_or_synthesize.py -v` | Wave 0 |
| REC-01 | Main-PID-first kill: SIGTERM to pid, then SIGKILL to pgid after grace | unit (mock os.kill) | `uv run pytest tests/test_handle_timeout.py -v` | Wave 0 |
| REC-01 | Kill with N completed folds → composite is mean of N folds, not 0.0 | integration | `uv run pytest tests/test_partial_fold_recovery.py -v` | Wave 0 |
| REC-02 | Normal completion writes all four artifacts | unit | `uv run pytest tests/test_terminal_writer.py -v` | Wave 0 |
| REC-02 | Cap-kill completion writes all four artifacts | unit | `uv run pytest tests/test_terminal_writer.py::test_cap_kill_writes_all_four -v` | Wave 0 |
| REC-02 | `automil rank` and `results.tsv` agree after completion | integration | `uv run pytest tests/test_terminal_writer_consistency.py -v` | Wave 0 |
| REC-02 | `reconcile --from-archive` refreshes existing node composite | unit | `uv run pytest tests/test_reconcile_from_archive.py -v` | Wave 0 |
| REC-03 | `partial` status validates against updated schema | unit | `uv run pytest tests/test_result_schema_validation.py -v` | Exists — extend |
| REC-03 | `termination_reason` field validates | unit | `uv run pytest tests/test_result_schema_validation.py -v` | Exists — extend |
| REC-03 | `crashed` canonicalized to `crash` in `_crashed_payload` | unit | `uv run pytest tests/test_crashed_canonicalization.py -v` | Wave 0 |
| REC-03 | `oom`/`timeout` synthesis produces canonical status + termination_reason | unit | `uv run pytest tests/test_collect_or_synthesize.py -v` | Wave 0 |
| REC-04 | `make_cell_id(dataset, encoder, mil_model)` produces deterministic id | unit | `uv run pytest tests/test_submit_cell_identity.py -v` | Exists — extend |
| REC-04 | Re-parenting joins same cell (not new cell) | unit | `uv run pytest tests/test_submit_cell_identity.py::test_reparent_joins_same_cell -v` | Wave 0 |
| REC-04 | `--mil-model` missing with no config fallback → ClickException | unit | `uv run pytest tests/test_submit_cell_identity.py -v` | Wave 0 |
| REC-04 | `mil_model` normalization collapses whitespace, lowercases | unit | `uv run pytest tests/test_mil_model_normalization.py -v` | Wave 0 |
| REC-04 | `automil cells migrate` merges elapsed budget without double-count | unit | `uv run pytest tests/cells/test_migrate.py -v` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_result_schema_validation.py tests/test_submit_cell_identity.py tests/test_terminal_writer.py -v`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_sigterm_flush.py` — covers REC-01 flush write target
- [ ] `tests/test_collect_or_synthesize.py` — covers REC-01 fold-first synthesis + REC-03 status canonicalization
- [ ] `tests/test_handle_timeout.py` — covers REC-01 D-04 main-PID-first signaling
- [ ] `tests/test_partial_fold_recovery.py` — covers REC-01 end-to-end kill simulation
- [ ] `tests/test_terminal_writer.py` — covers REC-02 all four artifacts (normal + cap-kill)
- [ ] `tests/test_terminal_writer_consistency.py` — covers REC-02 rank/TSV agreement
- [ ] `tests/test_reconcile_from_archive.py` — covers REC-02 D-11 opt-in refresh
- [ ] `tests/test_crashed_canonicalization.py` — covers REC-03 `crashed` canonicalization
- [ ] `tests/cells/test_migrate.py` — covers REC-04 budget-merge migration

**Extend existing:**
- `tests/test_result_schema_validation.py` — add `partial` status + `termination_reason` cases
- `tests/test_submit_cell_identity.py` — add re-parent test + `--mil-model` resolution tests + normalization

---

## Security Domain

`security_enforcement: true` in `.planning/config.json`. ASVS Level 1 applies.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Not applicable — no user-facing auth |
| V3 Session Management | No | Not applicable |
| V4 Access Control | No | Internal framework; no multi-user access control |
| V5 Input Validation | Yes | `mil_model` normalization (strip/lowercase); `--mil-model` flag validated before hashing; malformed fold files skipped in `aggregate_folds` |
| V6 Cryptography | No | SHA-256 used for deterministic ID derivation only (not security-sensitive hashing) |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `AUTOMIL_RESULTS_DIR` | Tampering | Validate that `Path(results_dir_env)` is within the archive dir on `register_sigterm_flush` |
| `mil_model` injection via `--mil-model` flag | Tampering | Normalization (strip/lowercase) + cell ID is a fixed-length hex hash; model name never used as a filesystem path |
| Stale cell file after migration failure | Elevation of privilege (budget circumvention) | Migration helper must be atomic: write new cell file before deleting old one; rollback on failure |

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `jsonschema` | Schema validation | Yes (already in pyproject.toml) | per installed env | None needed |
| `uv` | Test runner | Yes | per env | `pip install` |
| `fcntl` | Graph locking | Yes (Linux only, confirmed platform) | stdlib | None — project is Linux-only (PROJECT.md) |
| `signal.SIGTERM`, `os.kill`, `os.killpg` | Timeout signaling | Yes (Linux stdlib) | stdlib | None |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_handle_timeout` is called only for local backend (not SLURM/Ray) | Area A D-04 | If SLURM/Ray have a timeout polling path that also calls `_handle_timeout`, the main-PID-first change needs a backend-type guard |
| A2 | `Cell.parent_id` is not read anywhere else in `src/automil/` other than `cells/` and the CLI callers identified | Area D | If another module reads `cell.parent_id` directly, it breaks after field rename |
| A3 | Leo's live TCGA-LUAD/CCRCC cells use `wall_clock` mode (older cells pre-date `agent_active` default) | Area D D-15 | If some cells are `agent_active`, the migration helper must handle the mode-aware merge correctly — design already accounts for this |

All other claims are VERIFIED from direct source reads.

---

## Sources

### Primary (HIGH confidence — direct source reads)
- `src/automil/runtime_helpers.py` — `register_sigterm_flush` full body (L1-59)
- `src/automil/cells/reconcile.py` — `aggregate_folds`, `_crashed_payload`, `reconcile_budget_kill` (L1-148)
- `src/automil/backends/_orchestrator_daemon.py` — `_handle_timeout` (L1434-1456), `_handle_completion` (L1151-1235), `_handle_cap_killed_completion` (L1258-1321), `_collect_or_synthesize_result` (L1322-1386), `_append_results_tsv` (L1525-1575), `_check_running` (L1050-1067)
- `src/automil/cells/state.py` — `make_cell_id`, `Cell`, `write_cell`, `read_cell` (L1-163)
- `src/automil/cells/registry.py` — `get_or_create_cell`, `list_cells` (L1-156)
- `src/automil/schemas/result.schema.json` — full schema (L1-24)
- `src/automil/schemas/_result.py` — validator implementation (L1-39)
- `src/automil/graph.py` — `locked_update` (L44-63), `save()` (L975-989), `reconcile` existing-node skip (L610-612), archive recovery (L729-786)
- `src/automil/cli/reconcile.py` — full CLI (L1-79)
- `src/automil/cli/submit.py` — cell creation section (L326-378)
- `src/automil/cli/propose.py` — options and body (L80-122)
- `benchmarks/src/autobench/pipeline/clam/runner.py` — `_write_fold_result_json` (L16-70)
- `src/automil/backends/slurm.py` — SIGTERM delivery via `--signal=B:TERM@30` (L103-105, L193)
- `src/automil/backends/ray.py` — cancel path (L340-352)
- `.planning/config.json` — `nyquist_validation: true`, `security_enforcement: true`
- `tests/test_result_schema_validation.py` — existing schema test coverage

---

## Metadata

**Confidence breakdown:**
- Current defect signatures: HIGH — read directly from source files
- Decision feasibility: HIGH — all implementation points confirmed at file:line
- SLURM/Ray backend assessment: HIGH — `_handle_timeout` call chain confirmed as local-only
- Fold atomicity assessment: HIGH — `write_text` non-atomic confirmed; `aggregate_folds` malformed-skip guard confirmed as safe mitigation
- Cell migration design: MEDIUM — the `parent_id` field rename risk (A2) is architectural; mitigated by explicit grep confirming only two call sites

**Research date:** 2026-06-10
**Valid until:** 2026-07-10 (stable codebase, no external dependencies changing)
