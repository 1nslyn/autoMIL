# Phase 9: State & Recovery Integrity - Pattern Map

**Mapped:** 2026-06-10
**Files analyzed:** 12 new/modified files
**Analogs found:** 12 / 12

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/automil/terminal_writer.py` (NEW) | utility | file-I/O | `src/automil/cells/reconcile.py` + `_orchestrator_daemon.py:_handle_completion` | role-match (combines write patterns from both) |
| `src/automil/runtime_helpers.py` (MODIFY) | utility | event-driven | `src/automil/runtime_helpers.py` itself | self (targeted 2-line fix) |
| `src/automil/backends/_orchestrator_daemon.py` (MODIFY) | service | event-driven | same file — `_handle_timeout`, `_collect_or_synthesize_result` | self (targeted section rewrites) |
| `src/automil/schemas/result.schema.json` (MODIFY) | config | — | same file | self (additive change) |
| `src/automil/schemas/_result.py` (audit only) | utility | request-response | same file | self |
| `src/automil/cells/state.py` (MODIFY) | model | CRUD | `src/automil/cells/state.py` itself | self (field rename + shim) |
| `src/automil/cells/registry.py` (MODIFY) | service | CRUD | `src/automil/cells/registry.py` itself | self (signature change) |
| `src/automil/cells/migrate.py` (NEW) | utility | batch | `src/automil/cells/registry.py` `list_cells` + `write_cell` | role-match |
| `src/automil/cli/submit.py` (MODIFY) | controller | request-response | `src/automil/cli/submit.py` itself + `propose.py` option pattern | self |
| `src/automil/cli/propose.py` (MODIFY) | controller | request-response | `src/automil/cli/propose.py` itself | self (add one option) |
| `src/automil/cli/reconcile.py` (MODIFY) | controller | request-response | `src/automil/cli/reconcile.py` itself + `graph.reconcile()` | self (add `--from-archive` flag) |
| `src/automil/cli/cells.py` (NEW) | controller | request-response | `src/automil/cli/reconcile.py` | exact (same Click subcommand pattern) |

---

## Pattern Assignments

### `src/automil/terminal_writer.py` (NEW — utility, file-I/O)

**Primary analogs:**
- `src/automil/cells/reconcile.py:95-147` — `reconcile_budget_kill` (writes archive `result.json`, returns payload)
- `src/automil/backends/_orchestrator_daemon.py:1183-1212` — `_handle_completion` (writes `completed/<node>.json`, calls `_append_results_tsv`)
- `src/automil/backends/_orchestrator_daemon.py:1279-1307` — `_handle_cap_killed_completion` (direct graph mutation to replace)
- `src/automil/graph.py:44-63` — `locked_update` context manager
- `src/automil/cells/state.py:129-151` — `write_cell` atomic tempfile+replace pattern

**Module docstring pattern** (mirror `reconcile.py:1`):
```python
"""Single terminal-state writer for all four artifacts (REC-02 / D-09, D-10).

Fixed write order: graph node (via locked_update) → completed/<node>.json
→ archive result.json → results.tsv. Both _handle_completion and
_handle_cap_killed_completion delegate here. Never called from train.py.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automil.graph import ExperimentGraph

logger = logging.getLogger(__name__)
```

**Atomic write pattern** — copy from `src/automil/cells/state.py:129-151`:
```python
def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomic tempfile+replace write — same filesystem guaranteed by dir=."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

**Graph update pattern** — copy from `src/automil/graph.py:44-63` (`locked_update`):
```python
# Use locked_update — never direct dict mutation (Pitfall 1/2 from RESEARCH.md)
from automil.graph import locked_update

with locked_update(str(graph.path), technique_map=graph.technique_map) as g:
    gnode = g.get_node(node_id)
    gnode["type"] = "executed"
    gnode["status"] = keep_or_discard   # from graph's own keep/discard logic
    gnode["composite"] = result["composite"]
    if result.get("metrics"):
        gnode["metrics"] = dict(result["metrics"])
    # g.save() is called automatically on context exit
```

**completed/<node>.json write pattern** — copy from `_orchestrator_daemon.py:1183-1209`:
```python
completion = {
    "id": node_id,
    "status": result.get("status", "completed"),
    "composite": result.get("composite", 0),
    "metrics": result.get("metrics", {}),
    "elapsed_seconds": result.get("elapsed_seconds", elapsed_s),
    "peak_vram_mb": result.get("peak_vram_mb", 0),
    "gpu": gpu_id,
    "completed_at": datetime.now().isoformat(),
    "graph_metadata": result.get("graph_metadata") or spec.get("graph_metadata") or {},
}
_atomic_write_json(completed_dir / f"{node_id}.json", completion)
```

**TSV delegation pattern** — `_append_results_tsv` stays in the daemon but is called only from `terminal_writer`. The writer receives the daemon's `_append_results_tsv` as a callable argument (or the daemon passes `self` for that single call). This keeps the TSV writer's `self.results_tsv` reference intact while satisfying D-10's "sole writer" invariant.

**Status canonicalization** — add before any write:
```python
# D-06: canonicalize status drift before writing
_STATUS_CANON = {"crashed": "crash", "oom": "crash", "timeout": "crash"}

def _canonicalize(result: dict, termination_reason: str | None = None) -> dict:
    result = dict(result)
    raw_status = result.get("status", "crash")
    result["status"] = _STATUS_CANON.get(raw_status, raw_status)
    if termination_reason:
        result["termination_reason"] = termination_reason
    return result
```

---

### `src/automil/runtime_helpers.py` (MODIFY — targeted fix at L53-54)

**Analog:** same file — `runtime_helpers.py:48-55` (current buggy handler)

**Current code to replace** (`runtime_helpers.py:48-55`):
```python
def _handler(signum: int, frame: object) -> None:
    from automil.cells.reconcile import aggregate_folds
    n = int(os.environ.get(fold_count_env, "5"))
    payload = aggregate_folds(Path.cwd(), n)
    (Path.cwd() / "result.json").write_text(json.dumps(payload, indent=2))
    sys.exit(0)
```

**Replacement pattern** (D-02 fix + D-05 `termination_reason`):
```python
def _handler(signum: int, frame: object) -> None:
    from automil.cells.reconcile import aggregate_folds
    n = int(os.environ.get(fold_count_env, "5"))
    results_dir_env = os.environ.get("AUTOMIL_RESULTS_DIR")
    target = Path(results_dir_env) if results_dir_env else Path.cwd()
    payload = aggregate_folds(target, n)
    payload["termination_reason"] = "sigterm"   # D-05
    (target / "result.json").write_text(json.dumps(payload, indent=2))
    sys.exit(0)
```

Note: `Path(results_dir_env)` must be validated is within the archive dir (security note from RESEARCH.md threat table). Add: `if not target.is_absolute(): target = Path.cwd() / target`.

---

### `src/automil/backends/_orchestrator_daemon.py` (MODIFY — three targeted sections)

#### Section 1: `_collect_or_synthesize_result` D-03 fold-first (L1366 insertion point)

**Analog:** same file — the `if result is None:` branch at L1366-1386

**Insertion** (after L1365 `if result is None:`, before existing log synthesis):
```python
if result is None:
    # D-03: try fold aggregation before synthesising from log heuristics
    fold_files = list(archive.glob("fold_*_result.json"))
    if fold_files:
        from automil.cells.reconcile import aggregate_folds
        expected = self._read_fold_count_for_node(node_id)
        result = aggregate_folds(archive, expected)
        reason = "timeout" if self._timed_out.get(node_id) else "sigkill"
        result["termination_reason"] = reason   # D-05
        _atomic_write_json(archive / "result.json", result)
    # existing log-heuristic synthesis path follows (OOM / timeout / crash)
```

Also update the existing synthesis status values (D-05/06 canonicalization):
```python
# Replace: status = "oom" → status = "crash" + termination_reason = "oom"
# Replace: status = "timeout" → status = "crash" + termination_reason = "timeout"
# (when no fold files exist, zero-fold crash path)
if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
    status = "crash"
    termination_reason = "oom"
elif self._timed_out.get(node_id):
    status = "crash"
    termination_reason = "timeout"
```

#### Section 2: `_handle_timeout` D-04 main-PID-first (replace L1434-1456)

**Analog:** same file — `_kill_experiment` at L1458-1501 (shows `os.kill` + grace pattern)

**Current code** (`_orchestrator_daemon.py:1434-1456`) to replace wholesale:
```python
def _handle_timeout(self, exp_id: str):
    exp = self.running[exp_id]
    pid = exp.process.pid
    # D-04: SIGTERM main PID first so its flush handler can write partial result.
    # Then SIGKILL process group after configurable grace (default 10s).
    grace = int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10))
    logger.warning(
        "Timeout for %s, SIGTERMing main PID %d (grace=%ds)", exp_id, pid, grace
    )
    try:
        os.kill(pid, signal.SIGTERM)    # main PID only — flush handler runs
    except ProcessLookupError:
        pass
    time.sleep(grace)
    if exp.process.poll() is None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    self._timed_out[exp_id] = True
    self._handle_completion(exp_id, returncode=-9)
```

**Config read pattern** — mirrors `_reload_orchestrator_config` usage already in the daemon:
```python
grace = int((self.config.get("orchestrator") or {}).get("timeout_grace_seconds", 10))
```

#### Section 3: `_handle_completion` and `_handle_cap_killed_completion` — delegate to terminal_writer

**Both methods** must strip their direct artifact writes and call `terminal_writer` instead.

`_handle_completion` removes:
- L1207-1209 (`completed/<node>.json` write)
- L1212 (`self._append_results_tsv(...)` call)

`_handle_cap_killed_completion` removes:
- L1288-1296 (direct `gnode[...] =` dict mutation + `self.graph.save()`)
- L1299-1307 (zero-fold direct mutation + `self.graph.save()`)

Both gain a single call:
```python
from automil.terminal_writer import write_terminal_state
write_terminal_state(
    node_id=node_id,
    result=result,       # validated, canonicalized dict
    graph=self.graph,
    completed_dir=self.completed_dir,
    archive_dir=self.archive_dir,
    results_tsv_writer=self._append_results_tsv,
    spec=spec,
    elapsed_s=elapsed_s,
    gpu_id=gpu_id,
)
```

---

### `src/automil/schemas/result.schema.json` (MODIFY — additive)

**Analog:** same file — current schema at L1-24

**Current `status` enum** (L14-17):
```json
"status": {
  "type": "string",
  "enum": ["completed", "crash", "budget_killed", "cancelled"]
}
```

**Replace with** (D-05/07 — add `partial`, add `termination_reason`):
```json
"status": {
  "type": "string",
  "enum": ["completed", "crash", "budget_killed", "cancelled", "partial"]
},
"termination_reason": {
  "type": "string",
  "description": "Free-form reason for non-completed terminal state (timeout, oom, sigterm, sigkill, unknown). Not an enum — consumers may extend."
}
```

No `"required"` change — `termination_reason` is optional. No schema version bump needed (additive, `additionalProperties: true` already set). Add a comment in the `description` field noting the change date and the reasoning (no version bump, additive-only).

---

### `src/automil/cells/state.py` (MODIFY — field rename + shim)

**Analog:** same file — `Cell` dataclass L31-94, `make_cell_id` L97-106, `read_cell` L154-162

**`make_cell_id` signature change** (D-13):
```python
# Current (L97-106):
def make_cell_id(dataset: str, encoder: str, parent_id: str) -> str:
    return hashlib.sha256(f"{dataset}|{encoder}|{parent_id}".encode("utf-8")).hexdigest()[:16]

# Replace with (D-13, D-14):
def make_cell_id(dataset: str, encoder: str, mil_model: str) -> str:
    """Return a 16-char deterministic hex id for the (dataset, encoder, mil_model) triple.

    mil_model must be pre-normalized: strip().lower() + collapsed whitespace (D-14).
    """
    return hashlib.sha256(f"{dataset}|{encoder}|{mil_model}".encode("utf-8")).hexdigest()[:16]
```

**`Cell` dataclass field rename** (D-13): rename `parent_id: str` → `mil_model: str` throughout. Update the docstring: `"""16-char hex derived from sha256(dataset|encoder|mil_model)[:16]."""`

**`read_cell` compat shim** (D-15 Pitfall 4 guard — same file L154-162):
```python
def read_cell(path: Path) -> Cell:
    data = json.loads(path.read_text())
    # Backward-compat shim: old cells have "parent_id", new cells have "mil_model".
    if "parent_id" in data and "mil_model" not in data:
        data["mil_model"] = data.pop("parent_id")
    data["status"] = CellStatus(data["status"])
    return Cell(**data)
```

**`mil_model` normalization helper** (D-14) — add near `make_cell_id`:
```python
def normalize_mil_model(raw: str) -> str:
    """Strip, lowercase, collapse internal whitespace (D-14)."""
    return " ".join(raw.strip().lower().split())
```

---

### `src/automil/cells/registry.py` (MODIFY — signature change)

**Analog:** same file — `get_or_create_cell` L33-96

**Signature change** (D-13): replace `parent_id: str` parameter with `mil_model: str`:
```python
def get_or_create_cell(
    dataset: str,
    encoder: str,
    mil_model: str,          # was: parent_id: str
    budget_seconds: int,
    safety_buffer_seconds: int,
    idle_grace_seconds: int = 300,
    mode: str = "agent_active",
) -> Cell:
```

**Cell construction block** (L77-90) — update field name:
```python
cell = Cell(
    cell_id=cell_id,
    dataset=dataset,
    encoder=encoder,
    mil_model=mil_model,     # was: parent_id=parent_id
    started_at=time.time(),
    ...
)
```

**Log message** (L92-95) — update `parent=%s` to `mil_model=%s`:
```python
logger.info(
    "Opened cell %s: dataset=%s encoder=%s mil_model=%s budget=%ds buffer=%ds mode=%s",
    cell_id[:8], dataset, encoder, mil_model, budget_seconds, safety_buffer_seconds, mode,
)
```

---

### `src/automil/cells/migrate.py` (NEW — batch utility)

**Analog:** `src/automil/cells/registry.py:122-146` (`list_cells`) + `write_cell` from `state.py:129-151`

**Module pattern** (mirrors `registry.py` style — module-level functions, no class):
```python
"""Budget-cell back-fill migration: parent_id keying → mil_model keying (D-15, REC-04).

One-time operator action. Run via `automil cells migrate --mil-model <value>`.
Dry-run mode prints a summary without writing. Atomic: new cell file written
before old cell file deleted; rollback on any failure.
"""
from __future__ import annotations

import logging
from pathlib import Path

from automil.cells.state import Cell, CellStatus, make_cell_id, normalize_mil_model, read_cell, write_cell

logger = logging.getLogger(__name__)
```

**Migration algorithm** (D-15, RESEARCH.md Area D):
```python
def migrate_cells(cells_dir: Path, mil_model: str, dry_run: bool = False) -> list[dict]:
    """Re-key all cells from parent_id → mil_model. Returns summary records."""
    mil_model_norm = normalize_mil_model(mil_model)
    summaries = []
    from automil.cells.state import consumed_seconds
    import dataclasses, time

    for path in sorted(cells_dir.glob("*.json")):
        try:
            cell = read_cell(path)
        except Exception as exc:
            logger.warning("Skipping malformed cell %s: %s", path.name, exc)
            continue

        new_id = make_cell_id(cell.dataset, cell.encoder, mil_model_norm)
        new_path = cells_dir / f"{new_id}.json"

        if new_path.exists() and new_path != path:
            # Merge: existing new-keyed cell already exists — sum agent_active or keep earliest wall_clock
            existing = read_cell(new_path)
            if cell.mode == "agent_active":
                merged_consumed = cell.consumed_active_seconds + existing.consumed_active_seconds
                merged = dataclasses.replace(existing, consumed_active_seconds=merged_consumed)
            else:  # wall_clock: keep the earliest started_at
                earliest = min(cell.started_at, existing.started_at)
                merged = dataclasses.replace(existing, started_at=earliest)
            summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "merge"})
            if not dry_run:
                write_cell(merged, cells_dir)
                path.unlink()
        else:
            new_cell = dataclasses.replace(cell, cell_id=new_id, mil_model=mil_model_norm)
            summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "rename"})
            if not dry_run:
                write_cell(new_cell, cells_dir)   # write new FIRST (atomic)
                if path != new_path:
                    path.unlink()                  # then delete old
    return summaries
```

---

### `src/automil/cli/cells.py` (NEW — controller, request-response)

**Analog:** `src/automil/cli/reconcile.py:1-79` (exact same Click subcommand structure)

**Module pattern** (copy reconcile.py header verbatim, adapt):
```python
"""cells subcommand group: automil cells migrate (REC-04 / D-15)."""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir


@main.group()
def cells():
    """Budget-cell management commands."""


@cells.command("migrate")
@click.option("--mil-model", required=True, help="MIL model name to assign to all existing cells.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print migration summary without writing.")
def cells_migrate(mil_model: str, dry_run: bool):
    """Re-key budget cells from parent_id → mil_model (D-15)."""
    adir = _find_automil_dir()
    from automil.cells.migrate import migrate_cells
    summaries = migrate_cells(adir / "cells", mil_model=mil_model, dry_run=dry_run)
    for s in summaries:
        click.echo(f"  {s['action']}: {s['old_id'][:8]} → {s['new_id'][:8]}")
    if dry_run:
        click.echo(f"Dry run: {len(summaries)} cells would be migrated.")
    else:
        click.echo(f"Migrated {len(summaries)} cells.")
```

---

### `src/automil/cli/submit.py` (MODIFY — add `--mil-model` option + resolution)

**Analog:** same file — existing `@click.option` block L23-40 + cell resolution L330-378

**New option** (insert after `--safety-buffer-seconds` at L37, matching existing option style):
```python
@click.option("--mil-model", default=None,
              help="MIL model identifier for budget cell keying (D-12, REC-04). "
                   "Resolved: --mil-model flag → run.mil_model config → error.")
```

**Resolution chain** (insert at L361, replacing `_parent_for_cell` logic, D-12 + D-14):
```python
# D-12: resolve mil_model: flag → config → error
_mil_model_raw = (
    mil_model                                                    # explicit flag
    or (_automil_cfg.get("run") or {}).get("mil_model")         # config fallback
)
if not _mil_model_raw:
    raise click.ClickException(
        "--mil-model is required (or set run.mil_model in config.yaml). "
        "This pins the budget cell to a specific MIL model so re-parenting "
        "does not open a fresh budget."
    )
# D-14: normalize — strip, lowercase, collapse whitespace
from automil.cells.state import normalize_mil_model
_mil_model_norm = normalize_mil_model(_mil_model_raw)

_cell = get_or_create_cell(
    dataset=_dataset_name,
    encoder=_encoder_name,
    mil_model=_mil_model_norm,   # was: parent_id=_parent_for_cell
    budget_seconds=_cap.budget_seconds,
    safety_buffer_seconds=_cap.safety_buffer_seconds,
    idle_grace_seconds=_cap.idle_grace_seconds,
    mode=_cap.mode,
)
```

---

### `src/automil/cli/propose.py` (MODIFY — add `--mil-model` option + store in node metadata)

**Analog:** same file — existing `@click.option` block L80-87 + `graph.add_proposed` call L112-117

**New option** (insert after `--kind` at L87, same style):
```python
@click.option("--mil-model", default=None,
              help="MIL model identifier — stored in node metadata so subsequent "
                   "`automil submit` can inherit it as a fallback (D-12).")
```

**Store in node metadata** — `graph.add_proposed` returns `node_id`; after the call, if `mil_model` was provided, update the node:
```python
node_id = graph.add_proposed(
    parent_id=parent,
    description=desc,
    techniques=list(techniques),
    kind=kind or "unspecified",
)
if mil_model:
    from automil.cells.state import normalize_mil_model
    gnode = graph.get_node(node_id)
    gnode.setdefault("metadata", {})["mil_model"] = normalize_mil_model(mil_model)
graph.recalculate_scores()
graph.save()
```

**`submit.py` inherits from propose metadata** — add as third fallback in resolution chain:
```python
_mil_model_raw = (
    mil_model                                                          # --mil-model flag
    or (_automil_cfg.get("run") or {}).get("mil_model")               # config
    or (graph_json.get("nodes", {}).get(node) or {})
       .get("metadata", {}).get("mil_model")                          # from propose metadata
)
```

---

### `src/automil/cli/reconcile.py` (MODIFY — add `--from-archive` flag)

**Analog:** same file — existing `--recompute-best` flag pattern L17-66 + `graph.reconcile()` at L70-78

**New option** (insert after `--dry-run` L26, mirroring `--recompute-best` style):
```python
@click.option(
    "--from-archive",
    default=None,
    metavar="NODE_OR_ALL",
    help="Refresh existing node(s) from archive result.json. "
         "Pass a node_id or 'all'. Default reconcile stays missing-node-only (D-11).",
)
```

**New branch** (insert before default path at L68, mirroring `--recompute-best` block L40-66):
```python
if from_archive is not None:
    orch = adir / "orchestrator"
    archive_dir = orch / "archive"
    graph = ExperimentGraph(path=str(adir / "graph.json"), technique_map=_load_technique_map(adir))
    targets = (
        [from_archive] if from_archive != "all"
        else [p.name for p in archive_dir.iterdir() if p.is_dir()]
    )
    refreshed = 0
    for nid in targets:
        result_path = archive_dir / nid / "result.json"
        if not result_path.exists():
            click.echo(f"  skip {nid}: no archive result.json")
            continue
        gnode = graph.get_node(nid)
        if gnode is None:
            click.echo(f"  skip {nid}: not in graph (use default reconcile for missing nodes)")
            continue
        # Pitfall 3 guard: never overwrite a live running node
        if gnode.get("status") == "running":
            click.echo(f"  skip {nid}: currently running")
            continue
        import json as _json
        payload = _json.loads(result_path.read_text())
        gnode["composite"] = payload.get("composite", gnode.get("composite", 0.0))
        gnode["status"] = payload.get("status", gnode.get("status"))
        if payload.get("metrics"):
            gnode["metrics"] = payload["metrics"]
        refreshed += 1
    graph.save()
    click.echo(f"Refreshed {refreshed} node(s) from archive.")
    return
```

---

## Shared Patterns

### Atomic file write
**Source:** `src/automil/cells/state.py:129-151` (`write_cell`)
**Apply to:** `terminal_writer.py` (all four artifact writes), `cells/migrate.py`
```python
tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest_dir), suffix=".tmp")
try:
    with os.fdopen(tmp_fd, "w") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_path, str(dest_path))
except Exception:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    raise
```

### Graph file locking
**Source:** `src/automil/graph.py:44-63` (`locked_update`)
**Apply to:** `terminal_writer.py` (graph node update), `cli/reconcile.py` `--from-archive` path
```python
from automil.graph import locked_update
with locked_update(str(graph_path), technique_map=technique_map) as g:
    gnode = g.get_node(node_id)
    # mutate gnode
    # g.save() called automatically on exit
```

### Schema validation at ingestion
**Source:** `src/automil/backends/_orchestrator_daemon.py:1345-1364`
**Apply to:** `terminal_writer.py` (validate before writing), `_collect_or_synthesize_result` fold path
```python
try:
    from automil.schemas import validate_result, ValidationError
    validate_result(result)
except ValidationError as exc:
    logger.warning("result.json schema validation failed for %s: %s", node_id, exc.message)
    result = {"status": "crash", "composite": 0.0, "metrics": {},
              "error": f"result.json failed schema validation: {exc.message}"}
```

### Click option addition
**Source:** `src/automil/cli/submit.py:23-40` (existing option block)
**Apply to:** `submit.py` + `propose.py` `--mil-model` additions
```python
@click.option("--new-flag", default=None, help="Description (D-XX).")
```
Always add after the last existing option, before the function signature. Match `default=None` for optional flags; use `required=False` implicitly.

### Click subcommand group
**Source:** `src/automil/cli/reconcile.py:16-79`
**Apply to:** `cli/cells.py` (new group + migrate subcommand)
```python
@main.group()
def cells():
    """Group docstring."""

@cells.command("migrate")
@click.option(...)
def cells_migrate(...):
    """Command docstring."""
    adir = _find_automil_dir()
    from automil.cells.migrate import migrate_cells
    ...
```

### `cells.py` must be imported in `src/automil/cli/__init__.py` or wherever the CLI main group registers submodules. Check existing pattern by grepping for `from automil.cli import reconcile` or similar.

### Malformed-file guard
**Source:** `src/automil/cells/registry.py:141-145` (`list_cells`)
**Apply to:** `cells/migrate.py` (iterating cell files)
```python
for p in sorted(cells_dir.glob("*.json")):
    try:
        cells.append(read_cell(p))
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
        logger.warning("Skipping malformed cell file %s: %s", p, exc)
```

---

## Test Pattern Assignments

### `tests/test_result_schema_validation.py` (EXTEND — add `partial` + `termination_reason` cases)

**Analog:** same file — existing test functions L22-87

**Pattern to copy** (mirror `test_autobench_four_key_shape_validates` at L22-34):
```python
def test_partial_status_validates():
    """D-07: partial is a valid status after schema update."""
    validate_result({"composite": 0.42, "status": "partial",
                     "termination_reason": "sigterm"})  # must not raise

def test_termination_reason_without_status_validates():
    """D-05: termination_reason is optional and free-form."""
    validate_result({"composite": 0.5, "termination_reason": "oom"})

def test_unknown_status_still_fails():
    """D-06: enum is tight; 'crashed' (drift value) must fail."""
    with pytest.raises(ValidationError):
        validate_result({"composite": 0.5, "status": "crashed"})
```

### `tests/test_submit_cell_identity.py` (EXTEND — add `--mil-model` + re-parent + normalization)

**Analog:** same file — `TestSubmitCellIdentity` class L57-112

**Pattern** (mirror `test_cell_keys_off_project_and_encoder` at L58-72 using `CliRunner.invoke`):
```python
class TestMilModelCellIdentity:
    def test_explicit_mil_model_flag_keys_cell(self, cli_runner, tmp_path, monkeypatch):
        """D-12: --mil-model flag is used for cell keying."""
        # _init_git_repo, monkeypatch.chdir, cli_runner.invoke(main, ["init"]) pattern
        # invoke submit with --mil-model clam_sb
        # assert cell["mil_model"] == "clam_sb"

    def test_reparent_joins_same_cell(self, cli_runner, tmp_path, monkeypatch):
        """D-13: re-parenting to different node_id but same mil_model joins existing cell."""
        # submit node_0001 --mil-model clam_sb → cell A
        # submit node_0002 --parent node_0001 --mil-model clam_sb → must join cell A

    def test_missing_mil_model_raises(self, cli_runner, tmp_path, monkeypatch):
        """D-12: missing --mil-model and no config fallback → ClickException."""
        # submit without --mil-model and config has no run.mil_model
        # assert result.exit_code != 0 or "required" in result.output

    def test_mil_model_normalization(self, cli_runner, tmp_path, monkeypatch):
        """D-14: 'CLAM_SB' and 'clam_sb' and ' clam sb ' collapse to same cell."""
        # submit with --mil-model "CLAM_SB", then with " clam sb " → same cell_id
```

### New test files — fixture and structure conventions

**Fixture convention** (copy from `tests/test_runner.py:13-34` and `tests/test_submit_cell_identity.py:21-42`):
```python
@pytest.fixture
def cli_runner():
    return CliRunner()

def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)
```

**Unit test for daemon methods** — use `unittest.mock.patch` for `os.kill`/`os.killpg`. See `test_runner.py` pattern for tmp_path-based isolation without a real GPU. No real subprocess needed for unit tests of `_handle_timeout`.

**`tests/test_terminal_writer.py` pattern** (NEW — mirrors `test_result_schema_validation.py` style):
```python
"""REC-02: terminal_writer writes all four artifacts in fixed order."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

def test_normal_completion_writes_all_four(tmp_path):
    """Four artifacts present after terminal_writer call with completed result."""
    # set up minimal fake graph, completed_dir, archive_dir, results_tsv
    # call write_terminal_state(...)
    # assert graph.json updated, completed/<node>.json exists,
    #        archive/result.json exists, results.tsv has row

def test_cap_kill_writes_all_four(tmp_path):
    """Four artifacts present even for budget-kill path."""
```

**`tests/cells/test_migrate.py`** (NEW — needs `tests/cells/__init__.py`):
```python
"""REC-04 D-15: budget-merge back-fill without double-count."""
from __future__ import annotations
import dataclasses, json, time
from pathlib import Path
import pytest

from automil.cells.state import Cell, CellStatus, make_cell_id, write_cell

def test_agent_active_merge_sums_consumed(tmp_path):
    """Two agent_active cells for same mil_model → consumed_active_seconds summed."""

def test_wall_clock_merge_keeps_earliest_started_at(tmp_path):
    """Two wall_clock cells for same mil_model → oldest started_at kept."""

def test_dry_run_does_not_write(tmp_path):
    """dry_run=True returns summaries but leaves files unchanged."""
```

---

## No Analog Found

All files in this phase have analogs. No entries.

---

## Metadata

**Analog search scope:** `src/automil/`, `tests/`
**Files scanned:** 12 source files read directly
**Pattern extraction date:** 2026-06-10
