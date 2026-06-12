---
phase: 10-variant-application-integrity
reviewed: 2026-06-11T18:55:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - src/automil/cli/lifecycle/apply.py
  - benchmarks/src/autobench/pipeline/variant_dispatch.py
  - benchmarks/scripts/run_experiment.py
  - src/automil/registry/variants/model.py
  - examples/sklearn-iris/train.py
findings:
  critical: 2
  warning: 3
  info: 2
  total: 7
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-06-11T18:55:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 10 ships three interlocking pieces: the `automil apply` CLI command
(`apply.py`), the autobench consumer bridge (`variant_dispatch.py`), and
consumer-side wiring in `run_experiment.py` and the iris demo. The
framework-purity invariant (D-206) is clean — no `autobench` references
appear in `src/automil/`. The `ModelVariant` ABC is sound.

Two critical issues were found:

1. **A1 runtime reachability is broken for the forward path.** `apply.py`
   writes `applied_variant.json` into `archive/<node_id>/`. The
   orchestrator's `apply_overlay` copies `overlay_dir = archive/<node_id>/`
   into the *same* node's worktree. But `automil apply <node_id>` is run
   *after* that node's experiment has completed — the canonical use-case is
   "apply a good result, then submit the *next* experiment." The next
   experiment is submitted under a *new* node id (`archive/<new_node_id>/`).
   `applied_variant.json` lives only in `archive/<old_node_id>/` and is
   never copied into `archive/<new_node_id>/`, so it is absent from the
   new worktree. The fallback chain in `variant_dispatch.py` then reads
   from `config.yaml` (gitignored, not in worktree) or from
   `AUTOMIL_VARIANT_MODEL` (only set if the queue spec `env` injection in
   `apply.py` lines 216-231 fires AND the queue file exists). In the common
   case where `apply` is called before the next `submit`, the queue file for
   the new node does not yet exist, the env injection silently no-ops, and
   the variant is inert in production.

2. **iris `train.py` dereferences `_spec.loader` without a null check.**
   `importlib.util.spec_from_file_location` can return `None` for the
   loader when the file extension is unrecognised, and `module_from_spec`
   can produce a module whose loader is `None`, causing an `AttributeError`
   at `_spec.loader.exec_module(_mod)` at runtime.

---

## Critical Issues

### CR-01: `applied_variant.json` never reaches the new experiment's worktree — variant inert in production

**File:** `src/automil/cli/lifecycle/apply.py:204-213`

**Issue:** The intended A1 runtime path is:

```
automil apply <node_id>
  → writes  automil/orchestrator/archive/<node_id>/applied_variant.json
automil submit <new_node_id> ...
  → writes  automil/orchestrator/archive/<new_node_id>/<submitted_files>
  → spec: overlay_dir = "archive/<new_node_id>"
orchestrator runs new experiment
  → apply_overlay copies archive/<new_node_id>/* into worktree
  → variant_dispatch reads worktree/automil/applied_variant.json  ← MISSING
```

The file is written to `archive/<old_node_id>/`, but the new experiment's
`overlay_dir` points at `archive/<new_node_id>/`. `apply_overlay` in
`runner.py` copies *everything* from the overlay directory (except
`spec.json`, `run.log`, `result.json` — line 70 of runner.py) so the file
would transfer if it were in the right archive directory, but it is not.

The queue-spec env injection at lines 216-231 is supposed to act as the
last-resort fallback, but it also fails silently: it patches the queue
file for `<node_id>` (the already-completed node), which the orchestrator
has already consumed. The *new* node's queue spec is written by `submit`
*after* `apply` runs, so the env dict never receives `AUTOMIL_VARIANT_MODEL`
for the new node.

Result: in the primary production workflow, all three read paths in
`variant_dispatch.py` return `None` and the variant is silently not applied.
Tests pass because they inject `applied_variant.json` directly into a
temp worktree that the test controls.

**Fix:** `apply.py` must also write `applied_variant.json` into a
framework-level location that survives across node boundaries and is
reliably present in *every* future worktree. Two correct approaches:

Option A — write to a well-known path inside the overlay that every future
submit picks up automatically:
```python
# In apply.py, after writing the archive copy, ALSO write to a
# framework-level "active variant" file that submit.py explicitly
# copies into every new node's archive:
active_variant_path = adir / "active_variant.json"
_atomic_write_text(active_variant_path, json.dumps(selection, indent=2))
```
Then in `submit.py`, unconditionally copy `active_variant.json` from
`adir` into the new `archive/<new_node>/automil/applied_variant.json`
before writing the queue spec (so the overlay always contains it).

Option B — in `apply.py`, write directly to the overlay path the *next*
submitted node will use. This requires knowing the next node id at apply
time, which is not available, so Option A is the only clean fix.

The `variant_dispatch.py` fallback chain and the `AUTOMIL_VARIANT_MODEL`
env injection are both best-effort mitigations around this root gap; they
do not fix it.

---

### CR-02: `_spec.loader` null dereference in iris `train.py`

**File:** `examples/sklearn-iris/train.py:103-105`

**Issue:** `importlib.util.spec_from_file_location` can return a spec
whose `loader` attribute is `None` (documented Python behaviour when no
loader can be found for the file). `module_from_spec(_spec)` also produces
a module with `loader=None` in that case. The subsequent
`_spec.loader.exec_module(_mod)` raises `AttributeError: 'NoneType' object
has no attribute 'exec_module'` rather than a useful error, crashing the
entire experiment with no result.json written.

Additionally, `spec_from_file_location` can itself return `None` if the
path argument yields no spec at all, making `_mod = _ilu.module_from_spec(None)`
raise `AttributeError` one line earlier.

```python
# Current (lines 103-105) — unsafe
_spec = _ilu.spec_from_file_location(variant_name, _py_files[0])
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
```

**Fix:**
```python
_spec = _ilu.spec_from_file_location(variant_name, _py_files[0])
if _spec is None or _spec.loader is None:
    raise ValueError(
        f"Could not load variant module at {_py_files[0]}: "
        "importlib returned no spec or loader."
    )
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
if not hasattr(_mod, "make_classifier"):
    raise AttributeError(
        f"Variant module {_py_files[0]} has no make_classifier() function."
    )
```

---

## Warnings

### WR-01: `_classify_variant_route` fires AFTER the null-selection guard but calls `scan_variants` twice

**File:** `src/automil/cli/lifecycle/apply.py:45-68`

**Issue:** `scan_variants(variants_root)` is called once in the `loss_name`
branch (line 48) and again unconditionally in the `policy_name` branch
(line 60). `scan_variants` is documented as idempotent, but if the
directory does not exist (e.g., `automil init` not yet run), both calls
silently no-op, meaning a node with a loss or policy variant that is NOT
yet on disk passes the guard and reaches config mutation — the guard was
supposed to raise. The guard is effective only when the variant module is
already registered; it provides no protection against a variant name that
was specified but whose module was never committed.

**Fix:** Document this invariant explicitly in the function docstring, or
add an early check:
```python
if not variants_root.exists():
    logger.warning(
        "_classify_variant_route: variants directory %s does not exist; "
        "loss/policy guard cannot fire. Run `automil refresh-registry` "
        "after committing variant modules.", variants_root
    )
```

---

### WR-02: `variant_dispatch.py` config.yaml fallback silently masks a missing `applied_variant.json`

**File:** `benchmarks/src/autobench/pipeline/variant_dispatch.py:101-122`

**Issue:** The docstring labels the config.yaml fallback "DEPRECATED" and
says "New tests MUST NOT rely on this path." However, because CR-01 means
`applied_variant.json` is never in the worktree for the forward path, the
deprecated fallback becomes the de-facto primary path — but `config.yaml`
is also gitignored and absent from the worktree. This leaves the env-var
fallback as the only active path, and that path only fires if the queue
spec had the env key injected, which also fails (CR-01). The cascading
silent no-ops mean a mis-applied variant produces no error and no
diagnostic: the experiment runs baseline silently.

The DEPRECATED fallback should either be removed (forcing the A1 fix) or
emit a log message at WARNING level that is visible even without debug
logging, not just in the `if variant_name:` guard (line 117-122 logs
WARNING, but only if the fallback *succeeds*, not if it is attempted and
also misses).

**Fix:** At minimum, add a log line when `config.yaml` is attempted but
`model.variant` is also absent:
```python
else:
    logger.warning(
        "apply_model_variant_to_exp_cfg: config.yaml exists but has no "
        "model.variant key (deprecated fallback also missed)."
    )
```

---

### WR-03: `run_experiment.py` hard-codes `automil_dir = Path("automil")` — breaks when cwd is not the worktree root

**File:** `benchmarks/scripts/run_experiment.py:180`

**Issue:** `apply_model_variant_to_exp_cfg(exp_cfg, _Path("automil"))` uses
a bare relative path. The orchestrator sets `cwd=str(wt_path)` when
launching the subprocess (daemon line 904), so this is correct in
production. However, if the script is invoked manually from any other
directory (e.g., `python benchmarks/scripts/run_experiment.py --dataset ...`
from the repo root), `Path("automil")` resolves to
`<repo_root>/automil/` (which may or may not exist and will not contain
`applied_variant.json`). The script has no warning when `automil_dir` does
not exist — `variant_dispatch.py` silently skips the entire dispatch and
runs baseline.

**Fix:** Emit a diagnostic when the automil dir is absent:
```python
_automil_dir = _Path("automil")
if not _automil_dir.exists():
    print(
        "[automil] WARNING: automil/ directory not found in cwd "
        f"({os.getcwd()}); variant dispatch skipped (running baseline).",
        flush=True,
    )
apply_model_variant_to_exp_cfg(exp_cfg, _automil_dir)
```

---

## Info

### IN-01: iris `train.py` imports `json` twice — once at module level (implicit via `json` stdlib) and once inline

**File:** `examples/sklearn-iris/train.py:4,74`

**Issue:** `import json` appears at the top of the file (line 4). Inside
`main()`, the variant dispatch block re-imports it as `import json as _json`
(line 74). The inline import is redundant and the aliasing adds noise
without protection (there is no name collision risk here since `json` is a
stdlib module, not a local name).

**Fix:** Remove `import json as _json` from line 74 and use the top-level
`json` directly:
```python
_sel = json.loads(applied_path.read_text()) or {}
```

---

### IN-02: `variant_dispatch.py` MODEL_VARIANTS lookup key silently falls back to `(None, variant_name)` when `parent_name` is absent

**File:** `benchmarks/src/autobench/pipeline/variant_dispatch.py:155-162`

**Issue:** If the `applied_variant.json` was written without a `"parent"`
key in the model section, `parent_name` is `None`. The registry lookup
`MODEL_VARIANTS.get((None, variant_name))` returns `None` and raises
`ValueError` with the message "Variant not found after scanning". The error
message does not tell the operator that the key being looked up has
`parent=None`, making it hard to diagnose vs. a genuine missing module.

**Fix:** Add the key to the error message:
```python
raise ValueError(
    f"Variant {variant_name!r} (parent={parent_name!r}) not found in registry"
    f" after scanning {variants_root}."
    f" Registry key attempted: {key!r}."
    f" Run `automil refresh-registry` to re-scan the variants directory."
)
```

---

_Reviewed: 2026-06-11T18:55:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
