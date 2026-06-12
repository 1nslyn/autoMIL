# Phase 12: Scheduling & Overlay Isolation - Research

**Researched:** 2026-06-11
**Domain:** GPU scheduling policy dispatch + editable-install overlay guard
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Add `orchestrator.scheduling_policy` config (`best_fit | round_robin | least_loaded`,
  default `best_fit`). `_find_best_gpu` dispatches on the policy.
- **D-02:** `automil check` WARNS when an editable-installed consumer package path is snapshotted
  (in `files.editable`) WITHOUT a worktree import guard in place.
- **D-03 (D-199 caution):** Generic overlay guard is **opt-in**: `orchestrator.editable_overlay_guard: true`
  (default OFF) adds a daemon-side PYTHONPATH injection. The unconditional injection removed in
  D-199/DEC-01 MUST NOT be reinstated. Autobench's `run_experiment.py:38` consumer prepend STAYS.

### Claude's Discretion
- Round-robin cursor storage (daemon instance attr vs persisted) — researcher/planner choose.
- Exact config key names and how `automil check` detects editable paths without a guard
  (pip `.pth` scan vs `files.editable` inspection) — planner's call. D-02/D-03 shape must hold.

### Deferred Ideas (OUT OF SCOPE)
- Anti-starvation aging in the scheduler — explicitly out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCH-01 | `orchestrator.scheduling_policy` knob (`best_fit \| round_robin \| least_loaded`) so low-VRAM compute-bound jobs no longer over-stack one GPU. | §SCH-01 architecture, `_find_best_gpu` anatomy, config pattern, threading confirmation |
| SCH-02 | Generic daemon-side guard/injection for editable-installed consumer packages + `automil check` warning when editable paths are snapshotted without a worktree import guard. | §SCH-02 D-199 history, editable install detection mechanism, opt-in injection design |
</phase_requirements>

---

## Summary

Phase 12 has two independent fixes. SCH-01 is a clean strategy-dispatch refactor of a single
42-line function (`_find_best_gpu`, L743-766) plus one new config knob and its hot-reload hook.
SCH-02 is the subtler fix: a prior unconditional PYTHONPATH injection was deliberately removed
(D-199/DEC-01) because it silently forced the parent-checkout `benchmarks/src` path into every
experiment, defeating the env-whitelist security fix from Phase 0. The correct replacement is
detect-and-warn (always, via `automil check`) plus config-gated opt-in injection (daemon-side,
default OFF). Both changes are fully contained in `src/automil/` and are consumer-agnostic.

**Primary recommendation:** Implement SCH-01 as a strategy dispatch within `_find_best_gpu` with
a daemon instance attr `self._rr_cursor` (int, plain) for round_robin; implement SCH-02 as
(a) a `check.py` warning that scans `files.editable` entries against editable `.pth` files in
site-packages without a detected guard, and (b) a per-worktree PYTHONPATH prepend in
`_build_subprocess_env` when `orchestrator.editable_overlay_guard: true`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GPU scheduling policy selection | Orchestrator Daemon | Config (hot-reload) | `_find_best_gpu` is the sole placement decision point; policy read from `orchestrator.*` config section, hot-reloaded each tick |
| Round-robin cursor state | Orchestrator Daemon instance | — | Daemon is single-threaded (poll loop); plain `int` instance attr is safe, no locking needed |
| Editable install detection | CLI (`automil check`) | — | Detection is a static analysis step, not a runtime gate; belongs with the pre-flight check command |
| Worktree PYTHONPATH injection | Orchestrator Daemon | Config opt-in | `_build_subprocess_env` already owns all subprocess env construction; injection added here as a config-gated layer |
| Consumer self-protection guard | Consumer entrypoint (`run_experiment.py`) | — | Consumer owns its own import path; framework does not touch it |

---

## SCH-01: GPU Scheduling Policy

### `_find_best_gpu` — Exact Current Logic (VERIFIED: file:line)

**Location:** `src/automil/backends/_orchestrator_daemon.py:743-766`

```python
def _find_best_gpu(self, needed_gb: float) -> int | None:
    gpus = query_gpus()
    candidates = []
    for g in gpus:
        running_on = self.gpu_allocations.get(g.index, [])
        if len(running_on) >= self.max_per_gpu:          # L750: concurrency cap
            continue
        alloc_vram = sum(
            self.running[eid].estimated_vram_gb
            for eid in running_on if eid in self.running
        )
        schedulable = g.free_gb - self.safety_margin_gb - alloc_vram  # L757
        if schedulable >= needed_gb:                      # L758: fits?
            candidates.append((g.index, schedulable))    # L759
    if not candidates:
        return None
    # Best-fit: pick GPU with LEAST schedulable free (tightest fit)  L764
    candidates.sort(key=lambda x: x[1])                  # L765
    return candidates[0][0]                               # L766
```

**Candidate data shape:** `list[tuple[int, float]]` — `(gpu_index, schedulable_free_gb)`.
`schedulable_free_gb = g.free_gb - safety_margin_gb - sum(allocated_vram_for_running_on_that_gpu)`.
[VERIFIED: src/automil/backends/_orchestrator_daemon.py:743-766]

**Call site:** L1760 inside `tick()` — `gpu = self._find_best_gpu(needed_gb)` — single-threaded
poll loop. `needed_gb` comes from `spec.get("estimated_vram_gb", self.default_vram)`. [VERIFIED:
src/automil/backends/_orchestrator_daemon.py:1759-1760]

**Only one call site.** A grep for `_find_best_gpu` in the daemon file confirms L1760 is the
sole call site; no other path bypasses it. [VERIFIED: src/automil/backends/_orchestrator_daemon.py]

### Threading Model — Round-Robin Cursor Is Safe As Plain Attr

The orchestrator daemon is **single-threaded** (one `time.sleep(self.poll_interval)` poll loop
in `run()`; `tick()` executes synchronously per iteration). `_find_best_gpu` is called from
`tick()` only. No locks are needed for a plain `int` cursor. [VERIFIED: ARCHITECTURE.md §Threading
+ src/automil/backends/_orchestrator_daemon.py:1740-1766]

The viz server runs an asyncio loop with a watchdog thread, but it does NOT touch the scheduler
path at all. [VERIFIED: ARCHITECTURE.md §Visualization push]

### Config Pattern to Mirror (VERIFIED: file:line)

Existing `orchestrator.*` knobs read in `__init__` (L433-438) and hot-reloaded in
`_reload_orchestrator_config` (L1699-1738):

```python
# __init__ (L433-438):
orch_cfg = self.config.get("orchestrator", {}) if self.config else {}
self.poll_interval  = orch_cfg.get("poll_interval_sec",         POLL_INTERVAL_SEC)
self.safety_margin_gb = orch_cfg.get("safety_margin_gb",        SAFETY_MARGIN_GB)
self.default_timeout  = orch_cfg.get("default_timeout_min",     DEFAULT_TIMEOUT_MIN)
self.max_per_gpu    = orch_cfg.get("max_concurrent_per_gpu",    MAX_CONCURRENT_PER_GPU)
self.default_vram   = orch_cfg.get("default_vram_estimate_gb",  DEFAULT_VRAM_ESTIMATE_GB)

# _reload_orchestrator_config (L1720-1738) — same keys with log on change:
orch_cfg = (cfg.get("orchestrator") or {}) if isinstance(cfg, dict) else {}
new_max  = orch_cfg.get("max_concurrent_per_gpu", self.max_per_gpu)
# ... assigns self.max_per_gpu = new_max with logger.info on change
```

[VERIFIED: src/automil/backends/_orchestrator_daemon.py:433-438, 1720-1738]

**Pattern for SCH-01:**
1. Add module-level constant `SCHEDULING_POLICY = "best_fit"`.
2. In `__init__`: `self.scheduling_policy = orch_cfg.get("scheduling_policy", SCHEDULING_POLICY)`.
3. In `__init__`: `self._rr_cursor: int = 0` (round-robin cursor, default 0, not hot-reloaded —
   resetting the cursor on config reload would disrupt in-progress round-robin distribution; leave
   it sticky).
4. In `_reload_orchestrator_config`: add hot-reload for `scheduling_policy` with `logger.info`
   on change (same pattern as `max_per_gpu`). The cursor is NOT reset on policy change — if the
   operator switches from `round_robin` to `best_fit` mid-run, the stale cursor is harmless
   because it's only consulted when policy == `round_robin`.
5. Rewrite `_find_best_gpu` to dispatch on `self.scheduling_policy`:
   - `best_fit`: current behavior — `candidates.sort(key=lambda x: x[1])`, return `candidates[0][0]`.
   - `least_loaded`: reverse sort — `candidates.sort(key=lambda x: x[1], reverse=True)`, return
     `candidates[0][0]` (most schedulable free VRAM = least loaded).
   - `round_robin`: sort candidates by `gpu_index` (stable, predictable ordering), then pick the
     one whose index is `>= self._rr_cursor % len(all_gpu_indices)` — or more precisely, pick the
     eligible candidate with the smallest index `>= (self._rr_cursor % n_gpus)`, wrapping around
     if none qualify, then advance `self._rr_cursor`. Implementation detail in §Code Examples.

**Config template addition** (`config.yaml.j2`, `orchestrator:` section):
```yaml
orchestrator:
  # ...existing keys...
  scheduling_policy: "best_fit"  # best_fit | round_robin | least_loaded
```

### Round-Robin Cursor Design (Discretionary)

**Recommendation:** store cursor as `self._rr_cursor: int` initialized to `0` in `__init__`.
Advance it by `1` each time `round_robin` places a job. On each round-robin call, collect eligible
candidate `gpu_index` values (already filtered for VRAM fit + concurrency cap), sort by index,
then pick the one at position `self._rr_cursor % len(candidates)` and increment `self._rr_cursor`.

This is simpler than a global GPU-index cursor (which could advance to a non-eligible GPU):
cycling through the *eligible* candidates in index order ensures the cursor never skips over a
blocked GPU indefinitely. The cursor is intentionally not persisted — daemon restart resets to 0,
which is fine; round-robin fairness is a soft scheduling preference, not a hard invariant.

---

## SCH-02: Editable-Install Overlay Guard

### Why D-199/DEC-01 Removed Unconditional PYTHONPATH Injection (VERIFIED: file:line)

The comment at `_orchestrator_daemon.py:797-799` reads:
```
# D-199 / DEC-01: Consumer-specific env vars and PYTHONPATH overlay
# (formerly injected here in Phase 0) are removed; consumers wire
# them via env.passthrough in automil/config.yaml (D-202).
```

And at `_launch` (L885-888):
```
# CLN-02 / D-04 + DEC-01 / D-199: build env from explicit whitelist +
# config passthrough. Consumer-specific vars (formerly auto-injected
# by this block in Phase 0) are now opted in per project via
# automil/config.yaml: env.passthrough (D-202).
```

[VERIFIED: src/automil/backends/_orchestrator_daemon.py:797-799, 885-888]

**Root cause of removal:** Phase 0 injected `PYTHONPATH = <worktree>/benchmarks/src:...`
unconditionally, which:
1. **Violated framework purity (D-206):** `src/automil/` contained a hardcoded `benchmarks/src`
   path, making the framework depend on the autobench consumer layout. The framework purity test
   (`test_framework_purity.py`) enforces zero `autobench|AUTOBENCH_|benchmarks/` refs in
   `src/automil/` outside the allowlist.
2. **Violated the env-whitelist security fix (CLN-02/D-04):** Phase 0's `env = {**os.environ, ...}`
   blob was replaced by an explicit whitelist. Re-injecting a hardcoded worktree-relative path
   bypassed the new whitelist model and broke the `test_autobench_root_not_auto_injected_phase8`
   and `test_pythonpath_not_auto_injected_phase8` tests. [VERIFIED: tests/test_orchestrator_env_whitelist.py:135-196]
3. **Was consumer-specific:** the `benchmarks/src` path is autobench's layout; a generic MIL
   consumer (e.g. sklearn-iris) doesn't have that structure and would receive a broken PYTHONPATH.

**Consequence for SCH-02:** The generic guard CANNOT re-introduce a hardcoded consumer path.
The opt-in injection must be **generic**: it prepends the worktree-relative equivalent of the
**editable source root** that the framework can discover by inspecting the consumer's own editable
install metadata — NOT a hardcoded `benchmarks/src`.

### Editable Install Detection Mechanism (VERIFIED: filesystem inspection)

Modern `pip install -e .` (PEP 660, uv) creates a `_editable_impl_<pkg>.pth` file in
site-packages. For this project:

- `.venv/lib/python3.11/site-packages/_editable_impl_autobench.pth` → content: `/home/jma/Documents/yinshuol/autoMIL/benchmarks/src`
- `.venv/lib/python3.11/site-packages/_editable_impl_automil.pth` → content: `/home/jma/Documents/yinshuol/autoMIL/src`

[VERIFIED: filesystem — `_editable_impl_autobench.pth` content confirmed by direct read]

The `.pth` file contains the **main checkout** path that Python adds to `sys.path` at interpreter
startup. When a git worktree is created, it does NOT replicate the `.venv/` (only tracked files
are in the worktree), so this `.pth` still points at the **main checkout**, not the worktree —
which is why worktree overlays to `benchmarks/src/autobench/` are silently ignored unless
the worktree's `benchmarks/src` is prepended to `PYTHONPATH` first. [VERIFIED: ARCHITECTURE.md
§"Silent reliance on a parent venv pip install -e ."]

**Older pip** (setup.py develop) creates `<pkg>.egg-link`; very old uv may use
`__editable__<pkg>*.pth` (underscore-underscore prefix vs single-underscore-impl). Scanning
for both patterns is safe and backward-compatible.

Additionally, the `dist-info/direct_url.json` for editable installs contains
`{"dir_info": {"editable": true}}`. [VERIFIED: `.venv/lib/python3.11/site-packages/autobench-0.1.0.dist-info/direct_url.json`]

### How `files.editable` Connects to the Detection

`automil/config.yaml: files.editable` declares which files the agent may overlay. When a consumer
submits a file under `benchmarks/src/autobench/` (or any path under a package's editable source
root), the overlay could be shadowed by the editable install. The check warning fires when:

1. At least one path in the snapshotted overlay (`files.editable` or auto-detected) falls under
   the editable source root of an installed editable package, AND
2. No worktree import guard is detectable in the consumer's run script (the `run.script` or
   `run.command` entrypoint doesn't contain the `sys.path.insert(0, ...)` pattern, AND
   `orchestrator.editable_overlay_guard` is `false`/absent).

**Detection algorithm for `automil check`:**

```
1. Collect all editable source roots:
   scan site.getsitepackages() + [site.getusersitepackages()] for
   - _editable_impl_*.pth   (uv / modern pip PEP 660)
   - __editable__*.pth      (older pip PEP 660 variant)
   - *.egg-link             (legacy setup.py develop)
   Read each file's content to get the source root path.

2. For each editable source root R:
   - Check if any entry in config files.editable starts with R (or is under R).
   - If yes: this consumer overlays files from an editable-installed package.

3. Check if a guard is present (any of):
   - orchestrator.editable_overlay_guard == true in config  (framework-side opt-in)
   - run script contains "sys.path.insert" (consumer self-protection pattern)

4. If overlaying editable paths AND no guard: emit WARNING.
```

[VERIFIED: autobench pattern at benchmarks/scripts/run_experiment.py:37-38]
[ASSUMED: the `sys.path.insert` heuristic for consumer guard detection — simple text search is the
practical approach; a more sophisticated AST check is unnecessary given the narrow target.]

### Opt-In Injection Design (D-03)

**Config key:** `orchestrator.editable_overlay_guard: false` (default OFF, honoring D-199 caution).

**Where injected:** `_build_subprocess_env` at L836 (end of method, after all existing layers).
This is the correct location — it's after the hardcoded orchestrator vars (layer 3) and per-spec
env (layer 4), but the injection should be treated as layer 3.5: it overrides the passthrough
`PYTHONPATH` if any, because the worktree path must win. [VERIFIED: src/automil/backends/_orchestrator_daemon.py:836]

**What to inject:** When `editable_overlay_guard` is true AND the daemon can identify an editable
source root R (via the site-packages `.pth` scan above), AND the worktree contains R's equivalent
path, prepend `<worktree>/<relative_path_of_R_from_project_root>` to the existing `PYTHONPATH`
(or set it if absent). This is the generic version of the autobench-specific
`benchmarks/src` prepend.

**Concrete injection logic:**
```python
if self.editable_overlay_guard:
    wt_pythonpath_prepends = []
    for editable_root in self._editable_source_roots:  # computed at __init__
        # editable_root is absolute (main checkout); compute relative to project_root
        try:
            rel = Path(editable_root).relative_to(self.project_root)
            wt_rel = wt_path / rel   # wt_path passed in by _launch
        except ValueError:
            continue  # editable root not under project_root; skip
        if wt_rel.exists():
            wt_pythonpath_prepends.append(str(wt_rel))
    if wt_pythonpath_prepends:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(wt_pythonpath_prepends + ([existing] if existing else []))
```

**Note:** `_build_subprocess_env` currently does not receive `wt_path`. The planner must decide
whether to (a) pass `wt_path` as an additional parameter to `_build_subprocess_env`, or (b) apply
the injection in `_launch` after `_build_subprocess_env` returns. Option (b) is cleaner — it
keeps `_build_subprocess_env` as a pure env-dict builder and the injection as a post-processing
step in `_launch` where `wt_path` is already available (L872).

**Framework purity constraint:** The site-packages scan and PYTHONPATH injection logic must
contain zero references to `autobench`, `AUTOBENCH_`, or `benchmarks/`. [VERIFIED:
tests/test_framework_purity.py — the grep gate will catch any slip]

### `automil check` Warning — Where to Attach

`check.py:125` is the command entry point. The warning belongs in the existing `warnings.append`
pattern, after the `files.editable` check (currently at L163-166). The detection runs after
config is loaded (L138). [VERIFIED: src/automil/cli/check.py:125-406]

**The new check block template:**
```python
# SCH-02: warn when editable-installed package paths are snapshotted
# without a worktree import guard (ISSUE-010 / D-02).
editable_roots = _collect_editable_source_roots()  # pure fn, no I/O side effects
run_script_path = git_root / (config.get("run", {}).get("script") or "train.py")
run_command = config.get("run", {}).get("command")
has_consumer_guard = _has_consumer_guard(run_script_path, run_command)
overlay_guard_enabled = (config.get("orchestrator") or {}).get(
    "editable_overlay_guard", False)
for root in editable_roots:
    root_p = Path(root)
    for editable_glob in editable:   # editable already loaded at L164
        if (git_root / editable_glob).is_relative_to(root_p) or \
           root_p.is_relative_to(git_root) and ...:
            if not has_consumer_guard and not overlay_guard_enabled:
                warnings.append(
                    f"files.editable includes paths under editable-installed "
                    f"package root {root!r}. Worktree overlays to this path "
                    f"may be shadowed by the parent-venv editable install. "
                    f"Fix: add sys.path.insert(0, <worktree_src>) to your run "
                    f"script, or set orchestrator.editable_overlay_guard: true "
                    f"in automil/config.yaml."
                )
```

(The planner should refine the glob-matching logic; the above captures the intent.)

### D-199 Test Compatibility

The existing tests at `test_orchestrator_env_whitelist.py:135-196` assert:
- `AUTOBENCH_ROOT` is NOT auto-injected (`test_autobench_root_not_auto_injected_phase8`)
- `PYTHONPATH` is NOT force-set to a worktree-relative path (`test_pythonpath_not_auto_injected_phase8`)

The opt-in guard (default OFF) does NOT trigger in these tests because `editable_overlay_guard`
defaults to `false`. New tests for SCH-02 must use fixtures that set `editable_overlay_guard: true`
explicitly — they will not conflict with the existing D-199 tests. [VERIFIED:
tests/test_orchestrator_env_whitelist.py:178-196]

---

## Architecture Patterns

### System Architecture Diagram

```
tick() [single-threaded poll loop]
  │
  └─► _reload_orchestrator_config()
        reads orchestrator.scheduling_policy → self.scheduling_policy
        reads orchestrator.editable_overlay_guard → self.editable_overlay_guard
  │
  └─► for each pending spec:
        needed_gb = spec.estimated_vram_gb or self.default_vram
        gpu = _find_best_gpu(needed_gb)   ← DISPATCH on scheduling_policy
              │
              ├─ best_fit:     sort candidates by schedulable ASC  → tightest fit
              ├─ least_loaded: sort candidates by schedulable DESC → emptiest GPU
              └─ round_robin:  pick candidates[_rr_cursor % len] by index order
                               advance _rr_cursor by 1
        if gpu is not None:
          _launch(spec, gpu)
            │
            └─► _build_subprocess_env(gpu, node_id, archive, spec) → env dict
            │     (layers 1-4 as before)
            │
            └─► [SCH-02 opt-in] if editable_overlay_guard:
                  prepend worktree-relative editable src roots to PYTHONPATH in env
            │
            └─► subprocess.Popen(cmd, env=env, cwd=wt_path)

automil check [CLI, pre-flight]
  │
  └─► _collect_editable_source_roots()  ← scan site-packages for .pth / egg-link
  └─► compare against config files.editable
  └─► if overlap AND no guard: warnings.append(SCH-02 warning)
```

### Recommended Project Structure (changes only)

```
src/automil/
├── backends/
│   └── _orchestrator_daemon.py   # _find_best_gpu dispatch + _rr_cursor + editable guard inject
├── cli/
│   └── check.py                  # SCH-02 warning + _collect_editable_source_roots helper
└── templates/
    └── config.yaml.j2            # scheduling_policy + editable_overlay_guard keys added
tests/
├── test_scheduling_policy.py     # NEW: SCH-01 policy dispatch tests
└── test_editable_overlay_guard.py # NEW: SCH-02 detect-and-warn + opt-in injection tests
```

---

## Standard Stack

### Core (no new dependencies)

Both SCH-01 and SCH-02 use only stdlib and already-imported modules.

| Module | Already Used | Purpose in Phase 12 |
|--------|-------------|---------------------|
| `site` (stdlib) | No (new use in check.py) | `site.getsitepackages()` + `site.getusersitepackages()` for editable root scan |
| `pathlib.Path` (stdlib) | Yes | Path arithmetic for relative editable root detection |
| `os` (stdlib) | Yes | `os.environ`, `PYTHONPATH` construction |

No new pip dependencies. [VERIFIED: all used modules are stdlib or already in pyproject.toml]

### Package Legitimacy Audit

> No new packages are installed in this phase. All changes use stdlib + existing dependencies.

| Package | Verdict | Disposition |
|---------|---------|-------------|
| (none) | — | No new packages |

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Editable install path discovery | Custom `.dist-info` parser | `site.getsitepackages()` + glob `_editable_impl_*.pth` / `__editable__*.pth` / `*.egg-link` | The `.pth` file content IS the source root; one `.read_text()` gives the path |
| Consumer guard detection | Full AST parse of run script | Simple `"sys.path.insert" in script_content` text search | Same heuristic already used in `check.py:148` for `"result.json" in script_content`; sufficient for a warning |
| Policy object/enum | Full strategy class hierarchy | `if/elif` dispatch inside `_find_best_gpu` | 42-line function; strategy classes are over-engineering for 3 branches |

---

## Common Pitfalls

### Pitfall 1: Resetting `_rr_cursor` when `scheduling_policy` changes during hot-reload
**What goes wrong:** If `_reload_orchestrator_config` resets `self._rr_cursor = 0` when policy
changes, a switch from `round_robin` → `best_fit` → `round_robin` introduces a silent cursor reset
that disrupts any in-flight fairness expectation.
**Why it happens:** It seems "clean" to reset state on policy change.
**How to avoid:** Do NOT reset `_rr_cursor` in `_reload_orchestrator_config`. The cursor is a
monotonically-advancing int; policy-switch just stops consulting it.

### Pitfall 2: Injecting unconditional PYTHONPATH in `_build_subprocess_env` (repeating D-199)
**What goes wrong:** `test_pythonpath_not_auto_injected_phase8` and `test_autobench_root_not_auto_injected_phase8` fail. Framework purity gate may also fire.
**Why it happens:** Developer adds injection without checking the `editable_overlay_guard` flag.
**How to avoid:** Guard the injection with `if self.editable_overlay_guard:`. Default is `False`.
Keep the injection in `_launch` AFTER `_build_subprocess_env` returns (not inside it), so the
guard flag is checked only at launch time and existing tests don't need `wt_path` fixtures.
**Warning signs:** Either of the two D-199 env whitelist tests turns red.

### Pitfall 3: Framework purity gate firing on new `benchmarks/` or `autobench` references
**What goes wrong:** `test_framework_purity.py::test_framework_purity_no_autobench_refs` fails if
any `benchmarks/` path or `autobench` string appears in `src/automil/`.
**Why it happens:** Comment or log message uses consumer-specific terminology; or editable-root
scan code mentions `benchmarks/src` as an example.
**How to avoid:** Keep SCH-02 code entirely generic. The `.pth` scan reads paths from the
filesystem at runtime — no consumer path appears in source. Log messages must not mention
`autobench` or `benchmarks/`. If an informational comment is unavoidable, add it to
`_ALLOWLIST` in `test_framework_purity.py`. [VERIFIED: tests/test_framework_purity.py:43-67]

### Pitfall 4: `round_robin` cycling through ALL GPU indices, not just eligible candidates
**What goes wrong:** Cursor points to a GPU that is at `max_per_gpu` capacity or has insufficient
VRAM; `_find_best_gpu` returns `None` even though other GPUs have space.
**Why it happens:** Cursor implementation advances over the full GPU index space, not the
filtered-eligible candidates list.
**How to avoid:** Filter candidates first (same VRAM + concurrency checks as best_fit), then
round-robin among the *eligible* set sorted by index.

### Pitfall 5: `_collect_editable_source_roots()` called at every `check` invocation — expensive on large venvs
**What goes wrong:** `site.getsitepackages()` might return large directories with many `.pth`
files; the scan is slow.
**Why it happens:** Unbounded glob.
**How to avoid:** Filter to only `_editable_impl_*.pth`, `__editable__*.pth`, and `*.egg-link`
(three narrow patterns). Typical venv has < 20 such files. Not a real concern at this scale but
worth noting for correctness.

### Pitfall 6: `editable_overlay_guard` scan reads site-packages at construction time — stale on new installs
**What goes wrong:** If the operator runs `pip install -e .` after the daemon starts, the daemon's
cached `_editable_source_roots` is stale.
**Why it happens:** Reading site-packages at `__init__` and caching.
**How to avoid:** Compute `_editable_source_roots` at each `_launch` call (cheap glob scan),
not in `__init__`. Or if caching is preferred, re-scan during `_reload_orchestrator_config`.
Recommendation: compute at `_launch` time — it's called infrequently and the scan is trivial.

---

## Code Examples

### SCH-01: Refactored `_find_best_gpu` with policy dispatch

```python
# Source: verified against src/automil/backends/_orchestrator_daemon.py:743-766
def _find_best_gpu(self, needed_gb: float) -> int | None:
    """Pick a GPU according to self.scheduling_policy."""
    gpus = query_gpus()
    candidates: list[tuple[int, float]] = []  # (gpu_index, schedulable_free_gb)

    for g in gpus:
        running_on = self.gpu_allocations.get(g.index, [])
        if len(running_on) >= self.max_per_gpu:
            continue
        alloc_vram = sum(
            self.running[eid].estimated_vram_gb
            for eid in running_on if eid in self.running
        )
        schedulable = g.free_gb - self.safety_margin_gb - alloc_vram
        if schedulable >= needed_gb:
            candidates.append((g.index, schedulable))

    if not candidates:
        return None

    policy = self.scheduling_policy
    if policy == "least_loaded":
        # Most schedulable free VRAM = least loaded GPU
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    elif policy == "round_robin":
        # Cycle through eligible GPUs in stable index order
        candidates.sort(key=lambda x: x[0])  # sort by gpu index
        chosen = candidates[self._rr_cursor % len(candidates)][0]
        self._rr_cursor += 1
        return chosen
    else:
        # best_fit (default): tightest fit — preserves current behavior
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
```

### SCH-02: Editable source root scanner (pure function for `check.py`)

```python
# Source: derived from verified .pth file inspection
import site
from pathlib import Path

def _collect_editable_source_roots() -> list[str]:
    """Return editable source root paths from site-packages .pth / egg-link files.

    Scans for three editable-install file patterns:
      - _editable_impl_*.pth   (uv / pip PEP 660, modern)
      - __editable__*.pth      (older pip PEP 660 variant)
      - *.egg-link             (legacy setup.py develop)
    Returns the source root path strings (one per editable package).
    """
    roots: list[str] = []
    site_dirs = site.getsitepackages()
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs = site_dirs + [user_site]

    for site_dir in site_dirs:
        p = Path(site_dir)
        if not p.is_dir():
            continue
        for pattern in ("_editable_impl_*.pth", "__editable__*.pth", "*.egg-link"):
            for pth_file in p.glob(pattern):
                try:
                    content = pth_file.read_text().strip()
                    if content and Path(content).is_dir():
                        roots.append(content)
                except OSError:
                    continue
    return roots
```

### SCH-02: Opt-in PYTHONPATH injection in `_launch` (post-`_build_subprocess_env`)

```python
# Source: design derived from _launch at src/automil/backends/_orchestrator_daemon.py:837+
# Applied AFTER env = self._build_subprocess_env(...) and BEFORE subprocess.Popen(...)
if self.editable_overlay_guard:
    prepends: list[str] = []
    for editable_root in _collect_editable_source_roots():
        root_p = Path(editable_root)
        try:
            rel = root_p.relative_to(self.project_root)
        except ValueError:
            continue  # editable root not under project root; skip (framework-agnostic)
        wt_candidate = wt_path / rel
        if wt_candidate.is_dir():
            prepends.append(str(wt_candidate))
    if prepends:
        existing_pp = env.get("PYTHONPATH", "")
        parts = prepends + ([existing_pp] if existing_pp else [])
        env["PYTHONPATH"] = ":".join(parts)
        logger.debug(
            "editable_overlay_guard: prepended %d path(s) to PYTHONPATH for %s",
            len(prepends), node_id,
        )
```

---

## State of the Art

| Old Approach | Current Approach | Changed | Impact for Phase 12 |
|--------------|-----------------|---------|---------------------|
| Unconditional PYTHONPATH injection (Phase 0) | D-199 removal; consumers use `env.passthrough` | Phase 8 (D-199/DEC-01) | SCH-02 opt-in guard must respect this; existing tests enforce it |
| `env = {**os.environ, ...}` blob | Whitelist + passthrough (CLN-02/D-04) | Phase 0 cleanup | Any PYTHONPATH injection goes through the same env dict, after whitelist |
| Single best-fit policy | Policy knob (SCH-01) | Phase 12 (this) | Default unchanged; opt-in for round_robin/least_loaded |

---

## Validation Architecture

`workflow.nyquist_validation: true` in `.planning/config.json`. Validation section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (uv run pytest) |
| Config file | `pyproject.toml` (no separate pytest.ini) |
| Quick run command | `uv run pytest tests/test_scheduling_policy.py tests/test_editable_overlay_guard.py -v` |
| Full suite command | `uv run pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCH-01 | `best_fit` picks tightest-fit GPU (current behavior) | unit | `uv run pytest tests/test_scheduling_policy.py::test_best_fit_picks_tightest -x` | ❌ Wave 0 |
| SCH-01 | `least_loaded` picks emptiest GPU | unit | `uv run pytest tests/test_scheduling_policy.py::test_least_loaded_picks_emptiest -x` | ❌ Wave 0 |
| SCH-01 | `round_robin` cycles across eligible GPUs in index order | unit | `uv run pytest tests/test_scheduling_policy.py::test_round_robin_cycles_eligible -x` | ❌ Wave 0 |
| SCH-01 | `round_robin` wraps cursor correctly across calls | unit | `uv run pytest tests/test_scheduling_policy.py::test_round_robin_cursor_wraps -x` | ❌ Wave 0 |
| SCH-01 | `scheduling_policy` hot-reloaded by `_reload_orchestrator_config` | unit | `uv run pytest tests/test_scheduling_policy.py::test_policy_hot_reload -x` | ❌ Wave 0 |
| SCH-01 | Unknown policy string falls back to `best_fit` (defensive) | unit | `uv run pytest tests/test_scheduling_policy.py::test_unknown_policy_fallback -x` | ❌ Wave 0 |
| SCH-01 | `_rr_cursor` not reset when policy changes during hot-reload | unit | `uv run pytest tests/test_scheduling_policy.py::test_cursor_not_reset_on_policy_change -x` | ❌ Wave 0 |
| SCH-02 | `automil check` warns when `files.editable` overlaps editable source root and no guard | unit | `uv run pytest tests/test_editable_overlay_guard.py::test_check_warns_missing_guard -x` | ❌ Wave 0 |
| SCH-02 | `automil check` suppresses warning when `editable_overlay_guard: true` | unit | `uv run pytest tests/test_editable_overlay_guard.py::test_check_no_warn_when_guard_enabled -x` | ❌ Wave 0 |
| SCH-02 | `automil check` suppresses warning when run script has `sys.path.insert` | unit | `uv run pytest tests/test_editable_overlay_guard.py::test_check_no_warn_when_consumer_guard_present -x` | ❌ Wave 0 |
| SCH-02 | Opt-in injection prepends worktree editable root to PYTHONPATH when flag is true | unit | `uv run pytest tests/test_editable_overlay_guard.py::test_opt_in_injection_prepends_pythonpath -x` | ❌ Wave 0 |
| SCH-02 | Opt-in injection is a no-op (default OFF) — D-199 tests still pass | unit | `uv run pytest tests/test_orchestrator_env_whitelist.py -x` | ✅ existing |
| SCH-02 | Framework purity gate passes (no autobench/benchmarks/ in new SCH-02 code) | lint | `uv run pytest tests/test_framework_purity.py -x` | ✅ existing |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_scheduling_policy.py tests/test_editable_overlay_guard.py tests/test_orchestrator_env_whitelist.py tests/test_framework_purity.py -x`
- **Per wave merge:** `uv run pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_scheduling_policy.py` — covers SCH-01 (7 tests)
- [ ] `tests/test_editable_overlay_guard.py` — covers SCH-02 (5 new tests)

*(Existing `tests/test_orchestrator_env_whitelist.py` and `tests/test_framework_purity.py` already
cover the D-199 invariants that SCH-02 must not break.)*

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1` in `.planning/config.json`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | partial | Subprocess env whitelist (CLN-02/D-04) must not be weakened; `_SPEC_ENV_BLOCKED` prevents GPU-mask spoofing |
| V5 Input Validation | yes | `scheduling_policy` value from config must be validated against known strings before use (fall back to `best_fit` on unknown) |
| V6 Cryptography | no | — |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Config injection via `scheduling_policy: <arbitrary>` | Tampering | Treat unknown value as `best_fit` with a `logger.warning`; never `eval()` the policy string |
| PYTHONPATH hijack via `spec.env` | Elevation of Privilege | `PYTHONPATH` is in `_SYSTEM_ENV_WHITELIST_LITERAL` (L58) so it passes through the whitelist, but `_SPEC_ENV_BLOCKED` does NOT block it. The opt-in injection in `_launch` overrides whatever spec.env sets — which is correct behavior for the guard. [VERIFIED: src/automil/backends/_orchestrator_daemon.py:56-66] |
| Editable source root outside project_root | Tampering | `_collect_editable_source_roots` filter with `root_p.relative_to(self.project_root)` — skip roots that don't resolve under the project; `ValueError` is caught and the root is skipped |

---

## Project Constraints (from CLAUDE.md)

| Directive | Relevance to Phase 12 |
|-----------|----------------------|
| Address Leo at start of every response | Agent-level; note for executor |
| Plan mode for non-trivial tasks | Planner must produce PLAN.md before execution |
| Framework purity (D-206) | All new `src/automil/` code must pass `test_framework_purity.py`; zero `autobench/benchmarks/` refs |
| Keep suite green | Full `uv run pytest tests/ -v` must pass at phase gate |
| New behavior gets tests | SCH-01 policy dispatch + SCH-02 detect-and-warn + opt-in injection all require new test files |
| Minimal impact / no temporary fixes | SCH-01 changes one function + one config knob; SCH-02 changes one function + one CLI check block |
| Result contract: `results.tsv` sole writer is orchestrator | Not touched by this phase |
| `_recover_orphans()` only in `run()` | Not touched by this phase |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `sys.path.insert` text search is sufficient to detect consumer guard presence in run script | SCH-02 detection | Low — check.py already uses the same heuristic for `result.json`; a false negative means no warning suppression, which is safe (too many warnings, not too few) |
| A2 | Computing editable source roots at `_launch` time (not cached in `__init__`) is the right lifecycle point | SCH-02 injection | Low — scan is trivial (< 20 `.pth` files); worst case is a slightly stale cache if computed at init |

---

## Open Questions (RESOLVED)

> Both resolved inline below and implemented by the plans (12-03): post-processing injection in `_launch`; editable-detection limitation documented in the advisory warning.

1. **`_build_subprocess_env` signature change vs. `_launch` post-processing** — RESOLVED (post-processing in `_launch`).
   - What we know: `_build_subprocess_env` currently takes `gpu_id, node_id, archive, spec`.
     `wt_path` is needed for the injection but is not currently a parameter.
   - What's unclear: pass `wt_path` to `_build_subprocess_env` (changes signature + all test
     fixtures) vs. apply the injection as a 3-line post-processing block in `_launch` after
     `env = self._build_subprocess_env(...)` returns.
   - Recommendation: **post-processing in `_launch`** (option b). Avoids changing the
     `_build_subprocess_env` signature, keeps existing env whitelist tests untouched, and the
     injection logically belongs in `_launch` where `wt_path` is in scope.

2. **`automil check` editable root detection when running outside the project venv** — RESOLVED (advisory; document as known limitation).
   - What we know: `site.getsitepackages()` returns paths for the current Python interpreter.
     If `automil check` is run with a different interpreter than the project venv, the scan may
     find different (or no) `.pth` files.
   - What's unclear: Whether this matters in practice (operators typically run `uv run automil check`).
   - Recommendation: Document as a known limitation in the warning message. The check is advisory
     (warning, not issue), so a false-negative is acceptable.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `site` (stdlib) | SCH-02 editable root scan | ✓ | stdlib | — |
| `pathlib` (stdlib) | Both | ✓ | stdlib | — |
| `uv run pytest` | Tests | ✓ | (project standard) | — |

No missing dependencies.

---

## Sources

### Primary (HIGH confidence)
- `src/automil/backends/_orchestrator_daemon.py:743-766` — `_find_best_gpu` exact logic
- `src/automil/backends/_orchestrator_daemon.py:797-799, 885-888` — D-199/DEC-01 removal comment
- `src/automil/backends/_orchestrator_daemon.py:1699-1738` — `_reload_orchestrator_config` pattern
- `src/automil/backends/_orchestrator_daemon.py:433-438` — config knob read pattern in `__init__`
- `src/automil/backends/_orchestrator_daemon.py:1759-1766` — sole `_find_best_gpu` call site
- `src/automil/cli/check.py:125-406` — full `check` command, warning pattern, where to attach
- `benchmarks/scripts/run_experiment.py:32-38` — consumer self-protection pattern
- `tests/test_orchestrator_env_whitelist.py:135-196` — D-199 invariant tests SCH-02 must not break
- `tests/test_framework_purity.py:43-67` — purity gate allowlist
- `.venv/lib/python3.11/site-packages/_editable_impl_autobench.pth` — editable install mechanism confirmed
- `.venv/lib/python3.11/site-packages/autobench-0.1.0.dist-info/direct_url.json` — editable flag confirmed

### Secondary (MEDIUM confidence)
- `.planning/codebase/ARCHITECTURE.md` — threading model, env-var propagation history
- `.planning/codebase/CONCERNS.md` — PYTHONPATH/AUTOBENCH_ROOT historical mistake
- `tasks/test-run-issues.md` — ISSUE-005, ISSUE-010, ISSUE-021 history

---

## Metadata

**Confidence breakdown:**
- SCH-01 (scheduling policy dispatch): HIGH — all anchors verified in source, pattern mirrors existing config hot-reload exactly
- SCH-02 (detect-and-warn): HIGH — editable `.pth` mechanism confirmed on actual venv, D-199 history confirmed from code comments + test assertions
- SCH-02 (opt-in injection): HIGH — injection site and guard flag design confirmed against existing code structure; one open question on signature (resolved with recommendation)

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (stable framework; only risk is if a new phase changes `_build_subprocess_env` signature or `check.py` structure before this phase executes)
