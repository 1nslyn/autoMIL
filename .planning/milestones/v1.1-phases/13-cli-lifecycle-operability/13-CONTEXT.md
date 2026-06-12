# Phase 13: CLI Lifecycle & Operability - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning
**Mode:** Auto-decided (`/gsd-autonomous` → discuss `--auto`). Grey areas resolved via best-practice defaults; grounded in OPS requirements + verified code anchors (read during scout). The one non-trivial mechanism (OPS-01 cross-process direct-kill) is flagged for a short targeted research pass; OPS-02..05 are contained fixes.

<domain>
## Phase Boundary

Operators can drive the full node lifecycle from the CLI — cancel a daemon-launched
running job, cleanly dequeue a queued/pending node, have an existing pending proposal
transition to `running` on submit, target an overlay from outside its project root, and
reliably reach the viz dashboard via config-driven port resolution — without manual file
surgery. Covers OPS-01 (ISSUE-011), OPS-02 (ISSUE-016), OPS-03 (ISSUE-023),
OPS-04 (ISSUE-012), OPS-05 (ISSUE-004). Independent of Phases 9–12 and 14.

</domain>

<decisions>
## Implementation Decisions

### OPS-01 — `automil cancel` for daemon-launched local jobs (ISSUE-011)
- **D-01:** The CLI `cancel` command (`src/automil/cli/cancel.py`) must **signal the
  process group directly from the CLI process** using on-disk metadata, NOT route a
  running local job through `backend.cancel()` → `_kill_experiment()`. Root cause found
  during scout: `_kill_experiment` (`_orchestrator_daemon.py:1656`) resolves the target via
  `self.running.get(node_id)` — the daemon's **in-memory** map. A fresh CLI-spawned
  `LocalBackend` has an empty `self.running`, so `_kill_experiment` returns `False`
  ("not in self.running") and the running job is never killed. The on-disk running spec
  (`orchestrator/running/local/<node>.json`) carries `metadata.pid`, `metadata.pgid`, and
  `metadata.starttime_ticks` — enough to kill safely from any process.
- **D-02:** For a `local` running spec, when top-level `opaque_id` is absent, fall back to
  `metadata.pid` / `metadata.pgid`. Signal the **process group** (`os.killpg(pgid, SIGTERM)`)
  guarded by the same PID-reuse cross-check used by the daemon — `_read_proc_starttime(pid)`
  compared against `metadata.starttime_ticks` (`_is_pid_alive_with_starttime`,
  `_orchestrator_daemon.py:158`). Reuse those existing helpers; do NOT reimplement.
- **D-03:** Loud-fail only when the spec has **neither** top-level `opaque_id` **nor**
  `metadata.pid`/`metadata.pgid` (genuinely uncancellable corrupted state). Preserve the
  existing graph update (`status='cancelled'`, `cancelled_at`, `cancel_reason='cli'`) and
  the running-spec→archive move. Poll for the cancelled transition as today; observe via
  the starttime cross-check (PID gone or starttime mismatch ⇒ cancelled).
- **Rejected:** sentinel-file → daemon round-trip (CLI writes a cancel request, daemon
  actions it from its live map). More complex, async, and adds a daemon-liveness dependency
  for what on-disk pid/pgid metadata already makes a safe, synchronous direct kill.

### OPS-02 — `automil dequeue <node>` (ISSUE-016)
- **D-04:** Add a **new** top-level `automil dequeue <node_id>` command (its own module,
  `src/automil/cli/dequeue.py`, registered in `cli/__init__.py`). Do NOT overload `cancel`
  — keep the verbs clean: `cancel` = running jobs (backend signal), `dequeue` = queued/pending
  nodes (file removal + graph mark). It (1) removes the queue spec
  `orchestrator/queue/<node>.json` if present (and any backend-namespaced variant, mirroring
  D-169 namespacing if queue specs are namespaced), (2) marks the graph node `cancelled`
  via the official state-machine path (`graph.cancel()` / status set through `locked_update`),
  leaving no orphaned pending proposal.
- **D-05:** State guard: `dequeue` accepts non-running, non-terminal nodes
  (`proposed/pending`, `queued`). Hard-fail with a cross-referencing message if the node is
  `running` ("use `automil cancel`") or already terminal (`completed`/`cancelled`/`crashed`).
  Idempotent-friendly: if the node is pending with no queue spec on disk, still mark it
  `cancelled` (clears the orphaned proposal) rather than erroring.

### OPS-03 — pending proposal → running on submit (ISSUE-023)
- **D-06:** In `src/automil/cli/submit.py` (the `locked_update` block ~L495–519), the
  `graph.mark_running()` call currently lives **inside** `if not graph.get_node(node):`, so
  it is skipped when submitting against an existing `type=proposed, status=pending` node.
  Add an `else` branch: when the target node already exists as `type=proposed, status=pending`,
  call `graph.mark_running(node)` **after** the queue spec is written, within the same
  `locked_update`. `mark_running` (`graph.py:280`) already guards on type/status, so the call
  is a safe no-op for any other existing state. Keeps cancellation + portfolio accounting
  consistent (no launch while graph still says `proposed/pending`).

### OPS-04 — `--project PATH` for out-of-root targeting (ISSUE-012)
- **D-07:** Add a **group-level** `--project PATH` option on the `main` Click group
  (`src/automil/cli/__init__.py`) — invoked as `automil --project /path/to/overlay <cmd> …`.
  Resolve it in the group callback and bridge it so `_find_automil_dir()`
  (`src/automil/cli/_helpers.py:18`) honors the override **before** walking up from cwd.
  Lowest-impact: one option + one helper change covers all ~20 commands (vs. adding a
  `--project` option to every command signature). `PATH` may point at a project root
  (containing `automil/config.yaml`) or the `automil/` dir itself — resolve both; hard-fail
  with a clear message if no `automil/config.yaml` is found under the given path.
- **D-08 (bridge mechanism — planner's call, recommendation locked):** Prefer a
  module-level resolved-override set by the group callback and read by `_find_automil_dir()`
  (consumer-invisible, no `@click.pass_context` on every command). Click `ctx.obj` is the
  idiomatic alternative but forces `pass_context` threading through ~20 commands — higher
  impact. Whichever is chosen, `_find_automil_dir()` stays the single source of truth for
  project discovery so the override applies uniformly. Note: the `main` callback's existing
  best-effort `touch_last_action(_find_automil_dir())` must respect the override too.

### OPS-05 — viz port config fallback (ISSUE-004)
- **D-09:** Change the `viz start` CLI option default to `--port=None`
  (`src/automil/cli/viz.py:17`) and resolve the port in `cmd_start()`
  (`src/automil/viz/server.py:265`) mirroring the existing **host** fallback
  (server.py:285–295): explicit `--port` → `automil/config.yaml: viz.port` → default `8420`
  (`DEFAULT_PORT`). Co-locate the port resolution with the host resolution block so both read
  the same already-loaded config dict. Keep `viz stop`/`viz status` unchanged.

### Claude's Discretion
- OPS-01 direct-kill: exact signal escalation (SIGTERM then poll, optional SIGKILL after a
  grace) — researcher/planner may mirror the daemon's SIGTERM→grace→SIGKILL or keep the
  CLI's current single-SIGTERM-then-poll. Must remain PID-reuse-safe via starttime.
- OPS-04 bridge: module-global vs `ctx.obj` per D-08 (recommendation: module-global).
- Whether `dequeue` reuses any of `cancel.py`'s graph-update/archive helpers vs. inlines a
  smaller queued-node path — planner's call; keep it minimal.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope source
- `.planning/REQUIREMENTS.md` §"CLI Lifecycle & Operability (OPS)" — OPS-01..05 text.
- `tasks/test-run-issues.md` — ISSUE-011 (cancel can't reach daemon-launched local jobs),
  ISSUE-016 (no clean dequeue), ISSUE-023 (pending proposals don't transition on submit),
  ISSUE-012 (no out-of-root targeting), ISSUE-004 (viz ignores `viz.port`). Each entry has
  verified behavior + a proposed fix.
- `.planning/ROADMAP.md` §"Phase 13: CLI Lifecycle & Operability" — 5 success criteria.

### Code anchors (verified 2026-06-12 during scout)
- `src/automil/cli/cancel.py` — full flow; step-4 `opaque_id` hard-fail is the OPS-01 gate
  to relax; reuses `_get_node_or_die`, graph atomic update, running→archive move.
- `src/automil/backends/_orchestrator_daemon.py:1656` (`_kill_experiment`, in-memory
  `self.running` — the OPS-01 root cause), `:158` (`_is_pid_alive_with_starttime`),
  `:145` (`_read_proc_starttime`) — reuse for the CLI direct-kill + PID-reuse guard.
- `src/automil/backends/local.py:296` (`LocalBackend.cancel` — phase-1 queue unlink /
  phase-2 `_kill_experiment` delegate; confirms cancel resolves by `node_id`, not `opaque_id`).
- `src/automil/cli/submit.py:495–519` — the `locked_update` block where `mark_running` is
  gated behind `if not graph.get_node(node):` (OPS-03 fix site).
- `src/automil/graph.py:280` (`mark_running`, type/status-guarded), `:381` (`cancel`) — OPS-02/03.
- `src/automil/cli/__init__.py` (`main` group + `touch_last_action`) and
  `src/automil/cli/_helpers.py:18` (`_find_automil_dir`) — OPS-04 group option + override bridge.
- `src/automil/cli/viz.py:16-29` (`viz_start`, `--port default=8420`) and
  `src/automil/viz/server.py:265` (`cmd_start`), `:285-295` (host fallback to mirror) — OPS-05.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_is_pid_alive_with_starttime` / `_read_proc_starttime` (`_orchestrator_daemon.py`):
  PID-reuse-safe liveness check — OPS-01's direct-kill must reuse these, not reinvent.
- `cancel.py`'s graph atomic-update (tempfile + `os.replace`) and running→archive move:
  the dequeue command can reuse the same atomic-write idiom for its graph mark.
- `graph.mark_running` / `graph.cancel`: official state-machine transitions; route OPS-02/03
  through these (under `locked_update`) so `meta` counters stay consistent (the same reason
  submit.py routes through `add_proposed`+`mark_running` rather than dict-mutation).
- viz `cmd_start` host-fallback block (server.py:285-295): the exact pattern OPS-05 mirrors.

### Established Patterns
- D-169 backend-namespaced orchestrator paths: `running/<backend>/<node>.json`,
  `queue/<node>.json` — cancel.py already resolves `running/local/<node>.json`; dequeue must
  resolve the queue path the same way.
- Lazy intra-command imports (`from automil.backends import ...` inside the function body) to
  avoid CLI-load circular imports (PATTERNS §8 / D-69) — new `dequeue.py` should follow this.
- `locked_update(graph_path, technique_map=...)` serializes CLI graph writes against the
  daemon's `_handle_completion` — both OPS-02 and OPS-03 graph mutations must run inside it.

### Integration Points
- `_find_automil_dir()` is the single discovery seam every command calls — OPS-04 threads
  through exactly this one function, keeping the change surface tiny.
- The `main` group callback already calls `_find_automil_dir()` (for `touch_last_action`);
  the `--project` override must be resolved before that call so activity stamping targets the
  right overlay.

</code_context>

<specifics>
## Specific Ideas

- OPS-01 through-line: the running spec already carries everything needed to kill safely —
  the bug is that cancel routes through the daemon's in-memory map instead of the on-disk
  metadata. Fix = read pid/pgid from disk, `os.killpg` with starttime guard, from the CLI.
- OPS-04 through-line: one discovery seam (`_find_automil_dir`), one group option — resist
  the temptation to sprinkle `--project` onto 20 command signatures.
- OPS-03 through-line: a one-line gap (`mark_running` trapped inside the `if-not-exists`
  branch) — the smallest fix in the phase; add the `else`.

</specifics>

<deferred>
## Deferred Ideas

- Remote-backend (SLURM/Ray) cancel-from-CLI parity for jobs the local daemon didn't launch
  is already handled by those backends' own `cancel()` via `opaque_id` (scancel/ray.cancel);
  OPS-01 is scoped to the **local** daemon gap only. No change needed for remote backends.
- A unified `automil lifecycle`/`automil rm` super-verb that auto-dispatches cancel-vs-dequeue
  by state — nice ergonomics, but adds a verb; keep `cancel` + `dequeue` explicit for v1.1.

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 13-CLI Lifecycle & Operability*
*Context gathered: 2026-06-12 (auto-decided via /gsd-autonomous)*
