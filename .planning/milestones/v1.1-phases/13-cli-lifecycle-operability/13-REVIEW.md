---
phase: 13-cli-lifecycle-operability
reviewed: 2026-06-12T12:30:00Z
depth: deep
files_reviewed: 7
files_reviewed_list:
  - src/automil/cli/cancel.py
  - src/automil/cli/dequeue.py
  - src/automil/cli/__init__.py
  - src/automil/cli/_helpers.py
  - src/automil/cli/submit.py
  - src/automil/cli/viz.py
  - src/automil/viz/server.py
findings:
  critical: 1
  warning: 6
  info: 5
  total: 12
status: issues_found
---

# Phase 13: Code Review Report

**Reviewed:** 2026-06-12T12:30:00Z
**Depth:** deep
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Reviewed the five OPS bug-fixes (OPS-01 cancel direct-kill, OPS-02 dequeue, OPS-03
submit pending→running, OPS-04 `--project`, OPS-05 viz port resolution) plus the
shared `__init__.py`/`_helpers.py` edits, across the cancel.py ↔ orchestrator
helper call chain.

**Framework purity (D-206):** CLEAN. Zero `autobench`/`AUTOBENCH_`/`benchmarks`
references in any of the seven source files.

**Lazy-import discipline (D-69 / PATTERNS §8):** Correct. cancel.py, dequeue.py,
viz.py, and the `--project` bridge all defer heavy imports into function bodies;
`__init__.py`'s top-level command-module imports are idempotent and acyclic.

**Test-theater audit:** PASS on the high-risk cases.
`test_cancel_local_direct_kill` and `test_cancel_no_starttime_ticks` spawn a
**real** `subprocess.Popen(["sleep", 60], start_new_session=True)` and assert
`ProcessLookupError` from `os.kill(pid, 0)` — no mocked `os.killpg`. OPS-02/03
tests drive real graph/queue shapes. The `_PROJECT_OVERRIDE` autouse teardown
reset exists in `test_cli_project_option.py`. See WR-06 for the one viz test that
*is* theatrical (mocks `cmd_start`, so server.py's own resolution block is never
exercised by the suite).

The headline correctness defect is CR-01: cancel.py's graph mutation is a raw
tempfile write that (a) bypasses `locked_update` and so races the daemon, and
(b) skips the `meta.total_proposed` decrement that `graph.cancel()` performs —
permanently drifting the proposed counter on every cancel. dequeue.py does this
correctly; cancel.py does not, and the in-file docstring even brags about the
divergence.

## Critical Issues

### CR-01: cancel.py mutates graph.json with an unlocked raw write that races the daemon AND skips the `total_proposed` decrement

**File:** `src/automil/cli/cancel.py:273-305`
**Issue:** Two coupled defects in one block.

1. **Unsynchronized write races the daemon.** Every other graph writer in the
   codebase (dequeue.py:71, submit.py:497, the daemon's `_handle_completion`)
   serializes through `locked_update`. cancel.py instead does a bare
   `read → mutate dict → tempfile → os.replace`. The daemon writes graph.json
   concurrently (e.g. it may `mark_failed`/`promote` the very node being
   cancelled, or any sibling, in the same window). A read-modify-write with no
   lock means the cancel can clobber a concurrent daemon update (lost
   `total_executed`++ , lost sibling status flip) or be clobbered itself. The
   node-vanished `else` branch at line 288 acknowledges the data is racing but
   does nothing to actually serialize.

2. **`meta.total_proposed` counter drift.** The raw write sets
   `status="cancelled"` but never decrements `meta.total_proposed`. The
   canonical cancel path, `graph.cancel()` (graph.py:381-384), does
   `self.meta["total_proposed"] = max(0, total_proposed - 1)` precisely because
   a running/pending node was counted as proposed. A running node reached
   `running` via `mark_running`, which does NOT decrement `total_proposed`
   (graph.py:280-289) — so the proposed count still includes it. Cancelling via
   the raw write leaves `total_proposed` one too high, permanently, for every
   `automil cancel`. `mark_failed`/`promote` both decrement on the normal exit
   path; cancel is the only terminal transition that forgets to. Over a session
   this drifts the counter that gates proposal budgeting.

The file's own module docstring (line 4 of dequeue.py) and cancel.py's design
notes treat the raw write as intentional ("unlike cancel.py's raw tempfile
write"). It is a bug, not a feature.

**Fix:** Route cancel's terminal transition through the same locked path
dequeue uses, calling `graph.cancel()` so the counter decrement and lock both
apply. The post-kill / post-poll graph update should be:
```python
from automil.graph import locked_update  # noqa: PLC0415
from automil.cli._helpers import _load_technique_map  # noqa: PLC0415

graph_path = adir / "graph.json"
if graph_path.exists():
    with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
        node = graph.get_node(node_id)
        if node:
            graph.cancel(node_id)  # decrements total_proposed, sets status
            node.setdefault("metadata", {})["cancelled_at"] = datetime.now().isoformat()
            node["metadata"]["cancel_reason"] = "cli"
        else:
            logger.warning("cancel: node %s vanished from graph during lock", node_id)
```
This also removes the duplicated tempfile boilerplate at lines 291-305.

## Warnings

### WR-01: cancel.py direct-kill PID-reuse window — liveness is starttime-checked but the kill signal is not

**File:** `src/automil/cli/cancel.py:172-205`
**Issue:** `_is_alive(_pid, _starttime)` correctly cross-checks
`starttime_ticks` to defend against PID reuse before deciding to signal. But the
actual kill targets the **process group** via `_os.killpg(_pgid, ...)`, and
`_pgid` is read straight from disk with no starttime validation of the group
leader. Between the `_is_alive` probe and the `killpg`, if the original process
group leader has exited and its PGID has been recycled as the PGID of an
unrelated new process group, the SIGTERM/SIGKILL lands on the wrong group. The
window is small, but the whole point of the starttime cross-check (per the phase
risk note) is to be PID-reuse-safe; signalling a `_pgid` that was never
starttime-validated reintroduces the exact hazard for the group case. On Linux
the more robust target is the validated `_pid` itself (or re-probe liveness
immediately before each `killpg`). At minimum, document that the group-kill is
not reuse-protected.
**Fix:** Gate each `killpg` on a fresh `_is_alive(_pid, _starttime)` immediately
prior (the SIGTERM branch currently does not re-check), and/or prefer
`os.killpg(os.getpgid(_pid), sig)` only after confirming `_pid` is the validated
process, so the PGID is derived from a starttime-verified PID rather than from
unvalidated on-disk state.

### WR-02: `_try_reap(pid)` can reap an unrelated PID-reused child and mask a still-running job

**File:** `src/automil/cli/cancel.py:130-143, 210`
**Issue:** `_try_reap` calls `os.waitpid(pid, os.WNOHANG)` on the raw `_pid`
with no starttime guard. The docstring claims it is safe because
`ChildProcessError` is ignored when the CLI is not the parent. But the unsafe
case is the opposite: when the CLI process *is* the parent of a **different,
PID-reused** child that now happens to occupy `_pid`, `waitpid(WNOHANG)` will
silently reap that innocent child's exit status. In an in-process test runner or
any parent-of-many context, this can consume the wait result for a live job and
desync its bookkeeping. The reap is only meaningful when `_pid` is the
starttime-verified original; calling it unconditionally after the kill block
(line 210) widens the blast radius.
**Fix:** Only attempt the reap when the starttime cross-check still identifies
`_pid` as the original target (or skip the reap entirely when
`_starttime is None`), e.g. guard `_try_reap(_pid)` behind
`if _starttime is not None and _read_proc_starttime(_pid) == _starttime`. If
`/proc` is unavailable the reap should be skipped, not best-effort.

### WR-03: `_is_alive` non-Linux fallback treats `PermissionError` as dead → premature "already gone"

**File:** `src/automil/cli/cancel.py:162-168`
**Issue:** On the non-Linux branch (`state is None`), `os.kill(pid, 0)` is used
as a liveness probe and `PermissionError` is caught and mapped to `return False`
("dead"). EPERM from `kill(pid, 0)` means the process **exists** but is owned by
another user — i.e. it is alive, not dead. Mapping it to "dead" makes cancel
log "process already gone, skipping kill" (line 174) and then flip the graph to
`cancelled` while the real job keeps running and holding GPU/VRAM. This is the
same false-negative the daemon's `_is_pid_alive_with_starttime` is careful to
avoid. On Linux the `/proc` path masks this, but the non-Linux fallback exists
precisely for the case where it matters.
**Fix:** Treat `PermissionError`/`EPERM` as alive in the signal-0 probe:
```python
try:
    _os.kill(pid, 0)
    return True
except ProcessLookupError:
    return False
except PermissionError:
    return True  # exists but not ours — still alive
```

### WR-04: dequeue TOCTOU — queue unlink and graph mark are not atomic; a concurrent daemon pickup can run a "dequeued" node

**File:** `src/automil/cli/dequeue.py:53-76`
**Issue:** The state guard (`status == "running"` check, line 45) reads the node
*outside* any lock, then the command unlinks `queue/<node>.json` (line 61) and
*separately* enters `locked_update` to mark cancelled (line 71). Between the
out-of-lock guard read and the unlink, the daemon can pick the spec out of the
queue and launch it (the node was `pending` at guard-time). Now: the file is
gone but the process is live, and dequeue then marks the node `cancelled` in the
graph while the daemon believes it is running — orphaning a live GPU job under a
`cancelled` graph status (exactly the divergence OPS-01 was built to repair).
The unlink and the graph mark should both occur under the same `locked_update`,
and the running-state guard should be re-checked inside the lock.
**Fix:** Move the queue-spec unlink and the running/terminal guard *inside* the
`locked_update` block so the daemon (which also takes the graph lock before
dequeuing) cannot interleave:
```python
with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
    node = graph.get_node(node_id)
    if not node:
        ...
    if node.get("status") == "running":
        raise click.ClickException(...)  # re-check under lock
    if queue_spec.exists():
        queue_spec.unlink()
    graph.cancel(node_id)
```

### WR-05: dequeue calls `graph.cancel()` on terminal/non-proposed nodes without status-shape guard → counter underflow risk and wrong-state cancel

**File:** `src/automil/cli/dequeue.py:49-73`
**Issue:** The guard rejects `running` and the explicit `TERMINAL_STATES` set,
but `graph.cancel()` (graph.py:381) unconditionally sets `status="cancelled"`
and decrements `total_proposed` with no type/status check of its own. The
`TERMINAL_STATES` frozenset is `{completed, cancelled, crashed, keep, discard}`
— but the codebase also uses `crash`, `oom`, `timeout`, and `partial` as
statuses (see submit.py:160 `("crash", "oom", "timeout")` and graph.py:352
`"partial"`). A node in `crash`/`oom`/`timeout`/`partial` state is NOT in
`TERMINAL_STATES`, is not `running`, and so falls through to `graph.cancel()`,
which flips an already-executed node to `cancelled` and decrements
`total_proposed` for a node that was never counted as proposed (it had already
been promoted/failed, which decremented it once). That is a double-decrement
masked by the `max(0, ...)` clamp, plus a spurious status rewrite of executed
results.
**Fix:** Make the dequeue guard inclusive of all non-pending states. Either gate
on the positive set (only allow `type=="proposed"` with
`status in {"pending"}` / queued) or expand `TERMINAL_STATES` to include
`crash`, `oom`, `timeout`, `partial`, `registered`. Prefer the positive guard:
```python
if not (node.get("type") == "proposed" and state in {"pending", "queued"}):
    raise click.ClickException(
        f"Node {node_id!r} is {node.get('type')}/{state!r}; only pending "
        f"proposals can be dequeued. Use `automil cancel` for running nodes."
    )
```

### WR-06: OPS-05 server.py port-resolution block is dead-weight in the test suite — every viz test mocks `cmd_start`, so server.py's own resolution is never exercised

**File:** `src/automil/viz/server.py:288-307`, `tests/test_viz_port_config.py:56,81,111`
**Issue:** OPS-05 deliberately duplicated port resolution in two layers: viz.py
(lines 37-49) and server.py `cmd_start` (lines 302-307). All three
`test_viz_port_config` tests `patch("automil.viz.server.cmd_start", ...)`, so
they only verify the **CLI-layer** resolution and capture the int viz.py passes;
server.py's `port is None` fallback branch is never run by any test. This is the
"injected-dependency test theater" pattern: the test asserts on the resolved
value the CLI hands to the mock, giving false-green confidence that the
*server-side* fallback (the path direct callers like `viz.server.main()` at line
415 actually hit) works. The two blocks can silently diverge — e.g. if server's
`int(raw_port)` coercion regresses — and the suite stays green. Note also
server.py's legacy `main()` (line 414-419) still hard-codes `port = DEFAULT_PORT`
and never consults config, so a `python -m automil.viz.server start` invocation
bypasses config-based port entirely; only the Click path got OPS-05.
**Fix:** Add one test that calls `automil.viz.server.cmd_start(port=None,
project_root=..., host="127.0.0.1")` directly (no mock) against a config with
`viz.port: 9001` and asserts the bound port, exercising the server-side branch.
Separately, fold the `main()` CLI shim onto the same resolution or drop its
hard-coded default.

## Info

### IN-01: cancel.py duplicated/late stdlib imports (`os`, `tempfile`) mid-function

**File:** `src/automil/cli/cancel.py:117, 291-292`
**Issue:** `import os as _os` is done inside the direct-kill branch (line 117),
then `import os` (unaliased) again at line 291 for the graph write, and
`import tempfile` at 292. Two aliases of `os` in one function plus a mid-body
re-import is confusing and error-prone. If CR-01's fix routes through
`locked_update`, the tempfile import disappears entirely.
**Fix:** Hoist a single `import os` to module top (it is stdlib, no circular-
import risk) and drop the `_os` alias.

### IN-02: viz/server.py module docstring references the obsolete `autoMIL/viz/server.py` path

**File:** `src/automil/viz/server.py:8-10`
**Issue:** Usage examples cite `uv run python autoMIL/viz/server.py start` — the
package moved to `src/automil/viz/server.py` long ago. Stale operator guidance.
**Fix:** Update the docstring paths to `src/automil/viz/server.py` (or the
`automil viz start` CLI form).

### IN-03: `cancelled_at` uses naive `datetime.now()` (no tz), inconsistent with submitted_at ISO handling

**File:** `src/automil/cli/cancel.py:286`
**Issue:** `datetime.now().isoformat()` writes a naive local-time stamp, while
cancel.py elsewhere normalizes ISO strings to UTC (lines 220-222). Mixed
tz-aware/naive timestamps in metadata complicate downstream window/age math.
**Fix:** Use `datetime.now(timezone.utc).isoformat()` (`timezone` is already
imported at line 21).

### IN-04: dequeue success message is emitted even when graph.json is absent (no-op)

**File:** `src/automil/cli/dequeue.py:70-77`
**Issue:** If `graph_path` does not exist, the `locked_update` block is skipped
entirely but the command still prints `Dequeued {node_id}.` and exits 0. Since
`_get_node_or_die` (line 41) already requires graph.json to exist, this branch
is effectively unreachable today — but it is latent dead-defensive code that
would lie to the operator if the file were removed between the two reads.
**Fix:** Drop the `if graph_path.exists()` guard (the node lookup already proved
it exists), or downgrade the echo when the lock block is skipped.

### IN-05: `--project` group option does not catch `OSError` from `.resolve()` on a broken symlink path

**File:** `src/automil/cli/__init__.py:33-35`, `_helpers.py:32-42`
**Issue:** `click.Path(exists=True)` validates existence but `Path(project_path)
.resolve()` can still raise on pathological inputs (e.g. a symlink loop) on some
platforms; the `main` callback does not wrap it, so the user sees a raw
traceback rather than a `ClickException`. Low severity — `exists=True` filters
the common cases, and there is no traversal vuln (the resolved path is only used
to locate `automil/config.yaml`, not to read arbitrary user-supplied files).
**Fix:** Wrap the override assignment in a try/except that re-raises as
`click.ClickException` with the offending path. Ordering is otherwise correct:
the override is set (line 35) before `touch_last_action(_find_automil_dir())`
(line 39), so discovery honors it on the first call.

---

_Reviewed: 2026-06-12T12:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
