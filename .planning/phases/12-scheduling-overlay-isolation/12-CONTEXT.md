# Phase 12: Scheduling & Overlay Isolation - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-decided (`/gsd-autonomous` → discuss `--auto`). Grey areas resolved via best-practice defaults; grounded in SCH requirements + verified anchors. Research follows (SCH-02 mechanism).

<domain>
## Phase Boundary

The orchestrator offers a configurable GPU placement policy so compute-bound jobs don't
over-stack one GPU, and the daemon guards editable-installed consumer packages so experiments
import from their worktree overlay (with `automil check` warning when that guard is missing).
Covers SCH-01 (ISSUE-005), SCH-02 (ISSUE-010). Independent of Phases 10/11/13/14.

</domain>

<decisions>
## Implementation Decisions

### SCH-01 — GPU scheduling-policy knob (ISSUE-005)
- **D-01:** Add `orchestrator.scheduling_policy` config with values `best_fit | round_robin |
  least_loaded`, **default `best_fit`** (preserves current behavior — opt-in to the others).
  `_find_best_gpu` (`_orchestrator_daemon.py:743`) dispatches on the policy:
  - `best_fit` (current): pick the GPU with the LEAST schedulable free VRAM among those that fit
    (tightest fit — good for memory-bound packing).
  - `least_loaded`: pick the GPU with the MOST schedulable free VRAM (spreads load).
  - `round_robin`: cycle through eligible (fitting) GPUs in index order, tracking a daemon-level
    last-assigned cursor — so successive low-VRAM compute-bound jobs land on different GPUs.
  Rationale: best-fit over-stacks low-VRAM compute-bound jobs onto one GPU while others idle;
  round_robin/least_loaded fix throughput for that workload class.

### SCH-02 — generic editable-install overlay guard + check warning (ISSUE-010)
- **D-02:** `automil check` (`src/automil/cli/check.py:125`) WARNS when an editable-installed
  consumer package path is snapshotted (declared in `files.editable` / overlaid) WITHOUT a
  worktree import guard in place — so a consumer that lacks the autobench-style
  `run_experiment.py` sys.path prepend gets a loud heads-up that its overlay may be shadowed by
  the main-checkout editable install.
- **D-03 (the D-199 caution — do NOT repeat the removed mistake):** the daemon previously injected
  per-worktree `PYTHONPATH` UNCONDITIONALLY; that was REMOVED in D-199/DEC-01 (`_orchestrator_daemon.py:797`)
  because it caused problems. The generic guard is therefore **opt-in and safe**: a config-gated
  daemon-side injection (e.g. `orchestrator.editable_overlay_guard: true`, default OFF) that
  prepends the worktree's editable src to the experiment process's `PYTHONPATH` — PLUS the
  always-on `automil check` warning (D-02). The autobench consumer's own `run_experiment.py:38`
  prepend STAYS (consumer self-protection); this phase adds the GENERIC framework-side detect +
  opt-in guard so a different editable consumer is protected too.

### Claude's Discretion
- Round-robin cursor storage (daemon instance attr vs persisted) — researcher/planner choose;
  must distribute successive submits across eligible GPUs.
- Exact config key names (`scheduling_policy`, `editable_overlay_guard`) and how `automil check`
  detects "editable package path snapshotted without a guard" (pip editable `.pth`/`.egg-link`
  scan vs files.editable inspection) — planner's call; the detect-and-warn + opt-in-injection
  shape (D-02/D-03) must hold. Research to confirm the exact editable-install detection mechanism.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope source
- `.planning/REQUIREMENTS.md` §"Scheduling & overlay isolation (SCH)" — SCH-01/02 text.
- `tasks/test-run-issues.md` — ISSUE-005 (best-fit over-stacks compute-bound), ISSUE-010 (generic
  editable-install overlay protection missing) — verified behavior + proposed fixes; note ISSUE-010's
  "current autobench symptom mitigated; generic daemon gap kept" + ISSUE-021 history.

### Code anchors (verified 2026-06-11)
- `src/automil/backends/_orchestrator_daemon.py:743` (`_find_best_gpu`), L757-764 (best-fit logic),
  L1760 (call site) — SCH-01.
- `src/automil/backends/_orchestrator_daemon.py:797` (D-199/DEC-01 — the REMOVED unconditional
  PYTHONPATH overlay; the caution for SCH-02), L885 (env whitelist build) — SCH-02.
- `benchmarks/scripts/run_experiment.py:33-38` — the autobench consumer's own sys.path prepend
  (consumer self-protection; the pattern a generic guard generalizes; the workaround the check warns about absence of).
- `src/automil/cli/check.py:125` (`check`) — where the SCH-02 warning lives.

</canonical_refs>

<code_context>
## Existing Code Insights

- SCH-01 is a clean strategy-dispatch refactor of one function (`_find_best_gpu`) + a config knob;
  default `best_fit` keeps current behavior so it's non-breaking.
- SCH-02 is the subtle one: D-199/DEC-01 deliberately REMOVED unconditional PYTHONPATH injection.
  The fix is NOT to re-add it unconditionally — it's detect-and-warn (always) + opt-in injection
  (config-gated). Honor the history; research must confirm WHY D-199 removed it before re-introducing
  any injection.
- Framework purity: the generic guard + check live in `src/automil/` and must be consumer-agnostic
  (no autobench specifics); the autobench `run_experiment.py` prepend is the consumer's own concern.

</code_context>

<specifics>
## Specific Ideas

- SCH-01 through-line: the scheduler must serve BOTH memory-bound (best_fit packs) and
  compute-bound (round_robin/least_loaded spreads) workloads — one knob, three policies.
- SCH-02 through-line: don't repeat D-199. Detect + warn loudly; inject only when opted in.

</specifics>

<deferred>
## Deferred Ideas

- Anti-starvation aging in the scheduler — explicitly out of scope (PROJECT.md Out-of-Scope:
  "defer until observed in practice"). SCH-01 is policy-selection only, not fairness/aging.

</deferred>

---

*Phase: 12-Scheduling & Overlay Isolation*
*Context gathered: 2026-06-11 (auto-decided via /gsd-autonomous)*
