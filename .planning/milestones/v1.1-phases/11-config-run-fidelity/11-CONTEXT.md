# Phase 11: Config & Run Fidelity - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-decided (`/gsd-autonomous` → discuss `--auto`). Grey areas resolved via best-practice defaults; grounded in CFG requirements + verified code anchors.

<domain>
## Phase Boundary

Config-file and snapshot values **drive** experiment runs and are **not silently overridden**
by argparse/CLI defaults; per-node run-command overrides are expressible without editing
snapshotted code. Covers CFG-01 (ISSUE-015), CFG-02 (ISSUE-022), CFG-03 (ISSUE-008).
Independent of Phases 10/12/13/14.

</domain>

<decisions>
## Implementation Decisions

### CFG-01 — argparse defaults stop masking config snapshots (ISSUE-015)
- **D-01:** In `benchmarks/scripts/run_experiment.py::parse_args` (L51), the training-override
  args `--seed`, `--lr`, `--max_epochs`, `--patience`, `--stop_epoch`, `--n_folds` change to
  `default=None`. In `main()` (the `TrainConfig(...)` construction ~L155), each value is passed
  into `TrainConfig` **only when not None** (i.e. explicitly supplied on the CLI) — otherwise the
  snapshot/config `TrainConfig` dataclass default is honored. The Python idiom: argparse
  `default=None` → the arg is None unless the caller passed it; conditional-pass (`{k: v for k,v
  in ... if v is not None}` or per-field `if v is not None`) preserves dataclass defaults.

### CFG-02 — `submit --timeout` stops masking config default (ISSUE-022)
- **D-02:** `automil submit --timeout` (submit.py L29) defaults to `None`; `timeout_min` is OMITTED
  from the queue spec when `--timeout` is unset (L441), so the orchestrator's configured
  `orchestrator.default_timeout_min` controls per-job timeout.
- **D-03 (interaction guard):** the existing `--max-time` logic (D-195) detects an explicit
  `--timeout` via the sentinel `timeout != 150` (submit.py ~L60). That sentinel MUST change to
  `timeout is not None` so "`--max-time` wins when both provided" still holds with the new None
  default. Do not regress the D-195 ceil-div translation.

### CFG-03 — per-node run-command override (ISSUE-008)
- **D-04:** A queue spec can carry a per-node run-command override (e.g. `--seed 42 --lr 1e-4
  --encoder X --n_folds 3`) **layered on top of** the config `run.command` base, without editing
  any snapshotted file. Shape: `automil submit --override "<args>"` writes the override into the
  spec; the daemon (which launches `shlex.split(self.run_command)` at `_orchestrator_daemon.py`
  L899) **appends** the spec's override args after the base `run.command`. Suffix-append model
  (not full-command replacement) — keeps the config `run.command` as the authoritative base.

### Claude's Discretion
- Exact spec field name for the override (`run_command_override` vs `override_args`) and whether
  `--override` takes a quoted string vs repeatable `--override-arg` — planner's call; the
  suffix-append-onto-base semantics in D-04 must hold.
- Whether CFG-01's conditional-pass is a dict-comprehension filter or per-field guards — either,
  as long as None values never reach `TrainConfig` and override its dataclass defaults.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Scope source
- `.planning/REQUIREMENTS.md` §"Config & run fidelity (CFG)" — CFG-01/02/03 text.
- `tasks/test-run-issues.md` — ISSUE-015 (argparse masks config), ISSUE-022 (--timeout default 150),
  ISSUE-008 (per-node run-command overrides) — verified behavior + proposed fixes.

### Code anchors (verified 2026-06-11)
- `benchmarks/scripts/run_experiment.py:51` (`parse_args`) + ~L155 (`TrainConfig(...)`) — CFG-01.
- `benchmarks/src/autobench/pipeline/config.py` (`TrainConfig` dataclass defaults) — the values CFG-01 must honor.
- `src/automil/cli/submit.py:29` (`--timeout default=150`), ~L60 (`timeout != 150` sentinel), L441
  (`timeout_min` written) — CFG-02.
- `src/automil/backends/_orchestrator_daemon.py:431` (`self.run_command = run_config.get("command")`),
  L899-900 (`shlex.split(self.run_command)` launch) — CFG-03 base + append point.

</canonical_refs>

<code_context>
## Existing Code Insights

- CFG-01 and CFG-03 are both about "config/snapshot is authoritative; CLI is an explicit, optional
  override layered on top" — the same principle, two surfaces (TrainConfig fields; run.command).
- CFG-02's trap is the `150` sentinel coupling with `--max-time` (D-195) — changing the default
  without updating the sentinel would silently break the --max-time-wins path. The planner MUST
  cover the interaction with a test.
- All three are contained, single-file-ish fixes (run_experiment.py, submit.py, daemon launch).

</code_context>

<specifics>
## Specific Ideas

- Through-line: **explicit beats default, config beats hardcoded**. A value the operator did not
  explicitly pass must never override a snapshot/config value.

</specifics>

<deferred>
## Deferred Ideas

None — all three CFG items are in scope and contained. (Per-node overrides here are command-arg
suffixes, NOT registry variants — variant application is Phase 10.)

</deferred>

---

*Phase: 11-Config & Run Fidelity*
*Context gathered: 2026-06-11 (auto-decided via /gsd-autonomous)*
