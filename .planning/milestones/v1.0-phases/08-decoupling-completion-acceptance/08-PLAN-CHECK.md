# Phase 8 Plan Check, Iteration 1

**Reviewed:** 2026-05-07
**Phase:** 08-decoupling-completion-acceptance
**Reviewer:** plan-checker (goal-backward audit, FORCE stance)
**Plans audited:** 10 (08-01 through 08-10)
**Source artifacts read:** 08-CONTEXT.md (D-199..D-208), 08-RESEARCH.md (9 OQs), 08-PATTERNS.md, 08-PLAN-SUMMARY.md, REQUIREMENTS.md (DEC-01..07), ROADMAP.md (Phase 8 lines 192-203), src/automil/{graph.py, backends/_orchestrator_daemon.py, cli/check.py, cli/lifecycle/verify_repro.py, cli/lifecycle/revert_baseline.py, gate/manifest.py, viz/static/app.js, templates/config.yaml.j2}, tests/{test_orchestrator_env_whitelist.py, cli/__init__.py}, pyproject.toml, CHANGELOG.md, plus directory listings for tests/cli, tests/viz (absent), tests/acceptance (absent), examples/.

---

## 1. Verdict

**RETRY** (3 BLOCKER + 5 HIGH + 4 MEDIUM + 2 LOW; surgical fixes required before execute).

Plans are decision-traceable, file-disjoint across waves, and the dependency graph is sound. All 10 plans have valid frontmatter (1-indexed waves, `depends_on`, `files_modified`, `autonomous`, `requirements`). Every DEC-01..07 maps to at least one plan; every D-208 clause has a verifying clause-test in 08-10. The OQ resolutions (Draft202012Validator, OQ-9 Option B Pareto, sub-gate marker semantics, content-anchor allowlist) are pre-applied at plan-write time.

The 3 BLOCKERs are concrete-source mismatches that will produce broken executor output on first attempt:

1. The graph.py migration delta omits a write site that DOES contain the named-field copy (`_orchestrator_daemon.py:1055-1057` cap-killed-reconcile path inside the daemon, which writes `gnode["test_auc"]` etc. directly), and plan 08-02 explicitly defers this site per OQ-7/OQ-8 even though it would re-introduce `gnode["test_auc"]` AFTER 08-02 deletes them from add_executed/promote, leaving the system in an inconsistent post-D-200 state where the cap-killed branch contradicts the new storage shape.
2. Plan 08-08's framework purity grep gate's `_ALLOWLIST` is missing a third entry for `src/automil/cli/lifecycle/revert_baseline.py:87` (which contains `'benchmarks/lib/CLAM/**'`); the gate WILL fail on first run.
3. Plan 08-05 leaves `tests/test_orchestrator_env_whitelist.py::test_pythonpath_overrides_whitelist_value` (line 166-170) ALIVE and untouched, but that test asserts `env["PYTHONPATH"] == "/tmp/wt/benchmarks/src"` which can no longer be true after the PYTHONPATH injection is purged from `_build_subprocess_env`. The test will FAIL on first run; plan 08-05 only mentions the `test_autobench_root_still_injected_phase0` conversion.

The 5 HIGHs are coverage gaps and stale references that the executor would notice but should be locked at plan-write time. The MEDIUMs are correctness risks. The LOWs are style.

This is NOT a fundamental rework; the planner can patch all 8 BLOCKER+HIGH findings in one targeted revision pass.

---

## 2. Findings by Severity

| Severity | Count | What it means |
|----------|-------|---------------|
| **BLOCKER** | 3 | Plan WILL fail when executed; must fix before kickoff. |
| **HIGH** | 5 | Plan is likely to deviate or short out a DEC requirement; should fix. |
| **MEDIUM** | 4 | Correctness risk; nice to fix in same revision. |
| **LOW** | 2 | Style/clarity; ship as-is acceptable. |

| ID | Plan | Severity | One-line description |
|----|------|----------|----------------------|
| F-01 | 08-08 | BLOCKER | `_ALLOWLIST` missing entry for `src/automil/cli/lifecycle/revert_baseline.py:87` (`'benchmarks/lib/CLAM/**'`); first run of the framework purity gate FAILS. |
| F-02 | 08-05 | BLOCKER | `tests/test_orchestrator_env_whitelist.py::test_pythonpath_overrides_whitelist_value` (line 166-170) asserts the orchestrator-injected PYTHONPATH wins; after 08-05 deletes that injection the assertion is unreachable and the test FAILS. Plan only converts `test_autobench_root_still_injected_phase0`. |
| F-03 | 08-02 / 08-05 | BLOCKER | Migration delta omits `_orchestrator_daemon.py:1055-1057` cap-killed reconcile branch which writes `gnode["test_auc"]` etc. directly into the graph node. Post-D-200 the daemon will write these top-level keys for cap-killed nodes only, contradicting the dict-spread invariant and producing a hybrid node shape. The PLAN-SUMMARY explicitly defers this per OQ-7/OQ-8 to a "future phase," but the deferral leaves the framework in an inconsistent state where the cap-killed path silently re-introduces the very keys 08-02 just removed. |
| F-04 | 08-09 | HIGH | sub-gate B test `test_subgate_b_sklearn_iris_end_to_end` runs `python train.py` directly via subprocess but does NOT exercise the orchestrator daemon ingest path (per the test's own NOTE comment). The result.json is validated only by the in-test `validate_result(result)` call, NOT by the daemon's `_orchestrator_daemon.py:~1090` validation hook. CONTEXT D-205 sub-gate B contract requires "automil submit ... run via real LocalBackend". Without that, DEC-02 + DEC-07 are not actually verified end-to-end at the orchestrator-ingest level; only the script-side schema-conformance is tested. |
| F-05 | 08-04 | HIGH | The `_validate_env_required` insert site claimed at "after the env.passthrough block (around line 227)" matches reality (line 211-227 is the existing env.passthrough block ending at 227 with `click.echo("env.passthrough: (none declared)")`). HOWEVER plan 08-04's example `if not missing_required and isinstance(raw_required, list) and raw_required:` reads `raw_required` from a separately-fetched `env_section_chk`. The variable shadowing of `env_section` (line 212) vs `env_section_chk` is benign but confusing; more importantly, the plan's listing has `raw_required = env_section_chk.get("required", [])` but later `if raw_required and not isinstance(raw_required, list):` will only trigger when `raw_required` is truthy AND not-a-list (so empty lists `[]` correctly pass through to the validator). This logic is correct but the plan's prose says "type-mismatch handling: env.required value is not a list -> emit warning, do not crash" which the test `test_required_not_a_list_returns_empty` covers ONLY for the helper's return value, NOT for the warning emission at the call site. Add a test for the call-site warning. |
| F-06 | 08-04 | HIGH | The plan instructs the executor to write `required: []` ABOVE `passthrough:` in `config.yaml.j2`. Real file (lines 96-104) has the `env:` block with ONLY `passthrough:`. The plan's "Step B" replacement is correct, but the inline AUTOBENCH example comment (`# e.g. for an autobench consumer: [AUTOBENCH_OVARIAN_ROOT, AUTOBENCH_CCRCC_ROOT]`) introduces a NEW autobench reference inside `src/automil/templates/config.yaml.j2`. Plan 08-04 acknowledges this requires allowlist coordination with 08-08, and offers an "either path is acceptable" alternative. This is exactly the kind of decision the planner is supposed to lock pre-execute, not punt to the executor. The framework-purity grep gate scans `src/automil/`; templates/ is in scope. Lock the decision: drop the inline example or pre-add the allowlist entry to 08-08. |
| F-07 | 08-04 / 08-06 | HIGH | DEC-04 verbatim text in REQUIREMENTS.md and ROADMAP.md says "config-driven (`automil/config.yaml: scoring.formula` or `scoring.entry_point`)". Plan 08-04 covers `env.required` (DEC-05) and the config.yaml.j2 template extension, but NEITHER 08-02 NOR any other plan adds a `scoring:` block to `config.yaml.j2` (the framework template). The sklearn-iris consumer's config.yaml in 08-06 includes `scoring.formula: "accuracy"` but the framework template stays silent. Per CONTEXT D-200 the formula is "documentation-only", which is a valid simplification, but the framework template MUST surface the field for new consumers to discover the contract. Without a `scoring:` block in `config.yaml.j2`, fresh `automil init` outputs do not document the field; DEC-04 is partially delivered. |
| F-08 | 08-09 | HIGH | sub-gate A's autobench-project path probe `ccrcc_data_root.parent / "ccrcc"` is a guess. Real autobench layout (per CLAUDE.md monorepo section + benchmarks/datasets/) is `benchmarks/experiments/ccrcc/automil/config.yaml`; AUTOBENCH_CCRCC_ROOT is a dataset root, not a project root. The probe will not find the project on Leo's actual workstation, the test will pytest.skip with a misleading message, and sub-gate A will silently skip even WITH CCRCC data set. This degrades D-205 sub-gate A from "workstation-runnable" to "always skip". |
| F-09 | 08-10 | MEDIUM | clause 11 `test_d208_clause_11_state_roadmap_complete` body has a tortured assertion: `dec_complete_count = roadmap_text.count("Complete") + (...) .read_text().count("DEC-") if (...).exists() else 0`. The conditional-expression precedence makes this read as `roadmap_text.count("Complete") + (X if Y else 0)` so the right-hand `0` only kicks in when `REQUIREMENTS.md` is absent; otherwise it ALWAYS adds REQUIREMENTS.md `DEC-` count (which is 7+ per file) regardless of whether they're "Complete" or "Pending". Assertion `>= 1` will always pass trivially. The clause is not actually verifying ROADMAP completion. |
| F-10 | 08-10 | MEDIUM | Task 4 instructs "if REQUIREMENTS.md is in scope for this plan ... include the edit; if not ... best-effort". `files_modified` does NOT include `.planning/REQUIREMENTS.md`. Plan 08-10's frontmatter is the source of truth for wave file-disjointness audit; either commit to editing REQUIREMENTS.md (add to files_modified) or document the follow-up explicitly. Punting here breaks the milestone-acceptance signal (per ROADMAP line 199-203 "Final acceptance"). |
| F-11 | 08-02 | MEDIUM | Plan 08-02 promises the bootstrap-loader at lines 779-810 keeps backwards-compat reads. The actual `import_from_tsv` at graph.py lines 779-810 already passes `test_auc`/etc INTO the metrics dict that is fed to `add_executed`. After Step A's refactor, `add_executed` does `dict(metrics)` so the keys end up in `node["metrics"]`. Plan 08-02 says "the engineering rule is: any local variables named val_auc/etc parsed from results.tsv MUST flow into the metrics input of add_executed under those same key names". This is correct AS WRITTEN in the current source (lines 791-805 already pass them as `test_auc`/etc keys in the metrics dict). Step H is then a NO-OP: the "translate the four legacy columns into a metrics dict" instruction implies a code change that is unnecessary because the existing code already does this. Either drop Step H entirely or rewrite it as a verify-only acceptance criterion. |
| F-12 | 08-09 | MEDIUM | conftest.py's `ccrcc_data_root` fixture skips if `(root / "splits").exists()` is False. The benchmarks/datasets/cptac/ccrcc.yaml YAML and CCRCC dataset shape on Leo's workstation do not necessarily have a `splits/` subdirectory at the dataset root (per benchmarks/datasets/cptac/ccrcc.yaml the splits live at a separate path). Verify the actual layout before locking the skip-on-missing-splits guard, otherwise the fixture will skip on Leo's workstation too. |
| F-13 | All Wave 1+ | LOW | Em-dash audit: zero em or en dashes (U+2014/U+2013) found in any of the 10 Phase 8 plans, 08-CONTEXT.md, 08-RESEARCH.md, 08-PATTERNS.md, or 08-PLAN-SUMMARY.md. Pre-existing em-dashes survive in `src/automil/backends/_orchestrator_daemon.py` (24 lines, including line 50 and the allowlist anchor itself at line 55) and `src/automil/cli/check.py` (5 lines). Plans correctly do NOT instruct the executor to remove pre-existing dashes (out of scope per Phase 7 F-09 precedent). |
| F-14 | 08-10 | LOW | Task 3 STATE.md update template uses `progress.total_plans: 92` and `completed_plans: 92` as illustrative values, with a parenthetical "Plan count grows from 82 to 92 with Phase 8's 10 plans; adjust if the actual plan count differs." The actual plan count across phases 0-8 should be a deterministic figure resolvable now. Punting to the executor risks number drift across this and the final commit. Resolve at plan-write time. |

---

## 3. Per-Finding Details

### F-01, BLOCKER, plan 08-08 task 1, _ALLOWLIST entries

**Description:** The framework-purity grep `grep -rEn "autobench|AUTOBENCH_|benchmarks/" src/automil/` returns 7 hits in the current source pre-purge. After plan 08-05 lands (deleting lines 718-722 + 777-786 of `_orchestrator_daemon.py`), 5 of those will disappear. The 2 remaining "informational" comments per the plan are:

1. `src/automil/backends/_orchestrator_daemon.py:54` , `Consumer-specific vars (e.g. AUTOBENCH_*_ROOT) are opted in per project via`
2. `src/automil/cli/lifecycle/verify_repro.py:84` , `# Use a clean env (no AUTOBENCH_* leakage; CUDA visibility removed).`

Both verified at the cited line numbers in current source.

HOWEVER, a third match exists that no plan addresses: `src/automil/cli/lifecycle/revert_baseline.py:87` contains the literal substring `'benchmarks/lib/CLAM/**'` (inside a string used as part of an error/help message about protected paths). The grep regex `benchmarks/` will match this. Plan 08-08's `_ALLOWLIST` only enumerates 2 entries; on first run the test will FAIL with revert_baseline.py:87 as a non-allowlisted offender.

**Suggested fix:** Either:
- Add a third allowlist entry to plan 08-08:
  ```
  "src/automil/cli/lifecycle/revert_baseline.py:87":
      "must not touch (e.g., 'benchmarks/lib/CLAM/**')",
  ```
  AND coordinate the line-number anchor with revert_baseline.py at plan-execute time (the line may drift if pre-existing edits land before 08-08).
- Or rewrite the offending string in revert_baseline.py to use a generic example (e.g., `'examples/lib/CLAM/**'`) as a one-line cleanup task in plan 08-05 or a new sub-task in 08-08, before the gate test asserts the empty match list.

**Rationale:** Plan 08-08 explicitly says "If a hit is returned (executor of 08-04 retained the example comment), record the file:line and add a 3rd allowlist entry" but only audits config.yaml.j2 (executor of 08-04). It misses the pre-existing revert_baseline.py:87 hit which has been in the source since at least Phase 1.

---

### F-02, BLOCKER, plan 08-05 task 3 + missing task

**Description:** Plan 08-05 task 3 converts `test_autobench_root_still_injected_phase0` to a negative test. Plan 08-05 task 1 also instructs "drops `pythonpath` and `worktree_benchmarks` parameters" from `_build_subprocess_env`, and step C of task 3 says "audit helper signature ... update there as well" + step D says "find any other test in the file referencing `worktree_benchmarks` or `pythonpath` as parameters and update those".

This is the right instruction in spirit, but the plan does NOT name the specific test that will break (`test_pythonpath_overrides_whitelist_value` at lines 166-170 of `tests/test_orchestrator_env_whitelist.py`):

```python
def test_pythonpath_overrides_whitelist_value(orch, monkeypatch):
    """The orchestrator-injected PYTHONPATH wins over the whitelisted os.environ['PYTHONPATH']."""
    monkeypatch.setenv("PYTHONPATH", "/some/parent/path")
    env = _call_build(orch, pythonpath="/tmp/wt/benchmarks/src")
    assert env["PYTHONPATH"] == "/tmp/wt/benchmarks/src"
```

After 08-05's purge, `_build_subprocess_env` no longer accepts `pythonpath` AND no longer overrides PYTHONPATH (the orchestrator-injected layer 3 stops setting it). Two failure modes:
- If the executor follows step C and removes `pythonpath` from `_call_build` defaults, this test will fail with a TypeError or with the assertion (PYTHONPATH will equal `/some/parent/path` from the system whitelist).
- If the executor does not, `_build_subprocess_env(pythonpath=...)` will raise TypeError because the parameter was dropped from the signature.

Either way the test breaks; the plan does not specify what the new behavior of this test should be.

**Suggested fix:** Add an explicit step to plan 08-05 task 3:

> Step E, delete `test_pythonpath_overrides_whitelist_value` (lines 166-170) entirely. Per D-199, the orchestrator no longer injects PYTHONPATH; the test asserts a contract that no longer exists. If a replacement is needed, add `test_pythonpath_not_auto_injected_phase8` asserting `assert env.get("PYTHONPATH") == os.environ.get("PYTHONPATH", "")` (i.e., it equals the whitelisted system PYTHONPATH if any, with no orchestrator override).

**Rationale:** The plan's step D ("find any other test ... update those") is a correct instruction but punts the specific decision to the executor. Plan-write-time resolution removes the ambiguity and matches the same level of explicit naming the plan applies to `test_autobench_root_still_injected_phase0`.

---

### F-03, BLOCKER, plan 08-02 + 08-05 (migration delta + cap-killed reconcile)

**Description:** The PLAN-SUMMARY's Migration Delta Cross-Reference table (lines 110-128) explicitly defers two daemon sites:
- `_orchestrator_daemon.py:1055` cap-killed reconcile (per OQ-7) , deferred
- `_orchestrator_daemon.py:1289-1298` results.tsv writer , deferred per OQ-8

Verified line 1055 of the current source:
```python
for k in ("test_auc", "test_bacc", "val_auc", "val_bacc"):
    if k in payload.get("metrics", {}):
        gnode[k] = payload["metrics"][k]
```

This block writes `gnode["test_auc"]`, `gnode["test_bacc"]`, etc. directly into the graph node's TOP LEVEL. After plan 08-02 lands (`add_executed` and `promote` switch to dict-spread `node["metrics"] = dict(metrics)`), every NORMAL completion writes nothing at top level for the four named keys. But the cap-killed branch at line 1055 STILL writes them at top level, producing a heterogeneous node shape: cap-killed nodes have `gnode["test_auc"]` AND nothing in `gnode["metrics"]`; normal-completed nodes have everything in `gnode["metrics"]` AND nothing at top level.

The plan-08-03 viz reader migration assumes uniform `node.metrics` shape; the cap-killed-node reads will produce `undefined` and render as `'-'`. Worse, plan 08-02's `_reevaluate_descendants` migration to OQ-9 Option B uses `composite-only` dominance, but the cap-killed path's top-level write `gnode["test_auc"]` is incidental garbage that future code will read and propagate.

The deferral was made on the OQ-7 reasoning that "the cap-killed reconcile is low-traffic". But CONTEXT D-200 says verbatim: "graph.py stores ALL metric keys on the node via `node['metrics'] = dict(metrics)` (full dict spread), instead of named-field copy." The cap-killed branch in `_orchestrator_daemon.py` is functionally part of the graph mutation surface for D-200 even though it physically lives in the daemon module, and CONTEXT does not authorize a partial migration.

**Suggested fix:** Add a Task to plan 08-05 (or a new plan 08-05b in Wave 2):

> Task: migrate `_orchestrator_daemon.py:1055` cap-killed reconcile to dict-spread.
> Replace lines 1055-1057 with `gnode["metrics"] = dict(payload.get("metrics", {}))`.
> The orchestrator-measured scalars `composite`, `vram_gb`, `elapsed_min` stay at top-level via `gnode["composite"] = payload["composite"]` (already at line 1054).
> No new tests required (cap-killed integration tests cover this path indirectly).

**Rationale:** Without this, D-200 / DEC-04 is partially delivered. Pitfall 7b (no silent zero-default) is violated for cap-killed nodes specifically. The deferral note in PLAN-SUMMARY at line 126 says "deferred per OQ-8 deferred decision" but OQ-8 is about `results.tsv`, not the cap-killed branch. The deferral has no decision basis.

---

### F-04, HIGH, plan 08-09 task 3 (sub-gate B test body)

**Description:** Plan 08-09's `test_subgate_b_sklearn_iris_end_to_end` runs `subprocess.run([sys.executable, "train.py"], cwd=project)` and asserts on the result.json directly. The test file's own NOTE block says: "Step 3 is the minimum end-to-end proof. A fuller invocation through automil orchestrator start (Subprocess + sleep + poll) is feasible but not strictly required for sub-gate B".

CONTEXT D-205 specifies: "Sub-gate B (sklearn-iris end-to-end): same harness, automil submit --node iris_001 --files examples/sklearn-iris/train.py --max-time 60, assert terminal state `executed` AND composite >= 0.90."

The "same harness" reference points back to sub-gate A's harness which uses real `automil submit` + LocalBackend. The plan's actual implementation skips the orchestrator path and only runs train.py directly. Implications:
- DEC-02 ("plugs into autoMIL via documented contract and runs an experiment loop end-to-end") is partially delivered: the contract is exercised but the experiment loop (submit -> orchestrator start -> daemon ingest -> graph promote) is not.
- DEC-07's "as the decoupling proof" is weakened: a sklearn training script that runs standalone proves only that sklearn-iris works as a script, not that autoMIL can drive it.
- The validate_result hook in `_orchestrator_daemon.py:~1090` (added by 08-05) is NOT exercised by sub-gate B; only the standalone `from automil.schemas import validate_result` call inside the test runs. If the daemon hook breaks, sub-gate B remains green.

**Suggested fix:** Replace sub-gate B's body with the full submit + orchestrator path:

```python
def test_subgate_b_sklearn_iris_end_to_end(tmp_path: Path):
    project = tmp_path / "iris_project"
    shutil.copytree(_EXAMPLES_IRIS, project)
    _git_init_and_commit(project)
    # Real automil submit, then drive orchestrator one tick.
    out = _run(["automil", "init", "--non-interactive"], cwd=project)
    out.check_returncode()
    out = _run(["automil", "submit", "--node", "iris_001",
                "--files", "train.py", "--max-time", "60"], cwd=project)
    out.check_returncode()
    # Drive one tick of the orchestrator (synchronous helper for tests).
    out = _run(["automil", "orchestrator", "tick", "--once"], cwd=project, timeout=120)
    # Read graph.json, assert iris_001 terminal status executed + composite.
    graph = json.loads((project / "automil" / "graph.json").read_text())
    nodes = {n["id"]: n for n in graph.get("nodes", {}).values()}
    assert nodes["iris_001"]["type"] == "executed"
    assert nodes["iris_001"]["composite"] >= 0.90
```

This requires plan 08-09 to verify that `automil orchestrator tick --once` exists or to add it as a sub-task. If no synchronous tick exists today, the alternative is `subprocess.Popen(["automil", "orchestrator", "start"], ...)` + bounded `time.sleep(60)` + poll for `result.json` in the archive dir + send SIGTERM.

**Rationale:** D-205's CI sub-gate is the load-bearing verification of decoupling. A test that only runs the consumer script standalone misses the framework's own contract surface (daemon ingestion, graph promotion, schema validation). Plans for milestone-acceptance gates should not punt on the harness shape.

---

### F-05, HIGH, plan 08-04 task 1 (call-site warning test gap)

**Description:** Plan 08-04 task 1 prose says "type-mismatch handling: env.required value is not a list -> emit warning, do not crash" and the call-site code:

```python
if raw_required and not isinstance(raw_required, list):
    warnings.append(
        f"config.yaml: env.required must be a list of var names; "
        f"got {type(raw_required).__name__}, ignoring."
    )
```

The unit test `test_required_not_a_list_returns_empty` only verifies the helper returns `[]`; it does NOT assert that a warning is emitted. Plan 08-04's success criteria do not include a call-site warning emission test.

The existing pattern (`env.passthrough`) at check.py line 217-220 emits a similar warning; that wasn't tested either, but Phase 8 is the milestone-acceptance phase and DEC-05 explicitly is about "fail fast at startup" + "clear error". A non-list `env.required` should produce an OPERATOR-VISIBLE warning, not a silent empty list.

**Suggested fix:** Add an 8th test to plan 08-04 task 3:

```python
def test_required_not_a_list_emits_warning(monkeypatch, capsys):
    """D-202: type-mismatch surfaces as an operator-visible warning."""
    # Either invoke check() via CliRunner and capture stdout/stderr,
    # OR refactor the warning emission into a returned list and test that.
    # ...
```

Alternatively, refactor `_validate_env_required` to ALSO return a `(missing, warnings)` tuple so the helper-level test can cover it.

**Rationale:** "Decide engineering, ask features" (Leo memory): a warning emission with a known broken-config input is exactly the kind of testable invariant that should not depend on operator vigilance.

---

### F-06, HIGH, plan 08-04 + 08-08 (config.yaml.j2 inline AUTOBENCH example coordination)

**Description:** Plan 08-04 step C says:

> The framework purity grep gate (08-08) flags `AUTOBENCH_` as part of its bad-token tuple. The `# e.g. ...` comment is the most pragmatic way to communicate the example without baking the value into the default; if the purity test rejects it, plan 08-08 adds the comment line to the allowlist (one line, identical to the existing 2-comment allowlist for `_orchestrator_daemon.py:54` and `verify_repro.py:84`). The planner of 08-08 must coordinate this allowlist entry.

Plan 08-08's task 1 step B says:

> If a hit is returned (executor of 08-04 retained the example comment), record the file:line and add a 3rd allowlist entry. If empty (executor dropped the example comment), only 2 entries are needed.

This punts the lock-in to the executor. The result is that plan-execute-time has two valid behaviors for plan 08-04 (with example, without example) and two valid behaviors for plan 08-08 (3 allowlist entries, 2 allowlist entries), giving a 2x2 matrix where only 2 cells are consistent. If 08-04 ships with the example AND 08-08 ships with 2 allowlist entries (because the 08-08 executor checked before 08-04 landed in the same wave), the gate fails.

**Suggested fix:** Lock at plan-write time. Recommendation: DROP the inline AUTOBENCH example from `config.yaml.j2`. The CHANGELOG 8.0.0 BREAKING text in plan 08-10 already shows the migration recovery snippet. Adding the same example to a framework template that scans for AUTOBENCH-purity is friction. Plan 08-04 step B becomes:

```yaml
env:
  required: []         # Vars that must be set at startup (validated by automil check)

  passthrough:
    - AUTOMIL_*       # All automil framework variables (includes AUTOMIL_RUNTIME)
    - AUTOMIL_RUNTIME # Runtime declaration: explicit, never inferred (D-87)
```

No autobench reference; no allowlist coordination needed. Plan 08-08 keeps `_ALLOWLIST` at 2 entries (plus the F-01 fix for revert_baseline.py).

**Rationale:** "Decide engineering" (Leo memory): the inline-example-vs-allowlist matrix is exactly the kind of decision that compounds across executors if left ambiguous. Lock it.

---

### F-07, HIGH, plan 08-04 + missing config.yaml.j2 scoring block (DEC-04)

**Description:** REQUIREMENTS.md DEC-04 verbatim: "Composite scoring formula is config-driven (`automil/config.yaml: scoring.formula` or `scoring.entry_point`); not hardcoded to autobench's 4-key (val_auc + val_bacc + test_auc + test_bacc) recipe".

ROADMAP.md Phase 8 success criterion 3 says the same.

CONTEXT D-200 reduces the framework's role to "consumer's training script writes `composite` scalar to `result.json`. Config exists for documentation + JSON-Schema validation hints only." This is a valid interpretation of DEC-04 (scoring is config-DOCUMENTED rather than config-evaluated), but to honor the requirement's first half ("config-driven"), the framework template `config.yaml.j2` needs a `scoring:` block so new consumers SEE the field on `automil init`.

The sklearn-iris consumer config in plan 08-06 task 2 includes:
```yaml
scoring:
  formula: "accuracy"
```

Plan 08-04 task 2 extends `config.yaml.j2` with `env: required: [] passthrough: [...]`. NO plan extends `config.yaml.j2` with a `scoring:` block. The template currently has no `scoring:` field anywhere (verified at line 90 of the template). After Phase 8 ships, fresh `automil init` projects do not document the scoring contract.

**Suggested fix:** Add to plan 08-04 task 2 step B (or a new task 4):

```yaml
# --- Composite scoring (DEC-04 / D-200) ---
# scoring.formula is documentation-only: the framework does NOT evaluate it.
# Your training script computes composite and writes it to result.json directly.
# State the formula here so collaborators understand your composite recipe.
scoring:
  formula: "TODO: document your composite (e.g. accuracy, or 0.4*val_auc + 0.6*val_bacc)"
```

Plus a regression test in plan 08-04 task 3 (or 08-10 clause 3):

```python
def test_template_has_scoring_block():
    template = (REPO_ROOT / "src/automil/templates/config.yaml.j2").read_text()
    assert "scoring:" in template
    assert "formula:" in template
```

**Rationale:** DEC-04 is a Phase 8 success criterion. Without the template field, fresh consumers do not discover the contract. The ROADMAP wording "config-driven" is plainly violated by a template that omits the field entirely.

---

### F-08, HIGH, plan 08-09 task 3 (sub-gate A path probe)

**Description:** sub-gate A test body:

```python
autobench_project = ccrcc_data_root.parent / "ccrcc"
if not (autobench_project / "automil" / "config.yaml").exists():
    pytest.skip(...)
```

Per CLAUDE.md "Monorepo Structure", autobench experiments live at `benchmarks/experiments/<dataset>/automil/` (e.g. `benchmarks/experiments/ccrcc/automil/`). `AUTOBENCH_CCRCC_ROOT` is set to a DATASET path (per `benchmarks/datasets/cptac/ccrcc.yaml` and `benchmarks/.env.example`), not a project path. `ccrcc_data_root.parent / "ccrcc"` will not resolve to `benchmarks/experiments/ccrcc/`.

Result on Leo's workstation:
- AUTOBENCH_CCRCC_ROOT is set (e.g. to `/data/ccrcc`)
- `(autobench_project / "automil" / "config.yaml").exists()` checks `/data/automil/config.yaml` which does not exist
- Sub-gate A skips with the misleading message "autobench CCRCC project not at /data/ccrcc; this sub-gate is workstation-shape-dependent"
- D-205 sub-gate A silently never runs

**Suggested fix:** Change the path probe to use the autoMIL repo root + benchmarks layout:

```python
autobench_project = _REPO_ROOT / "benchmarks" / "experiments" / "ccrcc"
if not (autobench_project / "automil" / "config.yaml").exists():
    pytest.skip(
        f"autobench CCRCC experiment dir not found at {autobench_project}; "
        f"clone the autobench monorepo into benchmarks/experiments/ccrcc/."
    )
```

This is the deterministic shape per CLAUDE.md monorepo section.

**Rationale:** Sub-gate A is the load-bearing CCRCC reproduction test (D-205, D-208 clause 8). A wrong path probe degrades the gate from "verify-on-workstation" to "always-skip", silently dropping the milestone-acceptance verification.

---

### F-09, MEDIUM, plan 08-10 clause 11 (assertion logic bug)

**Description:** plan 08-10 task 1 clause 11:

```python
dec_complete_count = roadmap_text.count("Complete") + (
    _REPO_ROOT / ".planning" / "REQUIREMENTS.md"
).read_text().count("DEC-") if (_REPO_ROOT / ".planning" / "REQUIREMENTS.md").exists() else 0
assert dec_complete_count >= 1, (...)
```

Python operator precedence: this parses as
```python
dec_complete_count = (roadmap_text.count("Complete") + (...).read_text().count("DEC-")) if (...).exists() else 0
```

So when REQUIREMENTS.md exists (always, in this repo), `dec_complete_count` = `roadmap_text.count("Complete") + REQUIREMENTS_md.count("DEC-")`. REQUIREMENTS.md mentions DEC-XX many times (>= 7 for the requirement IDs alone, plus traceability rows). The assertion `>= 1` always passes.

The clause is supposed to verify that ROADMAP/REQUIREMENTS reflect Phase 8 completion. The current logic does not.

**Suggested fix:** Replace with a deterministic check on REQUIREMENTS.md traceability table rows. REQUIREMENTS.md lines 238-244 currently read:

```
| DEC-01 | Phase 8 | Pending |
...
| DEC-07 | Phase 8 | Pending |
```

The assertion becomes:

```python
req = (_REPO_ROOT / ".planning" / "REQUIREMENTS.md").read_text()
for dec_id in ("DEC-01", "DEC-02", "DEC-03", "DEC-04", "DEC-05", "DEC-06", "DEC-07"):
    pending = f"| {dec_id} | Phase 8 | Pending |"
    complete = f"| {dec_id} | Phase 8 | Complete |"
    assert pending not in req, f"{dec_id} still marked Pending in REQUIREMENTS.md"
    assert complete in req, f"{dec_id} not marked Complete in REQUIREMENTS.md"
```

**Rationale:** A clause assertion that always passes is worse than no clause at all because it gives false confidence. The fix also implicitly forces task 4 to commit to editing REQUIREMENTS.md (resolves F-10).

---

### F-10, MEDIUM, plan 08-10 task 4 (REQUIREMENTS.md scope ambiguity)

**Description:** Plan 08-10 task 4 step C says "if REQUIREMENTS.md is in scope for this plan ... include the edit; if not ... best-effort". `files_modified` lists only `tests/acceptance/test_phase8_acceptance.py`, `CHANGELOG.md`, `.planning/STATE.md`, `.planning/ROADMAP.md`. The "best-effort" language abandons milestone-completion to executor judgment.

**Suggested fix:** Add `.planning/REQUIREMENTS.md` to plan 08-10's `files_modified` frontmatter. Make task 4 step C explicit:

> Step C, update `.planning/REQUIREMENTS.md` traceability table rows for DEC-01 through DEC-07: change Status column from `Pending` to `Complete`. Match the existing pattern at lines 238-244.

The wave-4 file-disjointness audit holds: REQUIREMENTS.md is touched by no other Phase 8 plan.

**Rationale:** Cross-doc consistency at milestone is non-negotiable; punting fragments responsibility.

---

### F-11, MEDIUM, plan 08-02 task 1 step H (no-op instruction)

**Description:** Plan 08-02 task 1 step H tells the executor:

> any local variables named val_auc/test_auc/val_bacc/test_bacc parsed from results.tsv MUST flow into the metrics input of add_executed under those same key names. After Step A's refactor of add_executed, they end up at node["metrics"][...] automatically.

Verified in current source at graph.py:791-805: `add_executed(metrics={"composite": ..., "test_auc": test_auc, "test_bacc": test_bacc, "val_auc": val_auc, "val_bacc": val_bacc, ...})`. After Step A's refactor, this dict is passed through `dict(metrics)` and ends up under `node["metrics"]`. NO CHANGE is required to lines 779-810.

The plan's step H is then a no-op instruction: "ensure things continue to do what they already do." This is harmless but misleading: executors may interpret "MUST flow into" as needing a code change and add unnecessary edits.

**Suggested fix:** Replace step H with a verify-only instruction:

> Step H, verify the bootstrap loader at lines 779-810 needs NO code change. The `add_executed` call at lines 791-805 already passes `test_auc`/etc as metrics dict keys. After Step A's refactor of `add_executed` to dict-spread, those keys land at `node["metrics"][...]` automatically. Acceptance: `grep -A3 "add_executed" src/automil/graph.py | grep -c "test_auc"` returns at least 1 (inside the import_from_tsv call site).

**Rationale:** Plans should distinguish "code change" from "behavior assertion" instructions. Conflating them invites unnecessary edits.

---

### F-12, MEDIUM, plan 08-09 task 2 (ccrcc_data_root layout assumption)

**Description:** conftest.py:

```python
if not (root / "splits").exists():
    pytest.skip(f"CCRCC root {raw} does not have expected splits/ subdirectory.")
```

Per `benchmarks/datasets/cptac/ccrcc.yaml` and the autobench config layout, `AUTOBENCH_CCRCC_ROOT` typically points to a dataset root that has `features/`, `mapping.csv`, and split files at potentially-separate paths (configured per-dataset YAML). The `splits/` subdirectory layout is not universal.

**Suggested fix:** Drop the splits/ check, or make it advisory:

```python
@pytest.fixture
def ccrcc_data_root() -> Path:
    raw = os.environ.get("AUTOBENCH_CCRCC_ROOT")
    if not raw:
        pytest.skip("AUTOBENCH_CCRCC_ROOT not set; sub-gates A and C require real CCRCC data.")
    root = Path(raw)
    if not root.exists():
        pytest.skip(f"AUTOBENCH_CCRCC_ROOT={raw} does not exist on this host")
    return root
```

The "does this look like CCRCC?" check is sub-gate-A's responsibility (see F-08), not the fixture's.

**Rationale:** Fixtures should be liberal; tests should be specific.

---

### F-13, LOW, em-dash audit

**Description:** Em-dash audit performed via `grep -nP "\x{2014}|\x{2013}"`:
- All 10 Phase 8 plans: zero matches.
- 08-CONTEXT.md, 08-RESEARCH.md, 08-PATTERNS.md, 08-PLAN-SUMMARY.md: zero matches.
- Pre-existing em-dashes in source files (NOT touched by plans):
  - `src/automil/backends/_orchestrator_daemon.py`: 24 lines (lines 50, 55, 65, 76, 120, 181, 282, 289, 338, 349, 388, 429, 431, 439, 448, 539, 694, 853, 887, 1015, 1062, 1226, 1232, 1233, 1241).
  - `src/automil/cli/check.py`: 5 lines (355, 364, 381, 415, 724).
  - `src/automil/graph.py`: zero matches.
  - `src/automil/templates/config.yaml.j2`: zero matches.
  - `src/automil/viz/static/app.js`: zero matches in source; uses `&Delta;` HTML entity at line 244 (which is acceptable inside JS-rendered HTML; no plan modifies that line).

Plans correctly do not instruct the executor to remove pre-existing dashes (out of scope per Phase 7 F-09 precedent). HOWEVER, the allowlist anchor at `_orchestrator_daemon.py:54` itself is on a line whose neighbors (lines 50, 55) contain em-dashes, and any future cleanup that re-flows these comments WILL break the anchor substring lookup. Recommend a low-priority follow-up to re-flow the comment block (lines 45-55) without em-dashes; not a Phase 8 blocker.

**Suggested fix:** None blocking. Note the constraint in the Phase 8 follow-ups section of STATE.md (per plan 08-10 task 3).

---

### F-14, LOW, plan 08-10 task 3 (plan count drift)

**Description:** Plan 08-10 task 3 STATE.md frontmatter template:

```yaml
progress:
  total_phases: 9
  completed_phases: 9
  total_plans: 92
  completed_plans: 92
  percent: 100
```

with parenthetical "Plan count grows from 82 to 92 with Phase 8's 10 plans; adjust if the actual plan count differs."

The actual plan count across phases 0-8 is deterministic and resolvable now. Punting to the executor risks number drift.

**Suggested fix:** Run `find .planning/phases -name "*-PLAN.md" | wc -l` at plan-write time and lock the resulting count. From the directory listing observed: each phase has its own plan count. Plan 08-10 should pre-fill the actual integer, not a placeholder.

**Rationale:** Milestone-acceptance metadata should be exact, not estimated.

---

## 4. Goal-Backward Map (DEC-01..07 -> plans)

Every requirement has at least one implementing plan AND at least one verifying clause-test in 08-10:

| Req ID | Verbatim text from REQUIREMENTS.md | Implementing plans | D-208 verifying clause |
|--------|-----------------------------------|--------------------|-----------------------|
| **DEC-01** | zero autobench refs in `src/automil/` | 08-05 (purge), 08-08 (CI gate) | clause 1 (subprocess invokes 08-08) + clause 7 (allowlist anchors still present) |
| **DEC-02** | sklearn-iris second consumer end-to-end | 08-06 (consumer files), 08-09 sub-gate B | clause 5 (path checks) + clause 8 (sub-gate B passes) |
| **DEC-03** | result.json JSON-Schema validated at ingestion | 08-01 (schema + validator), 08-05 (daemon ingest call) | clause 2 (schema + daemon source grep + 08-01 tests) |
| **DEC-04** | composite scoring config-driven; not hardcoded to 4-key | 08-02 (graph dict-spread), 08-03 (viz reader); **PARTIAL: see F-07** for missing scoring block in template | clause 3 (graph source grep + 08-02 tests) |
| **DEC-05** | env.required validated by automil check | 08-04 (validator + template + tests) | clause 4 (check.py + template grep + 08-04 tests) |
| **DEC-06** | training-script contract documented | 08-07 (docs + tests) | clause 6 (subprocess invokes 08-07's docs-exist test) |
| **DEC-07** | final reproduction sanity (CCRCC + sklearn-iris) | 08-09 (3 sub-gates) | clause 8 (subprocess invokes sub-gate B; A+C skip cleanly) |

**Coverage assessment:** 6 of 7 DECs are fully covered. DEC-04 is partially delivered per F-07 (the framework template `config.yaml.j2` does not surface the `scoring:` field that the requirement names verbatim). Treat F-07 as a HIGH addressing this gap.

---

## 5. D-208 Acceptance Gate Map (11 clauses -> satisfying test)

| Clause | D-208 text | Satisfying test in 08-10 | Test source | Status |
|--------|-----------|---------------------------|-------------|--------|
| 1 | zero autobench refs allowlisted | `test_d208_clause_01_framework_purity` | invokes `tests/test_framework_purity.py` | OK pending F-01 |
| 2 | result.schema.json + daemon validates | `test_d208_clause_02_result_schema_validation` | static reads + `tests/test_result_schema_validation.py` | OK |
| 3 | graph.py dict-spread + composite-only Pareto | `test_d208_clause_03_graph_dict_spread` | static greps + `tests/test_graph_dict_spread.py` | OK pending F-03 (cap-killed branch) |
| 4 | env.required + env.passthrough; check validates | `test_d208_clause_04_env_required_validator` | static greps + `tests/cli/test_check_env_required.py` | OK pending F-05 (warning emission) |
| 5 | examples/sklearn-iris/ exists; decoupled | `test_d208_clause_05_sklearn_iris_consumer_exists` | path checks + `from automil` absence | OK |
| 6 | docs/training-script-contract.md covers 6 items | `test_d208_clause_06_training_script_contract_doc` | invokes `tests/test_phase8_docs_exist.py` | OK |
| 7 | tests/test_framework_purity.py PASSES | `test_d208_clause_07_framework_purity_grep_gate` | static greps + invokes purity test | OK pending F-01 |
| 8 | sub-gate B PASSES; A+C skip cleanly | `test_d208_clause_08_final_acceptance_gate` | invokes sub-gate B + asserts no FAILED | OK pending F-04 + F-08 |
| 9 | Phase 7 baseline (838+) preserved; +>=10 new | `test_d208_clause_09_baseline_plus_10_tests` | pytest --collect-only with floor 858 | OK |
| 10 | CHANGELOG 8.0.0 BREAKING entry | `test_d208_clause_10_changelog_8_0_0` | grep on CHANGELOG.md | OK |
| 11 | ROADMAP + STATE + REQUIREMENTS reflect complete | `test_d208_clause_11_state_roadmap_complete` | grep on STATE.md/ROADMAP.md/REQUIREMENTS.md | BUG see F-09 |

**Aggregator coverage:** All 11 clauses have a clause-test. 8 of 11 clause-tests are correct as written; 3 have issues (F-01, F-03, F-04, F-05, F-08, F-09 all flow into clause-test correctness).

---

## 6. Migration Delta Coverage

Per OQ-9 + Migration Delta Punch List in 08-RESEARCH.md, the named-field reader migration sites:

| Site | File:Line | Migration owner | Plan-level coverage |
|------|-----------|-----------------|---------------------|
| add_executed write site | graph.py:122-145 | 08-02 | Covered |
| promote write site | graph.py:205-219 | 08-02 | Covered |
| reconcile completion ingest assembly | graph.py:559-571 | 08-02 | Covered |
| archive recovery loop A | graph.py:607-630 | 08-02 | Covered |
| archive recovery loop B | graph.py:670-705 | 08-02 | Covered |
| results.tsv bootstrap loader | graph.py:779-810 | 08-02 | Covered (no-op per F-11) |
| _reevaluate_descendants Pareto | graph.py:241-270 | 08-02 | Covered (Option B) |
| reconcile Pareto | graph.py:546-552 | 08-02 | Covered (Option B) |
| archive-recovery Pareto | graph.py:674-681 | 08-02 | Covered (Option B) |
| viz/static/app.js metric reader | viz/static/app.js:227-237 | 08-03 | Covered (single-line change) |
| **_orchestrator_daemon.py:1055-1057 cap-killed reconcile** | **NOT migrated** | **deferred per OQ-7** | **F-03 BLOCKER (silent re-injection of top-level keys)** |
| _orchestrator_daemon.py:1289-1298 results.tsv writer | NOT migrated | deferred per OQ-8 | OK (autobench-shaped writer; sklearn-iris emits 0.0 for absent keys; degenerate behavior is correct) |

**Daemon-internal node mutation surface coverage:** Critically incomplete. The cap-killed branch at line 1055 writes named-field keys at top level AFTER plan 08-02 deletes them from add_executed/promote, producing a hybrid post-D-200 state. F-03 makes this BLOCKER; the deferral has no decision basis.

**Test migration coverage** (tests that DEPEND on top-level node["test_auc"] etc.):
- `tests/test_graph.py`: passes named keys as INPUT to add_executed; does NOT read them back from `node["test_auc"]`. After D-200 the input flow into `node["metrics"]` is identical for these tests; they continue to pass without modification. Verified by grep: zero `node["test_auc"]` reads in `tests/test_graph.py`.
- `tests/test_orchestrator_env_whitelist.py`: see F-02 (test_pythonpath_overrides_whitelist_value breakage).
- All other tests: zero direct reads of `node["test_auc"]`/etc. confirmed by grep.

---

## 7. Phase 0-7 Baseline Preservation

Phase 7 closed at "848+ tests" per 08-PLAN-SUMMARY (line 192) and 08-RESEARCH baseline. Phase 8 plans add:

| Plan | New tests | Cumulative |
|------|-----------|------------|
| 08-01 | +8 | 856 |
| 08-02 | +6 | 862 |
| 08-03 | +3 | 865 |
| 08-04 | +8 (+1 if F-05 fix) | 873-874 |
| 08-05 | +6 | 879-880 |
| 08-07 | +8 | 887-888 |
| 08-08 | +3 | 890-891 |
| 08-09 | +3 | 893-894 |
| 08-10 | +11 | 904-905 |

D-208 clause 9 floor: 858. Plans target ~904, well above the floor. **No regression risk if F-02 is fixed (it converts a passing test, not a deletion).**

Plan 08-05 task 3 (test conversion) does NOT subtract from the test count: `test_autobench_root_still_injected_phase0` becomes `test_autobench_root_not_auto_injected_phase8`. F-02's fix (delete `test_pythonpath_overrides_whitelist_value`) reduces by 1; if a replacement is added, net zero.

---

## 8. Backwards-Compatibility / CHANGELOG Migration Audit

Plan 08-10 task 2 ships a `## 8.0.0 - Phase 8 decoupling completion + final acceptance (unreleased)` entry with 3 BREAKING subsections:
1. `env.required` mandatory in automil/config.yaml
2. `node["test_auc"]` etc. no longer at top level
3. AUTOBENCH_ROOT no longer auto-injected

**Migration recovery snippets** are present and operator-facing.

**Anchor strings** (clause 10 grep targets): `## 8.0.0`, `BREAKING`, `env.required`, `node["metrics"]` OR `metrics dict` , all present in the proposed body.

**Pre-existing CHANGELOG.md em-dashes**: confirmed at lines 3, 50, 87, 89 (Phase 6 entries). Plan 08-10 correctly grandfathers these; new 8.0.0 section must not introduce new dashes.

**Backward-compat behavioral guarantees:**
- pre-D-200 graph.json with top-level `test_auc` keys: still readable; framework no longer writes them. Per CONTEXT D-200, this is forward-compatible cleanup. graph.json `schema_version` is NOT bumped (correct call).
- pre-D-200 `results.tsv`: bootstrap loader at graph.py:779-810 unchanged. New nodes written via the autobench-shaped writer (per OQ-8 deferred). Sklearn-iris consumers write 0.0 for absent keys. Degenerate but correct.
- pre-D-202 `automil/config.yaml` files (no env.required key): `_validate_env_required` returns `[]` for missing/None env.required. NO breakage. The CHANGELOG migration text correctly frames `env.required` as recommended-but-empty-default; the BREAKING claim is for the *consumer-side* (operator must add the key to surface dataset paths) not the framework-side (no immediate failure).

**Verdict**: CHANGELOG migration plan is correctly framed and operator-recoverable. No findings.

---

## 9. Final Acceptance Sub-Gate Map

| Sub-gate | D-205 description | Marker | CI behavior | Pre-conditions | Status |
|----------|-------------------|--------|-------------|----------------|--------|
| A | CCRCC node_0176 reproduces composite within +-0.005 of 0.502 | requires_ccrcc_data | SKIP | AUTOBENCH_CCRCC_ROOT set + autobench project at benchmarks/experiments/ccrcc/ | F-08 BLOCKER (path probe wrong) |
| B | sklearn-iris consumer end-to-end via real CLI | unmarked | RUN | examples/sklearn-iris/ exists | F-04 HIGH (test bypasses orchestrator path) |
| C | both consumers register in same project | requires_ccrcc_data | SKIP | A + B preconditions; pytest.skip in plan body acknowledges incomplete shape | OK as documented (executor-deferred per CONTEXT) |

CI runs sub-gate B unconditionally per `pytest -m "not requires_ccrcc_data and not requires_slurm and not requires_ray"`. Sub-gates A and C skip cleanly via the `requires_ccrcc_data` marker + `ccrcc_data_root` fixture. **F-04 + F-08 must be fixed for sub-gates B and A to deliver their D-205 contracts; sub-gate C is acceptable as workstation-deferred.**

---

## 10. Architectural Tier Compliance

Phase 8 RESEARCH.md does not contain an `## Architectural Responsibility Map` section. **Dimension 7c: SKIPPED (no responsibility map found).**

Per Phase 7 precedent and CLAUDE.md "Architecture" section, the implicit tier assignments in Phase 8 plans are:
- Schema validation (08-01): framework data layer (correct)
- AUTOBENCH purge + daemon ingest validate (08-05): backend orchestrator tier (correct)
- env.required validator (08-04): CLI tier (correct)
- Framework purity grep (08-08): test/lint tier (correct)
- Sklearn-iris consumer (08-06): consumer tier (correct, fully decoupled)

No tier mismatches detected.

---

## 11. Cross-Plan Data Contract Coverage

| Data flow | Source plan | Sink plan | Compatibility | Status |
|-----------|-------------|-----------|---------------|--------|
| `automil.schemas.validate_result` | 08-01 | 08-05 (daemon ingest), 08-09 sub-gate B | API stable; both import names verified | OK |
| `node["metrics"]` shape (post-D-200) | 08-02 | 08-03 (viz reader), 08-05 (daemon ingest cleanup if F-03 fix lands) | dict-spread + defensive `(node.metrics \|\| {})` in viz | OK if F-03 lands |
| `env.passthrough` consumer-driven AUTOBENCH_*_ROOT | 08-05 (daemon reads list) | 08-04 (config.yaml.j2 template surfaces field) | seam preserved; consumer-side migration documented in CHANGELOG | OK if F-06 lands |
| `result.schema.json` cross-link | 08-01 (file) | 08-05 (error message), 08-07 (doc link), 08-09 (sub-gate B validates) | path stable; no schema version field, schema is single-source | OK |
| `examples/sklearn-iris/` directory | 08-06 (creates) | 08-09 sub-gate B (copies into tmp), 08-07 doc cross-link | shutil.copytree path stable; cross-link grep stable | OK |
| `tests/test_framework_purity.py` | 08-08 (creates) | 08-10 clause 1+7 (subprocess invokes) | invocation stable | OK if F-01 lands |
| `tests/acceptance/test_final_phase8_acceptance.py::test_subgate_b_*` | 08-09 (creates) | 08-10 clause 8 (subprocess invokes) | invocation stable | OK if F-04 lands |

**Cross-plan contract risks:** All data flows are compatible at the contract level. The 3 contracts requiring F-01/F-03/F-04/F-06 fixes are individually flagged. No incompatible-transform conflicts (both viz reader and daemon read `node["metrics"]` as opaque dict; no transform applied).

---

## 12. CLAUDE.md Compliance

Audited Phase 8 plans against CLAUDE.md project instructions:

| CLAUDE.md directive | Compliance |
|---------------------|------------|
| Address Leo at the start of any response | N/A (plan content, not a response) |
| Plan first; check in before implementation | Plans exist; this audit runs the verification | OK |
| Use subagents liberally | Per Phase 8 plan-by-plan structure with wave parallelism | OK |
| Self-improvement loop (lessons.md updates) | NOT in scope for plans (post-execution work) | N/A |
| Verification before "done" | All plans have `<acceptance_criteria>` + 08-10 D-208 aggregator | OK |
| Demand elegance | Mostly; F-01..F-08 are exactly the surgical fixes "elegance" demands | RETRY |
| Autonomous bug fixing | N/A (no extant bugs to fix in plans) | N/A |
| **Never use em dashes** | All Phase 8 plans/CONTEXT/RESEARCH/PATTERNS/PLAN-SUMMARY: zero em-dashes (verified) | OK |
| **Never blind-checkout for rollback** | gate/manifest.py pattern (path.unlink) is the analog called out in 08-PATTERNS.md "Atomic write" section | OK |
| **autoMIL is generic; autobench is one consumer** | DEC-01..07 are exactly this principle; the framework-purity gate enforces it | OK (F-01/F-06 fixes tighten enforcement) |
| **Skills for setup, CLI for runtime** | N/A for Phase 8 (no skill changes) | N/A |
| **Decide engineering, ask features** | F-05/F-06/F-08/F-14 are decisions that should be locked at plan-write time, not executor time | RETRY |
| **No autobench-specific paths in src/automil/** | DEC-01 (the requirement); F-01/F-06/F-08 are the audit findings on this | RETRY |

---

## 13. Frontmatter Validity

All 10 plans have valid YAML frontmatter:

| Plan | wave | depends_on | files_modified count | autonomous | requirements |
|------|------|-----------|----------------------|------------|--------------|
| 08-01 | 1 | [] | 4 | true | [DEC-03] |
| 08-02 | 1 | [] | 2 | true | [DEC-04] |
| 08-03 | 1 | [] | 2 | true | [DEC-04] |
| 08-04 | 2 | [] | 3 | true | [DEC-05] |
| 08-05 | 2 | ["08-01"] | 3 | true | [DEC-01, DEC-03] |
| 08-06 | 3 | ["08-01"] | 6 | true | [DEC-02] |
| 08-07 | 3 | ["08-01"] | 2 | true | [DEC-06] |
| 08-08 | 3 | ["08-04", "08-05"] | 1 | true | [DEC-01] |
| 08-09 | 4 | ["08-01".."08-08"] | 4 | true | [DEC-02, DEC-07] |
| 08-10 | 4 | ["08-01".."08-09"] | 4 | true | [DEC-01..07] |

Wave numbering is 1-indexed (correct per Phase 7 precedent). Dependency graph is acyclic. File-disjointness confirmed within each wave (verified in PLAN-SUMMARY lines 21-40 and re-verified by direct file enumeration). **F-10 minor finding**: 08-10 should add `.planning/REQUIREMENTS.md` to `files_modified` if F-09/F-10 fixes land.

---

## 14. Recommended Next Action

**RETRY** with the following surgical fixes in a single revision pass:

1. **F-01**: plan 08-08 , add 3rd allowlist entry for `revert_baseline.py:87` (or rewrite the offending string to use a non-`benchmarks/` example).
2. **F-02**: plan 08-05 , add explicit task step to delete or rewrite `test_pythonpath_overrides_whitelist_value`.
3. **F-03**: plan 08-05 (or new plan 08-05b) , migrate `_orchestrator_daemon.py:1055-1057` cap-killed-reconcile branch to dict-spread.
4. **F-04**: plan 08-09 , replace sub-gate B body with full `automil submit` + orchestrator-tick invocation OR explicitly document why the standalone-script approach satisfies D-205 (the latter requires a CONTEXT amendment).
5. **F-05**: plan 08-04 , add 9th unit test for warning emission on non-list `env.required` (via CliRunner or refactored helper return shape).
6. **F-06**: plan 08-04 , drop the inline AUTOBENCH example comment from `config.yaml.j2`; lock allowlist size at 2 entries (plus F-01's 3rd entry for revert_baseline.py).
7. **F-07**: plan 08-04 , add `scoring:` block to `config.yaml.j2` template + regression test.
8. **F-08**: plan 08-09 , fix sub-gate A path probe to `_REPO_ROOT / "benchmarks" / "experiments" / "ccrcc"`.
9. **F-09**: plan 08-10 clause 11 , replace the always-passing assertion with the deterministic REQUIREMENTS.md row check.
10. **F-10**: plan 08-10 , add `.planning/REQUIREMENTS.md` to `files_modified`; commit task 4 step C to the deterministic edit.
11. **F-11**: plan 08-02 task 1 step H , rewrite as verify-only.
12. **F-12**: plan 08-09 , relax `ccrcc_data_root` fixture; drop the `splits/` check.

After revision, the 3 BLOCKERs and 5 HIGHs are addressable; the 4 MEDIUMs and 2 LOWs can ship as-is or fold into the same revision.

**Estimated revision effort:** ~30-45 minutes of planner work; surgical, no plan rewrites required.

**Once revisions land, re-run plan-checker** for iteration 2; expected pass on subsequent verification.

---

## 15. Audit Closure

| Dimension | Status |
|-----------|--------|
| 1. Goal-backward fidelity (DEC-01..07 covered) | RETRY (DEC-04 partial per F-07) |
| 2. API correctness vs source (line numbers verified) | RETRY (F-01: revert_baseline.py:87 missed; F-08: sub-gate A path wrong) |
| 3. File-disjointness for parallel waves | OK |
| 4. Acceptance criteria are bash-verifiable | OK |
| 5. Anti-pattern scan (em/en dashes, HTML entities, git checkout for rollback, autobench refs in framework) | OK (F-13 LOW informational) |
| 6. Frontmatter validity | OK (F-10 minor) |
| 7. Dependency-order soundness | OK |
| 8. Migration delta completeness | RETRY (F-03 cap-killed branch) |
| 9. D-208 11-clause aggregator | OK (F-09 clause 11 logic bug) |
| 10. Phase 0-7 baseline preservation | OK |
| 11. Backwards-compat (CHANGELOG migration) | OK |
| 12. Final acceptance sub-gates | RETRY (F-04 sub-gate B, F-08 sub-gate A) |

**Overall verdict: RETRY** , patch all 3 BLOCKERs + 5 HIGHs, then re-verify.

End of plan check, iteration 1.
