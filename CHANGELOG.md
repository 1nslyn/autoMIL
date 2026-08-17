# Changelog

autoMIL: F2-readiness framework refactor.

> Releases through v1.0 use the phase-based numbering below (`8.0.0` = the v1.0
> milestone). Post-v1.0 releases follow semver and map to git tags `v1.1.0`,
> `v1.2.0`, and `v1.2.1`.

## Unreleased

- **BREAKING: the "composite" concept is retired — single-metric optimization
  is the contract.** The selection signal IS the declared primary validation
  metric; the result-contract field `composite` is now `primary_value` and
  `composite_se` is `primary_se` (graph nodes, `meta.best_primary_value`,
  fold entries, results.tsv columns, viz, registry VariantSpec field +
  `Primary_value:` doc label, and the scoring API follow). The write-only
  `metrics.composite_formula` config key is deleted outright — the
  declaration pair is `scoring.formula` + `metrics.primary` — and
  `automil rank` prints the declared metric's name in its leaderboard
  header. Training scripts emitting the old field fail loudly at ingest;
  graph.json files predating the rename are migrated on load (schema 3).
  SMMILe is removed entirely (pipeline adapter, vendored lib, tests): it
  was never reachable from `--framework` and is not in the campaign.


- **Protocol `preprint-v3`: per-task-family reporting + val-loss checkpoint
  selection.** The campaign's reporting primary is now the field's canonical
  metric per task family (binary/multiclass → `test_auc`, ordinal grade →
  `test_qwk`, survival → `test_c_index`), with `task_family` frozen into the
  manifest cell identity (manifest schema 6, campaign `automil-preprint-130-v6`,
  census-locked 39/13/13/65) and family-EXACT schema locks at both ends of the
  sealed-evidence chain; selection everywhere stays the primary VALIDATION
  metric (`scoring.formula: val_auc` / `val_c_index`), so `test_qwk` reports
  but never selects. Classification checkpoint selection moved off the
  quantized epoch metric onto continuous validation loss on every tile arm
  (nnMIL early stopping, ABMIL/DTFD shared CE helper, CLAM upstream-loss
  with its tracker now UNCONDITIONAL — the tunable `early_stopping` flag
  only gates early termination, so no legal proposal can revert the arm to
  final-epoch weights; survival was already loss-selected), and the whole
  checkpoint-selection layer joined `registry.protected` in the 9 cohort
  templates. `revert-baseline` probes protected patterns with git's own
  pathspec engine against the base commit's TREE (read-tree into a
  throwaway index + `ls-files`), so a protected path with no baseline
  state can no longer fail the all-or-nothing checkout — and glob patterns
  still match.

- **Adversarial-review fix round 2 (Codex gpt-5.6-sol xhigh, 17 findings —
  16 real, 1 rejected).** The firewall's fold-level seams: held-out-named
  keys inside `validation_folds[*].metrics` now fail the node closed (they
  fed the recomputed per-fold values — under the mean reducer, straight
  into the paired margin); a fold entry with no metrics block is dropped,
  never trusted (bare reported fold values forged as parent+uniform-delta
  zeroed the paired SE and collapsed the keep bar to the δ floor); the
  CR-1b recompute/refusal moved OUTSIDE the graph lock so a missing node
  or lock failure can no longer republish the unvetted reported scalar;
  an unknown frozen formula refuses instead of trusting (one typo no
  longer disables CR-1b graph-wide); `verify-repro` sanitizes the repro
  result with the same ingest contract; `reconcile --from-archive`
  re-evaluates descendants and recomputes best; schema migrations scrub
  held-out-named keys and stale `composite` scalars out of node metrics;
  duplicate fold_index fails recovery closed (phantom "completed") while
  crash payloads keep measured telemetry; CLAM's upstream EarlyStopping
  gets a non-finite guard (NaN val loss was saved as an "improvement");
  the top-level ordinal summary treats a lost held-out component as
  unestimable; the fresh-rerun escape in repair_baselines requires
  family-exact evidence on the canonical tree; the examples' configs are
  brought onto the declaration contract (sklearn-iris could not even
  seed a graph — its formula "accuracy" fails validation); rank labels
  the SE basis the gate actually applied; and the init template stays
  vocabulary-agnostic (mean default) with explicit guidance to declare a
  single-metric selector once the consumer's metric names are known.
  Rejected: "the disagreement audit stamp leaks the reported scalar" —
  that value is agent-authored (the agent wrote result.json), so the
  stamp discloses nothing new and never drives selection.

- **Adversarial-review fix round (Opus 5 max-effort, 13 findings).**
  Highlights beyond the two above: a missing nnMIL `val/loss` now reads as
  NaN (skipped epoch), never a perfect 0.0 loss that captures the
  checkpoint; a `best_*.pth` left by a prior attempt is deleted at
  EarlyStopping construction so a same-node relaunch cannot certify stale
  weights; the CR-1b refusal propagates to `completed/`, archive
  `result.json` and `results.tsv` (not just the graph node); the schema-3
  migration also renames the baseline root's `metadata.validation_folds`
  entries (paired-margin topology); legacy `composite`/`composite_se`
  results.tsv columns migrate into their successors instead of becoming
  phantom metrics; per-fold validity spans `held_out`; `reconcile
  --from-archive`'s val-recompute is status-independent; the daemon PID
  file writes atomically; and the ordinal invalid-reuse deadlock in
  `repair_baselines` is resolved (audit accepts a valid fresh rerun when
  history is unreusable; migrate skips invalid cells per cell instead of
  refusing the dataset).

- **One-session-per-host limit removed.** Activity metering is now
  multiplexed per project instead of pinned to one host-wide endpoint:
  every project declares `activity.exporter_port` in `config.yaml`
  (default 9464; campaign materialization assigns a deterministic
  per-manifest-row port), the written Claude settings carry
  `OTEL_EXPORTER_PROMETHEUS_PORT`, and every consumer — runtime hooks,
  orchestrator scrape, operator `activity close`, `automil check`, the
  campaign audit, and the launcher port probe — resolves the same
  declared port. Concurrent cell orchestrators on one host partition
  GPUs with the host-local `AUTOMIL_VISIBLE_GPUS` env (malformed values
  refuse daemon startup). Discovery therefore parallelizes per cell on
  a single station, bounded by GPUs, not by a per-host session limit.

- **Frozen agent protocol sources + per-cell launcher (Gate 2).** The
  campaign's coding-agent policy is now buildable and executable instead of
  a template: `proposal_policy.md` (the exact per-cell instruction text,
  publication-specific — no free-mode, no in-cell certify) and
  `toolset.json` (machine-readable locked tool surface) are committed
  sources; `campaign_agent_protocol.py build/verify` assembles and checks
  the publication `agent_protocol.json` (model/runtime identities pinned
  from a real throwaway session, ancestor-memory hashes auto-refreshed);
  `campaign_launch.py` + `autobench.campaign_launch` launch the formal
  session as a pure executor of the locked protocol — protocol-derived
  `claude` flags, byte-exact `CLAUDE.md` instruction render, pinned CLI
  version, frozen repository-memory surface, no user memory/plugins, cell
  settings drift refusal, exporter-port exclusivity (per cell since the
  entry above), running-orchestrator and fresh-cell preconditions, all
  fail-closed.

- **Preprint campaign integration (#39 + #40 + #41).** Native Claude
  active-time metering (30 charged attempts + 12h agent-active, protocol
  `preprint-v2`, manifest `automil-preprint-130-v5`) merged with the
  claims–mechanism alignment fixes (A1–A9, B1–B8, C-\*; A10 superseded by
  the re-sized protocol) and the campaign operator runbook. Post-merge
  hardening from the adversarial review: atomic SessionEnd finalization
  (closes the observe-then-record TOCTOU), `automil activity close`
  operator recovery for runtimes that die without their hook, promotion's
  time wall re-declared as 7d pure containment (`PROTOCOL.
  promotion_wall_clock_containment`), declared-but-unreadable cell specs
  held instead of irreversibly cancelled, obsolete `idle_grace*` cap keys
  rejected, mode-less legacy cell layouts rejected, `init` preserves
  conflicting user telemetry env, single-session refusals state the real
  contract, runbook + claims-alignment audit reconciled to the merged
  protocol.

- **Preprint dataset roster pivot.** Dropped TCGA-SKCM (NRAS), TCGA-BLCA
  (PIK3CA), and TCGA-COAD (BRAF); added CPTAC-GBM (TP53, binary),
  CPTAC-PDAC (immune_class, 3-class), and TCGA-HNSC (tumor grade, 3-class).
  The roster now spans binary mutation, 3-class immune subtype, and 3-class
  tumor grade tasks across two data sources (TCGA + CPTAC), prioritizing
  classification-task diversity over the old ≥100-OS-deaths hard gate. Grid
  math is unchanged: 165 experiments (33/dataset), 825 fold-trainings.

## v1.2.1 (2026-07-15, `git tag v1.2.1`)

- **fix(packaging):** list all three authors (Shuolin Yin, Yeonwoo Seo, Jun Ma)
  in the package `Author` field.

## v1.2.0 (2026-07-15, `git tag v1.2.0`)

**Theme: the validation-firewall, the survival task family, and the preprint
dataset roster.**

- **Validation-selection firewall + Ladder keep-margin.** Keep/discard now
  selects on the **validation** composite only — a child is kept iff
  `child.composite > parent.composite + accept_margin` (δ, seeded from
  `config.yaml`'s `scoring.accept_margin`, default 0.0). Test metrics never
  drive search.
- **Born-sealed test artifacts (val-firewall).** Test metrics are sealed at
  ingest into a quarantined `archive/<node>/certify/` block, kept out of every
  agent-facing surface (results, `run.log`, trajectories) during search, and
  revealed exactly once via the new `automil certify` command. `result.json`
  now carries validation metrics in `metrics` and test in a separate `held_out`
  block.
- **Overall-survival task.** A second task family beyond classification:
  time-to-event OS prediction for TCGA-LUAD/LGG/SKCM/BLCA/COAD, with Cox and
  discrete-time-NLL losses, patient-level concordance index via
  `scikit-survival`, 5-fold CV, and val-loss checkpoint selection. Survival is
  wired through CLAM, nnMIL, ABMIL, DTFD-MIL, and TITAN.
- **Preprint dataset roster.** TCGA trimmed to the 5-member preprint slate
  (LUAD/LGG/SKCM/BLCA/COAD); the MIL roster pinned to 4 aggregators per
  framework (`clam_mb`, `simple_mil`, `ab_mil`, `dtfd_mil`); a TITAN
  slide-encoder arm added (CONCH v1.5 @512px -> TITAN 768-d); ABMIL promoted to
  its own framework; `benchmarks/datasets/` reorganized by program into
  `tcga/ cptac/ other/ templates/`. CV defaults to 5-fold globally.
- **Packaging: publish-ready.** PyPI trusted-publishing via tokenless OIDC,
  dynamic versioning, and a citation file.

## v1.1.0 — Bug Fixing (2026-06-24, `git tag v1.1.0`)

Hardening milestone (20 requirements, 6 phases).

- **New commands / knobs:** `automil dequeue` (remove a queued or pending node);
  a `--project` group option for driving a repo from outside its root; a
  `scheduling_policy` knob with best-GPU dispatch; an editable-overlay guard in
  the daemon and `automil check`; migrate-on-read for older `graph.json` files.
- **Orchestrator:** `--experiments_per_gpu` wired through with a raised default
  concurrency cap; viz port fallback (explicit > config > default).
- **Safety fixes:** starttime-validated PID/PGID kills, dequeue state guards,
  cancel graph-mutation routed through the locked-update path, and
  `CUBLAS_WORKSPACE_CONFIG` pinned before the torch import.

## 8.0.0 — v1.0 milestone (shipped 2026-05-08, `git tag v1.0`)

The F2-readiness framework refactor. Nine phases (Phase 0 cleanup through
Phase 8 acceptance), 92 plans executed, 69 v1 requirements delivered (100%
v1 coverage across CLN / REG / BCK / TRJ / MRT / CAP / GTE / CLI / STP / DEC).
Final acceptance: D-208 11-clause aggregator green in CI; sub-gate B
sklearn-iris end-to-end green via real orchestrator subprocess; sub-gates A
(CCRCC reproduction) and C (heterogeneous consumers) workstation-deferred
behind `@pytest.mark.requires_ccrcc_data`.

### BREAKING migrations summarised

These accumulate across Phase 6, 7, and 8 entries below; consolidated for
operators upgrading from a pre-v1.0 checkout:

1. **`Backend.healthcheck` is abstract** (Phase 7). Custom Backend subclasses
   must implement it or raise `NotImplementedError` with the locked message
   `"healthcheck deferred to Phase 7+ for distributed backends (use salloc/ray status directly)"`.
2. **`env.required` is mandatory** (Phase 8). `automil check` fails with
   `Missing required env var: <name>` if anything declared in
   `automil/config.yaml: env.required` is unset. Empty list (`required: []`) is fine.
3. **`AUTOBENCH_ROOT` is no longer auto-injected** (Phase 8). Consumers
   declare what they need under `env.passthrough`. Recovery snippet in the
   Phase 8 entry below.
4. **`node["test_auc"]` etc. moved to `node["metrics"]["test_auc"]`** (Phase 8).
   Custom code reading `graph.json` directly must update the access path.
5. **`orchestrator/running/` is per-backend namespaced** (Phase 6).
   `running/<id>.json` (flat) → `running/<backend>/<id>.json`. The daemon
   refuses to start if it detects flat layout. Stop the daemon, confirm
   `ls automil/orchestrator/running/*.json` returns zero, upgrade, restart.

### Phase 8. Decoupling completion + final acceptance (2026-05-08)

**Theme:** prove the framework is generic. Zero `autobench` references in
`src/automil/`; sklearn-iris second consumer end-to-end via the documented
contract; CCRCC `node_0176` ±0.005 reproduction on the registry path.

**BREAKING. `env.required` mandatory in `automil/config.yaml`.**
`automil check` fails with `Missing required env var: <name>` if anything
declared under `env.required` is unset. Catches missing dataset paths
(e.g. `AUTOBENCH_CCRCC_ROOT`) BEFORE submit rather than deep inside the
training script.

Recovery for autobench-shaped consumers:

```yaml
env:
  required:
    - AUTOBENCH_OVARIAN_ROOT
    - AUTOBENCH_CCRCC_ROOT
  passthrough:
    - AUTOBENCH_OVARIAN_ROOT
    - AUTOBENCH_CCRCC_ROOT
    - HF_HOME
```

For self-contained consumers (sklearn-iris-style, no env-var dependencies):
`required: []`, `passthrough: [AUTOMIL_*]`.

**BREAKING. `node["test_auc"]` etc. no longer at top level.** The
`graph.json` node payload migrates the autobench-named metrics
(`val_auc`, `val_bacc`, `test_auc`, `test_bacc`) from top-level fields into
a generic `node["metrics"]` dict. This removes the framework's hardcoded
coupling to the autobench 4-key composite recipe and unblocks non-autobench
consumers. Custom code reading these fields must change to
`node["metrics"]["test_auc"]`. Framework-internal viz, CLI, and the
cap-killed reconcile branch are all migrated.

**BREAKING. `AUTOBENCH_ROOT` is no longer auto-injected into experiment
env.** The orchestrator no longer auto-injects `AUTOBENCH_ROOT` or overlays
`PYTHONPATH` to point at `benchmarks/`. Consumers that need these declare
them under `env.passthrough` per the recovery snippet above.

**Added:**

- `src/automil/schemas/result.schema.json` (D-201, JSON Schema 2020-12)
  describing the `result.json` contract. Validated at ingest via
  `jsonschema.validate(...)`. Malformed payloads transition the node to
  `crashed` with a schema-location pointer.
- `examples/sklearn-iris/`, ~80-line `train.py` demonstrating the
  contract on a non-autobench consumer (sklearn LogisticRegression on iris).
- `docs/training-script-contract.md` (DEC-06) documenting the 6 contract items.
- `tests/test_framework_purity.py` (D-206) regression-prevents `autobench`
  leakage in `src/automil/` (5-entry content-anchor allowlist).
- `tests/acceptance/test_final_phase8_acceptance.py` (D-205) final 3-sub-gate
  acceptance (sub-gate B drives the full submit + orchestrator subprocess
  path so the daemon ingest validate hook is exercised end-to-end, F-04).
- `tests/acceptance/test_phase8_acceptance.py` (D-208) 11-clause acceptance
  aggregator.
- `automil/config.yaml: scoring.formula` field surfaced in the framework
  template `config.yaml.j2` per DEC-04 ROADMAP success criterion 3 (F-07).
  Documentation-only field; consumers describe their composite recipe.

**Compatibility:** pre-D-200 `graph.json` files round-trip via the
bootstrap loader; explicit `schema_version` bump deferred (forward-compatible
cleanup). `pyproject.toml` adds `requires_ccrcc_data` marker. CI default
filters `not requires_ccrcc_data and not requires_slurm and not requires_ray`.
`jsonschema` is no new top-level dep; transitive since Phase 5.

## 7.0.0 — Phase 7. Hardware autodetect + automil-setup skill (2026-05-07)

**BREAKING.** `Backend.healthcheck` is abstract in the Backend ABC. Custom
Backend subclasses must implement it; the locked `NotImplementedError`
message for distributed backends is
`"healthcheck deferred to Phase 7+ for distributed backends (use salloc/ray status directly)"`.

### Phase 7. Hardware autodetect + automil-setup skill (2026-05-07)

**Theme:** detect hardware once, surface it to the user, never decide
silently. The `/automil-setup` skill becomes idempotent and ships a
mandatory dry-run gate.

**BREAKING. `Backend.healthcheck` is now an abstract method.** Subclasses
without a concrete `healthcheck` are uninstantiable
(`TypeError: Can't instantiate abstract class ... with abstract method healthcheck`).

- **`LocalBackend`** implements it (probes hardware via `nvidia-smi` /
  `rocm-smi` / CPU-only fallback per D-190).
- **`SLURMBackend`** and **`RayBackend`** raise `NotImplementedError` with
  the locked message
  `"healthcheck deferred to Phase 7+ for distributed backends (use salloc/ray status directly)"`
  (D-189). Distributed-cluster healthcheck is deferred to a post-v1.0 phase.
- **`MockSLURMBackend`** raises the same `NotImplementedError` for test-fixture parity.

**Added:**

- `Backend.healthcheck() -> HealthReport` on the Backend ABC (D-189 / STP-01).
  `HealthReport` is a frozen dataclass with 8 fields: `gpu_count`, `gpu_vram_gb`,
  `accelerator` (`cuda` / `rocm` / `cpu`), `python_version`, `automil_version`,
  `detection_status` (`ok` / `partial` / `failed`), `detection_warnings`,
  `detected_at`.
- `automil init` calls `LocalBackend.healthcheck()` between the `--update`
  guard and template render. Detected values flow into
  `automil/config.yaml`'s `cap:` and `hardware:` sections (D-191 / STP-02).
- `automil init --no-healthcheck` flag for CI / smoke-test paths.
- `automil submit --max-time SECONDS` for seconds-precision timeouts (D-195).
  `--timeout MINUTES` is preserved verbatim; when both are passed, `--max-time`
  wins (translated via ceil-div to `--timeout`).
- `_shared/automil-setup/SKILL.md` expanded from a 122-line skeleton to ~282
  lines covering Inspection Heuristics, Drafting Conventions, Idempotency
  Protocol, Setup-Done Gate, and Failure Modes (D-192..D-196 / STP-04..06).
- `agent_assets/codex/skills/automil-setup/SKILL.md` empty-frontmatter overlay
  for Codex plain-markdown rendering (D-196 / STP-07 / Pitfall D).

**Fixed:** `automil init`'s template render now stamps detected hardware
defaults (`max_concurrent_per_gpu`, `default_vram_estimate_gb`) instead of
the prior hardcoded constants. Per the Pitfall 8 anti-acceptance,
`default_vram_estimate_gb` is computed from `numpy.quantile(.95)` of empirical
`vram_gb` observations in `automil/results.tsv` when ≥10 rows are present,
and from `max(8.0, min(gpu_vram_gb) / 8.0)` otherwise.

## 6.0.0 — Phase 6. SLURM backend (submitit) + Ray backend (raw `@ray.remote`) (2026-05-06)

**BREAKING.** Per-backend `running/` namespacing.
`orchestrator/running/<id>.json` (flat) → `orchestrator/running/<backend>/<id>.json`.
The daemon refuses to start if it detects the flat layout. Recovery:
stop the daemon (`automil orchestrator stop`), verify
`ls automil/orchestrator/running/*.json | wc -l` returns 0, then restart.

### Phase 6. SLURM backend (submitit) + Ray backend (raw `@ray.remote`) (2026-05-06)

**Theme:** distributed-backend support without freezing local-backend
semantics into the contract. Ships as opt-in extras so default
`pip install -e .` stays slim.

**BREAKING. Per-backend `running/` namespacing.**
`orchestrator/running/<id>.json` (flat) → `orchestrator/running/<backend>/<id>.json`
(namespaced). autoMIL does not auto-migrate; the daemon refuses to start
if it detects flat layout.

**Operator upgrade path:**

```bash
automil orchestrator stop
ls automil/orchestrator/running/*.json 2>/dev/null | wc -l   # must be 0
# upgrade
automil orchestrator start
```

**Added:**

- `SLURMBackend` (`src/automil/backends/slurm.py`), opt-in via
  `pip install -e '.[slurm]'`. Dispatches via submitit `AutoExecutor`;
  honors the Phase 4 cap contract via `slurm_additional_parameters={"signal": "B:TERM@30"}`
  (30s SIGTERM grace, framework-mandated; `automil check` rejects operator
  override). Walltime translated via `_walltime_to_timeout_min(walltime_seconds)`.
- `RayBackend` (`src/automil/backends/ray.py`), opt-in via
  `pip install -e '.[ray]'`. Dispatches via raw `@ray.remote` (NOT Ray Tune);
  hybrid `RAY_ADDRESS → local fallback` (`backend.ray.allow_local_fallback`);
  cancel via `ray.cancel(force=True, recursive=True)`; non-blocking poll via
  `ray.wait(timeout=0)`.
- `BackendNotInstalledError`, `SlurmDirectivesIncompleteError`,
  `RayClusterUnreachableError` in `automil.backends.errors`.
- `automil check` validates `backend.slurm.directives` completeness (rejects
  `TODO_FILL_IN`) and Ray cluster reachability (advisory).
- Cross-backend log unification: `archive/<id>/run.log` is orchestrator-owned
  and drained from `backend.log_iter()` on terminal-state observation.
- pytest markers `requires_slurm` / `requires_ray` for nightly real-cluster
  tests (`test_contract_real_slurm.py` / `test_contract_real_ray.py`).
- D-179 11-clause acceptance gate (`tests/backends/test_phase6_acceptance.py`):
  9 PASS + 2 SKIP (extras-gated when `[slurm]` / `[ray]` absent).

**Compatibility:** `pip install -e .` (no extras) still works; submitit and
ray are NOT pulled. `automil --help`, `automil submit`, `automil cancel`,
`automil resubmit` work unchanged for `backend.name: local` configs.

### Phase 5. Generalization gate (2026-05-06)

**Theme:** separate exploration from generalization with a pre-registered
held-out manifest and paired statistical test.

**Added:**

- `candidate` node status, set by `automil nominate <node_id>` (idempotent;
  mutates `keep` → `candidate`). Manual nomination by default
  (`gate.auto_nominate: false`, D-142).
- `automil gate manifest`, writes and git-commits `gate_manifest.json`
  BEFORE search via `write_manifest_committed`. Manifest schema carries
  `(cell_id, dataset, encoder, task)` 4-tuples.
- `automil promote <candidate_id>`, runs Stage B gate. Spawns held-out
  evaluations through `Backend.submit(spec)` (NOT a parallel mechanism)
  with `metadata.gate_eval='true'`. Statistics: paired Wilcoxon + BCa
  bootstrap CI (1000 reps, GTE-04 locked per F1 paper §4.4) + Bonferroni
  `alpha/K` (DIVIDE direction).
- Promotion-rate metric exposed via `viz/api/promotion-rate` SSE and
  `automil status` (GTE-06).
- Pitfall-6 single-file anti-acceptance gate
  (`tests/gate/test_pitfall6_held_out_isolation.py`), 35 D-149 assertions;
  enforces that held-out cells are invisible to the search agent.

**Deferred:** calibration pilot K-determination requires Leo workstation
with CCRCC `node_0176` + 3-5 fresh cells. The framework-side scaffold is
committed at `90011e8`; run `automil promote --calibrate <candidate_id>`
and read the delta matrix from `archive/<candidate_id>/gate_evaluation.jsonl`.

### Phase 4. 6h per-cell hard cap + cell concept (2026-05-05)

**Note on the "6h" in this phase's title:** 21600 seconds (6h) is the
autoMIL-paper campaign-wide default that motivated this milestone. It is
NOT a framework constant. The framework provides the cap *mechanism*; the
*value* is consumer-supplied via `cap.budget_seconds` in
`automil/config.yaml` (or `--budget-seconds` at submit time per D-134).
The sklearn-iris example uses 60s; external consumers pick their own.

**Theme:** make `(dataset, encoder, parent_id)` a first-class graph entity
with a framework-enforced per-cell wall-clock cap *mechanism* (consumer
supplies the value). Budget-killed runs reconcile gracefully via per-fold
checkpoints and are stored as `executed` (with partial composite), never
`crashed`.

**Added:**

- `cell_id` first-class on every node; `cells/get_or_create_cell` lookup
  BEFORE writing queue spec; `metadata.cell_id` stamped on every queued spec.
- Two-tier cap state machine (`active` → `refusing-new` at `T - safety_buffer`
  → `terminating` at `T`); SIGTERM with 30s grace is the cap contract.
- Per-fold checkpoint protocol: training scripts write `fold_<i>_result.json`
  after each fold; `register_sigterm_flush` (in `runtime_helpers.py`)
  installs a SIGTERM handler that aggregates completed folds into a single
  `result.json` with `"partial": true`.
- Reconciler reads `metadata.cancel_reason='cap'` written BEFORE the cancel
  (Pitfall-4 ordering guarantee) and assigns `JobState.BUDGET_KILLED` (NOT
  `crashed`).
- Per-cell budget overrides: `automil submit --budget-seconds N --safety-buffer-seconds M`
  honored only on the submit that opens the cell (D-134).
- `automil cell list` / `status` / `show <id>` CLI surfaces.
- D-115 21-test acceptance gate including Pitfall-4 anti-acceptance,
  daemon-restart (5/5), reconcile cascade (5/5).

### Phase 3. Trajectory recorder + multi-runtime asset reorg (2026-05-04)

**Theme:** capture per-submit agent trajectories without leaking secrets
or fossilising the format; reorganise `agent_assets/` so canonical content
lives once with per-runtime overlays.

**Added:**

- `archive/<node_id>/trajectory.jsonl` canonical artifact. First line is
  metadata `{schema_version, runtime, runtime_version, tool_schema_version,
  automil_version, automil_runtime_env}`; subsequent lines are one event
  each using OpenTelemetry `gen_ai.*` field names (no runtime
  `opentelemetry-sdk` dependency).
- Redaction-on-capture for `sk-…`, `hf_…`, `ghp_…`, AWS access keys,
  `*_API_KEY=…`, `*_TOKEN=…`. Per-event 8 KB cap; per-file 5 MB soft / 50 MB
  hard rotate producing `trajectory.<n>.jsonl` siblings. Trajectories
  gitignored by default.
- `automil trajectory record` / `export` / `status` CLI. `export` produces
  a redacted, schema-validated bundle.
- `src/automil/agent_assets/_shared/`, canonical SKILL/AGENTS content.
- Per-runtime overlay directories: `claude/hooks/on_stop.sh`,
  `codex/skills/automil-setup/`, `opencode/plugins/automil-trajectory.ts`,
  `deepseek/README.md` (DeepSeek is a *model* routed via opencode/codex).
- `automil init --runtime <claude|codex|opencode|deepseek-via-opencode|deepseek-via-codex|all>`
  with auto-detect from existing `.claude/`, `.codex/`, `.opencode/` dirs.
- `automil init --update` re-renders skills/hooks/AGENTS.md without
  re-scaffolding.
- `automil show-skill --runtime <r>` renders the merged per-runtime
  SKILL/AGENTS file (`--asset SKILL` or `AGENTS`).
- `AUTOMIL_RUNTIME` declared, never inferred (D-87), required in
  `env.passthrough` so the trajectory recorder inside the experiment sees
  the declared value.
- End-to-end smoke test: experiment loop submits, runs, completes, and
  writes a valid `result.json` under Claude Code AND under one of
  {opencode, codex}, trajectories captured for both.

### Phase 2. Backend ABC + LocalBackend re-export + MockSLURM fixture (2026-05-03)

**Theme:** lock the backend contract against ≥2 implementations IN-phase
so Phase 6 cannot accidentally inherit local-backend semantics (PIDs, sync
status, `killpg`).

**Added:**

- `Backend(ABC)` in `src/automil/backends/base.py` with 5 abstract methods
  (`submit`, `poll`, `list_running`, `cancel`, `log_iter`) plus the
  state-not-control-flow `JobState` enum
  (`pending | running | completed | crashed | cancelled | budget_killed`)
  and frozen `JobHandle` / `JobSpec` dataclasses.
- `LocalBackend` ships as a re-export shim over the existing 750-line
  orchestrator (renamed to `_orchestrator_daemon.py`); 48-test baseline
  suite stays green with empty behavioural diff.
- `MockSLURMBackend` test fixture: eventual-consistency status (5s poll
  lag), opaque `job_id`, fire-and-forget `cancel`, node-local filesystem.
- Parameterised contract test (≥12 scenarios × 2 backends) gates the ABC
  before Phase 6.
- `ruff`/AST custom rule lint-blocks `os.kill`, `Popen`, and `pid`
  references outside `backends/local.py` and `backends/_orchestrator_daemon.py`
  (BCK-04 allowlist; viz/server.py allowlisted for daemon PID lifecycle).
- `automil cancel <node_id>` and `automil resubmit <node_id>` wired through
  `Backend.cancel` and `Backend.submit`. Cancelled nodes archive with
  `status: cancelled`; resubmits get a fresh worktree.
- `BACKENDS` registry singleton + `@register("name")` decorator on each
  backend class.
- `metadata.backend` written on every queued spec (BCK-01 / CLI-03/04 prereq).

### Phase 1. Variant registry + config-driven train + reproduction sanity (2026-05-02)

**Theme:** the keystone phase. Variants live as committed code modules
selected via config; shared library files become read-only; the registry-only
path reproduces a known-good node within ±0.005 from a clean checkout.

**Added:**

- `Variant` ABC family + frozen `VariantSpec` dataclass + internal
  `Registry` class with `@register` decorator + `importlib.metadata.entry_points`
  discovery.
- `automil refresh-registry` regenerates per-parent `variants/__init__.py`
  deterministically and idempotently.
- Submit pre-validator chain: `identity` (mode-aware strict in
  `architecture-preserving`, lenient in `free`), `interface` (subclass of
  matching ABC, required-method signatures match), `purity` (no top-level
  I/O / network / mutable globals).
- `registry.protected` glob list, submit hard-rejects overlays touching
  these paths (D-34); `automil check` fails on uncommitted edits to them.
- `automil/config.yaml: registry.mode` selects `free` (default) or
  `architecture-preserving`; `repro_tolerance` (default ±0.005);
  `identity_constraints`.
- `train.py`-side: model class, loss class, training policies, and
  hyperparameters all read from `config.yaml`. Zero `args.X = literal`
  overrides remain in framework code (verifiable via grep).
- Variant manifest schema commits parent commit, composite, and node id.
- Variant lifecycle CLI: `apply`, `revert-baseline` (mandatory pre-stash ,
  never blind-checkout), `port-variant` (idempotent; rejects already-registered
  nodes), `promote-variant`, `refresh-registry`, `verify-repro`.
- Synthetic mini-consumer round-trip
  (`tests/test_synthetic_consumer_roundtrip.py`) is the framework-side
  acceptance gate per D-49/D-50: register → port → refresh → apply →
  verify-repro end-to-end.

### Phase 0. Tier 2 cleanup + CLI split + compat shim (2026-05-01)

**Theme:** clear CONCERNS HIGH-severity backlog and prepare the codebase
shape so new commands and modules have a place to land without disturbing
existing tests.

**Added:**

- Subprocess `env` no longer leaks full `os.environ` to children; explicit
  whitelist + `env.passthrough` config field (CLN-02).
- `python-dotenv` replaces inline dotenv parser (CLN-03).
- PID-file cross-checks process start time via `/proc/<pid>/stat` to detect
  stale PID reuse (CLN-04).
- `nvidia-smi` invocation is path-pinned via `shutil.which` and reported by
  `automil check` when missing (CLN-05).
- Monolithic `cli.py` split into `src/automil/cli/` per-command-group package
  with thin `__init__.py` aggregator (no individual file >300 lines, CLN-06).
- `compat.py` re-export shim with empty `Active` section + populated
  `_PLANNED_MIGRATIONS` dict so pre-split `from automil.X import Y` paths
  still resolve (CLN-07).
- `automil reconcile --recompute-best` rebuilds `meta.best_node_id` from
  the honest non-leaky composite by walking only `executed/keep` nodes
  (CLI-07).
- 48-test baseline suite stays green; no new behaviour beyond cleanup +
  restructure + reconcile flag.

---

The v1.0/v1.1 milestone documents (phase-by-phase roadmap, the 69 v1 REQ-IDs
and their traceability, and the cross-phase integration audit) lived under
`.planning/` and were removed from the working tree on 2026-08-06. They remain
recoverable from git history: `git show 7de69a8:.planning/milestones/v1.0-ROADMAP.md`
and siblings.
