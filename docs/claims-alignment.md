# Claims–mechanism alignment audit — preprint campaign

_Audit of whether the agent, driving autoMIL through the current harness and
constraints, can actually produce the evidence each preprint claim needs.
Run 2026-08-07 against `main` (post-#38, `automil-preprint-130-v4`), before any
campaign cell has launched. Every finding below was verified in code by at
least two independent readers; file:line citations are to the audited
checkout (pre-fix)._

**Status: the fix plan in §4 is implemented on this branch.** Every A/B item
below is FIXED unless its row says otherwise; scope adjustments made during
implementation (with reasons) are: A7 resolved docs-only under the ship-fast
constraint; B8 ships the arm-defaults equivalence gate where a static
frozen-tree default exists to compare against (post-H-3 dtfd/titan archives;
nnMIL's arm block is a data-computed plan, so its reuse stays the operator's
call) with the pre-H-3 rerun-vs-reuse call left to the operator;
C-d/C-e resolved as honesty labels rather than deletions (both artifacts are
load-bearing for acceptance tests / the stable config surface); C-j's
tolerance covers the controller-stamped lower bound only (`ended_at` is
operator-supplied and needs none); A10 shipped as a 7d `cell_time_budget`
containment value and was then superseded at the #39 merge (see the
addendum below and the A10 row); B6's shared contract
is `scoring.ingest_signal` at the three reconcile mouths, with the terminal
writer running the same key-guard/recompute primitives inline (Steps
2a/2b/2c). Two pre-firewall test fixtures and two tests that pinned pre-CR-1b
behavior were updated to the contract shape.

**Post-merge addendum (2026-08-08).** This branch subsequently integrated
PR #39 (native Claude active-time metering) and PR #40 (campaign operator
runbook). The campaign protocol is now `preprint-v2`: **30 charged attempts
+ 12h natively-metered agent-active time** per cell
(`automil-preprint-130-v5` manifest). Where a row below cites 60 attempts
or a 6h/7d time stance, that is the audited pre-merge state; the A10 and
C-i rows record their post-merge supersession explicitly, and A9's
bill-at-archive invariant now lives inside #39's fail-closed admission
gate as a billed-retry exemption (`_refuse_closed_cell_spec`).

**Method.** Three independent code sweeps (orchestrator enforcement surfaces;
campaign + consumer machinery; graph/gate/agent assets), reconciled against the
paper planning record (`paper/preprint/{PLAN,EXPERIMENT_GRID,PRELAUNCH_REVIEW}.md`,
the 2026-04-29 proposal, and `benchmarks/campaigns/preprint_130/analysis_plan.json`),
then each load-bearing finding re-verified first-hand before it entered this
document.

---

## 1. The claims, as currently executable

The pre-registered form of the paper's claims is
`benchmarks/campaigns/preprint_130/analysis_plan.json` (frozen, hash-locked by
the manifest). Mapped to the figure plan and the proposal's RQs:

| # | Claim | Pre-registered estimand | Mechanism that must carry it |
|---|---|---|---|
| C1 | The corrected full-pipeline benchmark exists and is complete (130 cells: 5 datasets × 2 tasks × {4 tile arms × 3 encoders + TITAN}) | `missingness` (fail-closed census) | manifest + `run-baseline`/`register-baseline` + certification chain |
| **C2** | **Equal-effort agentic recipe search produces a real, honestly-measured lift over the native default** (fig-3 successor; RQ1 evidence base) | `agentic_lift`: frozen winner − native baseline, sealed-test, per cell; sign counts poolable, magnitudes per task type | 30-attempt discovery on folds 0–2 (12h metered agent-active, preprint-v2) → ≤10 promotion on folds 3–4 → 5-fold-val winner → campaign-wide freeze → paired reveal |
| C3 | Recipe search changes aggregator *rankings* (recipe-bias claim, RQ1) | `tile_ranking_response`: rank shift, Kendall τ-b, top-arm-set change per (dataset, task, encoder) block | same, aggregated across the 30 tile blocks |
| C4 | The survival axis works; TITAN reported separately | `survival_primary`, `titan_lift` | survival trainers + nllsurv pin + separate-regime rule |
| C5 | The search is auditable and test never drives it (val-firewall, RQ4) | `search_process` census; `status: frozen-before-held-out-certification` | born-sealing, launch admissibility, process census, session attestation |
| — | Encoder ≫ aggregator variance decomposition | **not pre-registered** (dropped after PRELAUNCH_REVIEW O1) | — |
| — | Agentic search beats menu-AutoML / random (RQ2) | **not pre-registered; no comparison arm exists** | — |

"Second claim" throughout this document = **C2**, the claim the whole campaign
apparatus exists to support and the one most sensitive to what the harness
actually lets the agent do.

## 2. What the agent can actually do, per arm (the action surface)

In `mode: architecture-preserving` (all 130 cells), the agent's entire action
surface is:

1. **Declared scalars** via `submit --override '--hparams {...}'` — validated
   against `registry.identity_locked_hparams` at submit *and* launch
   (`admissibility.py:329-340`), applied per arm by
   `hparams.apply_overrides` (`benchmarks/src/autobench/pipeline/hparams.py:210`)
   against the declared space (`search_space.py:64-153`).
2. **PolicyVariants** under `automil/variants/_policies/` — exactly three
   guarded operations (`policy_dispatch.py:222-268`):
   `wrap_optimizer_for(opt, role)`, `wrap_scheduler_for(sched, role)`,
   `should_stop(default, epoch, metrics)`.

Reachability, as audited — the pre-fix snapshot, kept as the historical
record (post-fix note below the table):

| Recipe family (proposal §5 "allowed") | clam | abmil | dtfd | titan | nnmil |
|---|---|---|---|---|---|
| Scalar hp (lr, wd, epochs, patience, dropout…) | **✗ dead channel (A1)** | ✓ (8) | ✓ (13) | ✓ (5) | ✓ (10) |
| Optimizer wrappers (Lookahead, clipping, per-group LR) | ✓ | ✓ | ✓ | ✓ | ✓ |
| SAM-class two-step optimizers | ✗ (needs a second forward pass; no closure seam, and `PolicyVariant.step` is never called by any consumer) | ✗ | ✗ | ✗ | ✗ |
| LR schedules | via wrapped `step()` | via wrapped `step()` | ✓ native seam | via wrapped `step()` | via wrapped `step()` |
| Custom stopping policies | **✗ metrics={} (A2)** cls / ✓ surv | ✓ | ✓ | ✓ | ✓ |
| Loss shaping (label smoothing, focal, class-balanced) | ✗ no seam | ✗ | ✗ | ✗ | ✗ |
| Sampling / batch construction | `weighted_sample` bool only | ✗ | ✗ | ✗ | `batch_sampler` string |
| Ensembling / multi-init, val-threshold opt., augmentation | ✗ | ✗ | ✗ | ✗ | ✗ |

_Post-fix (shipped with §4): the three bolded ✗ cells are closed — clam's
declared scalar channel is live (A1: `apply_overrides_to_exp_cfg` at the top
of clam `run_experiment`), clam classification stopping receives real val
metrics (A2), and nnmil's `dropout` is actually applied (A8) — and A4's
identity locks remove the capacity knobs from the campaign-tunable sets (clam
`model_size`; abmil `M`,`L`; dtfd `mDim`,`numLayer_Res`; nnmil `hidden_dim`),
shrinking the per-arm scalar counts accordingly. Every other cell is
unchanged._

Two consequences worth stating plainly:

- The historical feasibility anchors (CCRCC 0.744→0.807, ovarian 0.814→0.851)
  were produced in **free mode**, and their winning ingredients — label
  smoothing 0.08, WeightedRandomSampler, multi-init N=3, threshold tuning, SAM —
  are **mostly unreachable** under the campaign harness. Anchor-sized lifts must
  not be projected onto C2.
- The proposal (`automil-proposal-2026-04-29.md` §5) *explicitly allows* loss
  terms outside the forward pass and sampling/batch construction. The
  implementation is **narrower than the paper's own declared protocol**. Either
  the paper's methods text narrows to match the code, or the seam widens to
  match the text (§4, A7).

## 3. Problem table

Severity: **P0** = a claim is factually broken or the campaign would burn its
one shot; **P1** = claim integrity/honesty weakened, cheap to fix now; **P2** =
docs/claims discipline or deferred decision.

| ID | Sev | Claim hit | Finding (evidence) | Elegant fix |
|---|---|---|---|---|
| A1 | P0 | C2, C3, C5 | **CLAM's `--hparams` channel is dead.** No `apply_overrides(..., arm="clam")` exists; `clam/train.py::_make_clam_args` reads config directly and `hparam_overrides` is parsed (`run_experiment.py:464`) but never consumed. On 30/130 cells the only admissible scalar channel is a **silent no-op** — attempts get charged for runs identical to baseline, and the archived `config.json` (`clam/runner.py:104` → `config.py:242`) records the unconsumed overrides as if applied, so provenance claims a tuned run that was byte-identical to baseline. This is the exact H-3 failure mode `hparams.py:1-42` exists to remove, live on the reference arm. (Policy variants can still reach lr/wd through `wrap_optimizer`, so the arm is not literally untunable — the *declared scalar channel* is.) | Apply **only the opaque channel** (`exp_cfg.hparam_overrides` — CLAM's canonical knobs are natively live in the transport, re-routing them would duplicate a path) at the top of CLAM `run_experiment`, covering classification + survival. Partition keys across ModelConfig + TrainConfig (disjoint field sets) via a small `apply_overrides_to_exp_cfg` helper in `hparams.py` with the same `_check_declared` semantics — which also makes a locked key raise at train time, not just admissibility. **Ordering is load-bearing**: before `resolve_results_dir` (`clam/runner.py:97`, CR-5b cache identity) and before `exp_cfg.save` (`:104`, honest provenance). Channel test mirroring the other arms. **Must land together with A4** (a live channel without the capacity locks opens `model_size`). |
| A2 | P0 | C2, C3 | **CLAM classification passes `metrics={}` to `should_stop`** (`benchmarks/lib/CLAM/utils/core_utils.py:199`), so metric-driven stopping policies are structurally impossible on 15 cells while every other arm supplies val metrics. Unequal policy surface = channel width reported as a model result. | Have `validate`/`validate_clam` return the val metrics they already compute; pass them through. Vendored file is already policy-patched at `:168-170,198-199`. |
| A3 | P0 | C2, C5 | **`.claude/skills/automil/SKILL.md` is stale** — predates `registry.mode`; tells the agent to aim ≥50% architecture/ensemble proposals and use `automil/variants/<parent>/` model variants, all inadmissible in the campaign. `.agents/` copy matches canonical `_shared`; the Claude copy (the primary runtime) diverged, and **no test guards any copy against drift**. A campaign session would spend its metered agent-active budget (12h post-merge) fighting admissibility. | Sync from `agent_assets/_shared` + one test asserting both in-repo copies equal the canonical render (kills the whole drift class). |
| A4 | P0 | C2, C5 | **Capacity/architecture knobs are tunable under `architecture-preserving`**: `model_size` (clam: attention width presets, `model_clam.py:81-82`), `M`,`L` (abmil: Linear dims), `mDim`,`numLayer_Res` (dtfd: width + residual depth), `hidden_dim` (nnmil) all change parameter counts, but `identity_locked_hparams` lists only CLAM's two loss switches (`config.yaml:84`). `audit_materialized_campaign` (`campaign.py:841-850`) never verifies `allowed_override_options`/`identity_locked_hparams`, so drift passes the 130-root audit. Proposal §5 excludes width/depth changes by omission from its allowed list plus the attribution catch-all (layers/attention/pooling/heads named verbatim) — reviewer Attack 4 is currently valid. (`numGroup`, `B`, `batch_size`, `max_seq_length`, droprates are correctly tunable: sampling/batch-construction, zero parameters.) | Extend `identity_locked_hparams` per roster template (final lists, from consumer code: clam `model_size`; abmil `M`,`L`; dtfd `mDim`,`numLayer_Res`; nnmil `hidden_dim` — `B`/`numGroup`/`max_seq_length`/droprates stay tunable as zero-parameter sampling/recipe knobs). **`search_space.py` stays untouched** (two-layer design: it declares what the plumbing routes, mode-independent — free mode legitimately tunes widths; the campaign constraint's single source of truth is the hash-locked `registry:` block; the paper's methods table is the composition `coverage_table()` ∩ not-identity-locked). Audit asserts the *values* against an `EXPECTED_IDENTITY_LOCKED_HPARAMS` constant (+ `allowed_override_options`); name-collision/alias assertions; template edits + regenerated manifest in one commit. |
| A5 | P0 | C5 | **`spec.env` can retarget or un-gate the seal**: `_SPEC_ENV_BLOCKED` (`_orchestrator_daemon.py:75-82`) omits `AUTOMIL_RESULTS_DIR` (born-sealing target, `:1169-1171`), `AUTOMIL_DIR_REL` (policy resolution root, `policy_dispatch.py:120`), and — most direct — `AUTOMIL_CERTIFY`, which flips the consumer's test-print gates (`core_utils.py:212-228`) so test metrics stream into agent-visible `run.log` during search. Queue specs are unvalidated (`:991-1005`) and launch revalidation never inspects `spec["env"]`, so even preserving mode's fail-closed path passes a post-submit env edit. (Weakest P0 — C-g already concedes shell access — but it is an internal inconsistency in an existing enforcement surface.) | Add `AUTOMIL_RESULTS_DIR`, `AUTOMIL_DIR_REL`, `AUTOMIL_NODE_ID`, `AUTOMIL_CERTIFY`, `AUTOMIL_FOLD_COUNT` to the frozenset (enumerated, not prefix-wide — `AUTOMIL_VARIANT_*` are legitimate spec env, `apply.py:250-252`); **also pop ambient `AUTOMIL_CERTIFY`** in `_build_subprocess_env` — the `AUTOMIL_` whitelist prefix (`:70-72`) would forward an operator's leftover shell export into every search child. Tests in `test_orchestrator_env_whitelist.py`. |
| A6 | P0 | C5, C2 | **`metrics` keys are unconstrained at ingest** — the firewall strips exactly `held_out`/`summary` (`terminal_writer.py:187-188`); a result carrying `metrics: {"test_auc": …}` would flow test into `graph.json`, `results.tsv`, SSE, *and into the recomputed composite* (`scoring.py:59-77` means over all metric keys, names never inspected), i.e. test driving selection with the firewall's blessing. Campaign's `_validation_folds` (`campaign_stages.py:361-394`) catches it only at freeze, after search already consumed it. | Fail closed at the existing schema-validation choke point in `write_terminal_state`: a held-out-named metrics key → node crashes with a val-firewall pointer (same path as schema failure). Key predicate lives in `firewall.py` (`is_held_out_metric_key`) and **aligns with the freeze-time predicate** (`"test" in key.lower()` + `held` markers, `campaign_stages.py:391`) so the two enforcement layers cannot disagree; overcatching a hypothetical legit key fails loud with a rename-me message, never silently. Verified no in-repo consumer collides (iris: `accuracy`/`f1`; autobench: `val_*`). Fixture at `test_tick_cells.py:309` updates with the guard's positive test. |
| A7 | P0→resolved | C2 | **The seam is narrower than the paper's declared protocol** (§2): no loss-shaping seam anywhere, though the proposal's allowed list names label smoothing/focal/class-balanced — the proven ingredients of both anchors. C2's effect size is capped by this. **Adversarial review refuted the seam design on the merits**: CLAM constructs one `loss_fn` shared by train *and* validation (`core_utils.py:117-123`, val use at `:363,:423` where val_loss drives early stopping), so a construction-site wrap is not train-only; DTFD builds a single `reduction="none"` criterion for both tiers (`dtfd/train.py:187`); 65/130 cells are survival where the criterion is the grid axis and must stay closed. | **Resolved under the ship-fast constraint (no protocol-surface growth at the preprint stage): docs-only.** Narrow the paper's methods text to the implemented surface; `wrap_criterion_for` goes to the journal ledger (§7). The `label_smoothing` declared-scalar alternative was reviewed as *small but not trivial* (two mandatory fail-loud guards: `bag_loss=svm` incompatibility; survival-task rejection — plus the CLAM smoothed-val_loss semantics caveat), which by the pre-stated adoption rule sends it to §7 as well. |
| A8 | P0 | C2, C5 | **nnMIL's declared-tunable `dropout` is a silent no-op on the campaign's model** (found during adversarial verification of A4): `classification_trainer.py:74-79` passes `dropout=self.config['dropout']` to the factory, but `model_factory.py:23-25` hardcodes `SimpleMIL(..., dropout=True)` — the tuned value is discarded and `nn.Dropout(0.25)` runs regardless. Same H-3 class as A1: on 30 nnmil cells an agent tuning `dropout` burns attempts on baseline-identical runs with provenance claiming otherwise. | Wire the configured value through the vendored factory if `SimpleMIL` accepts a rate; otherwise move `dropout` to `search_space.locked` for nnmil with the upstream-fidelity reason. Truth either way; decided by reading the vendored model at implementation time. |
| A9 | P0 | C1, C2, C5 | **Launch billing and the freeze census disagree; one pre-spawn failure permanently deadlocks the cell.** `freeze_discovery` requires *both* `consumed_evals == DISCOVERY_ATTEMPTS` — 60 at audit, 30 post-merge — (`campaign_stages.py:960-966`) *and* exactly that many archived non-`cap_refused` specs (`:973-985`, `:1074-1078`). But `_launch` archives `spec.json` at the top (`_orchestrator_daemon.py:1514-1519`) and bills only after `Popen` succeeds (`:1704`, policy at `:1359-1371`); every pre-spawn failure (admissibility `:1594`, missing base_commit `:1613`, **worktree failure** `:1624`, Popen `:1681`, orphan-recovery `:875-946`) archives without billing → archived > billed forever, the two freeze conditions become jointly unsatisfiable, and the cap is monotone with no in-protocol repair. Promotion is worse (jobs materialized exactly once, `:1489-1499`). Contradicts the README ("crashes … consume discovery attempts") and `cap.py`'s own documented bill-at-launch policy — the daemon implements a third policy. Across ~7,800 launches one transient git-lock failure is expected, and one is enough. | Move `_record_cell_launch` to immediately after the archive-spec write (cap-refusal check stays first): "archived non-`cap_refused` campaign spec ⇔ billed attempt" becomes a construction invariant; pre-spawn crashes become charged attempts in pre-registered census classes; restores the documented policy. One test: a worktree-failure attempt is billed and freeze-countable. *Post-#39 merge: the disagreement existed unchanged in #39's lineage (bill post-Popen at its `_orchestrator_daemon.py:1836`); the invariant now rides #39's fail-closed admission gate, with a billed-retry exemption in `_refuse_closed_cell_spec`.* |
| A10 | P0 → **SUPERSEDED at the #39 merge** | C1, C2 | **The 6h `agent_active` time cap contradicted the 60-attempt eval budget** (audited state): ~3–10 min of billed per-attempt activity × 60 attempts ≈ 3–10h against a 5.5h effective wall made the two predeclared budgets mutually unsatisfiable in expectation, with `REFUSING_NEW` one-way (`cap.py:110-118`) and `TERMINATING` killing in-flight billed work. The interim fix on this branch (7d `cell_time_budget` containment) shipped, then #39 replaced the premise: preprint-v2 re-sizes the axes to **30 attempts inside a 12h natively-metered agent-active budget** — time is a co-equal, frozen, session-attested budget, not containment, and the original arithmetic dissolves (3–10 min × 30 ≈ 1.5–5h ≪ 12h). | Superseded: `CELL_TIME_CONTAINMENT` / `PROTOCOL.cell_time_budget` removed at the merge; materialization + audit carry #39's `cap.budget: 12h`, `cap.mode: agent_active`, `cap.eval_budget: 30`. Residual documented posture (deliberate, tail-risk): if 12h exhausts below 30 attempts the freeze fails closed — the runbook documents it; no code path reopens a closed cell. |
| B1 | P1 | C2 | **The Ladder margin's noise floor is self-reported.** `composite_se` is read verbatim off `result.json` (`terminal_writer.py:283`); `scoring.cross_fold_se` is never called in `src/automil/`; `cells/reconcile.aggregate_folds` has fold composites in hand (`cells/reconcile.py:52-95`) and computes no SE, so budget-killed/partial nodes silently drop to the bare δ. The same machinery that refuses to trust the reported `composite` (CR-1b) trusts the reported SE that *gates* it. | Recompute SE at ingest from `result["validation_folds"]` (agent-visible, val-only, already emitted — `evaluate.py:213`); prefer recomputed, log disagreement — the exact CR-1b pattern. Same helper in `aggregate_folds`. |
| B2 | P1 | C5 | **`scoring.formula` fails open on a typo and the template teaches the failure.** `config.yaml.j2:148-155` says "documentation-only … NOT evaluate[d]" with examples (`"accuracy"`, `"(val_auc + val_bacc) / 2"`) that `scoring.py:59-66` rejects as reducer names; `terminal_writer.py:229-232` catches the ValueError and **trusts the reported composite** — CR-1b silently disabled by following the template's own comment. | Rewrite the template comment (reducer semantics: `mean`/`max`/`min`/`trust_reported`); validate the reducer name at graph seeding — **the raise must sit outside the blanket `except Exception` at `graph.py:258-260`** — and in `automil check`; fix the `scoring.py:56-58` docstring. (Reassurance from review: the materialized campaign configs never set `scoring.formula`, so the campaign runs CR-1b on the `mean` default and reproduces both declared composites exactly — this fix protects operators, not the campaign.) |
| B3 | P1 | C5 | **Remote-backend logs bypass H-1 redaction — root cause is call ordering.** `_handle_completion` collects (`:1994`) → redacts (`:2004`) → … → drains the remote log **last** (`:2048`, `_drain_remote_backend_log` `:2309-2353`), so on SLURM/Ray the redaction runs on a file that doesn't exist yet; submitit stdout/stderr are additionally symlinked raw (`:432-463`). (Local backend: the live-window gap is mitigated consumer-side by the `AUTOMIL_CERTIFY` print gate, `core_utils.py:211-217`.) | **Move the drain call before collection** — one moved line puts the drained log under the existing redaction *and* repairs remote OOM/timeout classification + error tails (`:2280-2303` currently reads a nonexistent file). Replace raw symlinks with a redacted copy at completion. |
| B4 | P1 | C2 | **`propose` admits `kind=None` in preserving mode; `portfolio` then hard-fails on "unspecified"** with a message that never names the offender (`propose.py:96-104` vs `:215-226`). Costs agent-active budget on a loop the skill mandates every batch. | Require `--kind` at the write when mode is architecture-preserving; error text lists the two allowed kinds. |
| B5 | P1 | RQ3 path | **Nominating a node evicts it from `best_node` and from `certify`'s default target** — `recompute_best` and `_sorted_keep_nodes` walk `status=="keep"` only (`graph.py:824-826`, `certify.py:44-48`); `candidate`/`registered` (better-validated states) silently vanish from "best". | Treat `{keep, candidate, registered}` as the keep-class in both walks. |
| B6 | P1 | C5 | **Reconcile paths trust the reported scalar** (`graph.py:982`, `:1112`, `cli/reconcile.py:85`) and `_mark_crashed` bypasses `write_terminal_state` entirely (`:2469-2492`) — so externally-written `completed/*.json` enters the graph without CR-1b or sealing. Publication numbers are safe (certification re-reads sealed folds), but the search-time graph is spoofable. **Review finding: the recompute must travel with A6's key-guard as one unit** — recomputing over stale `metrics` containing `test_*` keys would *average test into the composite*, worse than trusting reported. | Extract terminal-writer Step 2b (key-guard + `recompute_composite` + B1's SE recompute) into one shared ingest helper called from all four mouths (terminal writer + three reconcile paths). `_mark_crashed` gets the one-line sealed-key strip as a symmetry guard (vacuous today; keep it that way). Deduplicating an existing choke point, not a new layer. |
| B7 | P1 | C1, C2, C4 | **Baseline-before-search ordering is runbook-only.** `open_agent_session` requires pristine discovery state but **not a registered baseline** (`campaign_stages.py:2209-2262`); the initial phase is already `discovery` (`:194`). Attempts can burn with no incumbent, and a reconcile-created `graph.json` bricks baseline registration forever (`:544-547` "graph already has nodes but no registered baseline root"). TITAN is the live tripwire: `conch_v15` features are unextracted for all five cohorts, `titan/prepare.py:56-95` fails fast → 20 slide cells could each burn the full charged-attempt budget in crashes; the baseline run is the only fail-closed data preflight and nothing forces it first. | One line at the existing pristine-state choke point: `open_agent_session` additionally requires a registered baseline. Enforces README §3→§4, makes the baseline the guaranteed data preflight (TITAN included), closes the reconcile-bricking path. |
| B8 | P1 | C2, C3 | **Historical-baseline reuse is a hand-declared equivalence with no mechanical check.** `repair_baselines.py:36-37` hardcodes nnmil/dtfd/titan as reusable vs clam/abmil stale; the config/source fingerprint check exists only for the rerun pair (`:339-350`), and the reuse contract deliberately excludes code provenance — so code drift on a "reusable" arm (e.g. the L-2 determinism unification, which changed DTFD/ABMIL/TITAN reproducibility semantics per its own docstring) enters C2's paired contrast undetected. | Make the decision mechanical: extend the existing fingerprint gate (arm-config defaults + trainer-source digest vs the frozen tree) to the reusable frameworks; what fails the gate gets rerun by the runner that already automates reruns. No new experiments beyond what the gate itself demands. |
| C-a | P2 | RQ2 | **No Optuna/random/human-recipe arm exists** anywhere in the campaign. The paper must not claim superiority over menu AutoML (proposal Attack 1 has no empirical rebuttal); scope C2/C3 to lift-over-default and rank response. | Claims discipline now; a matched-budget Optuna arm on a cell subset is the Phase-2 answer. |
| C-b | P2 | C2 | **`PolicyVariant.step(loss, opt)` is documented and never called by any consumer**, and true SAM is unreachable regardless (needs a closure). The ABC docstring promises "SAM-style two-step" support the consumer loop cannot deliver. | Fix the docstring + canonical skill to the honest reachable set (§2); closure/step wiring is a Phase-2 decision, not a pre-campaign patch. |
| C-c | P2 | C5 | **The skill/AGENTS text never mentions the eval budget** — the campaign's actual unit of effort (60 at audit; 30 post-merge). Agents discover it by refusal. AGENTS.md also still claims "6h cap framework-enforced" while the template says consumer-supplied. | Canonical-skill paragraph: check remaining attempts (`automil cell status`), treat attempts as the budget; align AGENTS.md cap language. Propagates via A3's sync. |
| C-d | P2 | portability | **Codex assets are dead code**: skill install is `rt == "claude"`-gated (`init.py:111`), the codex overlay documents a merge that never runs, `show-skill` renders identical shared content for every runtime. README's "per-runtime SKILL/AGENTS overlay shipped" overstates. | Delete the unreachable overlay file; correct README/agent-compatibility wording to what is true (shared skill + per-runtime install targets). |
| C-e | P2 | hygiene | Dead/misleading knobs: `gate.auto_nominate` read+validated, acted on by nothing (`gate/config.py:41,99-106`); `gate/promote.py:15` docstring still says "Pareto"; exploration/novelty weights have no config seeding path. | Remove the dead knob from the template + config load; fix the docstring; leave weights (documented internal default). |
| C-f | P2 | positioning | **Fresh-session-per-cell protocol disables cross-session knowledge accumulation** — a README/paper selling point ("agents don't repeat mistakes") that the campaign deliberately turns off for independence. | Paper text: within-cell learning only for the campaign; cross-session learning is a framework feature demonstrated by the anchors, not by the 130-cell result. |
| C-g | P2 | C5 | **The firewall's threat model is anti-accident, not anti-adversary**: the seal is a naming convention in a tree the agent can shell into (`certify/` readable; no OS boundary). Publication numbers are protected by the certification chain (sealed-fold hashes bound at freeze, fail-closed report), not by secrecy. | State the threat model in the paper; keep the audit-trail framing (trajectories + archived policy code make violations detectable). |
| C-i | P2 → **SUPERSEDED at the #39 merge** | C5 | **Was: one-session-per-cell is attestation, not enforcement** — a replacement runtime session could silently inherit the on-disk `agent_session.json` binding (`submit.py:569-589`, audited state) and misattribute all proposals to session #1. #39 makes it enforcement: SessionStart must be the journal's exclusive first event, every submit requires a live metered sample from exactly the bound session, and the freeze requires that session closed with a durable SessionEnd — a replacement session can no longer inherit anything. | Residual is the inverse hazard: a session that dies without SessionEnd cannot be finalized against a dead exporter; recovery is `automil activity close` (operator-attested finalization from the last durable sample, added in this integration) plus runbook disclosure in `termination_reason`. |
| C-j | P2 | C1, C5 | **Freeze aborts permanently on ordinary clock skew**: `_freeze_discovery_unlocked` raises if any archived `submitted_at` precedes the controller's `bound_at` (`campaign_stages.py:986-1003`); submit host ≠ controller host + NTP-level seconds of skew around the first submit → a frozen-in artifact that can never pass freeze. | A small declared tolerance in the `PROTOCOL` dict (recorded in the audit row), accepted within it; beyond it still fails closed. Lower bound only — `ended_at` is operator-supplied at finalization and needs none. |
| C-k | P2 | C1, C4 | **A data-determined NaN validation fold makes the *baseline* permanently unregistrable**: an undefined per-fold c-index (single event censored past all others in a ~10-patient val fold) → `status: "partial"` (`run_experiment.py:317-343`) → `register_baseline` refuses (`campaign_stages.py:365-368`) and reruns reproduce it — every cell of that (dataset, task) is dead with no protocol answer. Low probability post-stratification; 13-cell blast radius. | Pre-launch, CPU-only: compute the five per-fold val c-index definabilities from the existing split CSVs + labels once per (dataset, task). Operator-runbook preflight; no training, no new experiments. |
| C-l | P2 | C2, C3 | **Concurrent baseline prep across encoder-cells of one (dataset, task) can race non-atomic task-CSV/split creation** (`prepare.py:258-271` documents the hazard and prescribes prep-once; `splits.py:246-248` plain `to_csv`) — the loser can silently freeze *different fold definitions*, invisible damage to exactly the cross-cell comparability pool. The campaign README never requires the prep step PRELAUNCH_REVIEW already prescribed. | Runbook: mandatory one-time `--prep_only` per dataset before any concurrent baseline launch (already the documented convention). The `flock` around `prepare_all` goes to the journal ledger. |
| C-h | P2 | feasibility | **Compute arithmetic**: 130 cells × (30×3-fold discovery + ≤10×2-fold promotion) ≤ 110 fold-trainings/cell ≈ 14,300 fold-trainings ≈ 700–1,200 GPU-h (at the measured 3–5 min/fold-training) → ~1–1.5 weeks on 4×H100, plus ≤12h agent-active × 130 sessions. The plan's own "pilot 12–18 cells" option remains the fallback; nothing in the machinery prevents certifying a predeclared subset — but the current manifest fails closed at 130, so a scope cut means a regenerated manifest, not an exception path. | Decide scale before launch; if cut, cut by regenerating the manifest (keeps fail-closed semantics). |

## 4. Fix plan (what ships with this audit)

Ordered so each change is independently verifiable; none adds a new layer —
every fix lands inside an existing choke point, seam pattern, or config field.

Commit sequence (review-approved; tree green at every step):

1. **A1** — CLAM opaque-channel wiring (`apply_overrides_to_exp_cfg` partition
   helper in `hparams.py`, applied at the top of clam `run_experiment`) +
   channel test. No manifest implications (trainers aren't hash-locked).
2. **A2** — vendored `validate`/`validate_clam` return
   `(stop, {val_loss, val_error, val_auc})`; thread into `should_stop` + test.
3. **A8** — wire or lock nnmil `dropout` (decided by reading `SimpleMIL`) +
   test.
4. **Canonical-asset truth pass** — C-c eval-budget paragraph + AGENTS.md cap
   wording; A7 reachable-family text; C-b `step()`/ABC docstring honesty.
5. **A3** — sync `.claude` copy from canonical + drift test over both copies ×
   both skills (carries step 4's content).
6. **A9** — bill-at-archive invariant: move `_record_cell_launch` to follow
   the archive-spec write in `_launch`; docstring update; billing test for a
   pre-spawn failure.
7. **A4 + A10 (atomic)** — five campaign template lock-list edits +
   off-manifest template consistency (ccrcc/ovarian_hrd/placeholder) +
   `EXPECTED_IDENTITY_LOCKED_HPARAMS` constant + `PROTOCOL`
   time-containment value applied at materialization (discovery + promotion;
   *this A10 portion superseded at the #39 merge — see the A10 row*)
   + `audit_materialized_campaign` extension (locks, override options, cap) +
   name-collision/alias assertions + **regenerated `manifest.json` +
   `.sha256` in the same commit** (`search_space.py` untouched except a
   cross-reference comment).
8. **A5** — widen `_SPEC_ENV_BLOCKED` (5 keys) + ambient `AUTOMIL_CERTIFY`
   pop + whitelist tests.
9. **A6** — `firewall.is_held_out_metric_key` + terminal-writer fail-closed
   check + `test_tick_cells.py:309` fixture + guard tests.
10. **B1 + B6 (one unit)** — shared ingest helper (key-guard + composite
    recompute + SE recompute) at the terminal writer and the three reconcile
    mouths; `aggregate_folds` SE; `_mark_crashed` strip; reconcile fixture
    updates.
11. **B2** — template comment + seeding validation (outside the blanket
    except) + `automil check` + `scoring.py` docstring.
12. **B3** — drain-call reorder + submitit copy-redact +
    `test_log_unification` updates.
13. **B4, B5** — propose/portfolio + `KEEP_CLASS` constant + test updates.
14. **B7** — `open_agent_session` requires a registered baseline + test.
15. **C-j** — declared `submitted_at` tolerance in `PROTOCOL`, classify
    within it + test. (Rides the same manifest regen as step 7 if sequenced
    before it; otherwise regenerate once more — the regen is deterministic.)
16. **B8** — extend the baseline-reuse fingerprint gate to the reusable
    frameworks in `repair_baselines.py`.
17. **C-d, C-e** — dead-asset deletions (checking
    `tests/skills/test_phase7_acceptance.py` references first) + docstring/
    template hygiene.

Doc-only items (A7 methods alignment, C-a, C-f, C-g, C-h, C-i, C-k, C-l)
live in this file, the campaign runbook, and the paper's methods checklist;
no code.

## 5. Verdict on the second claim

**Can the agent, through the current harness, achieve C2?** After A1/A2/A3
(without which the answer is *no by construction* — the reference arm's
declared scalar channel is a silent no-op, its stopping policies are blind,
and the primary runtime's instructions fight the mode; only the
optimizer-wrapper route would remain live on CLAM):

- The reachable space (scalars + optimizer wrappers + schedules + stopping
  policies + criterion shaping if A7 ships) demonstrably moves MIL AUC at the
  1–9 point scale in this codebase's own history (GOLDMARK exact-protocol
  deltas came from recipe knobs alone: optimizer, epochs, checkpoint policy).
- The pre-registered C2 is **descriptive and sign-based** — it does not promise
  anchor-sized lifts, and the campaign's design (promotion on unseen folds,
  parent-SE margin, native-baseline-wins-ties, sealed paired reveal, no
  p-values) makes a modest true lift honestly reportable and a null result
  survivable (the proposal's §8 pivot: "recipe bias is modest under
  architecture-preserving constraints among defaults-tuned arms" — still a
  publishable audit finding).
- The residual risk is **effect size on the two small cohorts** (≈10 val
  patients/fold): the 3-fold discovery mean + 2-fold promotion barrier is the
  right shape; expect visible val→test shrinkage and pre-commit to showing it
  (the census already captures the val trajectory per attempt).

So: **yes, conditionally** — the condition being the P0 list, which is exactly
what §4 ships. Under the ship-fast constraint (no protocol-surface growth at
the preprint stage), the honest expectation for C2 is "small but real lifts
concentrated in arms with wide scalar spaces (dtfd, nnmil), possible nulls on
titan/abmil" — and that outcome is *pre-registered as publishable*: the
analysis plan is descriptive and sign-based, and the proposal's §8 pivot
("recipe bias is modest under architecture-preserving constraints among
defaults-tuned arms") is a legitimate audit finding, not a failure. The
loss-shaping families that would raise the ceiling are journal-stage items
(§7); the preprint's job is to publish the claim honestly, not to maximize it.

## 6. Architecture-level patterns (beyond the itemized findings)

1. **Duplicated-artifact drift** (A3, C-d, and CLAUDE.md↔AGENTS.md): checked-in
   copies of generated/canonical content, with no byte-equality guard. The A3
   test establishes the pattern; apply it to any future runtime directory.
2. **Fail-open compatibility paths**: formula typo → trust-reported (B2);
   spec without `cell_id` → uncapped launch (deliberate `Backend.submit` seam;
   since the #39 merge a *declared* cell that is missing or unreadable fails
   closed instead of launching);
   legacy free-mode spec → admissible (`admissibility.py:761-782`); missing
   `cells/` dir → no cap ticks. Each is deliberate back-compat, each is also a
   quiet hole under the campaign's fail-closed philosophy. Preserving mode is
   fail-closed everywhere it matters; keep it that way and log the free-mode
   exceptions loudly.
3. **Doc-promises-code-doesn't**: `auto_nominate`, `PolicyVariant.step`, codex
   overlay, "documentation-only" formula, "Pareto" docstring. Every one is an
   agent- or operator-facing surface; each costs somebody a wrong decision.
   The fix class is: delete the promise or implement it — never leave it.
4. **Self-reported inputs to enforcement**: composite was fixed (CR-1b), SE was
   not (B1). Rule: any value that *gates* selection must be recomputed from
   evidence the framework already holds, or explicitly declared trusted.
5. **What is already right** (and the paper should say so): born-sealing at the
   subprocess boundary with a single ingest choke point; launch-time
   re-validation from live config (not recorded verdicts); the campaign's
   hash-locked, fail-closed certification chain; parent-SE keep-margins;
   eval-count as the portable effort unit — *once A9 made it well-defined at
   its edge* (bill-at-launch was mis-implemented; the companion time-cap
   tension resolved at the #39 merge by protocol re-sizing, 30 attempts
   inside a 12h metered budget); a no-p-values descriptive analysis
   plan with explicit missingness handling; D-139 held-out isolation in `rank`;
   the `AUTOMIL_CERTIFY` print gate inside the vendored trainer. One decorative
   key to note: the materialized configs' `metrics.composite_formula` is
   informational — CR-1b seeds from `scoring.formula` and runs the `mean`
   reducer, which reproduces both declared campaign composites exactly.
6. **Verified clean under adversarial sweep** (recorded so they are not
   re-litigated): fold definitions are protocol-identical across 5/3/2-fold
   invocations (pinned `--n_folds 5 --seed 42`, per-fold reseed); per-node
   overlay snapshots make `_policies` overwrites harmless to earlier
   candidates; promotion copies are hash-verified byte-exact; the top-10
   ordering is deterministic and independently recomputed at freeze;
   `resubmit` is refused in preserving mode; cap-refused specs are unbilled
   and census-excluded; exact winner ties prefer the native baseline; the
   `status` surfaces print validation-only values.

## 7. Journal-stage ledger (deferred by the ship-fast constraint)

Owner constraint (2026-08-07): the preprint adds **no experiments and no
protocol surface**; anything that would is recorded here for the journal phase
instead of being half-done now. Each entry names what it would buy and why it
waits.

| Item | What it buys | Why it waits |
|---|---|---|
| `wrap_criterion_for` policy seam (loss shaping: smoothing/focal/class-balanced) | The anchors' largest single ingredient on every arm → larger expected C2 lifts | Protocol-surface growth under the ship-fast rule. A de-risked **classification-only** design now exists (design-review-approved, recorded here for the journal implementation): ABC identity-default `wrap_criterion_for(criterion, role)` + `PolicyRuntime` guard mirroring `wrap_scheduler`; five bag-criterion sites (`core_utils.py:117-123` — never the instance loss; `abmil/train.py:120`; `dtfd/train.py:187` — wrapper must preserve `reduction="none"`; `titan/train.py:110`; nnmil `classification_trainer.py:186`), all `role="bag"`. Survival stays closed for three verified reasons: 3-arg losses aren't criterion-shaped; the same loss computes `_val_loss` = the selection signal; C4 is a separate regime so uniform closure adds no channel asymmetry. CLAM caveat: the wrapped criterion also computes val_loss (train objective == monitored objective — consistent; document). |
| `label_smoothing` as a declared scalar (`CrossEntropyLoss` kwarg via `--hparams`) | Cheapest loss-shaping ingredient, rides existing channel machinery | Reviewed *small but not trivial*: two mandatory fail-loud guards (`bag_loss=svm` incompatibility; survival-task rejection) + CLAM smoothed-val_loss semantics caveat + A1 dependency — protocol-surface growth under the ship-fast rule |
| Sampler / batch-construction seam | Second proposal-allowed family (WeightedRandomSampler beyond CLAM's bool) | Same class of risk; per-arm dataloader construction differs |
| Closure-passing `step` so SAM-class optimizers become reachable | The CCRCC anchor's SAM ingredient | Requires trainer-loop restructuring (second forward pass); heaviest of the seam items |
| Matched-budget Optuna / random-mutation / human-recipe arms | Empirical rebuttal of "just Optuna with Claude" (RQ2, proposal §6.2 methods 2–4) | Multiplies campaign compute; RQ2 is explicitly not pre-registered for the preprint |
| Agent-memory on/off + literature-retrieval on/off ablations | RQ2 ablation row | Additional sessions per cell |
| Regression task axis (CPTAC-PDAC continuous infiltration score, if recoverable) | Third task family; PLAN.md §4's own open check | New task type end-to-end |
| Encoder dynamic-range restoration (ResNet50 / CTransPath legacy arm) | Resurrects the encoder-axis question PRELAUNCH O1 killed | New extractions + cells |
| Extended-search ablation (5× budget on ~5 cells, proposal §6.3.5) | Saturation evidence for the 30-attempt budget choice | More attempts per cell by construction |
| Recipe-family / transfer analysis (RQ3 beyond the gate), recipe planner (proposal §11) | The nnU-Net-style upside | Needs the campaign's traces as input; journal by design |
| `flock` around shared-`benchmark_dir` prep (`prepare_all`) | Kills the concurrent-prep race class permanently (C-l) | Touches the protected prep path; the runbook prep-once rule covers the preprint |
| Live runtime-session evidence per submit (C-i mechanical half) | Detectable session resumption in the census | Attestation + runbook disclosure suffice for the preprint threat model |

**Post-integration residuals (2026-08-08, from the #39 adversarial review;
accepted, not blocking).** (a) `task.name` is hard-required by cell identity —
adding it to an older config re-keys the budget cell; upgrade edge, document
on the next breaking release. (b) One corrupt journal line or orphaned sample
degrades activity reads project-wide — now non-destructive (daemon holds, CLI
shows DEGRADED) but with no repair command. (c) Port 9464 single-tenancy is
documented for the campaign only; the generic degradation message doesn't name
port collision as a cause. (d) Generic (non-campaign) projects support at most
one agent-active cell, bound to its first session — stated in refusals, not in
the tutorial docs. (e) Per-tick journal replays are O(specs×cells) — bounded
by the tiny lifecycle-only journal. (f) `budget show` duplicates `cell`'s
registry-error epilogue. (g) A moved-value scrape can still lose a clock-skew
race and degrade one observation (self-heals next tick). (h) The Gate 1
real-GPU canary should name the exporter label contract
(`type∈{cli,user}` + session id) as an explicit checklist item.

## 8. Paper-side checklist (no code; carry into the manuscript)

- Scope C2/C3 as lift-over-default + rank response; no AutoML-superiority
  language anywhere (C-a).
- Methods table of the per-arm action surface = `search_space.coverage_table()`
  plus the seam list, with locked knobs and reasons — the declared-lock story
  *is* the rigor claim.
- Anchors labeled free-mode feasibility demos, off-roster, ingredient list
  annotated with what the campaign harness does/doesn't reach (§2).
- Threat model paragraph (C-g); knowledge-accumulation scoping (C-f);
  GPU-attached-job-hours wording (not "GPU utilization"); single-seed and
  dependence caveats verbatim from the analysis plan.
- Val→test shrinkage per cell shown as the selection-noise diagnostic.
