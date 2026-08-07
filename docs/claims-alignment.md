# Claims–mechanism alignment audit — preprint campaign

_Audit of whether the agent, driving autoMIL through the current harness and
constraints, can actually produce the evidence each preprint claim needs.
Run 2026-08-07 against `main` (post-#38, `automil-preprint-130-v4`), before any
campaign cell has launched. Every finding below was verified in code by at
least two independent readers; file:line citations are to this checkout._

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
| **C2** | **Equal-effort agentic recipe search produces a real, honestly-measured lift over the native default** (fig-3 successor; RQ1 evidence base) | `agentic_lift`: frozen winner − native baseline, sealed-test, per cell; sign counts poolable, magnitudes per task type | 60-attempt discovery on folds 0–2 → ≤10 promotion on folds 3–4 → 5-fold-val winner → campaign-wide freeze → paired reveal |
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

Reachability, as the code stands (before the fixes in §4):

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
| A1 | P0 | C2, C3, C5 | **CLAM's `--hparams` channel is dead.** No `apply_overrides(..., arm="clam")` exists; `clam/train.py::_make_clam_args` reads config directly and `hparam_overrides` is parsed (`run_experiment.py:464`) but never consumed. On 30/130 cells the only admissible scalar channel is a **silent no-op** — attempts get charged for runs identical to baseline. This is the exact H-3 failure mode `hparams.py:1-42` exists to remove, live on the reference arm. | One call at the CLAM entry, same pattern as `abmil/runner.py:73` / `dtfd/runner.py:79`, for classification + survival; channel test mirroring the other arms. |
| A2 | P0 | C2, C3 | **CLAM classification passes `metrics={}` to `should_stop`** (`benchmarks/lib/CLAM/utils/core_utils.py:199`), so metric-driven stopping policies are structurally impossible on 15 cells while every other arm supplies val metrics. Unequal policy surface = channel width reported as a model result. | Have `validate`/`validate_clam` return the val metrics they already compute; pass them through. Vendored file is already policy-patched at `:168-170,198-199`. |
| A3 | P0 | C2, C5 | **`.claude/skills/automil/SKILL.md` is stale** — predates `registry.mode`; tells the agent to aim ≥50% architecture/ensemble proposals and use `automil/variants/<parent>/` model variants, all inadmissible in the campaign. `.agents/` copy matches canonical `_shared`; the Claude copy (the primary runtime) diverged, and **no test guards any copy against drift**. A campaign session would spend its 6h agent-active budget fighting admissibility. | Sync from `agent_assets/_shared` + one test asserting both in-repo copies equal the canonical render (kills the whole drift class). |
| A4 | P0 | C2, C5 | **Capacity/architecture knobs are tunable under `architecture-preserving`**: `model_size` (clam), `M`,`L` (abmil), `mDim`,`numLayer_Res` (dtfd), `hidden_dim` (nnmil) change widths/depth, but `identity_locked_hparams` lists only CLAM's two loss switches (`config.yaml:84`). `audit_materialized_campaign` (`campaign.py:841-850`) never verifies `allowed_override_options`/`identity_locked_hparams`, so drift passes the 130-root audit. Proposal §5 forbids adding/removing layers and changing widths — reviewer Attack 4 is currently valid. | Config-only: extend `identity_locked_hparams` per roster template; regenerate the manifest; make the audit verify both fields. No new code paths. |
| A5 | P0 | C5 | **`spec.env` can retarget the seal**: `_SPEC_ENV_BLOCKED` (`_orchestrator_daemon.py:75-82`) omits `AUTOMIL_RESULTS_DIR` (born-sealing target, `:1169-1171`) and `AUTOMIL_DIR_REL` (policy resolution root, `policy_dispatch.py:120`). A hand-dropped queue spec (queue specs are unvalidated, `:991-1005`) un-seals a node or repoints policy resolution. | Add the two keys (plus `AUTOMIL_NODE_ID`) to the frozenset; one test. |
| A6 | P0 | C5, C2 | **`metrics` keys are unconstrained at ingest** — the firewall strips exactly `held_out`/`summary` (`terminal_writer.py:187-188`); a result carrying `metrics: {"test_auc": …}` would flow test into `graph.json`, `results.tsv`, SSE, *and into the recomputed composite* (`scoring.py:59-77` means over all metric keys), i.e. test driving selection with the firewall's blessing. Campaign's `_validation_folds` (`campaign_stages.py:361-394`) catches it only at freeze, after search already consumed it. | Fail closed at the existing schema-validation choke point in `write_terminal_state`: a metrics key matching the firewall's held-out markers → node crashes with a val-firewall pointer (same path as schema failure). No new layer — one more check where checks already live. |
| A7 | P0→decision | C2 | **The seam is narrower than the paper's declared protocol** (§2): no loss-shaping seam anywhere, though the proposal's allowed list names label smoothing/focal/class-balanced — the proven ingredients of both anchors. C2's effect size is capped by this. | Add **one** seam in the established pattern: `wrap_criterion_for(criterion, role)` on `PolicyVariant` (identity default) + `PolicyRuntime` guard + one call per trainer where the bag-level criterion is built. Role-scoped so defining losses (CLAM instance branch, DTFD tier structure) stay closed. Alternatively: narrow the paper's methods text to the implemented surface and accept smaller expected lifts. Recommendation: add the seam — it aligns code with the *already-declared* protocol rather than widening it. |
| B1 | P1 | C2 | **The Ladder margin's noise floor is self-reported.** `composite_se` is read verbatim off `result.json` (`terminal_writer.py:283`); `scoring.cross_fold_se` is never called in `src/automil/`; `cells/reconcile.aggregate_folds` has fold composites in hand (`cells/reconcile.py:52-95`) and computes no SE, so budget-killed/partial nodes silently drop to the bare δ. The same machinery that refuses to trust the reported `composite` (CR-1b) trusts the reported SE that *gates* it. | Recompute SE at ingest from `result["validation_folds"]` (agent-visible, val-only, already emitted — `evaluate.py:213`); prefer recomputed, log disagreement — the exact CR-1b pattern. Same helper in `aggregate_folds`. |
| B2 | P1 | C5 | **`scoring.formula` fails open on a typo and the template teaches the failure.** `config.yaml.j2:148-155` says "documentation-only … NOT evaluate[d]" with examples (`"accuracy"`, `"(val_auc + val_bacc) / 2"`) that `scoring.py:59-66` rejects as reducer names; `terminal_writer.py:229-232` catches the ValueError and **trusts the reported composite** — CR-1b silently disabled by following the template's own comment. | Rewrite the template comment (reducer semantics: `mean`/`max`/`min`/`trust_reported`); validate the reducer name at graph seeding and in `automil check` so the state is unrepresentable; fix the `scoring.py:56-58` docstring that claims fail-loud. |
| B3 | P1 | C5 | **Remote-backend logs bypass H-1 redaction.** `_handle_completion` redacts at `:2004`, *then* `_drain_remote_backend_log` (`:2309-2353`) writes `run.log` with no redaction; submitit stdout/stderr are symlinked raw (`:432-463`). On SLURM/Ray the log redaction is a no-op. (Local backend: the live-window gap is mitigated consumer-side by the `AUTOMIL_CERTIFY` print gate, `core_utils.py:211-217`.) | Redact after the drain write; replace raw symlinks with a redacted copy at completion. |
| B4 | P1 | C2 | **`propose` admits `kind=None` in preserving mode; `portfolio` then hard-fails on "unspecified"** with a message that never names the offender (`propose.py:96-104` vs `:215-226`). Costs agent-active budget on a loop the skill mandates every batch. | Require `--kind` at the write when mode is architecture-preserving; error text lists the two allowed kinds. |
| B5 | P1 | RQ3 path | **Nominating a node evicts it from `best_node` and from `certify`'s default target** — `recompute_best` and `_sorted_keep_nodes` walk `status=="keep"` only (`graph.py:824-826`, `certify.py:44-48`); `candidate`/`registered` (better-validated states) silently vanish from "best". | Treat `{keep, candidate, registered}` as the keep-class in both walks. |
| B6 | P1 | C5 | **Reconcile paths trust the reported scalar** (`graph.py:982`, `:1112`, `cli/reconcile.py:85`) and `_mark_crashed` bypasses `write_terminal_state` entirely (`:2469-2492`) — so externally-written `completed/*.json` enters the graph without CR-1b or sealing. Publication numbers are safe (certification re-reads sealed folds), but the search-time graph is spoofable. | Run the same `recompute_composite` in the reconcile ingest paths; strip sealed keys in `_mark_crashed` writes. Symmetry, not new machinery. |
| C-a | P2 | RQ2 | **No Optuna/random/human-recipe arm exists** anywhere in the campaign. The paper must not claim superiority over menu AutoML (proposal Attack 1 has no empirical rebuttal); scope C2/C3 to lift-over-default and rank response. | Claims discipline now; a matched-budget Optuna arm on a cell subset is the Phase-2 answer. |
| C-b | P2 | C2 | **`PolicyVariant.step(loss, opt)` is documented and never called by any consumer**, and true SAM is unreachable regardless (needs a closure). The ABC docstring promises "SAM-style two-step" support the consumer loop cannot deliver. | Fix the docstring + canonical skill to the honest reachable set (§2); closure/step wiring is a Phase-2 decision, not a pre-campaign patch. |
| C-c | P2 | C5 | **The skill/AGENTS text never mentions the eval budget** — the campaign's actual unit of effort (60 attempts). Agents discover it by refusal. AGENTS.md also still claims "6h cap framework-enforced" while the template says consumer-supplied. | Canonical-skill paragraph: check remaining attempts (`automil cell status`), treat attempts as the budget; align AGENTS.md cap language. Propagates via A3's sync. |
| C-d | P2 | portability | **Codex assets are dead code**: skill install is `rt == "claude"`-gated (`init.py:111`), the codex overlay documents a merge that never runs, `show-skill` renders identical shared content for every runtime. README's "per-runtime SKILL/AGENTS overlay shipped" overstates. | Delete the unreachable overlay file; correct README/agent-compatibility wording to what is true (shared skill + per-runtime install targets). |
| C-e | P2 | hygiene | Dead/misleading knobs: `gate.auto_nominate` read+validated, acted on by nothing (`gate/config.py:41,99-106`); `gate/promote.py:15` docstring still says "Pareto"; exploration/novelty weights have no config seeding path. | Remove the dead knob from the template + config load; fix the docstring; leave weights (documented internal default). |
| C-f | P2 | positioning | **Fresh-session-per-cell protocol disables cross-session knowledge accumulation** — a README/paper selling point ("agents don't repeat mistakes") that the campaign deliberately turns off for independence. | Paper text: within-cell learning only for the campaign; cross-session learning is a framework feature demonstrated by the anchors, not by the 130-cell result. |
| C-g | P2 | C5 | **The firewall's threat model is anti-accident, not anti-adversary**: the seal is a naming convention in a tree the agent can shell into (`certify/` readable; no OS boundary). Publication numbers are protected by the certification chain (sealed-fold hashes bound at freeze, fail-closed report), not by secrecy. | State the threat model in the paper; keep the audit-trail framing (trajectories + archived policy code make violations detectable). |
| C-h | P2 | feasibility | **Compute arithmetic**: 130 cells × (60×3-fold discovery + ≤10×2-fold promotion) ≤ 200 fold-trainings/cell ≈ 26,000 fold-trainings ≈ 1,300–2,200 GPU-h (at the measured 3–5 min/fold-training) → ~2–3 weeks on 4×H100, plus ≤6h agent-active × 130 sessions. The plan's own "pilot 12–18 cells" option remains the fallback; nothing in the machinery prevents certifying a predeclared subset — but the current manifest fails closed at 130, so a scope cut means a regenerated manifest, not an exception path. | Decide scale before launch; if cut, cut by regenerating the manifest (keeps fail-closed semantics). |

## 4. Fix plan (what ships with this audit)

Ordered so each change is independently verifiable; none adds a new layer —
every fix lands inside an existing choke point, seam pattern, or config field.

1. **A1** — wire `apply_overrides(..., arm="clam")` at the CLAM entry
   (classification + survival) + channel test.
2. **A2** — thread val metrics into CLAM classification `should_stop` + test.
3. **A3** — sync `.claude/skills/automil/SKILL.md` from canonical; add the
   drift test covering both in-repo copies; fold C-c's eval-budget paragraph
   into the canonical skill first so the sync carries it.
4. **A4** — extend `identity_locked_hparams` in the five roster templates
   (`model_size`; `M`,`L`; `mDim`,`numLayer_Res`; `hidden_dim`); teach
   `audit_materialized_campaign` to verify `allowed_override_options` +
   `identity_locked_hparams`; regenerate + re-lock the manifest.
5. **A5** — extend `_SPEC_ENV_BLOCKED` + test.
6. **A6** — held-out-marker check on `metrics` keys at the terminal-writer
   schema step, fail-closed + test.
7. **A7** — `wrap_criterion_for` seam (ABC default-identity → `PolicyRuntime`
   guard → one construction-site call per trainer, role-scoped) + tests; update
   `search_space.py` docs and the canonical skill's reachable-family list.
8. **B1–B6** — as specified in the table, each with a test.
9. **C-b/C-c/C-d/C-e** — docstring/docs/template truth fixes riding the same
   branch.
10. Doc-only items (C-a, C-f, C-g, C-h) live in this file and the paper's
    methods checklist; no code.

## 5. Verdict on the second claim

**Can the agent, through the current harness, achieve C2?** After A1/A2/A3
(without which the answer is *no by construction* — the reference arm can't be
tuned, its stopping policies are blind, and the primary runtime's instructions
fight the mode):

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
what §4 ships. Without A7 the honest expectation for C2 is "small but real
lifts concentrated in arms with wide scalar spaces (dtfd, nnmil), possible
nulls on titan/abmil"; with A7 the proven loss-shaping family
(both anchors' largest single ingredient) comes into reach on every arm.

## 6. Architecture-level patterns (beyond the itemized findings)

1. **Duplicated-artifact drift** (A3, C-d, and CLAUDE.md↔AGENTS.md): checked-in
   copies of generated/canonical content, with no byte-equality guard. The A3
   test establishes the pattern; apply it to any future runtime directory.
2. **Fail-open compatibility paths**: formula typo → trust-reported (B2);
   spec without `cell_id` → uncapped launch (`_orchestrator_daemon.py:1248-1268`);
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
   eval-count as the portable effort unit; a no-p-values descriptive analysis
   plan with explicit missingness handling; D-139 held-out isolation in `rank`;
   the `AUTOMIL_CERTIFY` print gate inside the vendored trainer.

## 7. Paper-side checklist (no code; carry into the manuscript)

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
