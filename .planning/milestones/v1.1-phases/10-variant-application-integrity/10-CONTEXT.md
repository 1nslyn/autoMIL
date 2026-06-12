# Phase 10: Variant Application Integrity - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning
**Mode:** Auto-decided (`/gsd-autonomous` → discuss `--auto`; grey areas resolved via best-practice defaults, no user prompts). Grounded in locked APL requirements + `tasks/lessons.md` variant-application rules.

<domain>
## Phase Boundary

A registered variant is **never silently inert** — when selected it applies to the
actual live model **through existing open seams**, and any variant whose only
application route would require **opening a closed training loop** fails **loudly**
instead of no-op'ing. Covers APL-01 (iris reference applies), APL-02 (CLAM variants
via the `clam_train` args seam), APL-03 (loud detection of loop-opening variants).

**Depends on:** Phase 9 REC-04 (`mil_model` is first-class; cell key is
`(dataset, encoder, mil_model)`) — APL-02 dispatches variants per cell/model.

**Hard scope fence (from `lessons.md` 2026-06-10):** application is through OPEN
seams only. Do **NOT** open CLAM's closed `train_loop_clam`. Do **NOT** add registry
runtime auto-scaffolding for arbitrary consumers (RTA-03 deferred). Verifying
application by running a real experiment is allowed/encouraged.

</domain>

<decisions>
## Implementation Decisions

### APL-01 — sklearn-iris reference applies (no longer inert)
- **D-01:** `automil apply <node>` (already exists at `src/automil/cli/lifecycle/apply.py:71`)
  resolves the node's registered variant and records the selection into the run
  config/snapshot the consumer's `train.py` reads. The iris reference `train.py` is
  fixed to **dispatch on that selection** — importing and instantiating the registered
  `classifier_v0` variant instead of always building the baseline model. Currently
  `classifier_v0` is registered but `train.py` never imports it (the inert bug).
- **D-02:** Observability contract: the applied run instantiates a **different model
  object** (`classifier_v0`) than the un-applied baseline — assertable in-CI (iris needs
  no external data, so APL-01 is a fully automated end-to-end gate).

### APL-02 — CLAM variants via the `clam_train` args seam
- **D-03:** A registered model/config/hyperparameter variant for the autobench CLAM
  consumer applies by **mapping variant fields → `clam_train` args** in the existing
  `_build_args` seam (`benchmarks/src/autobench/pipeline/clam/train.py:65`):
  `model_type`, `model_size`, `B`, `bag_weight`, `dropout`, optimizer/`lr`,
  `bag_loss`/`inst_loss`. The application/translation layer lives in the **consumer
  (autobench)**, NOT in `src/automil/` (framework stays generic) and NOT in `lib/`
  (no edits to vendored CLAM; no loop opening).
- **D-04 (verification split):** APL-02's "composite differs from baseline by more than
  noise" requires a real CLAM run on workstation data (`AUTOBENCH_CCRCC_ROOT`), which is
  **not available in CI** — gate that as a **human/workstation verification** (consistent
  with v1.0 sub-gate A deferrals). The **automated** layer asserts the variant's fields
  are actually threaded into the `clam_train` args namespace (unit/integration test with
  a stub/spy on `_build_args` / `clam_train`), so application correctness is proven in CI
  without the dataset.

### APL-03 — loud failure for loop-opening variants
- **D-05:** The apply/validate path **classifies a variant by its application route**.
  A variant whose only route requires injecting a callable **inside** CLAM's closed
  `train_loop_clam` (e.g. a custom `LossVariant` not expressible through the
  `bag_loss`/`inst_loss` selectors) is **detected and raises a loud, explicit error**:
  `"requires loop opening — deferred (ISSUE-007 / RTA)"`. It MUST never silently no-op.
  Classification key: can the variant be fully expressed through the open `clam_train`
  args seam? If yes → apply. If it needs mid-loop code injection → loud-fail.

### Claude's Discretion
- Exact mechanism for recording the applied selection (config key vs. snapshot field) and
  the consumer-side dispatch shape — planner/researcher choose, as long as D-01/D-03 hold
  (framework-generic, consumer-side translation, no `lib/` edits).
- Where the route-classification (D-05) lives (apply-path validator vs. a variant
  capability flag) — planner's call; it must fire loudly before any run starts.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked scope & rules (read first)
- `.planning/REQUIREMENTS.md` §"Variant application integrity (APL)" — APL-01/02/03 text;
  §"v2 Requirements (RTA)" + §"Out of Scope" — what is deferred (loop opening, RTA-03).
- `tasks/lessons.md` 2026-06-10 entries — "variants MUST apply through open seams" and
  "don't open the loop ≠ registry may stay inert". The governing rules for this phase.
- `tasks/test-run-issues.md` ISSUE-007 — the canonical blocked loss/attention case (the
  loop-opening route APL-03 must detect and defer).
- `.planning/phases/09-state-recovery-integrity/09-CONTEXT.md` — REC-04 `mil_model`
  first-class + cell keying that APL-02 dispatch depends on.

### Code anchors (verified 2026-06-11)
- `src/automil/cli/lifecycle/apply.py:71` — existing `automil apply <node>` command.
- `benchmarks/src/autobench/pipeline/clam/train.py:65` (`_build_args`) — the OPEN
  `clam_train` args seam for APL-02.
- `benchmarks/lib/CLAM/utils/core_utils.py:225` (`train_loop_clam`) — the CLOSED loop;
  out of scope (do not edit/open).
- `src/automil/registry/` (spec.py `VariantSpec`, scanner.py `@register`, manifest.py) —
  the registry the variant selection resolves against.
- sklearn-iris reference consumer (its `train.py` + registered `classifier_v0`) — the
  APL-01 target; researcher to locate exact path (under the shipped reference/example
  consumer).

</canonical_refs>

<code_context>
## Existing Code Insights

- **`automil apply` already exists** — the work is making application *take effect*, not
  adding the command.
- **Registry is live** — variants register via `@register(VariantSpec(...))`; the apply
  path resolves a node → its variant. The gap is the consumer-side dispatch that imports
  + instantiates the resolved variant.
- **Open seam vs closed loop** — `_build_args`→`clam_train(args)` is the supported
  injection point; `train_loop_clam` internals are off-limits. This boundary IS the
  APL-02 vs APL-03 dividing line.
- **Framework purity (D-206)** — zero autobench refs in `src/automil/`; the CLAM
  translation layer lives in autobench, not the framework.

</code_context>

<specifics>
## Specific Ideas

- The through-line: **registered ≠ applied**. Every APL item proves the variant reaches
  the real model object (iris: different model instantiated; CLAM: different `clam_train`
  args), or fails loudly when it can't without opening the loop.

</specifics>

<deferred>
## Deferred Ideas

- Opening CLAM's `train_loop_clam` (ISSUE-007 / RTA-01) and concrete mid-loop loss/attention
  variants (RTA-02) — future registry-adoption milestone. APL-03 only *detects + defers* them.
- Auto-scaffolding registry dispatch for arbitrary new consumers (RTA-03) — out of scope;
  the documented contract + fixed iris reference suffice.
- Real-data CLAM composite-delta verification (APL-02) — workstation-gated (`AUTOBENCH_CCRCC_ROOT`);
  CI proves arg-threading instead.

</deferred>

---

*Phase: 10-Variant Application Integrity*
*Context gathered: 2026-06-11 (auto-decided via /gsd-autonomous)*
