# Phase 10: Variant Application Integrity — Research

**Researched:** 2026-06-11
**Domain:** Registry dispatch / consumer-side variant application / open-seam classification
**Confidence:** HIGH — all findings are grounded in direct source-code reads at specific file:line locations.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** `automil apply <node>` (already exists at `src/automil/cli/lifecycle/apply.py:71`) resolves the node's registered variant and records the selection into `automil/config.yaml`. The iris reference `train.py` is fixed to **dispatch on that selection** — importing and instantiating the registered `classifier_v0` variant. Currently `classifier_v0` is registered but `train.py` never imports it.
- **D-02:** Observability contract: the applied run instantiates a **different model object** (`classifier_v0`) than the un-applied baseline — assertable in CI (iris needs no external data, so APL-01 is a fully automated end-to-end gate).
- **D-03:** A registered model/config/hyperparameter variant for autobench CLAM applies by **mapping variant fields → `clam_train` args** in `_make_clam_args` (`benchmarks/src/autobench/pipeline/clam/train.py:62`). The application/translation layer lives in the **consumer (autobench)**, NOT in `src/automil/`. Not in `lib/`. No loop opening.
- **D-04 (verification split):** APL-02's real-run composite delta check is **workstation-gated** (`AUTOBENCH_CCRCC_ROOT` not in CI). Automated layer: assert variant fields are threaded into the `clam_train` args namespace via stub/spy.
- **D-05:** The apply/validate path **classifies a variant by its application route**. A variant that can only inject a callable inside `train_loop_clam` is **detected and raises loudly**: `"requires loop opening — deferred (ISSUE-007 / RTA)"`. Never silently no-op.

### Claude's Discretion
- Exact mechanism for recording the applied selection (config key vs. snapshot field) and consumer-side dispatch shape — as long as D-01/D-03 hold (framework-generic, consumer-side translation, no `lib/` edits).
- Where the route-classification (D-05) lives (apply-path validator vs. a variant capability flag) — must fire loudly before any run starts.

### Deferred Ideas (OUT OF SCOPE)
- Opening CLAM's `train_loop_clam` (ISSUE-007 / RTA-01) and concrete mid-loop loss/attention variants (RTA-02).
- Auto-scaffolding registry dispatch for arbitrary new consumers (RTA-03).
- Real-data CLAM composite-delta verification (APL-02) — workstation-gated.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| APL-01 | `automil apply <node>` causes the live experiment to run with the registered variant applied to the actual model — never silently inert. Demonstrated end-to-end on sklearn-iris (`classifier_v0`). | §APL-01 deep-dive: dispatch mechanism, config key, iris train.py fix |
| APL-02 | Registered model/config/hyperparameter variants for autobench CLAM apply through the `_make_clam_args` seam without editing `lib/` or opening the closed loop. | §APL-02 deep-dive: arg-threading path, `ExperimentConfig.model` mutation point, test stub strategy |
| APL-03 | A variant that can only apply by injecting into `train_loop_clam` is detected and reported loudly — never silently no-op'd. | §APL-03 deep-dive: route classifier, LossVariant capability gap, error message, placement |
</phase_requirements>

---

## Summary

Phase 10 closes the "registered ≠ applied" gap across three surfaces: the sklearn-iris reference consumer (APL-01), the autobench CLAM consumer (APL-02), and the general loud-fail gate for loop-opening variants (APL-03).

**The core bug is simple.** `automil apply <node>` already writes `model.variant`, `loss.variant`, and `policy.variant` keys into `automil/config.yaml` correctly (`apply.py:115–136`). The iris `train.py` (`examples/sklearn-iris/train.py:69`) reads `automil/config.yaml` for the `data.seed` field but then hard-codes `LogisticRegression(max_iter=200, random_state=seed)` — it never reads `config["model"]["variant"]` and never imports the `classifier_v0` module. The variant directory exists and contains a working `make_classifier(seed)` function; it is simply never called.

The CLAM consumer gap is symmetric but lives in `run_experiment.py` + `_make_clam_args`. The config key `model.variant` written by `apply` is never read by `run_experiment.py`; `ExperimentConfig.model` is built purely from CLI args. The fix adds a variant-dispatch layer in autobench that reads the config key and patches `ModelConfig` fields before handing `ExperimentConfig` to `train_fold`.

For APL-03, `LossVariant.__call__` receives `(logits, targets, instance_logits, instance_labels)` — none of which `_make_clam_args` / `clam_train(args)` can deliver without reaching inside `train_loop_clam` (line 225 of `core_utils.py`). The classification rule is: *if applying this variant requires supplying a callable at a call-site inside `train_loop_clam`, it is loop-opening; raise loudly at apply-time, before any run starts.*

**Primary recommendation:** Add a `model.variant` read in iris `train.py` that dispatches to `classifier_v0.make_classifier`; add a `_apply_variant_to_model_cfg` helper in `autobench/pipeline/clam/train.py` that reads `automil/config.yaml` and patches `ModelConfig`; add a `_classify_variant_route` function (callable at `automil apply` time) that raises `ClickException` for any variant whose application route cannot be expressed through the open `_make_clam_args` seam.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Record variant selection | Framework CLI (`apply.py`) | — | Already exists; writes `model.variant` to `config.yaml` |
| Dispatch variant in iris | Consumer (`examples/sklearn-iris/train.py`) | — | Consumer reads its own config; framework stays generic (D-206) |
| Dispatch variant in CLAM | Consumer (`benchmarks/src/autobench/pipeline/clam/train.py`) | — | Translation layer owns `_make_clam_args`; no `lib/` or `src/automil/` edits |
| Route classification (APL-03) | Framework CLI (`apply.py` validation step) | — | Must fire before any run; framework-level because the seam boundary is framework-defined |
| Variant registry storage | Framework (`src/automil/registry/`) | — | `MODEL_VARIANTS`, `LOSS_VARIANTS`, `SPEC_STORE` — already populated at import time |

---

## Standard Stack

All work uses existing project dependencies. No new packages.

### Core (existing, no installation needed)
| Library | Role | Location |
|---------|------|----------|
| `sklearn` | iris classifier variants | `examples/sklearn-iris/` consumer |
| `yaml` (pyyaml) | config read/write for variant dispatch | already used in `apply.py`, `train.py` |
| `click` | ClickException for APL-03 loud-fail | already used throughout CLI |
| `automil.registry.registrar` | `resolve_model`, `resolve_loss`, `resolve_policy` | `src/automil/registry/registrar.py` |
| `automil.registry._state` | `MODEL_VARIANTS`, `LOSS_VARIANTS`, `SPEC_STORE` | `src/automil/registry/_state.py` |
| `automil.registry.scanner` | `scan_variants` — imports variant modules so `@register` fires | `src/automil/registry/scanner.py` |

### Package Legitimacy Audit

No new packages are installed in this phase. All referenced modules are already in the project.

| Package | Verdict | Disposition |
|---------|---------|-------------|
| (none new) | N/A | No installs required |

---

## Architecture Patterns

### System Architecture Diagram

```
automil apply <node_id>
        │
        ▼
apply.py: _derive_variant_selection(node)
  → reads node.variant_spec or node.recipe
  → writes config.yaml: model.variant / loss.variant / policy.variant
  → [NEW APL-03] _classify_variant_route(selection) raises ClickException if loop-opening
        │
        ▼ config.yaml now contains: model.variant = "classifier_v0" (iris)
                                  or model.variant = "clam_mb_v0176" (CLAM)
        │
        ├─── APL-01: iris train.py ──────────────────────────────────────────────┐
        │      reads automil/config.yaml                                          │
        │      reads config["model"]["variant"] → "classifier_v0"                │
        │      [NEW] scan_variants(variants_root)  ← imports logistic_v0.py     │
        │      [NEW] dispatch: from automil.variants.classifier_v0 import ...    │
        │      [NEW] clf = make_classifier(seed=seed) instead of hardcoded LR    │
        │      clf.fit(X_train, y_train) → different model object                │
        │      writes result.json                                                 │
        └─────────────────────────────────────────────────────────────────────────┘
        │
        ├─── APL-02: run_experiment.py + autobench CLAM path ────────────────────┐
        │      reads automil/config.yaml                                          │
        │      [NEW] reads config["model"]["variant"] → e.g. "clam_mb_v0001"    │
        │      [NEW] _resolve_variant_to_model_cfg(variant_name, model_cfg)      │
        │            looks up registered variant → patches ModelConfig fields    │
        │            (model_type, model_size, dropout, B, bag_weight, lr, etc.)  │
        │      ExperimentConfig.model = patched ModelConfig                      │
        │      train_fold() → _make_clam_args(exp_cfg, fold_dir) [UNCHANGED]    │
        │      clam_train(datasets, fold, args) [UNCHANGED — closed loop]        │
        │      writes result.json                                                 │
        └─────────────────────────────────────────────────────────────────────────┘
        │
        └─── APL-03: route classifier fires at apply time ───────────────────────┐
               if variant kind == "loss":                                         │
                 check: is loss name in {bag_loss selector set}?                  │
                   YES → can map to args.bag_loss or args.inst_loss → APPLY       │
                   NO  → requires LossVariant.__call__ inside train_loop_clam     │
                         → raise ClickException("requires loop opening...")       │
               result: loud error before any run starts, never silent no-op       │
               └────────────────────────────────────────────────────────────────── ┘
```

### Recommended Project Structure (new/changed files only)

```
examples/sklearn-iris/
└── train.py                    # EDIT: add model.variant dispatch (APL-01)

benchmarks/src/autobench/pipeline/clam/
└── train.py                    # EDIT: add _resolve_variant_to_model_cfg (APL-02)

src/automil/cli/lifecycle/
└── apply.py                    # EDIT: add _classify_variant_route + APL-03 guard

tests/
├── test_apl01_iris_dispatch.py         # NEW: iris end-to-end variant dispatch
├── test_apl02_clam_arg_threading.py    # NEW: arg-threading stub/spy test
└── test_apl03_loud_fail.py             # NEW: loud-fail for loop-opening variants

benchmarks/tests/
└── test_variant_dispatch_clam.py       # NEW: autobench-side dispatch test
```

---

## APL-01 Deep-Dive: iris Reference Consumer

### What `automil apply` currently does (VERIFIED)

`src/automil/cli/lifecycle/apply.py:71–144` [VERIFIED: direct read]:
- Reads `node.variant_spec` or `node.recipe` from graph.json
- Writes `config.yaml` sections: `model.variant`, `loss.variant`, `policy.variant`
- Atomic write via tempfile+rename. Backup to `config.yaml.bak`.
- Does NOT scan or import variants. Does NOT call into training code.

After a successful `automil apply node_0001`, `automil/config.yaml` contains:
```yaml
model:
  variant: classifier_v0
  parent: null   # (for iris, parent is None — it is not a CLAM model variant)
```

### The inert bug (VERIFIED: direct read of `examples/sklearn-iris/train.py`)

`examples/sklearn-iris/train.py:59–76` [VERIFIED]:
```python
config_path = Path("automil/config.yaml")
seed = 42
if config_path.exists():
    config = yaml.safe_load(config_path.read_text()) or {}
    seed = int((config.get("data") or {}).get("seed", 42))

# INERT BUG: reads only data.seed; never reads config["model"]["variant"]
clf = LogisticRegression(max_iter=200, random_state=seed).fit(X_train, y_train)
```

The `classifier_v0` module at `examples/sklearn-iris/automil/variants/classifier_v0/logistic_v0.py:7` [VERIFIED] provides:
```python
def make_classifier(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=200, random_state=seed)
```

This is never called. The variant directory `__init__.py` (line 1–6) contains only a docstring — no auto-imports.

### The minimal consumer-side dispatch (D-01)

The iris `train.py` must:
1. Read `config.get("model", {}).get("variant")` after loading `config.yaml`
2. If a variant name is set, import and call the variant's `make_classifier`
3. If no variant name, fall back to the baseline `LogisticRegression` (backward compatible)

**How the selection is passed from `apply` → the run:** Via `automil/config.yaml` `model.variant` key — the mechanism `apply.py` already writes. No additional propagation channel needed.

**Discovery mechanism for variant modules:** The simplest approach that requires no framework import in the consumer is direct filesystem path construction:
- `variants_root = config_path.parent / "variants" / variant_name`
- `module_path = variants_root / "logistic_v0.py"` (or whichever .py is present)
- Import via `importlib.util.spec_from_file_location` (same pattern as `scanner.py:46–67`)

Alternatively, the consumer can use the framework scanner directly — but that would require importing `automil`, which contradicts the existing iris philosophy ("No automil.* imports — consumer-decoupled" at `train.py:8`). **Recommendation:** Use a direct `importlib` path-based import, not the scanner. The variant module exports `make_classifier` by convention; iris `train.py` calls it.

**The exact fix (iris `train.py` addition):**
```python
# After reading config, before building clf:
variant_name = (config.get("model") or {}).get("variant")
if variant_name:
    import importlib.util as _ilu
    _variants_dir = config_path.parent / "variants" / variant_name
    _py_files = [f for f in _variants_dir.glob("*.py") if not f.name.startswith("_")]
    if _py_files:
        _spec = _ilu.spec_from_file_location(variant_name, _py_files[0])
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        clf = _mod.make_classifier(seed=seed).fit(X_train, y_train)
    # else: no variant module found — fall through to baseline
else:
    clf = LogisticRegression(max_iter=200, random_state=seed).fit(X_train, y_train)
```

**APL-01 is a fully automated CI gate** — iris uses only `sklearn` (no GPU, no external data). Confirmed by `examples/sklearn-iris/automil/config.yaml:38–40` [VERIFIED]: `hardware.accelerator: cpu`, `hardware.gpu_count: 0`.

### D-02 Observability

The test asserts that when `model.variant = "classifier_v0"` is in config.yaml, the model object instantiated is the `make_classifier` return value (not the hardcoded baseline). Cheapest way: monkeypatch `make_classifier` in the test and assert it was called; or check the class name / identity of the fitted model.

---

## APL-02 Deep-Dive: CLAM Variant via the `_make_clam_args` Seam

### The open seam (VERIFIED: direct read)

`benchmarks/src/autobench/pipeline/clam/train.py:62–92` [VERIFIED] — `_make_clam_args(exp_cfg, fold_dir)` builds the full `SimpleNamespace` that `clam_train(datasets, fold, args)` consumes. The fields controllable through this seam are:

| Seam field | Source in `_make_clam_args` | Variant can override |
|---|---|---|
| `args.model_type` | `exp_cfg.model.model_type` | YES — `ModelConfig.model_type` |
| `args.model_size` | `exp_cfg.model.model_size` | YES — `ModelConfig.model_size` |
| `args.drop_out` | `exp_cfg.model.dropout` | YES — `ModelConfig.dropout` |
| `args.B` | `exp_cfg.model.B` | YES — `ModelConfig.B` |
| `args.bag_weight` | `exp_cfg.model.bag_weight` | YES — `ModelConfig.bag_weight` |
| `args.bag_loss` | hardcoded `"ce"` (line 79) | YES — can be changed here |
| `args.inst_loss` | hardcoded `None` (line 75) | YES — can be changed here |
| `args.lr` | `exp_cfg.train.lr` | YES — `TrainConfig.lr` |
| `args.opt` | `exp_cfg.train.optimizer` | YES — `TrainConfig.optimizer` |
| `args.max_epochs` | `exp_cfg.train.max_epochs` | YES — `TrainConfig.max_epochs` |

**The closed call-site** (OUT OF SCOPE):
`benchmarks/lib/CLAM/utils/core_utils.py:225` [VERIFIED] — `train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer, loss_fn)` — `loss_fn` is passed in from `clam_train`. The `loss_fn` is a string-selected standard loss (`nn.CrossEntropyLoss` or `nn.BCEWithLogitsLoss`), constructed inside `clam_train` from `args.bag_loss`. Injecting a custom `LossVariant` callable requires either replacing the string selector OR patching the loss construction inside `clam_train` — both are loop-opening.

### The gap (VERIFIED)

`benchmarks/scripts/run_experiment.py:151–172` [VERIFIED] — `model_cfg` is built from `registries.model_registry.get(args.model, ...)` and CLI args. `automil/config.yaml` is NOT read. The `model.variant` key written by `apply` is never consulted. `ExperimentConfig` is constructed directly from argparse values.

### Where the translation layer lives (D-03)

The translation layer belongs in **autobench**, specifically in `benchmarks/src/autobench/pipeline/clam/train.py`. The planner should add a function `_apply_variant_to_exp_cfg(exp_cfg, automil_config_path)` that:

1. Reads `automil/config.yaml` for `model.variant`, `loss.variant`, `policy.variant`
2. Looks up the registered variant by name in `MODEL_VARIANTS` (after scanning the variants directory with `scan_variants`)
3. Reads the variant's `VariantSpec` from `SPEC_STORE` to get its declared fields
4. Maps those fields onto `ExperimentConfig.model` (a mutable `ModelConfig` dataclass)
5. Returns the patched `ExperimentConfig`

**Important:** The consumer side must call `scan_variants(variants_root)` first so `@register` decorators fire and `MODEL_VARIANTS` is populated. This is exactly what `scanner.scan_variants` does (`scanner.py:70–117`).

**Where to call it:** In `run_experiment.py` after `exp_cfg` is built (line 163) and before the `run_experiment` call (line 201). Or alternatively, inside `train_fold` in `train.py` — but `run_experiment.py` is cleaner because it has access to both `automil_config_path` and `exp_cfg`.

**Alternative: patch `_make_clam_args` directly.** If the variant specifies overrides as a dict (e.g., `{"model_type": "clam_sb", "dropout": 0.5}`), the patching can happen inside `_make_clam_args` by reading an optional `variant_overrides` attribute on `exp_cfg`. This is cleaner because it keeps the seam co-located with the args construction. **Recommendation for planner:** patch `ExperimentConfig.model` fields before `_make_clam_args` is called — this is the most transparent approach and leaves `_make_clam_args` unmodified.

**VariantSpec does NOT carry `model_type`/`dropout` fields.** `VariantSpec` (`spec.py`) carries only: `name`, `kind`, `parent`, `base_commit`, `composite`, `node_id`, `created_at`, `mutations`. The variant class itself (a `ModelVariant` subclass) must declare what it overrides. **Planner decision required:** how do registered model variants communicate the CLAM args they override? Options:
- **Option A (recommended):** A class-level dict attribute `CLAM_ARGS: dict[str, Any]` on the `ModelVariant` subclass specifying field overrides. The translation layer reads `cls.CLAM_ARGS` and applies to `ModelConfig`.
- **Option B:** A `@classmethod` `clam_overrides(cls) -> dict`. Same semantics, more explicit.
- **Option C:** The variant module exports a standalone `CLAM_ARGS` dict (not class-level). Simpler for iris-style variants; works without subclassing `ModelVariant`.

Option A is recommended because it keeps the override specification on the class (which is what the registry stores) and is consistent with the `ModelVariant` ABC pattern already in place.

### Automated test strategy (D-04, CI-safe)

The test does NOT need `AUTOBENCH_CCRCC_ROOT`. It:
1. Constructs a minimal `ExperimentConfig` with baseline `ModelConfig`
2. Registers a dummy model variant with `CLAM_ARGS = {"model_size": "big", "dropout": 0.5}`
3. Calls `_apply_variant_to_exp_cfg(exp_cfg, fake_config_path)`
4. Asserts `exp_cfg.model.model_size == "big"` and `exp_cfg.model.dropout == 0.5`
5. Calls `_make_clam_args(exp_cfg, "/tmp")` and asserts `args.model_size == "big"` and `args.drop_out == 0.5`

No CLAM imports needed. `_make_clam_args` only uses `SimpleNamespace` and reads `exp_cfg` fields — fully testable without GPU or data.

**Framework purity (D-206):** The `_apply_variant_to_exp_cfg` function will live in `benchmarks/src/autobench/pipeline/clam/train.py` or a new `benchmarks/src/autobench/pipeline/variant_dispatch.py`. It will import from `automil.registry` (registrar, scanner, _state). This is autobench importing automil — that direction is allowed. The purity gate (`test_framework_purity.py`) only blocks autobench references *inside* `src/automil/`. No allowlist update needed.

---

## APL-03 Deep-Dive: Loud Detection of Loop-Opening Variants

### The classification boundary (VERIFIED)

The `train_loop_clam` function at `core_utils.py:225` [VERIFIED]:
```python
def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    ...
    loss = loss_fn(logits, label)           # line 241 — bag-level loss
    instance_loss = instance_dict['instance_loss']  # line 244 — from model internals
    total_loss = bag_weight * loss + (1-bag_weight) * instance_loss  # line 249
```

The `loss_fn` at line 241 receives only `(logits, label)` — bag-level logits and integer label. **`LossVariant.__call__` signature** at `variants/loss.py:20–39` [VERIFIED] requires: `(self, logits, targets, *, instance_logits=None, instance_labels=None)`. The `instance_logits` and `instance_labels` are NOT available at the `train_loop_clam` call-site — they come from `instance_dict` which is computed inside the model forward pass and stored in `train_loop_clam`'s local scope (line 244). No path exists from `_make_clam_args` → `clam_train` → `train_loop_clam` to deliver these to a custom loss callable without modifying `train_loop_clam` internals.

**Classification rules:**

| Variant kind | Route | Expressible through seam? |
|---|---|---|
| `model` | `ModelConfig.model_type/model_size/dropout/B/bag_weight` → `_make_clam_args` | YES — apply |
| `loss` (string selector) | `args.bag_loss = "ce"/"svm"` or `args.inst_loss` | YES — apply (string overrides only) |
| `loss` (custom callable) | Requires `loss_fn = LossVariant()` inside `train_loop_clam` | NO — loop-opening |
| `policy` | N/A — no policy injection point in CLAM's closed seam | Depends on policy type |

**The classification function** (`_classify_variant_route`):

```python
# In src/automil/cli/lifecycle/apply.py (or a shared validator module)
def _classify_variant_route(selection: dict, variants_root: Path) -> None:
    """Raise ClickException if any selected variant requires loop opening.

    Classification: a loss variant is loop-opening iff it is a custom callable
    (subclass of LossVariant) rather than a string selector expressible through
    args.bag_loss / args.inst_loss. Model and policy variants are seam-expressible
    iff they declare only CLAM_ARGS-compatible field overrides.
    """
    loss_name = selection.get("loss", {}).get("variant")
    if loss_name is not None:
        # Scan to see if it is a registered LossVariant subclass
        scan_variants(variants_root)  # populates LOSS_VARIANTS
        if loss_name in LOSS_VARIANTS:
            # It is a custom LossVariant callable — requires loop opening
            raise click.ClickException(
                f"Loss variant '{loss_name}' is a custom LossVariant callable. "
                f"Applying it requires injecting into CLAM's train_loop_clam, "
                f"which is a closed training loop. "
                f"This is deferred (ISSUE-007 / RTA). "
                f"To apply it, wait for the RTA milestone that opens the loop. "
                f"This variant has NOT been applied and no run has been started."
            )
```

**Where this fires:** Inside `apply()` at `apply.py`, AFTER `_derive_variant_selection` and BEFORE writing `config.yaml`. This ensures the error is raised before any state mutation.

**Important scoping note:** The scanner (`scan_variants`) must be called against the correct `variants_root` for the consumer. `apply.py` has access to `adir` (the `automil/` directory) from which `variants_root = adir / "variants"` can be derived. This is already the pattern used by `refresh-registry`.

**Framework purity of the classifier:** If `_classify_variant_route` calls `scan_variants` and checks `LOSS_VARIANTS`, it imports `automil.registry.scanner` and `automil.registry._state` — both in `src/automil/`. No autobench reference. Framework-pure.

**Risk:** The current `LOSS_VARIANTS` dict in `_state.py` has no concrete registered subclasses at time of research [VERIFIED: observation 1299]. If no loss variant is registered, `LOSS_VARIANTS` is empty and the check trivially passes. The planner should note that APL-03 only fires when a loss variant IS registered; for now (no registered loss variants), the test must register one explicitly.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Variant module discovery | Custom glob+import loop | `automil.registry.scanner.scan_variants` (already exists, handles errors, deduplication, skips `_private` files) |
| Variant class lookup | dict walk | `automil.registry.registrar.resolve_model/resolve_loss` (already exists, gives clean KeyError messages) |
| Atomic config writes | `open("config.yaml", "w")` | `_atomic_write_text` from `cli/lifecycle/_shared.py` (already used in `apply.py`) |
| Variant field → args mapping | Reflection/introspection gymnastics | Class-level `CLAM_ARGS` dict on `ModelVariant` subclass; read with `getattr(cls, "CLAM_ARGS", {})` |

---

## Common Pitfalls

### Pitfall 1: Scanning variants BEFORE the variants_root is on sys.path

**What goes wrong:** `scan_variants` uses `importlib.util.spec_from_file_location` with a unique module name — no `sys.path` manipulation needed. BUT if a variant module imports from a sibling (e.g., `from . import helpers`), relative imports fail unless the package's parent directory is on `sys.path`. `scanner._import_path` registers the module in `sys.modules` before `exec_module`, which covers the relative import case for direct siblings.
**How to avoid:** Variant modules should not use relative imports to non-variant code. The iris `logistic_v0.py` imports only from `sklearn` — no issue.

### Pitfall 2: `_clear_registry()` required between tests

**What goes wrong:** `MODEL_VARIANTS`, `LOSS_VARIANTS`, `SPEC_STORE` are module-level singletons (`_state.py`). Tests that register variants will pollute subsequent tests.
**How to avoid:** Every test that calls `@register` or `scan_variants` must call `_clear_registry()` in a `finally` block or fixture teardown. The existing registry tests already do this — new APL tests must follow the same pattern.

### Pitfall 3: `apply` writes config.yaml BEFORE classification fires

**What goes wrong:** If `_classify_variant_route` is called AFTER `_atomic_write_text(config_path, new_text)`, a loop-opening variant's name has already been written to config.yaml. The next `automil submit` may pick it up silently.
**How to avoid:** The route classifier MUST fire before any config mutation. In `apply.py`, the order must be: derive selection → classify route (raise if loop-opening) → backup → write. [VERIFIED: the current `apply.py:101–136` order is derive → hard-fail for empty → backup → write. Insert classification between lines 113 and 134.]

### Pitfall 4: Iris dispatch breaks the "no automil.* imports" consumer contract

**What goes wrong:** The iris `train.py` docstring explicitly states "No automil.* imports (consumer-decoupled)." If the dispatch uses `from automil.registry.scanner import scan_variants`, it violates this contract and introduces a hard dependency on the framework being installed.
**How to avoid:** Use raw `importlib.util.spec_from_file_location` for the dispatch — same logic as the scanner but inlined in `train.py`. The variant convention (a callable named `make_classifier`) is documented in the consumer's README, not enforced by the framework. [VERIFIED: `train.py:8` states the no-import constraint; scanner logic at `scanner.py:46–67` can be replicated in ~10 lines inline.]

### Pitfall 5: `bag_loss` string selector vs. `LossVariant` callable — conflation

**What goes wrong:** The CLAM seam CAN change the loss function via `args.bag_loss = "svm"` (a string selector that CLAM handles inside `train_loop_clam`). A model variant's `CLAM_ARGS` dict could legitimately include `{"bag_loss": "svm"}`. This is seam-expressible and must NOT be flagged as loop-opening. Only a variant that is a registered `LossVariant` *subclass* (in `LOSS_VARIANTS`) requires loop-opening.
**How to avoid:** The classification check must distinguish: string bag_loss override (seam-expressible, allowed) vs. `LossVariant` subclass (requires loop injection, blocked). The classifier checks `LOSS_VARIANTS[loss_name]` — if found, it is a custom callable; if not found, it may be a string selector in a model variant's `CLAM_ARGS`.

### Pitfall 6: `ExperimentConfig.model` is a mutable dataclass — safe to patch

`ModelConfig` is a plain `@dataclass` (not frozen) at `config.py:58`. Field assignment is safe. No need for a copy or factory pattern. [VERIFIED: `config.py:58` — no `frozen=True`.]

### Pitfall 7: Lessons.md 2026-06-11 verification trap

**What goes wrong:** All tests pass but the dispatch never fires in production because the variant name is read from config.yaml by the test (which injects the key) but by `run_experiment.py` in production from CLI args (which never reads config.yaml for `model.variant`).
**How to avoid:** Following the 2026-06-11 lessons.md rule: verify the component supplies its OWN dependencies in the real entry path. The test for APL-02 must exercise `run_experiment.py`'s real path (or the `_apply_variant_to_exp_cfg` call within it) not just `_make_clam_args` in isolation. At minimum, a test that calls the full `_apply_variant_to_exp_cfg` → `_make_clam_args` chain with a real stubbed variant proves the wiring.

---

## Code Examples

### APL-01: Iris dispatch (minimal, no framework imports)
```python
# Source: direct derivation from scanner.py:46-67 + apply.py:114-115
# In examples/sklearn-iris/train.py, after reading config:
variant_name = (config.get("model") or {}).get("variant")
clf = None
if variant_name:
    import importlib.util as _ilu
    _variants_dir = config_path.parent / "variants" / variant_name
    _py_files = sorted(
        f for f in _variants_dir.glob("*.py")
        if not f.name.startswith("_") and f.name != "__init__.py"
    )
    if _py_files:
        _spec = _ilu.spec_from_file_location(variant_name, _py_files[0])
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        clf = _mod.make_classifier(seed=seed)
if clf is None:
    clf = LogisticRegression(max_iter=200, random_state=seed)
clf.fit(X_train, y_train)
```

### APL-02: Class-level CLAM_ARGS on a ModelVariant subclass
```python
# Source: derived from ModelVariant ABC (src/automil/registry/variants/model.py)
# In a hypothetical benchmarks/experiments/<dataset>/automil/variants/clam_mb/clam_mb_v0001.py:
from automil.registry.registrar import register
from automil.registry.spec import VariantSpec
from automil.registry.variants.model import ModelVariant

@register(VariantSpec(
    name="clam_mb_v0001", kind="model", parent="clam_mb",
    base_commit="abc123", composite=0.87, node_id="node_0001",
    created_at="2026-06-11T00:00:00+00:00",
))
class ClamMbV0001(ModelVariant):
    # Fields expressible through _make_clam_args seam:
    CLAM_ARGS: dict = {
        "model_size": "big",
        "dropout": 0.5,
        "bag_weight": 0.8,
        "B": 16,
    }

    def forward(self, features, coords=None):
        raise NotImplementedError("CLAM uses its own model internals; forward() unused")
```

### APL-02: Translation layer in autobench
```python
# Source: derived from registrar.py + _state.py + config.py
# In benchmarks/src/autobench/pipeline/variant_dispatch.py:
import yaml
from pathlib import Path
from automil.registry.scanner import scan_variants
from automil.registry._state import MODEL_VARIANTS

def apply_model_variant_to_exp_cfg(exp_cfg, automil_dir: Path) -> None:
    """Mutate exp_cfg.model in-place from automil/config.yaml model.variant.
    No-op if no variant is selected.
    """
    config_path = automil_dir / "config.yaml"
    if not config_path.exists():
        return
    config = yaml.safe_load(config_path.read_text()) or {}
    variant_name = (config.get("model") or {}).get("variant")
    parent_name  = (config.get("model") or {}).get("parent")
    if not variant_name:
        return

    variants_root = automil_dir / "variants"
    scan_variants(variants_root)  # populates MODEL_VARIANTS via @register

    key = (parent_name, variant_name)
    variant_cls = MODEL_VARIANTS.get(key)
    if variant_cls is None:
        raise ValueError(
            f"Variant '{variant_name}' (parent='{parent_name}') not found in registry "
            f"after scanning {variants_root}. Run `automil refresh-registry`."
        )

    clam_args = getattr(variant_cls, "CLAM_ARGS", {})
    for field, value in clam_args.items():
        if hasattr(exp_cfg.model, field):
            setattr(exp_cfg.model, field, value)
        elif hasattr(exp_cfg.train, field):
            setattr(exp_cfg.train, field, value)
```

### APL-03: Route classifier in apply.py
```python
# Source: derived from apply.py + _state.py + loss.py
# In src/automil/cli/lifecycle/apply.py, before config mutation:
def _classify_variant_route(
    selection: dict,
    variants_root: Path,
) -> None:
    """Raise ClickException if any selected variant requires loop opening."""
    from automil.registry.scanner import scan_variants
    from automil.registry._state import LOSS_VARIANTS

    loss_name = (selection.get("loss") or {}).get("variant")
    if loss_name is not None:
        scan_variants(variants_root)
        if loss_name in LOSS_VARIANTS:
            raise click.ClickException(
                f"Loss variant '{loss_name}' is a custom LossVariant callable that "
                f"requires injecting into a closed MIL training loop "
                f"(ISSUE-007 / RTA). It cannot be applied through the open "
                f"_make_clam_args seam. This variant has NOT been applied. "
                f"Deferred to the RTA milestone."
            )
```

---

## Runtime State Inventory

This is not a rename/refactor phase. The only runtime state relevant to Phase 10 is the `automil/config.yaml` file that `apply` already modifies atomically. No data migration required.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pyproject.toml` (workspace root) |
| Quick run command | `uv run pytest tests/test_apl01_iris_dispatch.py tests/test_apl03_loud_fail.py -v` |
| Full suite command | `uv run pytest tests/ benchmarks/tests/ -v` |
| Autobench test run | `uv run pytest benchmarks/tests/test_variant_dispatch_clam.py -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| APL-01 | iris `train.py` instantiates `classifier_v0` when `model.variant=classifier_v0` | integration (subprocess or direct call) | `uv run pytest tests/test_apl01_iris_dispatch.py -v` | No external data; fully CI-gated |
| APL-01 | iris baseline still works when no variant is set | unit | same file | Regression guard |
| APL-02 | `_apply_variant_to_exp_cfg` patches `ModelConfig` from registry | unit | `uv run pytest benchmarks/tests/test_variant_dispatch_clam.py -v` | No GPU, no data |
| APL-02 | variant fields flow through `_make_clam_args` into args namespace | unit (spy on args output) | same file | CI-gated; stubs `clam_train` |
| APL-02 | real CLAM composite differs from baseline | workstation-only | `AUTOBENCH_CCRCC_ROOT=... uv run pytest benchmarks/tests/test_apl02_real_run.py -v` | Marked `@pytest.mark.workstation` |
| APL-03 | registered `LossVariant` raises `ClickException` at `apply` time | unit | `uv run pytest tests/test_apl03_loud_fail.py -v` | CI-gated |
| APL-03 | no-op model/string-selector variants do NOT raise | unit | same file | Regression guard |
| APL-03 | error raised BEFORE `config.yaml` is mutated | unit | same file | Verifies ordering |
| APL-01 | existing `test_lifecycle_apply.py` (14 tests) still passes | regression | `uv run pytest tests/test_lifecycle_apply.py -v` | No changes to apply.py API |

### Wave 0 Gaps

- [ ] `tests/test_apl01_iris_dispatch.py` — covers APL-01 (iris end-to-end dispatch)
- [ ] `tests/test_apl03_loud_fail.py` — covers APL-03 (loud fail + ordering)
- [ ] `benchmarks/tests/test_variant_dispatch_clam.py` — covers APL-02 (arg-threading via stub)
- [ ] `benchmarks/src/autobench/pipeline/variant_dispatch.py` — the translation layer module (APL-02)
- [ ] Class-level `CLAM_ARGS` convention on `ModelVariant` — must be documented (or added as an optional `@classmethod` hook on the ABC for discoverability)

---

## Security Domain

| ASVS Category | Applies | Standard Control |
|---|---|---|
| V5 Input Validation | yes | Variant name is a string from config.yaml; only used for dict lookup and `Path` construction — validate it does not contain `..` or `/` (path traversal into variant files) |
| V4 Access Control | no | No auth surface |
| V2 Authentication | no | No auth surface |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `model.variant` value containing `../` | Tampering | Validate `variant_name` contains no `/` or `..` before constructing `_variants_dir / variant_name`; use `Path(variant_name).name == variant_name` check |
| Arbitrary code execution via a malicious variant module | Tampering | Variant modules are committed to git (registry-first invariant) — not user-supplied at runtime. Low risk for the current threat model. |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| Manual model config edits in train.py | `automil apply` writes `config.yaml`; consumer reads and dispatches | This phase | Variants become reproducible and attributable |
| Inert variant registration | Variant application through open seams, loud-fail for loop-opening | This phase | Registry-first thesis delivered |

---

## Open Questions

1. **`CLAM_ARGS` convention on `ModelVariant`**
   - What we know: `ModelVariant.forward()` is abstract and unused by CLAM (CLAM uses its own internal model classes). The ABC is the type check for `@register` but has no field-override mechanism.
   - What's unclear: Should `CLAM_ARGS` be a required class attribute (enforced in `__init_subclass__`), an optional one (checked with `getattr`), or a `@classmethod` override? A required attribute breaks existing zero-variant registrations.
   - Recommendation: Optional class attribute `CLAM_ARGS: dict = {}` via `getattr(cls, "CLAM_ARGS", {})`. Existing `ModelVariant` subclasses (currently none in the registry) are unaffected. Document in the variant-authoring guide.

2. **`automil/config.yaml` availability at run time in worktrees**
   - What we know: `apply` writes `automil/config.yaml` in the project root. The orchestrator's `apply_overlay` copies overlay files from `automil/orchestrator/archive/<node_id>/` into the worktree — NOT from `automil/config.yaml`. The training script runs in the worktree root.
   - What's unclear: Does `automil/config.yaml` end up in the worktree? The overlay model copies only files in `automil/orchestrator/archive/<node_id>/`. The base commit checkout (`git worktree add --detach`) provides whatever `automil/config.yaml` was committed to git — but runtime config.yaml is gitignored.
   - Risk: If `config.yaml` is gitignored and not in the overlay, the variant dispatch in `train.py` silently falls back to baseline (variant_name is None), making APL-01/APL-02 inert in real runs.
   - Recommendation: **The planner must verify whether `automil/config.yaml` is in `files.editable` (making it auto-included in the overlay) or needs to be explicitly submitted.** Check `examples/sklearn-iris/automil/config.yaml:39` — `files.editable` lists only `automil/variants/`. Config.yaml is NOT auto-included. The fix: either add `automil/config.yaml` to `files.editable`, or snapshot the variant selection into the spec (env var set by orchestrator, read by train.py). **This is the highest-risk gap for the planner to resolve before implementation.**

3. **Policy variants for APL-03**
   - What we know: `POLICY_VARIANTS` is registered like `LOSS_VARIANTS`. The current APL-03 scope focuses on `LossVariant` (the concrete ISSUE-007 case).
   - What's unclear: Are policy variants (e.g., SAM optimizer) expressible through the `opt=` field in `_make_clam_args`? If SAM is not a string selector that `get_optim` recognizes, it also requires loop opening.
   - Recommendation: APL-03 classification should also check `POLICY_VARIANTS` for the same reason. If a `PolicyVariant` is not a string selector expressible through `args.opt`, classify as loop-opening. Extend the classifier to handle policy variants symmetrically with loss variants.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `automil/config.yaml` written by `apply` reaches the worktree's `train.py` at runtime | Open Questions #2 | APL-01/APL-02 would be inert in real orchestrated runs even after the fix |
| A2 | `ModelConfig` fields are sufficient to express all model variants through `_make_clam_args` (no variant needs args not listed in the seam table) | APL-02 deep-dive | Variant that needs e.g. `embed_dim` or `n_classes` cannot be expressed — would need APL-03 classification |
| A3 | No existing CLAM consumer variant modules exist that define `CLAM_ARGS` | Based on zero registered variants in `MODEL_VARIANTS` | If any exist, the `CLAM_ARGS` convention must be retroactively documented/enforced |

---

## Sources

### Primary (HIGH confidence — direct source reads)
- `src/automil/cli/lifecycle/apply.py` — apply command full implementation (lines 1–144)
- `examples/sklearn-iris/train.py` — iris consumer, inert bug confirmed (lines 51–76)
- `examples/sklearn-iris/automil/variants/classifier_v0/logistic_v0.py` — variant module (lines 1–12)
- `examples/sklearn-iris/automil/config.yaml` — consumer config, files.editable (all lines)
- `benchmarks/src/autobench/pipeline/clam/train.py` — `_make_clam_args` seam (lines 62–92)
- `benchmarks/lib/CLAM/utils/core_utils.py` — `train_loop_clam` closed loop (lines 225–276)
- `benchmarks/src/autobench/pipeline/clam/_imports.py` — `clam_train` alias (lines 38–46)
- `benchmarks/src/autobench/pipeline/clam/runner.py` — `run_experiment` fold loop (lines 73–129)
- `benchmarks/scripts/run_experiment.py` — config NOT read for variant (lines 151–172)
- `src/automil/registry/_state.py` — `MODEL_VARIANTS`, `LOSS_VARIANTS`, `SPEC_STORE` singletons
- `src/automil/registry/spec.py` — `VariantSpec` fields (all lines)
- `src/automil/registry/registrar.py` — `@register`, `resolve_model/loss/policy` (all lines)
- `src/automil/registry/scanner.py` — `scan_variants`, `_import_path` (all lines)
- `src/automil/registry/variants/loss.py` — `LossVariant.__call__` signature (lines 11–39)
- `src/automil/registry/variants/model.py` — `ModelVariant` ABC (all lines)
- `benchmarks/src/autobench/pipeline/config.py` — `ModelConfig`, `ExperimentConfig` (all lines)
- `tests/test_lifecycle_apply.py` — 14 existing apply tests (all lines)
- `tests/test_framework_purity.py` — purity gate allowlist (all lines)
- `.planning/config.json` — `nyquist_validation: true` confirmed

### Secondary (MEDIUM confidence — planning docs)
- `.planning/phases/10-variant-application-integrity/10-CONTEXT.md` — locked decisions D-01..D-05
- `.planning/REQUIREMENTS.md` — APL-01/02/03 text; RTA + Out-of-Scope
- `tasks/lessons.md` — 2026-06-10 and 2026-06-11 entries
- `tasks/test-run-issues.md` — ISSUE-007 detail

---

## Metadata

**Confidence breakdown:**
- APL-01 iris fix: HIGH — bug confirmed at file:line, fix is minimal and contained
- APL-02 arg-threading: HIGH — seam fully mapped; open question on config.yaml propagation (A1) must be resolved by planner before execution
- APL-03 loud detection: HIGH — `train_loop_clam` call-site confirmed; classification rule is unambiguous for `LossVariant` subclasses
- Config.yaml → worktree propagation: LOW — not verified at runtime; marked as highest-risk open question

**Research date:** 2026-06-11
**Valid until:** 2026-07-11 (stable codebase — no fast-moving external dependencies)
