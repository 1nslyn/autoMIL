# autoMIL training-script contract

A training script is autoMIL-compatible when it honors the 6 contract items
below. The framework treats it as an opaque process; the contract is the
seam between framework and consumer.

Any language, any ML library: if the process writes a conforming `result.json`
and exits cleanly, the experiment tree records it correctly. The framework
never reads model weights, loss curves, or intermediate checkpoints.

## The contract

A training script must:

1. **Read `automil/config.yaml`** (or honor a `--config` flag if exposed).
   The orchestrator runs the script from the experiment's working directory
   (a git worktree); `automil/config.yaml` is reachable at the relative path.
   Consumer configuration (hyperparameters, dataset paths, scoring formula
   documentation) all lives here. The framework does not inject config values
   as command-line args.

2. **Honor the framework-owned device masks.** Never override or unset
   `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, or
   `GPU_DEVICE_ORDINAL`. CUDA receives its physical ID in
   `CUDA_VISIBLE_DEVICES`. On ROCm/Linux, `ROCR_VISIBLE_DEVICES` selects the
   physical host GPU and the HIP/CUDA compatibility masks expose it as logical
   device 0. CPU execution receives empty masks.

3. **Branch on `AUTOMIL_ACCELERATOR`.** Its value is `cpu`, `cuda`, or `rocm`.
   `AUTOMIL_GPU=0` remains a backward-compatible logical slot, including on
   CPU, and is not evidence that a physical GPU exists. Accelerator consumers
   use logical device 0 after masking (`torch.device("cuda:0")` works for both
   PyTorch CUDA and ROCm builds); CPU consumers ignore `AUTOMIL_GPU`.

4. **Exit cleanly on `SIGTERM`** with partial output written to result.json.
   The orchestrator sends SIGTERM at the cap boundary (Phase 4 / D-115);
   scripts that ignore it lose work and corrupt the cell budget bookkeeping.
   Write a result.json with `"partial": true` and `"status": "budget_killed"`
   before exiting. See the SIGTERM handling section below.

5. **Write `result.json`** to the working directory matching
   `automil/schemas/result.schema.json`. The framework validates this at
   ingestion; malformed payloads transition the node to `crashed` with
   the schema location in the error message. The minimum valid payload is
   `{"composite": <float>}`. All other fields are optional.

6. **Declared env vars are present at startup.** The framework's
   `automil check` validates `automil/config.yaml: env.required` BEFORE
   submit; missing vars fail with a named error rather than crashing the
   training script deep in execution. If your script reads `MY_DATASET_ROOT`,
   declare it under `env.required` in config.yaml and `automil check` catches
   an absent value before the experiment is even submitted.

## Minimal sklearn-iris example

The shipped reference is `examples/sklearn-iris/train.py` (~75 lines). It
demonstrates contract items 1, 2, 4, 5; item 3 selects CPU and item 6 is empty.
Read it as the executable spec.

The script follows this structure:

- install SIGTERM handler (Pattern B below)
- load iris dataset
- train LogisticRegression
- compute accuracy and F1
- write `result.json` with `{"composite": accuracy, "metrics": {...}, "status": "completed"}`

To run it end-to-end: `uv run automil submit --node iris_001 --files examples/sklearn-iris/train.py`.

## Minimal pytorch skeleton

A pytorch consumer adds GPU mask handling; the rest mirrors sklearn-iris.

```python
import json, os, signal, sys
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_state = {"completed": False, "loss": 0.0}

def _write_result(*, status: str, partial: bool):
    payload = {"status": status, "composite": -_state["loss"], "partial": partial}
    open("result.json", "w").write(json.dumps(payload))

def _on_sigterm(signum, frame):
    _write_result(status="budget_killed" if not _state["completed"] else "completed",
                  partial=not _state["completed"])
    sys.exit(0)

signal.signal(signal.SIGTERM, _on_sigterm)
# ... train loop ...
_state["completed"] = True
_write_result(status="completed", partial=False)
```

Key points for pytorch consumers:

- Branch on `AUTOMIL_ACCELERATOR`; for either CUDA or ROCm,
  `device = torch.device("cuda:0")` addresses the single masked accelerator.
  Never infer accelerator presence from `AUTOMIL_GPU`, which is also `0` on CPU.
- `_state["loss"]` updates at each epoch; SIGTERM during training writes the best-so-far loss.
- `sys.exit(0)` from the SIGTERM handler signals a graceful flush; the daemon distinguishes
  exit code 0 from crash (non-zero) and from silent process death.
- Move training loop logic between `signal.signal(...)` and `_write_result(status="completed")`.

## SIGTERM handling

Two patterns. Pick by fold count.

### Pattern A: multi-fold via `automil.runtime_helpers.register_sigterm_flush`

For consumers that train multiple folds and write `fold_<i>_result.json`
per fold, the framework provides an aggregator. Call
`register_sigterm_flush()` once at startup; the helper installs a SIGTERM
handler that aggregates completed-fold files into result.json.

```python
from automil.runtime_helpers import register_sigterm_flush

def main():
    register_sigterm_flush()   # must be called in main thread, before DataLoader init
    for fold_i in range(get_fold_count()):
        # ... train fold ...
        # write fold_{i}_result.json
```

The handler (in `src/automil/runtime_helpers.py`) calls `aggregate_folds()` from
`automil.cells.reconcile`, merges completed folds into a single result.json with
`"partial": true`, and calls `sys.exit(0)`. The orchestrator daemon then records
`status: executed` with `metadata.budget_killed = True`.

Constraint: call `register_sigterm_flush()` in the main thread, before creating
any `DataLoader` or `threading.Thread`. Python's `signal.signal()` raises
`ValueError` if called from a non-main thread.

### Pattern B: single-shot via inline `signal.signal`

For single-shot consumers (no fold structure), install your own handler.
The handler closes over a `_state` dict updated as training progresses.
Idempotent: a late SIGTERM after `_state["completed"] = True` writes
`status: completed` instead of `status: budget_killed`.

`examples/sklearn-iris/train.py` uses Pattern B.

### SIGTERM handler rules

Regardless of pattern, the handler must:

1. Write `result.json` FIRST (before any cleanup).
2. Exit via `sys.exit(0)` NOT `sys.exit(130)`. Exit code 0 tells the daemon
   the process completed gracefully; code 130 (or non-zero) is treated as a crash.
3. Be idempotent: if `result.json` already exists (normal completion before SIGTERM),
   the handler should either skip the write or overwrite with the same payload.

## Result.json schema

The contract is `automil/schemas/result.schema.json` (Draft 2020-12).
Required: `composite` (number). Optional: `metrics` (dict of
str -> number; validation-only under the val-firewall), `held_out` (dict of
str -> number; sealed test metrics, quarantined from search and revealed only
via `automil certify`), `status` (one of completed, crash, budget_killed,
cancelled), `elapsed_seconds`, `peak_vram_mb`, `fold_results`, `partial`.
`additionalProperties: true` means consumers may extend.

Minimum valid payload:

```json
{"composite": 0.912}
```

Full payload example (autobench / CCRCC consumer):

```json
{
  "composite": 0.845,
  "metrics": {
    "val_auc": 0.87,
    "val_bacc": 0.81
  },
  "held_out": {
    "test_auc": 0.87,
    "test_bacc": 0.83
  },
  "status": "completed",
  "elapsed_seconds": 4098,
  "peak_vram_mb": 4500
}
```

The framework validates result.json at ingestion via `jsonschema.validate(...)`.
Malformed payloads transition the node to `crashed` with an error that
references this schema's location. Schema path in the error:
`see automil/schemas/result.schema.json`.

The `composite` field is the single scalar used by the experiment tree for
ranking and keep/discard (UCB scoring, Ladder margin). It is the
**validation** selection signal, never test. The framework trusts it as an
opaque number and never recomputes or inspects it; test metrics belong in
`held_out` instead, sealed away from search and revealed once via
`automil certify`. Higher is always better. For loss minimization, negate:
`"composite": -val_loss`.

## Required env vars

Declare vars under `automil/config.yaml: env.required`:

```yaml
env:
  required:
    - MY_DATASET_ROOT
    - HF_HOME       # optional cache; only declare if your script reads it
  passthrough:
    - MY_DATASET_ROOT
    - HF_HOME
```

`automil check` validates `env.required` at startup; any missing var fails
with `Missing required env var: <name>; see automil/config.yaml: env.required`.

`env.passthrough` controls forwarding from the orchestrator process into
experiment subprocesses. Use this list to opt in consumer-specific vars
(formerly auto-injected `AUTOBENCH_ROOT` is now consumer-declared via
this list per Phase 8 / DEC-01).

For CPU-only consumers with no external data dependencies (e.g. sklearn-iris
where the dataset is bundled), both lists stay empty:

```yaml
env:
  required: []
  passthrough: []
```

Running `automil check` before submitting experiments verifies all required
vars are set. Integrate it into your workflow setup script or CI environment
validation to catch missing env vars before GPU time is wasted.

## Common pitfalls

1. **Writing result.json AFTER cleanup.** If your training loop closes
   files, releases CUDA memory, then writes result.json, a SIGTERM during
   cleanup loses the partial result. Always write FIRST, then clean up.
   Pattern: `_write_result()` at the top of the SIGTERM handler; cleanup
   (closing file handles, del model, torch.cuda.empty_cache()) only after
   the write returns.

2. **`sys.exit(0)` without writing partial.** A SIGTERM handler that
   exits before the writer fires produces an empty archive directory; the
   orchestrator sees `result.json` missing and synthesises a crash status.
   The handler must call the writer explicitly. Common mistake: reusing a
   `KeyboardInterrupt` handler that just calls `sys.exit(0)` without
   adapting it for SIGTERM semantics.

Both pitfalls have the same fix: install the SIGTERM handler early (top of
main), before any resource acquisition, and always write result.json from
inside the handler before exiting.

## See also

- `examples/sklearn-iris/train.py`: shipped reference (DEC-02).
- `src/automil/schemas/result.schema.json`: result.json contract.
- `src/automil/cli/check.py`: `automil check` env.required validator.
- `src/automil/runtime_helpers.py`: multi-fold SIGTERM helper.
- `docs/getting-started.md`: project initialisation and first submit.
