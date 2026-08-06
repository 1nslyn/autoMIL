# Ovarian HRD Example

Pre-v1.0 reference run: HRD status prediction from ovarian cancer WSIs.
Binary classification using CLAM-MB with H-optimus-1 encoder features.

## What's here

This example shows only the `automil/` subdirectory that autoMIL adds to
an existing project. The full project would also contain:

- Model code (CLAM-MB architecture, attention pooling)
- Data loading (slide_id → bag-of-features mapping)
- Feature extraction pipeline (TRIDENT + H-optimus-1)
- Training script honoring the [training-script contract](../../docs/training-script-contract.md)

## Results (pre-v1.0 autonomous run)

- **189 experiments** executed autonomously
- **Best composite: 0.851** (from 0.814 baseline, +4.5%)
- Discovered techniques: R-Drop, focal loss, gradient clipping, coordinate
  positional encoding, none of which the human researchers had tried
- See `automil/graph.json` for the full experiment tree
- See `automil/learnings.md` for the accumulated insights

## Running this with v1.0

To run a similar campaign on this example with the v1.0 framework:

```bash
cd examples/ovarian_hrd
uv run automil init --runtime claude # or your runtime; uses the existing config.yaml
uv run automil check                 # verifies env.required, registry, backend
uv run automil orchestrator start

# In another tmux session
claude --dangerously-skip-permissions
# Then type: /automil
```

The same pattern works for any consumer that honors the contract; see
[`examples/sklearn-iris/`](../sklearn-iris/) for a minimal non-pathology
reference (~80 LOC, no `automil.*` imports).
