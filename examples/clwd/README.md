# CLWD Example

Lung adenocarcinoma subtype classification (7-class) from the CLWD dataset.
408 WSIs with WHO histological subtype labels.

This example shows only the `automil/` subdirectory. The full project
would also contain the model code, data loading, and a training script
that honors the [training-script contract](../../docs/training-script-contract.md).

## Status

Skeleton, config and program.md only. No autonomous run results shipped
with this example; it serves as a v1.0 multi-class reference layout.

## Running this

```bash
cd examples/clwd
# Edit automil/config.yaml: set data paths, env.required, scoring.formula
uv run automil check
uv run automil orchestrator start

# In another tmux session
claude --dangerously-skip-permissions
# Then type: /automil-setup (first time) then /automil
```

For a minimal end-to-end reference with no real dataset dependency, see
[`examples/sklearn-iris/`](../sklearn-iris/).
