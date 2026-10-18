# Placeholder Example

Minimal project layout to start from. Copy and edit
`automil/config.yaml` for your project.

## Quick start from this template

```bash
cp -r examples/placeholder/automil /path/to/your/project/
cd /path/to/your/project

# Edit automil/config.yaml: at minimum, set
#   run.script
#   files.editable / files.readonly
#   env.required + env.passthrough
#   scoring.formula
#   cap.budget + cap.safety_buffer (durations such as 6h and 30m)
#   baseline.primary_value

uv run automil check                # validates protected files, env.required, backend
uv run automil orchestrator start
```

Ensure your training script honors the
[training-script contract](../../docs/training-script-contract.md): write
`result.json` matching the autoMIL result schema (`result.schema.json`, shipped
with the package; `automil init` does not scaffold a project-local copy), exit
cleanly on SIGTERM with a partial result, declare env vars under `env.required`.

For a worked minimal reference (about 130 lines, no `automil.*` imports), see
[`examples/sklearn-iris/`](../sklearn-iris/).
