# sklearn-iris, a second autoMIL consumer

A short training script (about 130 lines) that demonstrates plugging a non-autobench
training pipeline into autoMIL via the documented contract
(`docs/training-script-contract.md`).

## What this demonstrates

- The framework is consumer-agnostic. autobench is one consumer; this
  sklearn-iris demo is another. They run side-by-side in the same project
  via the variant registry.
- `result.json` is JSON-Schema-validated at ingestion.
- Required env vars are declared in `automil/config.yaml: env.required`
  For this consumer, the list is empty.

## How to run (local development)

```
uv sync --extra examples-iris
cd examples/sklearn-iris
uv run python train.py              # writes result.json to CWD
cat result.json
```

Expected: `primary_value` between 0.93 and 0.98 with `random_state=42`.

## How autoMIL uses it

Copy `train.py` into a scratch git repository, scaffold `automil/` there, and
submit the script as the baseline node. The `agent_active` budget clock needs
Claude telemetry, so this demo switches to `cap.mode: wall_clock`.

```
mkdir /tmp/iris_project && cp examples/sklearn-iris/train.py /tmp/iris_project/
cd /tmp/iris_project
git init -q && git add -A && git commit -q -m initial
automil init --no-healthcheck       # scaffolds automil/ (needs uv tool install)
# set cap.mode: wall_clock in automil/config.yaml, then commit the scaffold
automil submit --node iris_001 --desc "baseline" --files train.py --max-time 60 --mil-model sklearn_iris
automil orchestrator start
automil status
```

The acceptance test (`tests/acceptance/test_final_phase8_acceptance.py`)
runs this flow end to end on every CI run.

## Files

| Path | Role |
|------|------|
| `train.py` | minimal training script (about 130 lines) |
| `automil/config.yaml` | consumer config: env.required=[], scoring.formula="val_accuracy" |
| `automil/program.md` | narrative for the agent: what to search, primary_value definition |
| `automil/variants/classifier_v0/logistic_v0.py` | starter variant exporting `make_classifier` |
