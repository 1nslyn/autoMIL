# sklearn-iris, autoMIL second consumer (DEC-02)

A ~80-line training script that demonstrates plugging a non-autobench
training pipeline into autoMIL via the documented contract
(`docs/training-script-contract.md`, DEC-06).

## What this demonstrates

- The framework is consumer-agnostic. autobench is one consumer; this
  sklearn-iris demo is another. They run side-by-side in the same project
  via the registry path (Phase 1).
- `result.json` is JSON-Schema-validated at ingestion (D-201).
- Required env vars are declared in `automil/config.yaml: env.required`
  (D-202). For this consumer, the list is empty.

## How to run (local development)

```
pip install -e '.[examples-iris]'   # OR: pip install scikit-learn pyyaml
cd examples/sklearn-iris
python train.py                     # writes result.json to CWD
cat result.json
```

Expected: `composite` between 0.93 and 0.98 with `random_state=42`.

## How autoMIL uses it

```
cd examples/sklearn-iris
automil init                        # already initialised; this is idempotent
automil submit --node iris_001 --desc "baseline" --files train.py --max-time 60
automil orchestrator start
automil status
```

The acceptance gate (`tests/acceptance/test_final_phase8_acceptance.py`)
exercises this exact path on every Phase 8+ commit.

## Files

| Path | Role |
|------|------|
| `train.py` | minimal training script (~80 lines) |
| `automil/config.yaml` | consumer config: env.required=[], scoring.formula="accuracy" |
| `automil/program.md` | narrative for the agent: what to search, composite definition |
| `automil/variants/classifier_v0/logistic_v0.py` | starter variant exporting `make_classifier` |
