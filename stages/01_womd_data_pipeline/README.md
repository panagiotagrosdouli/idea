# Stage 1 - WOMD data pipeline

Audits the causal true-SDC trajectory samples, shapes, coordinate semantics,
split labels, finiteness and motion distributions for both training and
untouched official validation.

Inputs: `TRAIN_NPZ`, `VALIDATION_NPZ`. Outputs: `artifacts/paper_final/01_data/`.

Before setting those inputs, discover and verify available files with:

```bash
PYTHONPATH=src python scripts/womd_preflight.py /path/to/data \
  --output womd_preflight.json
```

The command fails closed. Finding an arbitrary NPZ never implies readiness,
and raw TFRecords produce `BLOCKED_EXPORT_REQUIRED` until frozen training and
official-validation NPZ files pass the complete Stage-1 corpus verifier.

```bash
python stages/01_womd_data_pipeline/run.py
```
