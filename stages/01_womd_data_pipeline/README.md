# Stage 1 - WOMD data pipeline

Audits the causal true-SDC trajectory samples, shapes, coordinate semantics,
split labels, finiteness and motion distributions for both training and
untouched official validation.

Inputs: `TRAIN_NPZ`, `VALIDATION_NPZ`. Outputs: `artifacts/paper_final/01_data/`.

```bash
python stages/01_womd_data_pipeline/run.py
```
