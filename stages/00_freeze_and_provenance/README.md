# Stage 0 - Freeze and provenance

Locks training/official-validation hashes, the experiment configuration and the
scenario-level leakage policy. The stage fails if any `scenario_id` appears in
both corpora. It never tunes a model or reads official-validation outcomes.

Inputs: `TRAIN_NPZ`, `VALIDATION_NPZ`. Outputs: `artifacts/paper_final/00_freeze/`.

```bash
python stages/00_freeze_and_provenance/run.py
```
