# Stage 1 — WOMD data pipeline

## Scientific purpose
Materialize and verify the frozen causal WOMD v1.3.1 training/development corpus and untouched official-validation corpus, including source-shard provenance, geometry/shape audits and split integrity.

## Required inputs
- `WOMD_DATA_ROOT`
- `TRAIN_NPZ`
- `VALIDATION_NPZ`

## Outputs
Canonical evidence is written under `artifacts/paper_final/01_data/`: source-shard manifest, training/validation audits, historical fingerprint comparison and corpus verification.

## Acceptance criteria
All declared source shards are present and hashed; arrays are finite and follow the frozen causal true-SDC contract; split labels are scenario-safe; training and official validation have zero overlapping scenario IDs; corpus SHA-256 values are recorded. Historical `249,137 samples / 24,182 scenarios` is compared as provenance, not enforced as a target.

## Canonical command
```bash
make canonical-stage1 \
  WOMD_DATA_ROOT=/data/womd \
  TRAIN_NPZ=/data/womd/womd_training_paper.npz \
  VALIDATION_NPZ=/data/womd/womd_validation_paper.npz
```
Direct stage entrypoint: `python stages/01_womd_data_pipeline/run.py`.

## Forbidden
Do not depend on Stage 2, CUDA or model checkpoints. Do not treat a merely non-empty partial Drive file as complete. Do not alter official-validation role, fabricate historical counts, or bypass Stage-1 SHA/provenance verification.

## Dependencies
Stage 0 contract/provenance. Stage 1 intentionally has no Stage-2 execution dependency.
