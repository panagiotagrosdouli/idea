# Canonical WOMD execution runbook

This runbook is the operator path for paper-producing WOMD runs. It assumes authorized access to WOMD v1.3.1 and an environment where the required TFRecords and derived NPZ corpora are available.

## Required environment

Set:

```bash
export WOMD_DATA_ROOT=/data/womd
export TRAIN_NPZ=/data/womd/womd_training_paper.npz
export VALIDATION_NPZ=/data/womd/womd_validation_paper.npz
export VALIDATION_GLOB='/data/womd/validation/*.tfrecord-*'
```

The canonical corpus contract is frozen in `configs/womd_paper_corpus.json`: WOMD v1.3.1, true-SDC geometry, 11 history states, 80 future states, up to 16 vehicles per scenario, first 50/1000 training shards and first 40/150 official-validation shards.

## Stage 1 gate

Run:

```bash
make canonical-stage1
```

This first executes the canonical preflight and then writes the Stage-1 shard manifest, per-corpus audits, historical-fingerprint comparison, and cross-corpus leakage verification under `artifacts/paper_final/01_data/`.

A historical fingerprint deviation is evidence requiring explanation, not a reason to alter deterministic preprocessing merely to reproduce old counts.

## Full canonical run

Install ML dependencies and verify CUDA first:

```bash
python -m pip install -e '.[dev,ml,paper]'
make canonical-full
```

`canonical-full` requires the frozen Stage-2 BER LUT and link verification, all validation TFRecords, PyTorch, and CUDA. It then runs the canonical Stage 1–7 pipeline. Lambda weights are never silently chosen: if `lambda_selection.json` does not already exist, provide development-selected values explicitly, for example:

```bash
make canonical-full \
  LAMBDA_ARGS='--lambda-link 0.2 --lambda-outage 0.1 \
  --selection-rationale "Selected from the declared development-only sweep."'
```

Only use values chosen from the declared development-only sweep. Official validation must not influence lambda, architecture, early-stopping, or model-selection decisions.

## Release gate

After Stages 1–7 have produced canonical evidence, run Stage 8 through the stage runner or the declared Stage-8 commands. Release readiness requires semantic PASS/canonical evidence, not merely files with the expected names.

Do not claim paper-ready evidence unless the Stage-8 release-readiness report passes. Model-based optical-channel limitations and negative or regime-dependent results remain part of the final evidence set.
