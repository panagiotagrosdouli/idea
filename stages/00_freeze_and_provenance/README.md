# Stage 0 — Freeze and provenance

## Scientific purpose
Freeze the WOMD release, corpus roles, hashes, experiment assumptions and scenario-level leakage policy before development or held-out evaluation can influence them.

## Required inputs
- `TRAIN_NPZ`
- `VALIDATION_NPZ`
- `configs/default.json`

## Outputs
- `artifacts/paper_final/00_freeze/split_audit.json`
- `artifacts/paper_final/00_freeze/paper_manifest.json`

## Acceptance criteria
Training and official-validation hashes are recorded, the frozen release is WOMD v1.3.1, scenario overlap is zero, and the official-validation role remains evaluator-only. Historical counts/hashes may be recorded as provenance fingerprints but are not target values to reproduce artificially.

## Canonical command
```bash
python stages/00_freeze_and_provenance/run.py
```

## Forbidden
Do not tune models, select hyperparameters, inspect official-validation outcomes for decisions, rewrite corpus roles, or weaken the zero-overlap gate.

## Dependencies
None. Stage 0 is the root scientific contract.
