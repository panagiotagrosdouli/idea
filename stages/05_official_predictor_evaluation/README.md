# Stage 5 — Official predictor evaluation

## Scientific purpose
Evaluate the already frozen Stage-4 predictor archive on the untouched official-validation corpus and persist scenario-level trajectory, link and uncertainty metrics.

## Required inputs
- `TRAIN_NPZ` for development-fitted calibration/provenance only
- `VALIDATION_NPZ` with `split=official_validation`
- exactly 20 verified Stage-4 checkpoints
- Stage-4 completion manifest
- verified frozen Stage-2 BER LUT/evidence

## Outputs
- `artifacts/paper_final/05_heldout/heldout_summary.json`
- `artifacts/paper_final/05_heldout/heldout_metrics_by_scenario.csv`
- `artifacts/paper_final/05_heldout/heldout_provenance.json`

## Acceptance criteria
Exactly 20 frozen checkpoints match the Stage-4 completion manifest; official validation remains untouched until this evaluation; ADE/FDE, geometry/link fidelity, outage metrics, NLL and coverage are recorded using development-fitted calibration where applicable.

## Canonical command
```bash
python stages/05_official_predictor_evaluation/run.py
```

## Forbidden
Do not retune lambdas, choose checkpoints, alter hyperparameters or select models using official-validation outcomes. Future ground truth is evaluator truth, not a deployable model input.

## Dependencies
Stage 4 must be scientifically complete; verified Stage-2 link evidence and Stage-1 official-validation provenance remain binding.
