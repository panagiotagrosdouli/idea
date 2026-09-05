# Stage 3 — Classical baselines

## Scientific purpose
Evaluate causal classical motion predictors on development data and propagate their forecasts through the same frozen Stage-2 link model used by learned predictors.

## Required inputs
- `TRAIN_NPZ`
- `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`
- `artifacts/paper_final/02_link/link_verification.json`

## Outputs
- `artifacts/paper_final/03_baselines/forecast_summary.json`
- `artifacts/paper_final/03_baselines/forecast_metrics_by_scenario.csv`

## Acceptance criteria
Last Position, Constant Velocity, Constant Acceleration, Kalman CV and IMM are evaluated on the development split; ADE/FDE and link metrics are recorded; the verified frozen BER LUT is used; no future state enters a deployable predictor.

## Canonical command
```bash
python stages/03_classical_baselines/run.py
```

## Forbidden
Do not tune against official validation, substitute a different link model, use future ground truth as predictor input, or hide negative baseline comparisons.

## Dependencies
Stages 0–2, especially verified Stage-2 link evidence and the Stage-1 development corpus.
