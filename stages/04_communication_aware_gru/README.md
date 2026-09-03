# Stage 4 - Communication-aware GRU

Runs the development-only communication-loss sweep, freezes selected weights
and trains trajectory-only, link-aware, outage-aware and full GRUs over five
seeds. Completion requires 20 verified checkpoints and a completion manifest.

Inputs: `TRAIN_NPZ`, `LAMBDA_LINK`, `LAMBDA_OUTAGE`. Outputs:
`artifacts/paper_final/04_learning/`.

```bash
python stages/04_communication_aware_gru/run.py
```
