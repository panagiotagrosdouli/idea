# Stage 3 - Classical baselines

Evaluates Last Position, Constant Velocity, Constant Acceleration, Kalman CV
and IMM on the development split with the same causal history and link model.

Input: `TRAIN_NPZ`. Outputs: `artifacts/paper_final/03_baselines/`.

```bash
python stages/03_classical_baselines/run.py
```
