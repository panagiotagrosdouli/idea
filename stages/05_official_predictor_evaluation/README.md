# Stage 5 - Official predictor evaluation

Evaluates every frozen checkpoint exactly once on untouched official validation.
It reports per-scenario trajectory, geometry, SNR, goodput, outage, lifetime,
Gaussian NLL and 50/90/95% coverage metrics.

Inputs: validation NPZ and checkpoint glob. Outputs: `artifacts/paper_final/05_heldout/`.

```bash
python stages/05_official_predictor_evaluation/run.py
```
