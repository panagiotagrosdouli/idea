# Stage 6 — Packet scheduling

## Scientific purpose
Run the canonical packet-level comparison on the same official held-out scenario population using paired traffic/channel randomness.

## Required inputs
- official-validation TFRecords
- `TRAIN_NPZ` for frozen model provenance
- exactly 20 verified Stage-4 checkpoints and completion manifest
- Stage-5 held-out metrics/provenance
- verified frozen Stage-2 BER LUT/evidence

## Outputs
Canonical raw and completion evidence is written under `artifacts/paper_final/06_scheduling/`, including `completed_runs.json` and `evaluation_manifest.json`.

## Acceptance criteria
Canonical mode is enabled; raw validation scenario IDs match Stage-5 held-out IDs; exactly eight frozen scheduler families are evaluated over exactly five paired traffic seeds; there is no canonical `max-scenarios` truncation; per-scenario raw metrics are retained. Ground-truth future may realize packet outcomes and evaluator/oracle references only.

## Canonical command
```bash
python stages/06_packet_scheduling/run.py
```

## Forbidden
Do not add/remove scheduler families or traffic seeds in the canonical protocol, unpair randomness, truncate the official scenario population, use future truth in a deployable scheduler, or suppress negative scheduling regimes.

## Dependencies
Stage 5 official predictor evaluation, Stage 4 frozen checkpoints, and verified Stage-2 link evidence.
