# Stage 4 — Communication-aware GRU

## Scientific purpose
Select communication-loss weights on development data only, freeze that selection, and train the canonical four GRU objectives over the five frozen seeds.

## Required inputs
- `TRAIN_NPZ`
- verified `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`
- verified `artifacts/paper_final/02_link/link_verification.json`
- development-only lambda selection inputs (`LAMBDA_LINK`, `LAMBDA_OUTAGE`, rationale) when a valid frozen selection is not already cached

## Outputs
All evidence lives under `artifacts/paper_final/04_learning/`: lambda sweep, frozen `lambda_selection.json`, learned ablation, 20 checkpoints, completion manifest and operational execution-state files.

## Acceptance criteria
The lambda sweep uses the five frozen seeds and development data only; the selected pair is a verified swept setting with rationale recorded before Stage 5; learned training is exactly four objectives × five frozen seeds; all 20 result/checkpoint pairs validate against objective, seed and dataset provenance; the Stage-4 completion manifest passes its existing scientific gate.

## Canonical command
```bash
python stages/04_communication_aware_gru/run.py
```
The production full-run path invokes the same scripts through `make canonical-full`.

## Resume semantics
A cached run is reused only when `training_result.json` parses, metadata matches the expected objective/seed/dataset, and its non-empty checkpoint is inside the expected run directory. Missing, corrupt or inconsistent state is recomputed. `execution_state.json` records operational reasons but never substitutes for the completion manifest.

## Forbidden
Do not use official validation for lambda/model selection, reduce the 4×5 archive, accept result-file existence as completion, reuse a stale selection from another dataset, or weaken Stage-2/Stage-4 gates to avoid reruns.

## Dependencies
Stage 3 and the frozen verified Stage-2 link; data/provenance constraints flow from Stages 0–1.
