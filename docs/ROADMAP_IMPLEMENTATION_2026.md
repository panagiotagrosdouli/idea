# Publication roadmap implementation status

This document maps the 24-page completion roadmap to executable repository
components. It deliberately distinguishes software completion from experimental
evidence: code being present does not mean that an official-WOMD result exists.

## Research questions

- **RQ1:** Under which operating conditions does motion prediction improve
  packet scheduling?
- **RQ2:** Does lower ADE/FDE imply better communication utility?
- **RQ3:** Does communication-aware predictor training improve downstream
  scheduling?
- **RQ4 (optional):** Does predictive RL add value beyond a strong heuristic?

## Work-package status

| Gate | Repository implementation | Evidence status |
|---|---|---|
| Immutable code/config/data hashes | `00_freeze_paper_manifest.py` now records code, config, dataset and BER-LUT hashes | Implemented; final artifacts must be frozen after training |
| Zero scenario overlap | `00_audit_womd_split_integrity.py` compares named NPZ corpora by `scenario_id` and fails on overlap | Implemented and unit-tested; official-validation NPZ still required |
| Official WOMD training corpus | Audited 249,137-sample, 24,182-scenario true-SDC corpus | Executed |
| Official WOMD validation corpus | Fixed-split export and hard held-out label gate | Implemented; export/evaluation not yet executed |
| Four GRU objectives | trajectory, link, outage and full communication-aware objectives | 7/12 earlier three-seed paper runs completed; five-seed canonical archive incomplete |
| Communication-loss sweep | Five unique lambda settings, resumable runner | Implemented; not executed at canonical five-seed scale |
| Learned NLL/calibration | Development-fitted per-horizon diagonal residual Gaussian; NLL and 50/90/95% joint coverage | Implemented and unit-tested; official values pending |
| Official learned predictor evaluation | Per-scenario trajectory, geometry, link, outage, lifetime and probabilistic metrics | Implemented; pending checkpoints and validation NPZ |
| Official packet scheduling | Reactive, PF, CV, Kalman, IMM, Link-Lifetime, learned and information-oracle under five paired traffic seeds | Implemented; official runs pending |
| Scenario-cluster statistics | Traffic-seed differences averaged within `scenario_id`; bootstrap/Wilcoxon/t-test/effect size/Holm | Implemented and tested |
| Horizon/N/load/slice studies | Existing experiment matrices and scenario labels | Executed on controlled/proxy data; official learned matrix pending |
| Part-A PC-FMCW/DPSK LUT | Receiver-derived, confidence-aware 31-point mapping | Executed; final regeneration/hash freeze pending |
| Optical measurements | Model-based channel with explicit uncalibrated-power flag | No measurement dataset; measured-channel claim prohibited |
| Predictive RL | Proposed only as a P1 extension | Not implemented; not on the core-paper critical path |

## Canonical execution order

1. Export official WOMD validation with
   `split=official_validation`.
2. Run the split-integrity audit and require zero overlapping scenario IDs.
3. Complete the development-only five-seed lambda sweep and freeze the selected
   weights before examining official validation.
4. Produce 20 checkpoints: four objectives times five seeds.
5. Fit residual uncertainty on development samples only and evaluate all frozen
   checkpoints once on official validation.
6. Run the paired packet-level scheduler matrix on official validation.
7. Join predictor and packet metrics by objective, model seed and scenario;
   perform scenario-clustered inference.
8. Regenerate final vector figures, LaTeX tables, manuscript and reproducibility
   manifest.

## Canonical commands

```bash
# Leakage gate
make split-audit \
  TRAIN_NPZ=data/processed/womd_v131_training.npz \
  VALIDATION_NPZ=data/processed/womd_v131_official_validation.npz

# Freeze data, configuration and physical-layer artifact
PYTHONPATH=src python scripts/00_freeze_paper_manifest.py \
  --training-npz data/processed/womd_v131_training.npz \
  --validation-npz data/processed/womd_v131_official_validation.npz \
  --ber-lut artifacts/paper_final/ber/dbpsk_ber_lut.csv \
  --womd-release v1.3.1 \
  --output artifacts/paper_final/paper_manifest.json

# Development-only uncertainty fit plus untouched held-out evaluation
PYTHONPATH=src python scripts/06_evaluate_learned_checkpoints.py \
  data/processed/womd_v131_official_validation.npz \
  artifacts/paper_final/learned_ablation/*/seed_*/best_comm_aware_gru.pt \
  --development-npz data/processed/womd_v131_training.npz \
  --output artifacts/paper_final/heldout_learned

# Paired packet-level experiment (defaults to eight schedulers and five seeds)
PYTHONPATH=src python scripts/06_evaluate_learned_scheduler_womd.py \
  /data/womd/validation/*.tfrecord \
  --checkpoints \
  artifacts/paper_final/learned_ablation/*/seed_*/best_comm_aware_gru.pt \
  --output artifacts/paper_final/official_scheduler
```

## Publication claim gate

The repository must not claim that communication-aware prediction improves
scheduling until all of the following are present: 20 verified checkpoints,
untouched official-validation results, paired packet experiments, scenario-level
confidence intervals, predeclared Holm families, measured GRU runtime, and a
clean reproduction. Negative or regime-dependent results remain valid research
results and must not be hidden.
