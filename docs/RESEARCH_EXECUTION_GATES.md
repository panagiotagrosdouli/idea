# Canonical Research Execution Gates

This document defines the operational gates for paper-producing runs. A stage is not complete because code executed; it is complete only when its declared evidence and acceptance checks pass.

## Canonical entrypoint

For authorized paper-scale WOMD data, use `scripts/run_canonical_womd_pipeline.py` rather than manually composing Stage 1–7 commands. The Colab notebook uses the same runner. `--mode stage1` performs provenance and leakage gates only. `--mode full` continues through baselines, learning, untouched validation, canonical packet scheduling, and scenario-level statistics. A missing development-only lambda freeze stops the full run instead of silently selecting on validation.

## Invariants

- Deployable predictors and schedulers never consume future ground truth.
- WOMD scenario IDs define the independent statistical cluster.
- Training/development and official validation have zero scenario overlap.
- Lambda weights, architecture, early stopping and other model-selection choices are frozen on development data before Stage 5.
- The canonical learned archive is four objectives × five seeds = 20 verified checkpoints.
- Scheduler comparisons use paired traffic/channel randomness whenever policies are compared.
- All five paired traffic realizations are retained in raw Stage-7 joins; inference collapses them within the WOMD scenario cluster rather than treating them as independent samples.
- Packet success is realized from the ground-truth-derived link; prediction affects decisions, not the realized channel.
- Historical corpus counts are provenance evidence, not optimization targets.
- Negative results, reversals and operating-region limitations remain visible in the paper.

## Stage 1 — Data freeze

Required evidence:

- frozen source-shard manifest with per-TFRecord SHA-256 and byte size;
- training and official-validation NPZ files;
- per-corpus SHA-256;
- 11 history and 80 future states;
- finite numeric arrays;
- true-SDC source and declared coordinate frame;
- scenario-safe internal split labels;
- official validation labeled only `official_validation`;
- zero cross-corpus scenario overlap;
- historical training fingerprint comparison report.

The historical fingerprint is 249,137 samples, 24,182 scenarios and SHA-256 `b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee`. Exact reproduction is desirable but is not obtained by changing scientifically justified deterministic preprocessing to force old counts. Deviations must be explained and frozen.

## Stage 2 — Link freeze

The BER LUT and link verification artifacts must exist before downstream learned or scheduling runs. Known burst-regime and normalized-power limitations remain disclosed.

## Stage 3 — Classical baselines

Last Position, CV, CA, Kalman CV and IMM are evaluated under the canonical data contract. Diagnostic toy archives are not paper evidence.

## Stage 4 — Learning

Lambda selection uses development data only. Canonical training produces exactly 20 checkpoints. The completion manifest must verify objectives, seeds, training-corpus hash, checkpoint paths and link configuration before Stage 5.

## Stage 5 — Untouched validation

Official validation is opened only after model selection is frozen. Report trajectory metrics and communication-state fidelity/calibration. Model seeds characterize training variability; they are not independent statistical replicates of WOMD scenarios.

## Stage 6 — Packet scheduling

Canonical mode requires exactly eight scheduler families, exactly five paired traffic seeds, all 20 verified checkpoints, the frozen BER LUT, the Stage-5 held-out scenario set, 16 vehicles maximum per scenario, and no `--max-scenarios` truncation. Prediction changes scheduler information, not ground-truth packet success.

## Stage 7 — Statistics

Inference is paired and scenario-clustered. Model-seed and traffic-realization rows remain available as raw evidence, but confirmatory inference first aggregates dependent observations within each scenario. The ADE-to-goodput association is computed on one scenario-level observation per WOMD scenario. Confirmatory families use declared confidence intervals and Holm multiplicity correction. The analysis explicitly tests the chain trajectory error → link-state fidelity → scheduling utility.

## Stage 8 — Release

The manuscript is generated only from frozen accepted artifacts. A submission-ready release requires source-shard and corpus provenance, verified Stage-2 link evidence, the 20-checkpoint completion manifest, untouched-validation outputs, paired scheduling evidence, scenario-clustered statistics, figures/tables and reproducibility instructions.
