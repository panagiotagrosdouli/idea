# Canonical Research Execution Gates

This document defines the operational gates for paper-producing runs. A stage is not complete because code executed; it is complete only when its declared evidence and acceptance checks pass.

## Invariants

- Deployable predictors and schedulers never consume future ground truth.
- WOMD scenario IDs define the independent statistical cluster.
- Training/development and official validation have zero scenario overlap.
- Lambda weights, architecture, early stopping and other model-selection choices are frozen on development data before Stage 5.
- The canonical learned archive is four objectives × five seeds = 20 verified checkpoints.
- Scheduler comparisons use paired traffic/channel randomness whenever policies are compared.
- Packet success is realized from the ground-truth-derived link; prediction affects decisions, not the realized channel.
- Historical corpus counts are provenance evidence, not optimization targets.
- Negative results, reversals and operating-region limitations remain visible in the paper.

## Stage 1 — Data freeze

Required evidence:

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

Policies are compared on paired scenario/traffic/channel realizations. Primary outcomes include timely goodput, PDR, deadline misses, latency, outage and fairness. Prediction changes scheduler information, not ground-truth packet success.

## Stage 7 — Statistics

Inference is paired and scenario-clustered. Figures may display multiple model-seed rows, but inferential procedures must not treat those rows as independent scenarios. Confirmatory families use declared confidence intervals and multiplicity correction. The analysis explicitly tests the chain trajectory error → link-state fidelity → scheduling utility.

## Stage 8 — Release

The manuscript is generated only from frozen accepted artifacts. A submission-ready release requires data/provenance manifests, 20 verified checkpoints, untouched-validation outputs, paired scheduling evidence, clustered statistics, figures/tables and reproducibility instructions.
