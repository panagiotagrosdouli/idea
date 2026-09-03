# Stage 0 repository inventory

## Freeze

- Repository: `panagiotagrosdouli/predictive-pc-fmcw-vehicular-communications.`
- Default branch: `main`
- Audited development branch: `feature/rl-scheduling`
- Audit baseline head before audit-artifact commits: `7bed967e6ea7ce3a104f82fac0856c38c52fb1ed`
- Base branch SHA observed by PR CI: `35e0f24615e01f0c1c0f93ce013f7734918ccb56`
- Language: Python
- Package: `predictive-pc-fmcw` 0.1.0
- Supported Python: >=3.10
- CI Python: 3.11
- CLI: `pcfmcw = predictive_pc_fmcw.cli:main`

## Top-level structure observed

`.github/`, `artifacts/`, `configs/`, `data/`, `docs/`, `notebooks/`, `output/`, `paper/`, `scripts/`, `src/`, `stages/`, `tests/`, plus `Makefile`, `README.md`, `README_GR.md`, and `pyproject.toml`.

## Scientific component status

| Area | Status | Stage-0 interpretation |
|---|---|---|
| WOMD training pipeline | IMPLEMENTED / EXECUTED | Status documentation records an audited training corpus; numerical claims remain tied to existing manifests/artifacts. |
| Official validation export | IMPLEMENTED / NOT EXECUTED AS FINAL EVIDENCE | Untouched official-validation artifact remains missing. |
| Part-A PC-FMCW/DPSK link | IMPLEMENTED / EXECUTED | Receiver-derived BER LUT and calibrated/model-based link path exist. Do not call it a measured optical channel. |
| Classical predictors | IMPLEMENTED / EXECUTED | Last/CV/CA/Kalman/IMM paths and evaluations exist. |
| Communication-aware GRU | IMPLEMENTED / PARTIAL EVIDENCE | Code/loss/checkpoint schema exists; canonical 4 objectives x 5 seeds = 20 verified checkpoints are not present. |
| Probabilistic calibration | PARTIAL | Classical residual-Gaussian calibration exists; learned held-out calibration evidence is missing. |
| Packet simulator | IMPLEMENTED / EXECUTED | Traffic, queues, deadlines, retries, packet metrics and paired realization logic exist. |
| Classical/predictive schedulers | IMPLEMENTED / EXECUTED | Random, RR, Reactive, PF, predictive families, Lifetime and Oracle exist. |
| Statistics | IMPLEMENTED / EXECUTED ON EXISTING STUDIES | Paired/scenario-aware statistics are implemented; final official-WOMD analysis still depends on missing held-out runs. |
| RL state/reward | IMPLEMENTED / UNIT-LEVEL | Causal observation and reward modules exist. |
| RL environment | IMPLEMENTED / UNVERIFIED END-TO-END | Gym-style reset/step API, action mask and oracle guard exist. |
| RL simulation backend | IMPLEMENTED / UNVERIFIED END-TO-END | Reuses canonical LinkModel, PacketQueues, TrafficTrace and causal forecast helper. Equivalence gate with canonical runner is still missing. |
| DQN/PPO | MISSING | Neural RL agents are intentionally not started before environment equivalence. |
| Final paper/release | PARTIAL | Existing figures/tables/manifests exist, but final official held-out evidence and final venue-formatted submission are missing. |

## Existing artifact families observed

The repository contains versioned `corrected_v1`/`corrected_v2`-era evidence including BER, figures, motion baselines, ablations, scenario slices, staged experiments and reproducibility manifests. These are existing evidence only; they do not substitute for the missing official-validation corpus or canonical learned checkpoint archive.

## Current CI definition

CI installs `.[dev]`, then runs:

1. `ruff check src tests scripts stages`
2. `pytest`
3. `pcfmcw validate --config configs/default.json --output results/validation.json`

At the start of Stage 0, CI failed at ruff before tests/validation. Fifteen reported formatting/import blockers were identified. Stage-0 commits corrected those known lint blockers without intentionally changing scientific semantics. A new CI run is required to establish the next baseline.

## P0 gaps

- frozen untouched official-validation processed artifact and cross-corpus zero-overlap evidence;
- development-only lambda selection record;
- 20 verified paper-scale GRU checkpoints (4 objectives x 5 paired seeds);
- learned probabilistic calibration on development and one frozen held-out evaluation;
- official-WOMD learned/reactive/oracle scheduler matrix with paired traffic seeds;
- final joined predictor-fidelity vs realized packet-utility analysis;
- final statistical artifacts and paper/release manifest based on official evidence.

## P1 RL gaps

- canonical-runner vs step-backend equivalence tests;
- deterministic replay/integration coverage on real simulation fixtures;
- DQN-current and PPO-current;
- DQN-predictive and PPO-predictive;
- uncertainty-aware predictive RL after calibration is frozen;
- train/dev-only RL tuning and one frozen official-validation evaluation.

## Gate 0

Current status at inventory creation: **PARTIAL**.

Reason: repository identity and evidence gaps are understood and lint blockers have been addressed, but the post-fix CI run must complete successfully before Gate 0 can be promoted to PASS.
