# Stage 0 repository inventory

## Freeze

- Repository: `panagiotagrosdouli/predictive-pc-fmcw-vehicular-communications.`
- Default branch: `main`
- Audited development branch: `feature/rl-scheduling`
- Audit baseline before cleanup: `7bed967e6ea7ce3a104f82fac0856c38c52fb1ed`
- Verified code SHA: `3cdb108ec5c3b50b96b2c61ffbfa579297f1dd8b`
- Base `main` SHA observed by CI: `35e0f24615e01f0c1c0f93ce013f7734918ccb56`
- Package: `predictive-pc-fmcw` 0.1.0
- Supported Python: >=3.10
- Verified CI Python: 3.11.16 on Ubuntu 24.04
- CLI: `pcfmcw = predictive_pc_fmcw.cli:main`

## Top-level structure observed

`.github/`, `artifacts/`, `configs/`, `data/`, `docs/`, `notebooks/`, `output/`, `paper/`, `scripts/`, `src/`, `stages/`, `tests/`, plus `Makefile`, `README.md`, `README_GR.md`, and `pyproject.toml`.

## Scientific component status

| Area | Status | Stage-0 interpretation |
|---|---|---|
| WOMD training pipeline | IMPLEMENTED / EXECUTED | Status documentation records an audited training corpus; final claims remain tied to manifests/artifacts. |
| Official validation export | IMPLEMENTED / FINAL ARTIFACT MISSING | Untouched official-validation corpus remains a Stage-1 blocker. |
| Part-A PC-FMCW/DPSK link | IMPLEMENTED / EXECUTED | Receiver-derived BER LUT and calibrated/model-based link path exist. It is not a measured optical channel. |
| Classical predictors | IMPLEMENTED / EXECUTED | Last/CV/CA/Kalman/IMM paths and evaluations exist. |
| Communication-aware GRU | IMPLEMENTED / PARTIAL EVIDENCE | Code/loss/checkpoint schema exists; canonical 4 objectives x 5 seeds = 20 verified checkpoints are not present. |
| Probabilistic calibration | PARTIAL | Classical residual-Gaussian calibration exists; learned held-out calibration evidence is missing. |
| Packet simulator | IMPLEMENTED / EXECUTED | Traffic, queues, deadlines, retries, packet metrics and paired realization logic exist. |
| Classical/predictive schedulers | IMPLEMENTED / EXECUTED | Random, RR, Reactive, PF, predictive families, Lifetime and Oracle exist. |
| Statistics | IMPLEMENTED / EXISTING EVIDENCE | Final official-WOMD analysis still depends on missing held-out runs. |
| RL state/reward | IMPLEMENTED / UNIT VERIFIED | Causal observation and reward tests are part of the passing suite. |
| RL environment | IMPLEMENTED / UNIT VERIFIED | Reset/step API, action mask and oracle guard exist; canonical simulator equivalence is still pending. |
| RL transition backend | IMPLEMENTED / INTEGRATION PENDING | Reuses LinkModel, PacketQueues, TrafficTrace and the causal forecast helper; equivalence gate remains required. |
| DQN/PPO | MISSING | Neural agents are intentionally deferred until environment equivalence. |
| Final paper/release | PARTIAL | Final official held-out evidence and final venue-formatted submission remain incomplete. |

## Verified CI baseline

CI run `33731714105` verified code SHA `3cdb108ec5c3b50b96b2c61ffbfa579297f1dd8b` using the repository workflow:

1. `ruff check src tests scripts stages` — **PASS** (`All checks passed!`)
2. `pytest` — **PASS: 71 passed in 4.97 s**
3. `pcfmcw validate --config configs/default.json --output results/validation.json` — **PASS**

Scientific validation reported:

- distance monotonicity: PASS
- pointing monotonicity: PASS
- BER monotonicity: PASS
- causal forecast invariance: PASS
- oracle forecast sensitivity: PASS

The Stage-0 cleanup fixed lint/import blockers only and did not intentionally change the scientific protocol.

## Existing artifact families observed

The repository contains versioned `corrected_v1`/`corrected_v2` evidence including BER, figures, motion baselines, ablations, scenario slices, staged experiments and reproducibility manifests. These artifacts do not substitute for the missing untouched official-validation corpus or the missing canonical learned checkpoint archive.

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
- deterministic replay/integration coverage on canonical simulation fixtures;
- DQN-current and PPO-current;
- DQN-predictive and PPO-predictive;
- uncertainty-aware predictive RL after calibration is frozen;
- train/dev-only RL tuning and one frozen official-validation evaluation.

## Gate 0

**PASS**.

Repository identity, inventory, baseline verification, CI status and known evidence gaps are now explicit. The current code baseline has clean lint, 71 passing tests and passing scientific validation. Missing experimental evidence remains explicitly classified and is not treated as completed work.
