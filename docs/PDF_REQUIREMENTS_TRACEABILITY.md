# Non-Joint PDF requirements traceability

This repository follows only the three predictive PC-FMCW/DPSK communication
plans supplied for this assignment. The separate *Joint uncertainty-aware
trajectory-to-beam-and-ADB control* direction is explicitly excluded.

| PDF requirement | Implementation | Verification / artifact | Status |
|---|---|---|---|
| Causal WOMD history to future trajectory | `predictors.py`, WOMD adapter | future-mutation tests | Implemented |
| Last/CV/Kalman/IMM/CA/oracle baselines | `predictors.py` | forecast summary, unit tests | Implemented |
| ADE/FDE | `forecast_evaluation.py` | motion forecast summary | Implemented |
| Trajectory to range/bearing/link | `geometry.py`, `link.py` | perfect-forecast and monotonicity tests | Implemented |
| SNR/BER/PER/goodput/outage | analytical DBPSK plus LUT | validation report and LUT | Implemented |
| Range/SNR/outage/lifetime metrics | `forecast_evaluation.py` | MAE, F1, AUROC, lifetime error | Implemented |
| Link-lifetime prediction | `LinkModel.link_lifetime_steps` | oracle zero-error test | Implemented |
| Queue, deadlines and conservation | `traffic.py`, simulation engine | conservation tests | Implemented |
| Poisson/periodic/Markov traffic | traffic generator | reproducibility tests | Implemented |
| Random/RR/current/PF baselines | scheduling policies | common-random benchmark | Implemented |
| CV/Kalman/IMM predictive scheduling | scheduling policies | benchmark and ablation rows | Implemented |
| Predictive utility/lifetime scheduling | scheduling policies | paired matrix | Implemented |
| Perfect-future information reference | lifetime-aware oracle | forecast/scheduler artifacts | Implemented |
| Delivered before expiry / left at disconnect | simulation metrics | benchmark summaries | Implemented |
| Channel fidelity ablations | range, pointing/FoV, atmosphere | ablation summary/figure | Implemented |
| Part-A BER calibration path | Monte Carlo LUT | LUT-source ablation | Implemented |
| Sensing/forecast robustness | deterministic noise injection | 0.5/1/2 m sweeps | Implemented |
| Horizon/load/vehicles/slot/seeds matrix | paper matrix JSON | quick/full commands | Implemented |
| Paired statistics | bootstrap, t-test, Wilcoxon, Cohen dz | matrix summary | Implemented |
| Communication-aware GRU objective | PyTorch training/inference | NumPy loss tests | Implemented, optional ML dependency |
| Scenario split and manifest | deterministic SHA256 grouping | dataset manifest | Implemented |
| Numbered reproducibility scripts | `scripts/00` through `07` | one-command pipeline | Implemented |
| LaTeX tables and paper figures | `paper_artifacts.py` | `artifacts/paper_run` | Implemented |
| Official full WOMD evaluation | requires official shards/SDC metadata | absent from supplied files | Blocked by input data |
| Frozen upstream trained checkpoint | requires original `.pt` file | absent from supplied archives | Blocked by input data |
| Measured optical-channel validation | requires measurement campaign | outside WOMD/supplied code | Out of available evidence |

“Implemented” means the code path and a reproducible verification exist. It
does not mean a final scientific claim has been established on the full WOMD
benchmark. The three blocked rows define the remaining evidence gap before a
submission-quality paper.
