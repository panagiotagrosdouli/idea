# Non-Joint PDF requirements traceability

The three supplied predictive PC-FMCW/DPSK plans define the scope. The separate
Joint trajectory-to-beam-and-ADB document is excluded.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Causal trajectory prediction | `predictors.py`, future-mutation gates | Complete |
| Last/CV/CA/Kalman/IMM/Oracle | predictor library and forecast artifacts | Complete |
| ADE/FDE | `forecast_evaluation.py` | Complete |
| Range and ego-relative bearing | `geometry.py`, rotation/heading tests | Complete |
| SNR/BER/PER/goodput | `link.py`, analytical and adaptive LUT | Complete, model-based |
| Explicit outage definition | BER/PER/goodput modes and tests | Complete |
| Link lifetime in steps/seconds | link and forecast modules | Complete |
| Packet queues/deadlines/retries | traffic and simulation engine | Complete |
| Poisson/periodic/Markov/saturated traffic | traffic generator | Complete |
| Random/RR/Reactive/PF | scheduling policies | Complete |
| CV/Kalman/IMM predictive policies | scheduling policies and benchmark | Complete |
| Predictive utility and lifetime urgency | scheduling policies and exact tests | Complete |
| Oracle-information reference | perfect future with heuristic utility | Complete |
| Common random numbers | shared trace and per-attempt uniforms | Complete |
| Ground-truth delivery evaluation | current true geometry drives PER | Complete |
| P50/P95/P99, censoring and normalized fairness | expanded metrics | Complete |
| Scheduled SNR/BER/PER/power | expanded metrics | Complete |
| Bearing/lifetime/support forecast metrics | forecast evaluator | Complete |
| Scenario regimes | deterministic slices and figures | Complete |
| PC-FMCW-like sensing uncertainty | declared range/bearing + AR(1) model | Complete, assumed |
| Covariance-aware Kalman robustness | equivalent `R` integration | Complete, assumed |
| Adaptive BER confidence | error-targeted LUT and Wilson upper bound | Complete |
| Physical deadline/duration invariance | configs and regression tests | Complete |
| Clustered paired statistics and Holm | metrics and staged summaries | Complete |
| Staged load/horizon/traffic/channel studies | 43-setting runner | Complete; quick run only |
| Four communication-aware objectives | GRU training ablation runner | Infrastructure complete |
| Scenario-safe split/checkpoint schema | hashes and version validation | Complete |
| True-SDC official WOMD adapter | optional TFRecord/proto path | Complete, data blocked |
| Official full-WOMD experiment | raw shards absent | Blocked |
| Frozen compatible learned checkpoint | no checkpoint supplied | Blocked |
| Multi-seed learned result | data/checkpoint/PyTorch absent | Blocked |
| Measured optical-channel validation | no measurements supplied | Outside evidence |
| Final submission paper | evidence/metadata gates remain | Not complete |

“Complete” means an executable code path plus relevant test or artifact. It
does not mean that a favorable scientific hypothesis was observed.
