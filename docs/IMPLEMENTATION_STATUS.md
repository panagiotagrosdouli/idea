# Exact implementation status

## Implemented and executed in `corrected_v2`

| Area | Concrete implementation | Verification |
|---|---|---|
| Causality | history-only predictor API and forecast/evaluation separation | future-mutation + H=0 tests |
| Coordinate frame | ego heading in geometry and link-aware loss | rotation/heading tests |
| Physical time | seconds for duration, horizon and deadlines | slot-invariance tests |
| Part-A receiver | chirp generation, FFT-carrier extraction, refinement, phase compensation and differential DPSK decisions | notebook-receiver BER tests |
| BER confidence | adaptive Monte Carlo, Wilson upper bound, conservative monotone LUT | BER monotonicity tests |
| Channel semantics | normalized power and explicit calibration flag | link tests and manifest |
| Outage | BER/PER/goodput modes | semantic tests and staged study |
| Traffic | Poisson, periodic, bursty, saturated, urgent/bulk deadlines | reproducibility and class tests |
| Schedulers | Random, RR, Reactive, PF, CV, Kalman, IMM, Utility, Lifetime, Oracle | policy tests and benchmarks |
| Metrics | goodput/PDR, P50/P95/P99, class misses, censoring, fairness, scheduled link state | end-to-end tests |
| Statistics | paired clusters, bootstrap, direction-aware tests and Holm | exact statistical tests |
| Sensing | perfect, Cartesian and range/bearing AR(1) assumptions | deterministic tests |
| Official WOMD | true SDC, vehicle filtering and validity masks | fixture tests |
| Learning | GRU plus trajectory/link/outage/full losses and checkpoint schema | loss/schema tests |
| Probabilistic baseline | per-horizon residual Gaussian CV/CA calibration on disjoint scenarios | NLL/coverage tests and figure |
| Artifacts | tables, architecture, traces, distributions, calibration, failures and complexity | regenerated figures/CSV/JSON/PDF |

## Executed evidence

- 51/51 tests, lint, compilation and 5/5 scientific gates;
- 31-point Part-A receiver BER LUT from -5 to 25 dB;
- 120 controlled policy episodes (12 scenarios × 10 policies);
- synthetic and compact-proxy motion/link evaluation;
- full paper ablations;
- 1,125 staged rows (45 settings × 5 seeds × 5 policies);
- 12 study axes including urgent/bulk traffic;
- CPU complexity diagnostics for five predictors and ten schedulers;
- analytical GRU parameter count: 169,620 parameters.

## Implemented but not executed with the supplied inputs

- official-WOMD TFRecord export and true-SDC training samples;
- four GRU objectives over three or more training seeds;
- compatible learned-checkpoint evaluation in the packet scheduler.

The Stage-4 artifact contains 70 causal predicted trajectories but no model
weights and no paired future ground truth. It cannot substitute for a trained
checkpoint or a held-out dataset.

## Not completed because the evidence does not exist

- official full-WOMD held-out results;
- probabilistic learned calibration results;
- measured optical-channel validation or calibrated absolute received power;
- learned GRU runtime on this machine;
- final author/venue-formatted submission.

These are reported as blockers and are not silently replaced by synthetic
claims.
