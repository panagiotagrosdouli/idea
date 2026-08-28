# Trajectory-Predictive Link-Lifetime Scheduling for PC-FMCW/DPSK Vehicular Optical Communications

**Manuscript status:** reproducible research draft; author and affiliation to be inserted.

## Abstract

Phase-coded frequency-modulated continuous-wave (PC-FMCW) laser headlamps can combine sensing, illumination, and gigabit-per-second differential phase-shift-keying communication. Their narrow optical field of view also makes communication availability strongly dependent on vehicle motion. This paper studies a communication-control problem that is distinct from beam selection and adaptive driving-beam control: which connected receiver should be served in each scheduling slot when its future optical-link lifetime can be inferred from a causal trajectory forecast? We implement a reproducible pipeline that maps vehicle-history observations to future trajectories, range and bearing, PC-FMCW/DPSK link quality, packet-error probability, and link lifetime. The scheduling layer combines discounted future goodput, outage risk, queue and deadline state, fairness, switching cost, and proactive pressure to serve packets before disconnection. Last-position, constant-velocity, Kalman, interacting-multiple-model, constant-acceleration, reactive, proportional-fair, and perfect-future references are included. In a controlled 12-episode benchmark, the proposed link-lifetime scheduler increases goodput from 2.293 to 2.336 Mbps and delivered-before-expiry ratio from 0.801 to 0.814. In a 224-row quick paired matrix, the mean gain is 0.0427 Mbps with a bootstrap 95% interval of [0.0229, 0.0629] Mbps. A compact three-scene WOMD proxy benchmark does not reproduce this gain, showing that the full official split, true self-driving-car identity, and frozen trained checkpoint are required before a full-WOMD performance claim. The contribution is therefore a complete research codebase, protocol, and mechanism evaluation rather than a claim of measured optical-channel performance.

**Index terms:** predictive communication, vehicular optical communication, PC-FMCW, DPSK, trajectory forecasting, link lifetime, packet scheduling.

## I. Introduction

Laser-headlamp platforms are attractive for integrated sensing, communication, and illumination because a shared optical front end can support precise ranging, high-rate data transfer, and road lighting. The supplied PC-FMCW study establishes a physical-layer premise with phase-coded FMCW signaling and 1 Gbit/s DPSK communication. That work does not, however, define a mobility-aware packet scheduler.

Vehicular optical links are highly geometry dependent. A receiver may have a strong current SNR but approach the field-of-view boundary, while another receiver with lower instantaneous quality may remain connected for longer. A scheduler that observes only the current link can therefore miss a limited opportunity to deliver deadline-sensitive packets before a predictable disconnection.

This work asks a focused question: **does causal knowledge of future vehicle motion improve packet/resource scheduling for a PC-FMCW/DPSK vehicular optical link?** The decision variable is the receiver served in the current slot. Beam selection, top-K beam prediction, adaptive driving-beam control, and illumination optimization are outside the scope of this paper.

The main contributions are:

- a causal trajectory-to-link transformation producing future range, bearing, SNR, BER, PER, outage, and link lifetime;
- a packet-level simulator with common random arrivals, deadlines, retransmission failures, and conservation checks;
- an interpretable link-lifetime scheduler that prioritizes packets at risk of becoming undeliverable;
- classical and perfect-future references, channel/traffic/noise ablations, paired statistics, and deterministic paper artifacts;
- a precise separation between real mobility input and model-based optical communication output.

![Compact WOMD mobility example](artifacts/paper_run/figures/example_womd_motion.png)

*Fig. 1. Example compact-WOMD mobility scene. Solid markers are causal history and dashed square markers are the hidden future used only for evaluation. The proxy ego is required because the supplied compact export omits the official SDC identifier.*

## II. System and Problem Formulation

At scheduling slot *t*, the system observes the ego and candidate-receiver positions up to *t*, together with packet queues and deadlines. A deployable predictor estimates positions for the next *H* steps. The predicted absolute trajectories are transformed into ego-relative range and bearing. The link mapper then estimates received power, SNR, BER, PER, goodput, outage, and the first future outage step.

For target *i* and future step *k*, range is the Euclidean distance between predicted target and ego positions. Bearing is the wrapped angle of their relative displacement after subtracting predicted ego heading. No ground-truth sample after *t* enters a deployable forecast. A future-mutation test changes all hidden future positions and requires each causal predictor to return bit-identical outputs.

The scheduler chooses at most one receiver per slot. Its information includes current realized link quality, predicted future link traces, predicted link lifetime, queue length, earliest deadline, accumulated delivered bits, and the previously selected receiver. Packet outcomes are evaluated using the ground-truth-trajectory-derived link rather than the predicted link. This prevents self-confirming evaluation bias.

## III. PC-FMCW/DPSK Link Abstraction

The supplied physical-layer paper uses a 193.4 THz optical carrier, 10 GHz chirp bandwidth, 10 microsecond chirp period, and 1 Gbit/s DPSK data rate. The present repository preserves the data-rate premise while using a transparent reference-SNR link abstraction because an experimentally calibrated vehicle-to-vehicle optical budget is not included in the supplied archives.

Relative optical gain combines inverse footprint area, atmospheric attenuation, Gaussian pointing loss, and a hard receiver field of view. SNR is calibrated at a declared reference distance and on-axis bearing. Differential binary PSK BER in AWGN is modeled as one half times the exponential of negative linear SNR. For an *L*-bit packet, PER is one minus the probability that all *L* bits are correct. Effective goodput equals the nominal bit rate multiplied by one minus PER.

Three channel levels are evaluated: range only; range with pointing and field of view; and the full model including atmospheric attenuation. BER can come from the analytical expression or a fixed-seed Monte Carlo lookup table. These alternatives expose how much of a scheduling result depends on channel-model fidelity.

## IV. Motion and Link Prediction

The baseline set contains Last Position, Constant Velocity, a position-only constant-velocity Kalman filter, a causal CV/CA interacting-multiple-model approximation, Constant Acceleration, and a perfect-future reference. An optional GRU path is provided for trajectory-only or communication-aware training. The communication-aware loss augments trajectory error with log-SNR error and a differentiable outage-classification loss.

Motion accuracy is reported with ADE and FDE. Communication relevance is reported with range MAE, SNR MAE, outage F1, outage AUROC, and absolute link-lifetime error. This distinction is necessary because a small geometric error can cross the optical field-of-view boundary and create a large link-state error.

| Predictor | Synthetic ADE (m) | Synthetic SNR MAE (dB) | WOMD-proxy ADE (m) | WOMD lifetime error (steps) |
|---|---:|---:|---:|---:|
| Last position | 1.711 | 0.698 | 1.903 | 1.167 |
| Constant velocity | 0.087 | 0.031 | 0.097 | 0.560 |
| Kalman CV | 0.352 | 0.083 | 0.328 | 0.260 |
| IMM | 0.176 | 0.053 | 0.212 | 0.540 |
| Constant acceleration | 0.028 | 0.002 | 0.196 | 0.507 |
| Perfect future | 0.000 | 0.000 | 0.000 | 0.000 |

![Forecast communication tradeoff](artifacts/paper_run/figures/forecast_link_tradeoff.png)

*Fig. 2. Trajectory ADE and derived SNR MAE. The ranking depends on the motion regime, motivating communication-aware evaluation rather than ADE-only selection.*

## V. Traffic and Scheduling

Each receiver has a bounded FIFO queue. The simulator supports Poisson, periodic, and two-state Markov-modulated arrivals. Packets receive absolute deadlines with configurable jitter. All policies in a paired episode receive identical arrivals, deadlines, and per-attempt packet-success random numbers.

Reactive baselines include Random, Round Robin, current-link Greedy, and Proportional Fair. Predictive references apply CV, Kalman, IMM, or constant-acceleration forecasts to a finite-horizon utility. The proposed link-lifetime policy adds urgency when a currently usable link is predicted to expire within the horizon. The urgency increases with queue occupancy and with decreasing predicted lifetime. A perfect-future oracle uses the same lifetime-aware objective with ground-truth future motion; it is an information reference, not a proof of globally optimal offline scheduling.

Primary packet metrics are successfully delivered goodput, packet-delivery ratio, scheduled outage, mean and P95 latency, deadline-miss ratio, Jain fairness, delivered-before-expiry ratio, and packets left in the queue at first disconnection.

## VI. Experimental Protocol

The controlled benchmark uses 12 episodes, five receivers, 120 scheduling slots, 100 ms slots, a one-second forecast horizon, and a normalized Poisson load of 0.72. The declared paper matrix sweeps horizons from 0.1 to 3 s, 3/5/10 receivers, loads from 0.3 to 1.1, 50/100/200 ms slots, and five seeds. The quick integration run selects the first two values of each axis and produces 224 policy rows. The full declared matrix produces 11,340 rows.

The compact real-motion adapter contains 56 actor records in three WOMD scenario IDs. Each actor has one second of history and one second of future at 10 Hz. Because the export omits the official SDC identifier, a deterministic current-position medoid is used as proxy ego. These results are labeled real mobility with proxy geometry and model-based communication.

Every result is paired by scenario/configuration and seed. Reports include bootstrap confidence intervals, a paired t-test, a Wilcoxon signed-rank test, win fraction, and Cohen's paired effect size. The quick-matrix tests are diagnostic because configurations share only two seeds; the full matrix is required for final inference.

## VII. Results

| Policy | Goodput (Mbps) | PDR | Delivered before expiry | Left at disconnect |
|---|---:|---:|---:|---:|
| Reactive Greedy | 2.293 | 0.644 | 0.801 | 49.75 |
| CV Predictive | 2.303 | 0.647 | 0.801 | 57.67 |
| Kalman Predictive | 2.300 | 0.646 | 0.800 | 57.25 |
| IMM Predictive | 2.301 | 0.646 | 0.800 | 57.25 |
| Predictive Utility | 2.304 | 0.647 | 0.801 | 57.67 |
| Link Lifetime | 2.336 | 0.656 | 0.814 | 51.92 |
| Perfect-Future Oracle | 2.339 | 0.657 | 0.815 | 51.92 |

The controlled default result shows that generic trajectory prediction alone produces little communication gain. The proposed lifetime term changes the scheduling objective and increases goodput by about 1.9%, while approaching the perfect-future information reference.

![Scheduling ablation](artifacts/paper_run/figures/scheduler_ablation.png)

*Fig. 3. Controlled quick ablation. Link-lifetime urgency, rather than generic future-rate utility alone, explains most of the observed gain.*

In the quick paired matrix, link lifetime gains 0.0427 Mbps on average over Reactive Greedy, with bootstrap 95% interval [0.0229, 0.0629] Mbps, an 87.5% win fraction, Wilcoxon p=0.00009, and paired Cohen effect size 0.725. CV, Kalman, IMM, and generic predictive utility have intervals crossing zero. The perfect-future lifetime reference has a similar 0.0423 Mbps mean gain.

![Communication performance versus load](artifacts/paper_run/experiment_matrix/communication_metrics_vs_load.png)

*Fig. 4. Goodput and P95 latency versus offered load for the quick matrix. These plots are integration evidence and must be regenerated using the complete axis set for submission.*

History-measurement noise is more harmful than direct future perturbation in the controlled generator. At 0.5, 1, and 2 m history noise, link-lifetime goodput falls to 1.887, 1.798, and 1.759 Mbps from 2.238 Mbps. Range-only and range-plus-pointing channels yield about 3.16 Mbps, confirming that atmospheric and field-of-view assumptions materially affect the operating region.

![Robustness to uncertainty](artifacts/paper_run/figures/robustness_noise.png)

*Fig. 5. Goodput sensitivity to causal history noise and direct forecast degradation.*

On the compact three-scene WOMD proxy benchmark, Reactive Greedy obtains 1.160 Mbps while Link Lifetime obtains 1.056 Mbps and the perfect-future reference obtains 0.976 Mbps. This negative result means the compact sample does not provide a scheduling opportunity that the current heuristic exploits. It must not be presented as evidence of generalization.

## VIII. Limitations and Claim Boundaries

WOMD supplies real vehicle motion, not measured PC-FMCW optical communication. All power, SNR, BER, PER, packet, and queue outcomes are simulated from declared assumptions. The compact export is inadequate for final validation because it contains only three scenarios, one second of future, no true SDC identity, no map context, and no upstream trained checkpoint.

The communication-aware GRU code is complete, but PyTorch and the intended full training data/checkpoint are optional external inputs and were not available in the executed environment. No learned-model result is fabricated. The full 11,340-row experiment matrix has also not been executed in this quick draft.

The next empirical gate is therefore to freeze the exact official WOMD release and scenario splits, use true SDC coordinates, provide or retrain the upstream predictor, execute trajectory-only versus communication-aware objectives, run the full matrix, and replace provisional physical constants if a measured Part-A calibration becomes available.

## IX. Conclusion

This work converts causal motion forecasting into an explicit communication-control mechanism for PC-FMCW/DPSK vehicular optical links. The complete pipeline evaluates future link quality and lifetime, simulates packet queues and deadlines, and schedules transmissions before predictable disconnection. Controlled results isolate a gain from lifetime-aware urgency rather than from generic trajectory prediction alone. The compact WOMD proxy result is negative and is reported as such. The repository is ready for reproducible full-data experimentation, while the empirical paper claim remains conditional on official WOMD inputs and frozen trained models.

## References

[1] S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, “Phase-coded FMCW Laser Headlamp for Integrated Sensing, Communication, and Illumination,” IEEE Photonics Technology Letters, accepted manuscript, DOI: 10.1109/LPT.2025.3649597, 2025.

[2] S. Ettinger et al., “Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset,” Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

[3] Project documentation, methodology, experiment protocol, data provenance, and non-Joint PDF requirements traceability in this repository, 2026.
