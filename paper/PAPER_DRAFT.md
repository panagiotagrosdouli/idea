# When Does Trajectory Prediction Help PC-FMCW/DPSK Vehicular Optical Scheduling?

**Manuscript status:** corrected reproducible research draft; authors,
affiliations and target venue remain to be inserted.

## Abstract

Phase-coded frequency-modulated continuous-wave (PC-FMCW) laser headlamps can
combine sensing, illumination and differential phase-shift-keying communication,
but the corresponding optical link is strongly geometry dependent. This work
studies a non-Joint communication-control question: whether a scheduler should
serve a receiver now because causal trajectory prediction indicates that its
link will soon become unusable. We implement a packet-level pipeline from
observed motion to trajectory forecasts, ego-relative range and bearing,
normalized optical gain, SNR, DBPSK bit and packet errors, link lifetime, queues,
deadlines and receiver scheduling. Frame consistency, stationary headings,
end-of-record horizons, physical time units, censoring and clustered statistics
are covered by 44 regression tests and five scientific gates. In a corrected
12-episode controlled benchmark, Link-Lifetime scheduling changes mean goodput
from 2.293 to 2.307 Mbps relative to Reactive Greedy, but the paired bootstrap
95% interval [-0.0319, 0.0593] Mbps includes zero and Holm-adjusted Wilcoxon
`p=1.0`. Its P95 latency is 269 ms worse on average. A three-scene compact WOMD
proxy benchmark is also negative: 1.056 versus 1.160 Mbps. A 430-row two-seed
staged diagnostic shows operating-region sign changes but is insufficient for
inference. The current evidence therefore rejects a universal prediction-gain
claim and motivates a narrower question: under which link-closure, traffic and
uncertainty regimes does future mobility information have communication value?
Official WOMD shards and a compatible learned checkpoint are still required for
a submission-quality empirical answer.

**Index terms:** predictive communication, vehicular optical communication,
PC-FMCW, DPSK, trajectory forecasting, link lifetime, packet scheduling.

## I. Introduction

Integrated sensing, communication and illumination can reuse an optical vehicle
front end for ranging, lighting and high-rate data transmission. The supplied
Part-A PC-FMCW study provides a physical-layer premise with a 193.4 THz optical
carrier, 10 GHz chirp bandwidth, 10 μs chirp duration and 1 Gbit/s DPSK data
rate. It does not define a mobility-aware packet scheduler.

A current-link scheduler sees only the instantaneous communication opportunity.
If one receiver is approaching an optical field-of-view boundary, packets left
in its queue may soon become undeliverable. A predictor could expose that
closing window. Prediction can also harm scheduling: trajectory error can cross
a hard optical boundary, an urgency heuristic can sacrifice low-latency service,
and traffic saturation can eliminate scheduling flexibility.

This paper asks: **when does causal future-motion information improve receiver
scheduling for a PC-FMCW/DPSK-informed optical link?** The decision is the
receiver served in the current slot. Beam-index selection, adaptive driving-beam
control and illumination optimization are explicitly excluded.

The contributions are:

- an auditable causal motion-to-link-to-packet pipeline;
- frame- and time-consistent link-aware prediction and scheduling;
- classical, reactive, predictive and perfect-future information references;
- explicit BER/PER/goodput outage semantics and normalized-power boundaries;
- packet censoring, scenario slices and cluster-aware paired inference;
- a true-SDC official-WOMD adapter and versioned learned-checkpoint contract;
- corrected controlled and compact-proxy results, including null and negative
  findings.

## II. System Model

At slot `t`, the system observes ego and target positions through `t`, packet
queues, deadlines and past service. A deployable predictor estimates the next
`H` positions. Predicted target and ego trajectories are converted to range
`d(i,k) = ||p_hat(i,t+k) - p_hat(e,t+k)||_2` and wrapped ego-relative bearing
`phi(i,k) = wrap[atan2(delta_y, delta_x) - psi(e,t+k)]`.

Stationary ego samples retain their last valid heading. Unavailable future
steps are truncated for every policy. A deployable predictor never receives a
ground-truth sample after `t`; future-mutation tests enforce this invariant.

The scheduler selects at most one target. Its forecast may influence the
decision, but packet success is sampled from the current ground-truth-derived
link. This separates decision information from evaluation and prevents
self-confirming results.

## III. PC-FMCW/DPSK-Informed Link

Because the supplied files contain no calibrated vehicle-to-vehicle optical
budget, the model is anchored by a declared reference SNR rather than absolute
measurement. Relative gain combines inverse footprint area, atmospheric loss,
Gaussian pointing loss and a hard field of view. In compact notation,
`h(d,phi)` is proportional to inverse footprint area multiplied by
`exp(-kappa*d)`, Gaussian pointing loss and the indicator
`1(|phi| <= FOV/2)`.

The SNR is `gamma_ref h(d,phi)/h(d_ref,0)`. The power output is a normalized
reference quantity with `received_power_calibrated=false`.

Analytical DBPSK uses `P_b=0.5 exp(-gamma)`. For `L` packet bits,
`PER=1-(1-P_b)^L`, and successful goodput is `R_b(1-PER)`. A second BER path
uses fixed-seed symbol-level Monte Carlo from -5 to 25 dB with adaptive bit
counts and a Wilson 95% upper bound when no error is observed.

BER, PER and minimum-goodput outage flags are all stored. The configured mode
determines link lifetime, defined as the first predicted outage step, or the
censored horizon if no outage is predicted.

## IV. Motion, Sensing and Learning

The deterministic baselines are Last Position, Constant Velocity, Constant
Acceleration, a position-only Kalman CV filter and a lightweight causal CV/CA
IMM. Perfect future motion is an information reference.

The optional GRU can be trained under four objectives: trajectory-only,
trajectory+link, trajectory+outage and full. The link loss receives explicit ego
heading; the outage loss combines a smooth field-of-view surrogate for gradient
flow with exact hard-boundary evaluation. Each local checkpoint records its
feature schema, dataset hash, split and seed. The upstream Stage-4 reports use a
different eight-feature interface and no compatible checkpoint was supplied.

Robustness studies distinguish future forecast degradation from observation
uncertainty. The declared sensing model supports IID Cartesian noise and
range/bearing noise with range-dependent radial variance and AR(1) temporal
correlation. The Kalman path can use an isotropic-equivalent covariance. These
are simulation assumptions, not measured sensor specifications.

| Predictor | Synthetic ADE (m) | Synthetic SNR MAE (dB) | Proxy ADE (m) | Proxy lifetime MAE (s) |
|---|---:|---:|---:|---:|
| Last Position | 1.711 | 0.698 | 1.903 | 0.117 |
| Constant Velocity | 0.087 | 0.031 | 0.097 | 0.056 |
| Kalman CV | 0.352 | 0.083 | 0.328 | 0.026 |
| IMM | 0.176 | 0.053 | 0.212 | 0.054 |
| Constant Acceleration | 0.028 | 0.002 | 0.196 | 0.051 |
| Perfect Future | 0.000 | 0.000 | 0.000 | 0.000 |

![Forecast trade-off](artifacts/corrected_v1/figures/forecast_link_tradeoff.png)

*Fig. 1. Motion ADE and derived-link SNR error on controlled trajectories. Link
metrics are evaluated separately because a small position error can cross a hard
FoV boundary.*

## V. Traffic and Scheduling

Each target has a bounded FIFO queue. Poisson, periodic, Markov-modulated and
saturated arrivals are supported. Deadlines and episode duration are expressed
in physical seconds and converted only after slot duration is chosen. Failed
packets return to the queue. Packets remaining at termination are censored and
reported explicitly.

Reactive references are Random, Round Robin, Reactive Greedy and Proportional
Fair. Predictive policies apply CV, Kalman, IMM or constant-acceleration
forecasts to a utility containing discounted future goodput, outage, queue,
deadline, fairness, opportunity and switching terms.

Link Lifetime adds
`U_life(i) = w_life * normalized_queue(i) * max[0, 1 - T_link(i)/H]`
only when the current link is usable. The term is dimensionless and invariant
to proportional rescaling of horizon steps. The Oracle receives perfect future
positions but uses the same heuristic; it is not a global scheduling optimum.

## VI. Experimental Protocol

The controlled benchmark has 12 independent 12 s episodes, five receivers,
100 ms slots, 1 s horizon, load 0.72 and packet deadlines of 1.2 +/- 0.4 s. All
policies share motion seeds, arrivals, deadlines and per-attempt random draws.

The compact export contains 56 actors and three WOMD scenario IDs with one
second of history and one second of future. Because it omits the official SDC,
the adapter uses a deterministic medoid proxy ego. The communication link is
simulated.

A staged design changes one axis around the reference: load, physical horizon,
slot duration, receiver count, traffic model, packet size, deadline, reference
SNR, FoV, outage mode or sensing model. Its quick run uses two seeds and is
diagnostic. Final inference requires five seeds and official held-out scenes.

Metrics include goodput, PDR, availability and scheduled outage, P50/P95/P99
latency, deadline miss, censoring, delivered-before-expiry, queue at first
disconnection, demand-normalized fairness, and scheduled SNR/BER/PER. Paired
statistics operate at independent scenario/seed level, use metric-specific
direction and apply Holm correction.

## VII. Corrected Results

| Policy | Goodput (Mbps) | PDR | P95 latency (ms) | Deadline+censored | Demand Jain |
|---|---:|---:|---:|---:|---:|
| Reactive Greedy | 2.2930 | 0.6439 | 699.6 | 0.3561 | 0.6717 |
| CV Predictive | 2.3053 | 0.6473 | 948.7 | 0.3527 | 0.6962 |
| Kalman Predictive | 2.3047 | 0.6472 | 933.3 | 0.3528 | 0.6948 |
| IMM Predictive | 2.3073 | 0.6479 | 948.7 | 0.3521 | 0.6949 |
| Predictive Utility | 2.3063 | 0.6476 | 965.4 | 0.3524 | 0.6972 |
| Link Lifetime | 2.3070 | 0.6478 | 968.3 | 0.3522 | 0.6993 |
| Oracle Information | 2.3078 | 0.6480 | 968.3 | 0.3520 | 0.7000 |

![Corrected benchmark](artifacts/corrected_v1/figures/corrected_benchmark_tradeoff.png)

*Fig. 2. Corrected controlled benchmark. Predictive policies show slightly
higher mean goodput and fairness but substantially worse P95 latency.*

Link Lifetime minus Reactive goodput is +0.0140 Mbps with paired bootstrap 95%
interval [-0.0319, 0.0593], win fraction 66.7%, Cohen `dz=0.161`, Wilcoxon
`p=0.677` and Holm-adjusted `p=1.0`. The data do not establish a goodput gain.

The P95 difference is +268.8 ms, unfavorable. Its paired interval is entirely
positive; Holm-adjusted Wilcoxon `p=0.00439`. Thus the clearest controlled
effect is a latency cost, not a throughput improvement.

![Paired gain ECDF](artifacts/corrected_v1/figures/paired_goodput_difference_ecdf.png)

*Fig. 3. Episode-level Link-Lifetime minus Reactive goodput differences include
both positive and negative outcomes.*

In the quick lifetime ablation, Predictive Utility, Link Lifetime and Link
Lifetime with zero lifetime weight all obtain 2.1655 Mbps. The explicit lifetime
term is inactive in those default episodes because it does not change a
decision inside the available closure horizon. This null result prevents the
earlier interpretation that lifetime urgency explained a default gain.

The compact WOMD proxy result is also negative: Reactive obtains 1.160 Mbps and
0.329 PDR, while Link Lifetime obtains 1.056 Mbps and 0.299 PDR. Approximately
70% of packets are deadline-dropped or censored in the one-second evaluation
window. The three scenes cannot establish generalization.

![Compact WOMD example](artifacts/corrected_v1/figures/example_womd_motion.png)

*Fig. 4. Compact real-motion example. The proxy ego is an integration device,
not the official SDC identity.*

The 430-row staged quick run contains 43 settings, two seeds and five policies.
The Link-Lifetime minus Reactive difference changes sign across axes - for
example, +0.018 Mbps at load 0.3 and -0.172 Mbps at load 0.9. Such changes are
hypothesis-generating only because two seeds do not identify an effect.

![Staged operating region](artifacts/corrected_v1/figures/staged_operating_region_diagnostic.png)

*Fig. 5. Two-seed staged diagnostic. Shading spans the two seed outcomes, not a
publication confidence interval.*

## VIII. Limitations

WOMD provides motion, not optical measurements. Every link and packet outcome
is model-based. The normalized power reference cannot support an absolute watt
claim. The compact export lacks true SDC identity, raw validity masks, map
context and a long evaluation window.

The official-WOMD adapter and learned ablation runner are implemented, but raw
WOMD shards, a compatible checkpoint and PyTorch were unavailable in the
executed environment. No learned result is reported. The supplied Stage-4 JSON
reports are provenance rather than locally reproduced evidence.

The Oracle is only an information reference and can be worse than Reactive
because the heuristic is not an exact offline optimum. A small-instance exact
optimizer remains future work. Channel parameters, traffic and scheduler
weights require development-set freezing before final testing.

## IX. Conclusion

The corrected codebase demonstrates how causal mobility forecasts can be
translated into optical-link forecasts and packet scheduling decisions. It also
shows why a favorable controlled result cannot be assumed: the mean goodput
difference is uncertain, tail latency is worse, the explicit lifetime ablation
is null at the default point and the compact proxy benchmark is negative.

The scientifically useful next question is conditional rather than universal:
which link-closure geometries, deadline pressures and uncertainty levels create
enough scheduling flexibility for prediction to help? Answering it requires the
official WOMD held-out evaluation and a real multi-seed learned-model ablation.

## References

[1] S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, "Phase-coded FMCW Laser
Headlamp for Integrated Sensing, Communication, and Illumination," IEEE
Photonics Technology Letters, DOI: 10.1109/LPT.2025.3649597, 2025.

[2] S. Ettinger et al., "Large Scale Interactive Motion Forecasting for
Autonomous Driving: The Waymo Open Motion Dataset," Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021.

[3] Project methodology, data provenance, experiment protocol and corrected
machine-readable artifacts in this repository, 2026.
