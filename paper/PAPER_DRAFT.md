# When Does Trajectory Prediction Help PC-FMCW/DPSK Vehicular Optical Scheduling?

**Manuscript status:** reproducible research draft. Authors, affiliations,
target venue, official WOMD results and learned-model results remain to be
inserted before submission.

## Abstract

Phase-coded frequency-modulated continuous-wave (PC-FMCW) laser headlamps can
combine sensing, illumination and differential phase-shift-keying (DPSK)
communication, but their optical link is strongly geometry dependent. This
work studies whether a packet scheduler should serve a receiver now because a
causal trajectory forecast indicates that its link will soon disappear. We
implement an auditable packet-level path from observed motion to future
trajectory, ego-relative range and bearing, normalized optical gain, SNR,
DPSK BER, packet error, link lifetime, queues, deadlines and receiver
scheduling. The BER calibration reproduces the supplied Part-A FFT-carrier and
differential receiver with confidence-limited Monte Carlo estimates. Fifty-one
regression tests and five scientific gates protect causality, coordinate/time
consistency, queue conservation and monotonicity. In 12 controlled episodes,
Link-Lifetime scheduling changes mean goodput from 2.293 to 2.307 Mbps versus
Reactive Greedy, but the paired bootstrap 95% interval [-0.0319, 0.0593] Mbps
includes zero and Holm-adjusted Wilcoxon `p=1.0`; P95 latency is 269 ms worse.
A compact three-scene WOMD proxy is also negative. A full 1,125-row, five-seed
staged study reveals positive and negative operating regions, but no cited
family survives Holm correction. In urgent/bulk traffic, Link Lifetime harms
urgent PDR because link urgency is not packet-class urgency. The evidence
rejects a universal prediction-gain claim and supports a conditional research
question. Official WOMD shards and a compatible learned checkpoint remain
necessary for submission-quality empirical evidence.

**Index terms:** predictive communication, vehicular optical communication,
PC-FMCW, DPSK, trajectory forecasting, link lifetime, packet scheduling.

## I. Introduction

Integrated sensing, communication and illumination can reuse an optical
vehicle front end for ranging, lighting and data transmission. The supplied
Part-A PC-FMCW study provides the physical-layer premise: a 193.4 THz optical
carrier, 10 GHz chirp bandwidth, 10 us chirp duration and 1 Gbit/s DPSK data
rate. It does not define a mobility-aware packet scheduler.

A current-link scheduler sees only the instantaneous opportunity. If a
receiver is approaching the optical field-of-view boundary, packets left in
its queue may soon become undeliverable. A predictor could expose that closing
window. Prediction can also hurt: position error can cross a hard boundary, a
link-urgency rule can conflict with packet deadlines, and saturation can remove
the flexibility needed to prefetch traffic.

This paper asks: **under which conditions does causal future-motion information
improve receiver scheduling for a PC-FMCW/DPSK-informed optical link?** The
decision variable is the receiver served in the current slot. Beam-index
selection, Adaptive Driving Beam control and illumination optimization belong
to the separate Joint project and are excluded.

The contributions are:

- a causal trajectory-to-link-to-packet implementation with ground-truth link
  realization separated from the forecast used for decision-making;
- a notebook-derived Part-A FFT/DPSK receiver LUT with explicit SNR semantics
  and confidence handling;
- classical motion baselines, ten scheduling policies, urgent/bulk traffic,
  censoring and boundary-sensitive communication metrics;
- a true-SDC official-WOMD adapter and a versioned communication-aware GRU
  training/checkpoint contract;
- controlled, compact-proxy and five-seed staged results that preserve null and
  negative findings.

![System architecture](artifacts/corrected_v2/figures/system_architecture.png)

*Fig. 1. Executed system boundary. Forecasts inform decisions; current
ground-truth-derived links determine realized packet outcomes.*

## II. System Model and Causality

At slot `t`, the system observes ego and target positions through `t`, packet
queues, deadlines and past service. A deployable predictor estimates the next
`H` target positions. For target `i` and horizon step `k`, predicted range and
ego-relative bearing are

`d(i,k)=||p_hat(i,t+k)-p_hat(e,t+k)||_2`,

`phi(i,k)=wrap[atan2(delta_y,delta_x)-psi(e,t+k)]`.

Stationary ego samples retain the last valid heading. When a record contains
fewer than `H` future samples, every method is truncated consistently. No
deployable policy receives repeated hidden tail samples. Tests mutate the
entire future after `t` and require deployable outputs to remain invariant.
The perfect-future Oracle must change under the same mutation. At `H=0`, future
arrays cannot change Reactive Greedy's decision.

## III. Part-A PC-FMCW/DPSK Link Abstraction

The supplied inputs contain no calibrated end-to-end vehicle optical budget.
Relative gain therefore combines inverse footprint area, atmospheric loss,
Gaussian pointing loss and a hard field of view, normalized at a declared
reference distance and bearing. SNR is `gamma_ref*h(d,phi)/h(d_ref,0)`. The
reported power is explicitly relative and carries
`received_power_calibrated=false`.

The analytical DBPSK ablation uses `P_b=0.5*exp(-gamma)`. For packet length
`L`, `PER=1-(1-P_b)^L`, and successful goodput is `R_b*(1-PER)`.

The primary LUT reproduces the Part-A receiver sequence: PC-FMCW chirp
construction, FFT carrier-bin extraction, parabolic sub-bin refinement,
rotation compensation and differential phase decisions. Its axis is named
**waveform-sample SNR**, not `Eb/N0`. Monte Carlo runs use adaptive bit support;
when zero errors are observed, the LUT uses a one-sided 95% Wilson upper bound
and never claims exact zero BER. A conservative monotone envelope prevents
sampling noise from making modeled BER improve in the wrong direction.

![DPSK BER calibration](artifacts/corrected_v2/figures/dpsk_ber_curve.png)

*Fig. 2. Part-A receiver BER LUT. Zero-error samples are confidence-limited.*

BER-, PER- and minimum-goodput outage flags are stored separately. The selected
mode defines link lifetime: the first predicted outage step, or a censored
horizon when no outage occurs.

## IV. Motion, Sensing and Communication-Aware Learning

The deployable baselines are Last Position, Constant Velocity, Constant
Acceleration, a position-only Kalman CV filter and a lightweight causal CV/CA
IMM. Perfect Future is an information reference. Motion evaluation reports
ADE/FDE, while link evaluation reports range/bearing/SNR error, outage
F1/AUROC and link-lifetime error.

| Predictor | Synthetic ADE (m) | Synthetic SNR MAE (dB) | Proxy ADE (m) | Proxy lifetime MAE (s) |
|---|---:|---:|---:|---:|
| Last Position | 1.711 | 0.698 | 1.903 | 0.117 |
| Constant Velocity | 0.087 | 0.031 | 0.097 | 0.056 |
| Kalman CV | 0.352 | 0.083 | 0.328 | 0.026 |
| IMM | 0.176 | 0.053 | 0.212 | 0.054 |
| Constant Acceleration | 0.028 | 0.002 | 0.196 | 0.051 |
| Perfect Future | 0.000 | 0.000 | 0.000 | 0.000 |

![Forecast trade-off](artifacts/corrected_v2/figures/forecast_link_tradeoff.png)

*Fig. 3. Geometric accuracy and derived-link accuracy are related but not
identical objectives.*

To provide an executable uncertainty baseline without inventing a missing
neural checkpoint, per-horizon isotropic residual variance is fitted for CV and
CA on six controlled scenario IDs and evaluated on six disjoint IDs. Each test
set contains 7,050 target-step samples. CV obtains RMSE 0.167 m, mean NLL
-2.859 and calibration error 0.118; CA obtains RMSE 0.078 m, mean NLL -4.522
and calibration error 0.121. A negative continuous-density NLL is possible for
narrow distributions. Both models over-cover at nominal 50% and under-cover at
95%; calibration is therefore diagnosed rather than assumed.

![Probabilistic calibration](artifacts/corrected_v2/figures/probabilistic_calibration.png)

*Fig. 4. Scenario-safe held-out coverage of classical Gaussian CV/CA residual
wrappers. These are not the missing learned Gaussian/GMM checkpoints.*

The optional GRU supports four preregistered objectives: trajectory-only,
trajectory+link, trajectory+outage and full. Link loss uses the same ego frame
as evaluation; outage loss uses a smooth FoV surrogate during optimization and
the hard boundary at test time. Every checkpoint records feature schema,
dataset SHA-256, scenario split, seed and objective. The supplied Stage-4 file
contains 70 causal predictions but no weights or paired future ground truth, so
it cannot be used as a compatible checkpoint or honest accuracy dataset.

Sensing robustness is separated from forecast degradation. Supported declared
assumptions are perfect observations, IID Cartesian error, and range/bearing
error with range-dependent radial variance and AR(1) temporal correlation.
They are simulation assumptions, not sensor measurements.

## V. Traffic and Scheduling

Each target has a bounded FIFO queue. Poisson, periodic, Markov-modulated and
saturated arrivals are supported. The simulator also supports a best-effort
class or urgent/bulk classes with distinct physical deadlines. Failed packets
return to the queue; delivered, deadline-dropped, overflow-dropped and remaining
packets must conserve total generation. Remaining packets are right-censored.

Reactive baselines are Random, Round Robin, Reactive Greedy and Proportional
Fair. Predictive policies apply CV, Kalman, IMM or constant-acceleration
forecasts to discounted future goodput, outage, queue, deadline, fairness,
opportunity and switching terms. Link Lifetime adds

`U_life(i)=w_life*Q_tilde(i)*max(0,1-T_link(i)/H)`

only when the current link is usable. Oracle uses perfect future positions but
the same heuristic utility; it is not a global offline optimum.

![Trajectory to link](artifacts/corrected_v2/figures/trajectory_to_link_trace.png)

*Fig. 5. Example mapping from real-motion proxy geometry to the modeled optical
link. Hard FoV transitions amplify small bearing errors.*

![Scheduler timeline](artifacts/corrected_v2/figures/scheduler_timeline.png)

*Fig. 6. Exact controlled-episode SNR, receiver selection and queue timeline.*

## VI. Experimental and Statistical Protocol

The controlled benchmark has 12 independent 12 s episodes, five receivers,
100 ms slots, 1 s horizon, offered load 0.72 and packet deadlines of
1.2+/-0.4 s. All policies share scenarios, arrivals, deadlines and packet
success uniform draws.

The compact export contains 56 actors and three WOMD scenario IDs with one
second of history and future. Because it omits official SDC identity, a
deterministic medoid proxy ego is used. The optical link remains simulated.

The full staged study changes one axis around the reference: load, horizon,
slot duration, receiver count, traffic model, traffic class, packet size,
deadline, reference SNR, FoV, outage definition or sensing model. It contains
45 settings, five seeds and five policies: 1,125 episode rows.

Communication metrics include goodput, PDR, scheduled/availability outage,
P50/P95/P99 latency, miss and censoring, delivered-before-expiry, queue at
disconnection, urgent/bulk PDR, demand-normalized Jain fairness and scheduled
SNR/BER/PER. Policy tests operate on paired scenario/seed clusters and receive
Holm correction. With only five pairs, even five same-direction differences
have minimum two-sided Wilcoxon `p=0.0625`, so staged results identify candidate
regions rather than final official-WOMD effects.

## VII. Controlled and Proxy Results

| Policy | Goodput (Mbps) | PDR | P95 latency (ms) | Deadline+censored | Demand Jain |
|---|---:|---:|---:|---:|---:|
| Reactive Greedy | 2.2930 | 0.6439 | 699.6 | 0.3561 | 0.6717 |
| CV Predictive | 2.3053 | 0.6473 | 948.7 | 0.3527 | 0.6962 |
| Kalman Predictive | 2.3047 | 0.6472 | 933.3 | 0.3528 | 0.6948 |
| IMM Predictive | 2.3073 | 0.6479 | 948.7 | 0.3521 | 0.6949 |
| Predictive Utility | 2.3063 | 0.6476 | 965.4 | 0.3524 | 0.6972 |
| Link Lifetime | 2.3070 | 0.6478 | 968.3 | 0.3522 | 0.6993 |
| Oracle information | 2.3078 | 0.6480 | 968.3 | 0.3520 | 0.7000 |

![Corrected benchmark](artifacts/corrected_v2/figures/corrected_benchmark_tradeoff.png)

*Fig. 7. Predictive policies slightly change mean goodput and fairness but
substantially worsen tail latency at the default point.*

Link Lifetime minus Reactive goodput is +0.0140 Mbps with paired bootstrap 95%
interval [-0.0319, 0.0593], win fraction 66.7%, Cohen `dz=0.161`, raw
Wilcoxon `p=0.677` and Holm-adjusted `p=1.0`. P95 latency is +268.8 ms worse,
with Holm-adjusted `p=0.00439`. The explicit lifetime-weight ablation is null at
this operating point because predicted closure inside the horizon does not
change additional decisions.

![Goodput ECDF](artifacts/corrected_v2/figures/policy_goodput_ecdf.png)

*Fig. 8. Policy-level goodput distributions preserve episode variability.*

In the compact real-motion proxy, Reactive obtains 1.160 Mbps and 0.329 PDR,
whereas Link Lifetime obtains 1.056 Mbps and 0.299 PDR. Roughly 70% of packets
are deadline-dropped or censored in the short window. Three proxy scenes cannot
establish generalization.

## VIII. Conditional Operating Regions and Failure Cases

| Setting | Link Lifetime - Reactive (Mbps) | Bootstrap 95% interval | Win fraction | Holm p |
|---|---:|---:|---:|---:|
| Deadline 0.5 s | +0.1392 | [+0.0920, +0.1872] | 1.0 | 0.375 |
| Reference SNR +3 dB | +0.0504 | [+0.0442, +0.0560] | 1.0 | 0.375 |
| Reference SNR +6 dB | +0.0428 | [+0.0244, +0.0612] | 1.0 | 0.375 |
| Load 1.1 | -0.1644 | [-0.2572, -0.0626] | 0.2 | 0.625 |
| Deadline 0.05 s | -0.2318 | [-0.3438, -0.1032] | 0.2 | 0.375 |
| Urgent/bulk | -0.1356 | [-0.2746, -0.0268] | 0.2 | 0.500 |

![Staged regions](artifacts/corrected_v2/figures/staged_operating_region_diagnostic.png)

*Fig. 9. Five-seed operating-region sign changes. Shading spans the seed range,
not a publication confidence band.*

Urgent/bulk traffic exposes a mechanism failure. Reactive averages 2.167 Mbps,
urgent PDR 0.497 and bulk PDR 0.676. Link Lifetime averages 2.031 Mbps, urgent
PDR 0.293 and bulk PDR 0.730. Link-closure urgency serves bulk opportunities at
the expense of short-deadline urgent packets; a class-aware utility is required.

![Forecast failures](artifacts/corrected_v2/figures/forecast_failure_cases.png)

*Fig. 10. Largest uncensored link-lifetime errors reveal optical-boundary cases
where a modest geometric error can cause a large communication error.*

![Throughput fairness](artifacts/corrected_v2/figures/throughput_fairness_pareto.png)

*Fig. 11. Throughput-fairness operating points; highlighted points are
nondominated within the evaluated heuristic set.*

## IX. Complexity and Reproducibility

| Component | Median runtime (us) | P95 runtime (us) |
|---|---:|---:|
| Last Position | 1.8 | 4.9 |
| Constant Velocity | 6.4 | 11.6 |
| Constant Acceleration | 17.5 | 33.6 |
| Kalman CV | 976.4 | 2919.7 |
| IMM | 1028.7 | 3435.2 |
| Reactive Greedy scheduler | 9.7 | 16.9 |
| Link Lifetime scheduler | 37.8 | 53.0 |

These are single-process CPU diagnostics on the recorded runtime, not hardware-
independent guarantees. The GRU architecture has 169,620 analytical parameters;
its runtime is intentionally absent because the learned execution path was not
run. Machine-readable manifests hash code/configuration and record whether
official data, checkpoints and measured channels were used.

## X. Limitations and Publication Gate

WOMD supplies motion, not optical measurements. All link and packet outcomes
are model-based, and normalized power cannot support an absolute watt claim.
The compact export lacks the true SDC, raw masks, map context and a long test
window. Although the official loader and GRU ablation code exist, the supplied
files do not include official WOMD TFRecord shards, a compatible checkpoint or
paired ground truth for the Stage-4 forecasts. Consequently no learned result,
probabilistic calibration or official-WOMD generalization is reported.

Before submission, the official shards must be hashed and split by scenario;
the four learned objectives must be trained for at least three seeds; frozen
checkpoints must be evaluated on held-out motion, boundary-link and packet
metrics; and authors, affiliations and venue formatting must be added.

## XI. Conclusion

The project demonstrates a complete causal translation from mobility forecasts
to PC-FMCW/DPSK-informed packet scheduling. It also shows why prediction cannot
be assumed beneficial: default goodput uncertainty, increased tail latency, a
negative compact proxy, deadline/load sign changes and urgent-class harm all
appear in the executed evidence. The publishable question is conditional:
which geometries, deadline pressures, channel regimes and uncertainty levels
create enough scheduling flexibility for future motion to be valuable?

## References

[1] S. Liu, T. Sun, X. Shu, J. Song, and Y. Dong, "Phase-coded FMCW Laser
Headlamp for Integrated Sensing, Communication, and Illumination," IEEE
Photonics Technology Letters, DOI: 10.1109/LPT.2025.3649597, 2025.

[2] S. Ettinger et al., "Large Scale Interactive Motion Forecasting for
Autonomous Driving: The Waymo Open Motion Dataset," ICCV, 2021.

[3] S. Shi et al., "Motion Transformer with Global Intention Localization and
Local Movement Refinement," NeurIPS, 2022.

[4] F. P. Kelly, A. K. Maulloo, and D. K. H. Tan, "Rate Control for
Communication Networks: Shadow Prices, Proportional Fairness and Stability,"
Journal of the Operational Research Society, vol. 49, no. 3, 1998.

[5] Supplied Part-A PC-FMCW/DPSK notebook, source code and research-plan
documents archived with this repository's provenance notes, 2026.

## Appendix A. Reproduction and Artifact Contract

The controlled evidence can be regenerated from a clean environment with
`make test`, `make lint`, `make corrected-full`, `make paper-draft` and
`make reproducibility`. The full run writes only to `artifacts/corrected_v2/`;
the quick integration target writes to a different directory and cannot
overwrite it.

Every policy in a paired episode receives the same motion, packet arrivals,
deadlines and per-attempt random numbers. The corrected-run manifest declares
whether official WOMD, a learned checkpoint, a measured optical channel and the
Part-A notebook receiver were used. Machine-readable outputs are retained in
CSV and JSON; the PDF reads only current corrected artifacts.

| Artifact group | Primary content |
|---|---|
| `synthetic_benchmark` | 12-scenario policy rows and paired statistics |
| `motion_baselines` | ADE/FDE and derived-link errors |
| `probabilistic` | disjoint-scenario Gaussian NLL/coverage |
| `staged_experiments` | 1,125 one-axis study rows |
| `complexity` | CPU runtime and parameter diagnostics |
| `figures` | exact publication plots generated from those rows |

## Appendix B. Open Evidence Gates

The repository intentionally fails the publication gate until all of the
following are present: hashed official WOMD TFRecords; true-SDC scenario-safe
train/development/test lists; at least three seeds for every learned objective;
a compatible versioned checkpoint; held-out motion, link-boundary and packet
evaluation; and author/venue metadata. A measured optical link is required only
for absolute-power or real-channel claims. The missing learned K=5 GMM is not
silently replaced by the classical Gaussian residual baseline.
