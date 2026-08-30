# Methodology

## 1. Scope and decision variable

The system schedules at most one candidate receiver in each communication
slot. Beam-index selection, Adaptive Driving Beam control, illumination quality
and the separate Joint project are outside the optimization problem.

The executed chain is:

```text
causal position history
→ future trajectory
→ ego-relative range and bearing
→ normalized optical link
→ SNR, BER, PER, goodput and outage
→ queue/deadline-aware scheduling
→ packet outcome on the ground-truth-derived current link
```

## 2. Causal motion forecasting

At decision time `t`, every deployable predictor receives samples only through
`t`. For target `i` and look-ahead step `k`, it estimates

```math
\hat{\mathbf p}_{i,t+k}.
```

The deployable baselines are Last Position, Constant Velocity, Constant
Acceleration, position-only Kalman CV and a lightweight CV/CA IMM. A learned GRU
adapter is optional. The Oracle reads the hidden future and is used only as an
information reference.

Tests mutate every state after `t` and require all deployable forecasts to
remain unchanged. Forecasts are truncated uniformly when fewer than `H` future
states exist; no policy receives repeated clamped tail samples.

## 3. Frames, heading and relative geometry

Range and bearing are

```math
d_{i,k}=\|\hat{\mathbf p}_{i,k}-\hat{\mathbf p}_{e,k}\|_2,
```

```math
\phi_{i,k}=\operatorname{wrap}\left[
\operatorname{atan2}(\Delta y_{i,k},\Delta x_{i,k})-\psi_{e,k}
\right].
```

Stationary ego samples reuse the last valid heading rather than resetting to
zero. The communication-aware training loss receives explicit ego heading, so
its link geometry is rotation-consistent with scheduler evaluation.

## 4. PC-FMCW/DPSK-informed link abstraction

The frozen Part-A assumptions are 193.4 THz carrier, 10 GHz chirp bandwidth,
10 μs chirp duration and 1 Gbit/s data rate. Their notebook hash and upstream
commit are recorded in `configs/part_a_physical_layer.json`.

The available inputs do not contain a measured end-to-end vehicle optical link
budget. The implementation therefore uses a reference-SNR abstraction:

```math
h(d,\phi) \propto
\frac{1}{\pi[d\tan(\theta_b)]^2}
\exp(-\kappa d)
\exp\left[-\frac{1}{2}\left(\frac{\phi}{\sigma_\phi}\right)^2\right]
\mathbf 1(|\phi|\leq \Phi_{FOV}/2),
```

```math
\gamma(d,\phi)=\gamma_{ref}\frac{h(d,\phi)}{h(d_{ref},0)}.
```

The reported power is normalized to a declared reference and carries
`received_power_calibrated=false`. It is not a measured watt-level claim.

For analytical DBPSK in AWGN,

```math
P_b=\frac{1}{2}e^{-\gamma},\qquad
PER=1-(1-P_b)^L,\qquad
G=R_b(1-PER).
```

The primary Monte Carlo LUT follows the supplied Part-A implementation: it
constructs the PC-FMCW chirp, detects the carrier using the receiver FFT,
applies parabolic sub-bin refinement and rotation compensation, and then makes
differential phase decisions. Its horizontal axis is explicitly
waveform-sample SNR rather than silently relabeled `Eb/N0`. Adaptive bit counts,
an error target and one-sided Wilson 95% upper bounds handle zero-error points;
a conservative monotone envelope is used by the packet model. The generated
grid covers −5 to 25 dB. The analytical DBPSK path remains an explicit
ablation.

Outage is explicitly selectable as BER-, PER- or goodput-based. All three flags
are stored, even when only one drives the scheduler.

## 5. Traffic and packet simulation

The simulator supports Poisson, periodic, Markov-modulated and saturated
traffic. It can use one best-effort class or an urgent/bulk mixture with
separate physical deadlines and class-level PDR/miss metrics. Deadline slots
are derived only after the slot duration is selected. Physical episode duration
is also fixed in seconds, preventing slot-size sweeps from changing the
simulated time.

All policies in a paired episode share arrivals, deadlines and packet-success
uniform random values. Failed transmissions return packets to the FIFO queue.
Delivered, deadline-dropped, overflow-dropped and remaining packets must sum to
generated packets.

Packets still queued at episode end are reported as censored. Latency is
interpreted jointly with PDR and censoring because it is defined only for
delivered packets.

## 6. Scheduling policies

- Random and Round Robin provide lower/reference baselines.
- Reactive Greedy uses current goodput, outage and queue state.
- Proportional Fair uses current normalized rate divided by historical
  normalized service per elapsed slot.
- CV, Kalman and IMM Predictive policies apply the same future-link utility to
  different causal forecasts.
- Predictive Utility combines discounted future goodput, outage risk, queue,
  deadline, fairness, opportunity and switching terms.
- Link Lifetime adds a dimensionless urgency term for a currently usable link:

```math
U_i^{life}=w_{life}\,\tilde Q_i\,
\max\left(0,1-\frac{T_i^{link}}{H}\right).
```

- Oracle uses perfect future positions with the same heuristic utility. It is
  not a globally optimal offline schedule.

The explicit lifetime term is zero when no outage lies inside the forecast
horizon. The corrected default ablation records this null behavior instead of
claiming an effect that did not occur.

## 7. Sensing uncertainty

The default assumes perfect observed positions. Robustness studies can use:

- IID Cartesian target-position noise;
- range/bearing noise with range-dependent radial variance;
- temporally correlated AR(1) errors;
- an isotropic-equivalent measurement covariance for the Kalman baseline.

The ego pose remains exact in these tests. These are declared synthetic
assumptions, not measured PC-FMCW sensor specifications.

## 8. Communication-aware learning

The trainable objective is

```math
\mathcal L=\lambda_{traj}\mathcal L_{traj}
+\lambda_{link}\mathcal L_{link}
+\lambda_{out}\mathcal L_{out}.
```

The link term compares log-SNR in the correct ego-heading frame. The outage
term uses a smooth FoV surrogate during training and exact hard-FoV evaluation.
Four pre-registered modes are supported: trajectory-only, trajectory+link,
trajectory+outage and full. The multi-seed runner requires at least three seeds.

Every local checkpoint stores a versioned feature schema, dataset SHA-256,
scenario-safe split metadata and objective. An upstream checkpoint without the
matching schema is rejected rather than silently adapted.

## 9. Metrics and inference

Motion/link metrics include ADE, FDE, range MAE, bearing MAE, SNR MAE, outage
F1/AUROC with positive support, and link-lifetime error in steps and seconds.

Communication metrics include goodput, PDR, availability and scheduled outage,
mean/P50/P95/P99 latency, deadline miss, censoring, delivered before expiry,
queue at disconnection, raw and demand-normalized Jain fairness, scheduled
SNR/BER/PER/relative power and switching count.

Policy comparisons are paired and direction-aware. Resampling and tests operate
at independent scenario/seed-cluster level, not at every correlated Cartesian
row. Wilcoxon families receive Holm correction. The completed staged study uses
five seeds and 1,125 policy episodes; its rank tests still have limited
resolution and are not substituted for official-WOMD held-out inference.
