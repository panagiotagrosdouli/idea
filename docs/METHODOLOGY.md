# Methodology

## 1. Causal motion forecast

At decision slot `t`, a deployable predictor receives states only through `t`.
For target `i` and look-ahead step `k`, it produces

```math
\hat{\mathbf p}_{i,t+k}.
```

The future-mutation gate in `validation.py` changes every future ground-truth
position and verifies that a causal forecast remains bit-identical. Only the
oracle-information policy reads the mutated future.

## 2. Relative geometry

The forecast is mapped to ego-relative distance and bearing:

```math
d_{i,k}=\|\hat{\mathbf p}_{i,k}-\hat{\mathbf p}_{e,k}\|_2,
```

```math
\phi_{i,k}=\operatorname{wrap}\left(
\operatorname{atan2}(\Delta y_{i,k},\Delta x_{i,k})-\psi_{e,k}
\right).
```

## 3. PC-FMCW/DPSK link abstraction

The original Assignment 1 demonstrates the PC-FMCW waveform and DPSK receiver,
but it does not supply an experimentally calibrated absolute vehicle-to-vehicle
optical link budget. This repository therefore uses a transparent reference-SNR
calibration.

The relative link gain is

```math
h(d,\phi) \propto
\frac{1}{\pi[d\tan(\theta_b)]^2}
\exp(-\kappa d)
\exp\left[-\frac{1}{2}\left(\frac{\phi}{\sigma_\phi}\right)^2\right]
\mathbb{1}(|\phi|\leq \Phi_{FOV}/2).
```

The SNR is anchored at `(d_ref, phi=0)`:

```math
\gamma(d,\phi)=\gamma_{ref}\frac{h(d,\phi)}{h(d_{ref},0)}.
```

This makes every absolute result conditional on the declared reference SNR,
while preserving the required physical monotonicity.

For binary differential detection in AWGN:

```math
P_b=\frac{1}{2}\exp(-E_b/N_0).
```

For a packet of `L` bits:

```math
PER=1-(1-P_b)^L,
```

and effective goodput is

```math
G=R_b(1-PER).
```

Outage is declared when BER exceeds the configured threshold. The Monte Carlo
LUT uses differential encoding, complex AWGN and adjacent-symbol differential
detection; it is kept separate from the analytical expression for validation.

## 4. Traffic and packet delivery

Each vehicle has an independent FIFO queue. Poisson arrivals are generated from
the normalized offered load. Every packet has an absolute deadline. For all
schedulers in the same episode, arrivals, deadlines and per-attempt uniform
random variables are identical. This common-random-number design prevents a
policy from receiving an easier traffic or channel realization.

At most one vehicle is selected per slot. Failed packets return to the head of
the queue; successful packets contribute their payload bits and latency.

## 5. Scheduling policies

- Random: lower baseline.
- Round Robin: equal-turn baseline.
- Reactive Greedy: current goodput and queue only.
- Proportional Fair: current rate divided by past service.
- CV Predictive: finite-horizon link estimate from constant velocity.
- Predictive Utility: constant-acceleration forecast and finite-horizon utility.
- Link-Lifetime: predictive utility plus proactive drain pressure.
- Oracle: perfect future positions used by the same utility heuristic.

The predictive score combines discounted expected goodput, predicted outage,
queue size, deadline urgency, fairness, switching cost and the opportunity loss
between the current and final predicted link. The link-lifetime variant adds a
pressure term only while the current link is usable:

```math
U_i^{life}\propto
Q_i\left(1-\frac{T_i^{link}}{H}\right)
+\frac{Q_i}{1+T_i^{link}}.
```

## 6. Communication-aware learning

The trainable objective is

```math
\mathcal L = \mathcal L_{traj}
+\lambda_{link}\mathcal L_{link}
+\lambda_{out}\mathcal L_{out}.
```

`L_traj` is Smooth L1 trajectory loss. `L_link` is Smooth L1 error between
predicted and ground-truth log-SNR. `L_out` is binary cross entropy for a smooth
outage probability. The architecture is a compact GRU encoder with a
multi-step decoder. Scenario IDs, rather than independent trajectory rows, are
used for the train/validation split.

## 7. Evaluation

Primary metrics are successfully delivered goodput, PDR, scheduled outage,
availability outage, mean/P95 latency, deadline-miss ratio and Jain fairness.
Results are paired by scenario and seed. The summary reports nonparametric
bootstrap confidence intervals and paired differences relative to Reactive
Greedy.

