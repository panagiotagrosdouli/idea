# Reproduced results

These results come from `make paper-quick` and the checked-in default config.
They are deterministic integration/mechanism evidence, not final full-WOMD
paper results.

## Motion and derived-link forecasts

| Predictor | Synthetic ADE (m) | Synthetic SNR MAE (dB) | WOMD-proxy ADE (m) | WOMD-proxy lifetime error (steps) |
|---|---:|---:|---:|---:|
| Last position | 1.711 | 0.698 | 1.903 | 1.167 |
| Constant velocity | 0.087 | 0.031 | 0.097 | 0.560 |
| Kalman CV | 0.352 | 0.083 | 0.328 | 0.260 |
| IMM | 0.176 | 0.053 | 0.212 | 0.540 |
| Constant acceleration | 0.028 | 0.002 | 0.196 | 0.507 |
| Oracle | 0.000 | 0.000 | 0.000 | 0.000 |

The large WOMD-proxy SNR errors near the FoV edge show why ADE alone is not a
sufficient communication metric: small geometric deviations can cross a hard
optical visibility boundary.

## Controlled motion benchmark

| Policy | Goodput (Mbps) | PDR | Delivered before expiry | Left at disconnect |
|---|---:|---:|---:|---:|
| Reactive Greedy | 2.293 | 0.644 | 0.801 | 49.75 |
| CV Predictive | 2.303 | 0.647 | 0.801 | 57.67 |
| Kalman Predictive | 2.300 | 0.646 | 0.800 | 57.25 |
| IMM Predictive | 2.301 | 0.646 | 0.800 | 57.25 |
| Predictive Utility | 2.304 | 0.647 | 0.801 | 57.67 |
| Link-Lifetime | 2.336 | 0.656 | 0.814 | 51.92 |
| Perfect-Future Lifetime Oracle | 2.339 | 0.657 | 0.815 | 51.92 |

The link-lifetime policy gains 0.0428 Mbps (about 1.9%) over Reactive Greedy at
the default operating point and approaches the perfect-future reference. The
result supports the mechanism in controlled motion; it is not a real-world
effect-size claim.

## Quick paired matrix

The quick matrix contains 224 policy rows: 32 paired operating points per
policy, spanning the first two values of every configured axis.

| Policy vs reactive | Mean difference (Mbps) | Bootstrap 95% CI | Win fraction | Wilcoxon p |
|---|---:|---:|---:|---:|
| CV Predictive | -0.0032 | [-0.0214, 0.0150] | 59.4% | 0.9702 |
| Kalman Predictive | -0.0033 | [-0.0220, 0.0155] | 56.2% | 0.9404 |
| IMM Predictive | -0.0028 | [-0.0209, 0.0153] | 59.4% | 0.9553 |
| Predictive Utility | -0.0030 | [-0.0213, 0.0153] | 56.2% | 0.9844 |
| Link-Lifetime | +0.0427 | [0.0229, 0.0629] | 87.5% | 0.00009 |
| Perfect-Future Oracle | +0.0423 | [0.0229, 0.0623] | 87.5% | 0.00009 |

Because this is a quick integration subset with only two seeds and related
operating points, its p-values are diagnostic rather than submission evidence.
The full 11,340-row matrix must be run for the final analysis.

## Robustness and channel ablations

At the two-episode quick setting, link-lifetime goodput is 2.238 Mbps with the
full channel. Removing lifetime urgency reduces it to 2.165 Mbps. History noise
has a strong effect: 0.5, 1 and 2 m reduce goodput to 1.887, 1.798 and 1.759
Mbps. Direct forecast perturbations at the same scales are less damaging in
this controlled generator (2.220, 2.194 and 2.202 Mbps). Range-only and
range-plus-pointing models yield about 3.16 Mbps, demonstrating that channel
fidelity materially changes the apparent scheduler opportunity.

## Compact real-WOMD motion / proxy-ego benchmark

| Policy | Goodput (Mbps) | PDR |
|---|---:|---:|
| Reactive Greedy | 1.160 | 0.329 |
| Link-Lifetime | 1.056 | 0.299 |
| Perfect-Future Lifetime Oracle | 0.976 | 0.277 |

This three-scene, one-second proxy benchmark does not support a predictive-
goodput claim. It lacks the true SDC, full official splits and a learned
checkpoint. The negative result is retained because it defines the remaining
evidence gap and prevents overclaiming.

Machine-readable summaries, LaTeX tables and generated figures are under
`artifacts/paper_run/`.
