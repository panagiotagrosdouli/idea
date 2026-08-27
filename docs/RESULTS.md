# Reproduced results

These results were generated from the checked-in default configuration.

## Controlled motion benchmark

| Policy | Goodput (Mbps) | PDR | Scheduled outage | Deadline miss | Jain fairness |
|---|---:|---:|---:|---:|---:|
| Reactive Greedy | 2.293 | 0.644 | 0.000 | 0.308 | 0.536 |
| CV Predictive | 2.303 | 0.647 | 0.000 | 0.298 | 0.564 |
| Predictive Utility | 2.304 | 0.647 | 0.000 | 0.298 | 0.564 |
| Link-Lifetime | 2.336 | 0.656 | 0.001 | 0.292 | 0.568 |
| Oracle Information | 2.307 | 0.648 | 0.000 | 0.298 | 0.564 |

The link-lifetime policy gains about 1.9% goodput over Reactive Greedy in this
controlled operating point, while also improving PDR, deadline misses and
fairness. This supports mechanism validity, not a real-world effect-size claim.

## Compact real-WOMD motion / proxy-ego benchmark

| Policy | Goodput (Mbps) | PDR | Scheduled outage | Deadline miss | Jain fairness |
|---|---:|---:|---:|---:|---:|
| Reactive Greedy | 1.160 | 0.329 | 0.067 | 0.005 | 0.378 |
| CV Predictive | 1.056 | 0.299 | 0.033 | 0.005 | 0.411 |
| Predictive Utility | 0.892 | 0.253 | 0.067 | 0.006 | 0.354 |
| Link-Lifetime | 1.056 | 0.299 | 0.067 | 0.005 | 0.411 |
| Oracle Information | 0.852 | 0.242 | 0.100 | 0.006 | 0.321 |

The compact proxy benchmark does not support a predictive-goodput claim. It has
only three scenes, one second of evaluation, no true SDC identifier and no
learned checkpoint. It is retained because negative or inconclusive evidence is
scientifically important and reveals exactly what must be added before a paper
submission.

Machine-readable episode rows, bootstrap intervals, paired differences, LaTeX
tables and figures are under `artifacts/`.

## Full experiment matrix

The full scripted matrix contains 3,600 policy results and 720 paired operating
points per predictive policy. No non-finite values occur in the primary metrics.

Across every horizon, vehicle count, load, slot duration and seed, the
link-lifetime policy has a mean goodput difference of +0.0055 Mbps relative to
Reactive Greedy and wins 61.5% of paired operating points. The average
difference is +0.0230 Mbps at load 0.35, +0.0241 Mbps at 0.55, +0.0039 Mbps at
0.75 and -0.0292 Mbps at 0.90. Thus the predictive advantage is load-dependent,
and it should not be claimed as universal.
