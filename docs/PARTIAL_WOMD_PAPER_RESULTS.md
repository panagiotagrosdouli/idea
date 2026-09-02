# Partial WOMD learned-ablation results (7/12 runs)

These are real paper-scale training diagnostics produced from the audited
249,137-sample WOMD corpus. They are intentionally labelled **partial** and
must not be presented as final held-out or packet-level results.

All completed runs use the same dataset SHA-256:

`b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee`

## Completion state

| Objective | Completed seeds |
|---|---:|
| Trajectory only | 3/3 |
| Trajectory + link | 3/3 |
| Trajectory + outage | 1/3 |
| Full communication-aware | 0/3 |

## Internal development diagnostics

| Objective | Seeds | ADE (m) | FDE (m) | Link loss | Outage loss |
|---|---:|---:|---:|---:|---:|
| Trajectory only | 3 | 3.9406 +/- 0.0103 | 11.2930 +/- 0.0283 | 2.6630 +/- 0.0046 | 0.6777 +/- 0.0099 |
| Trajectory + link | 3 | 3.9433 +/- 0.0104 | 11.2611 +/- 0.0359 | 2.6480 +/- 0.0164 | 0.6779 +/- 0.0162 |
| Trajectory + outage | 1 | 3.9488 | 11.2750 | 2.6669 | 0.6643 |

For the only complete paired comparison, trajectory + link versus trajectory
only over three identical seeds:

- ADE difference: +0.0028 m (paired t-test p=0.786; Wilcoxon p=1.0);
- FDE difference: -0.0319 m (paired t-test p=0.251; Wilcoxon p=0.5);
- link-loss difference: -0.0150 (paired t-test p=0.165; Wilcoxon p=0.25).

The direction is scientifically interesting: link-aware training slightly
reduced the internal link loss and FDE while leaving ADE effectively unchanged.
The evidence is not statistically significant with three training seeds and
does not yet establish realized scheduler goodput.

## Evidence boundary

These values use the internal scenario-safe development split of the training
corpus. Final claims still require:

1. the five missing paper runs;
2. untouched official-WOMD validation evaluation;
3. learned/reactive/oracle packet-level experiments on the same scenarios;
4. paired scenario statistics and Holm correction;
5. the communication-loss lambda sweep.
