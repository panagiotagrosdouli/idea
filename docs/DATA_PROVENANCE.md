# Data provenance and claim boundaries

## Assignment 1

`reference/assignment1` is an unchanged copy of the supplied GitHub archive. It
contains the original PC-FMCW/DPSK notebook, technical report and requirements.
The new package does not rewrite its results. It reuses the 1 Gbit/s DPSK
communication premise and the physical interpretation of phase-coded FMCW.

## Compact WOMD export

`data/example/womd_trajectories.json` is copied from the supplied Part-B archive.
It contains 56 actors across three scenario IDs, with ten past and ten future
positions per actor. It supports integration testing with real motion.

The export does not contain:

- raw WOMD TFRecords;
- the SDC/ego track identifier;
- Stage-4 trained checkpoints;
- optical received-power measurements;
- PC-FMCW/DPSK packets or channel impulse responses.

The adapter therefore selects the current-position medoid as a deterministic
proxy ego. Every resulting artifact records the source string
`real_WOMD_motion_proxy_ego_geometry`.

`pcfmcw dataset-manifest` records the input SHA256, record/scenario counts,
declared release, license boundary and deterministic SHA256 scenario split.
The supplied compact file has 56 actor records across three scenarios. This
manifest must be regenerated with the exact official release label when full
WOMD data are supplied.

## Large upstream results

The supplied Part-B archive reports later Stage-4 Gaussian/GMM results, but the
referenced `.pt` checkpoints and TFRecord shards are absent. This repository
does not relabel those report files as rerun results. The included PyTorch code
can train and evaluate communication-aware checkpoints when the real data is
provided.

## Controlled motion

The synthetic generator creates closing-range vehicles, acceleration and lane
changes. It exists to validate causality and exercise predictive scheduling
opportunities under exactly controlled conditions. These results are software
and mechanism evidence, not a real-world performance claim.

## Communication model

All link, BER, PER and packet-delivery outputs are simulated. Absolute values
depend on the declared reference SNR, packet size, field of view, offered load
and resource fraction in `configs/default.json`. Sensitivity sweeps must precede
any publication claim.
