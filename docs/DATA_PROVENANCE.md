# Data provenance and claim boundaries

## Part A physical-layer reference

The user-supplied Assignment-1 repository contains the PC-FMCW notebook and
technical report. Frozen provenance is recorded in
`configs/part_a_physical_layer.json`:

- upstream commit `44d62e3478e3818d1757b00971890f844cb032f7`;
- notebook SHA-256
  `b5a80a6d3441de6d571db4f65b4a43ed4052cc2b3ccba935ad31b5dd51316ef3`;
- 193.4 THz carrier, 10 GHz bandwidth, 10 μs chirp and 1 Gbit/s data rate.

The supplied notebook contains no stored executed code-cell outputs. The local
pipeline uses a reference-SNR abstraction and does not relabel it as the full
waveform receiver or a measured link budget.

## Compact WOMD export

`data/example/womd_trajectories.json` contains 56 actors in three scenario IDs,
with ten past and ten future XY samples. It was copied from the supplied Part-B
archive.

It omits raw TFRecords, official SDC identity, validity masks, map context,
optical measurements and compatible trained checkpoints. The compact adapter
therefore chooses a deterministic medoid proxy ego. Every artifact carries the
source label `real_WOMD_motion_proxy_ego_geometry`.

These results mean “real motion + proxy geometry + model-based communication,”
not real optical validation and not official-WOMD generalization.

## Official WOMD adapter

`data/womd_official.py` reads TFRecord payloads without TensorFlow, parses Waymo
Scenario protos when the optional proto package is installed, uses the true
`sdc_track_index`, filters vehicle tracks and requires valid finite states over
the entire retained window.

No official shard is present in the supplied files, so this path is implemented
and unit-tested with a schema-compatible fixture but not empirically executed.

## Supplied Stage-4 reports

`reference/part_b_stage4/` preserves five supplied JSON reports. The archive
reports WOMD/WOMD-LiDAR v1.3.0, 116,182 selected scenarios and trained-model
metrics. The raw shards and `.pt`/`.pth`/`.ckpt` files are absent.

The upstream model interface uses eight state features; the local optional GRU
uses ego-relative XY histories and explicit ego heading for link loss. Upstream
numbers are provenance only. They are not local reruns, scheduler results or
compatible checkpoints.

## Controlled motion

Synthetic scenes provide repeatable closing, receding, acceleration and lateral
motion. They support invariants and mechanism tests. Their numerical effect
sizes do not establish real-world performance.

## Sensing and communication outputs

All observation-noise, link, BER, PER, packet and scheduling outcomes are
simulated. The assumed range/bearing sensing model is explicitly marked
`measured_data=false`. Absolute optical power is uncalibrated. Any paper must
state these boundaries beside the relevant results.
