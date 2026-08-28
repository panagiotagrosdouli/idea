# Paper-readiness assessment

## Current status

The software and experimental protocol are paper-oriented and reproducible.
They are not yet sufficient for an honest submission claim on full WOMD.

Ready now:

- complete causal trajectory-to-link-to-scheduler pipeline;
- strong classical, reactive, predictive and perfect-future references;
- traffic, channel, noise and horizon/load ablations;
- packet-level communication KPIs and paired statistical analysis;
- deterministic manifests, configs, tests, tables and figures;
- controlled-motion and compact real-motion/proxy-geometry evidence.

Not supplied and therefore not fabricated:

- official WOMD train/validation shards and exact release metadata;
- true SDC identity in the compact export;
- the upstream trained Part-B checkpoint referenced by the plans;
- measured vehicular PC-FMCW optical-channel traces;
- compute output from the full 11,340-row paper matrix.

## What can be claimed now

The code demonstrates mechanism validity under controlled motion and provides
an executable integration check on a compact real-WOMD motion export. The
optical link is physics-informed and calibrated to declared reference
parameters. It is not a measured channel.

## Final submission gate

Before paper submission:

1. provide and freeze the exact official WOMD release and scenario lists;
2. use the true SDC and rerun eligibility filtering in ego/headlamp coordinates;
3. provide or retrain the intended upstream predictor with scenario-safe splits;
4. run trajectory-only versus communication-aware loss sweeps;
5. execute the full paper matrix and freeze its commit/config/hardware manifest;
6. inspect paired confidence intervals and report negative regimes as well as
   positive ones;
7. replace provisional Part-A constants if measured/supplied values become
   available.

Until these gates are complete, the correct label is **paper-ready research
code and protocol**, not **paper-ready empirical evidence**.
