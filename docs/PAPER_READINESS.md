# Paper-readiness assessment

## Verdict

The repository is now a tested, reproducible research prototype and an honest
draft-paper package. It is **not yet a submission-ready empirical paper**.

## Completed

- 44 deterministic tests and five scientific gates;
- corrected frame, heading, horizon and physical-time handling;
- explicit BER/PER/goodput outage semantics and normalized-power labeling;
- packet censoring, expanded KPIs and clustered direction-aware statistics;
- classical predictors, ten schedulers and true-SDC official-WOMD adapter;
- adaptive BER LUT with confidence handling;
- scenario slices and staged load/horizon/traffic/channel/sensing design;
- communication-aware four-objective multi-seed training infrastructure;
- corrected synthetic, compact proxy, ablation and 430-row diagnostic artifacts;
- beginner README, figures, tables, provenance and manuscript source.

## Empirical blockers

| Blocker | Why it matters |
|---|---|
| Official WOMD v1.3.0 shards absent | No true-SDC large held-out evaluation |
| Compatible checkpoint absent | No learned or communication-aware result |
| PyTorch absent in executed runtime | Training path not smoke-run here |
| Measured/calibrated optical link absent | No absolute power or real-channel claim |
| Only two staged quick seeds executed | Operating-region results are diagnostic |
| Author/venue metadata absent | Manuscript cannot be submitted as-is |

## Current evidence verdict

The corrected 12-episode benchmark shows a small, uncertain goodput difference
and a clear tail-latency penalty. The explicit lifetime-weight ablation is null
at the default operating point. The compact proxy result is negative. These are
valid outcomes and replace the earlier stronger provisional story.

## Submission gate

Before submission:

1. attach/fetch and hash official WOMD shards;
2. build frozen scenario-safe train/development/test samples with true SDC;
3. install PyTorch and run four objectives for at least three seeds;
4. evaluate motion, boundary-sensitive link and scheduler metrics on held-out
   scenes;
5. execute the five-seed staged design and inspect corrected families;
6. freeze a real repository commit and clean CI run;
7. update the manuscript from those immutable artifacts;
8. add authors, affiliations, venue formatting and a complete bibliography.

The defensible label today is **paper-oriented research code with corrected
controlled and proxy evidence**.
