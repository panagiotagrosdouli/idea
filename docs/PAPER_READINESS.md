# Paper-readiness assessment

## Verdict

The repository is a reproducible, paper-oriented research prototype with a
complete controlled experiment. It is **not yet a submission-ready empirical
paper** because the official data and learned-model evidence requested by the
plans are unavailable.

## Completed and executed

- 51 deterministic tests and five scientific gates;
- a notebook-derived Part-A FFT-carrier/DPSK BER receiver and confidence-aware
  LUT;
- causal motion → geometry → link → packet → scheduler execution;
- classical predictors, ten schedulers, urgent/bulk traffic, censoring and
  boundary-sensitive link metrics;
- a controlled 12-episode benchmark and compact three-scene proxy integration;
- scenario-safe classical Gaussian CV/CA calibration on disjoint controlled
  scenarios;
- 12 one-axis studies, five independent seeds and 1,125 policy episodes;
- clustered paired statistics, bootstrap intervals and Holm correction;
- architecture, timeline, BER, ECDF, Pareto, lifetime, failure-case and
  complexity artifacts;
- English/Greek README, methods/results docs, traceability and paper PDF.

## Remaining blockers

| Blocker | Consequence |
|---|---|
| Official WOMD shards absent | no true-SDC, large held-out evaluation |
| Compatible learned checkpoint absent | no learned/communication-aware comparison |
| Supplied Stage-4 JSON lacks paired ground truth | no honest ADE/FDE or probabilistic calibration from those 70 forecasts |
| Measured/calibrated optical link absent | no absolute-power or real-channel claim |
| Authors/venue metadata absent | draft cannot be submitted as-is |

PyTorch is optional for the executed classical experiment, but it is required
to train the GRU ablation once the official samples are present.

## Evidence verdict

The default 12-episode benchmark gives a small uncertain goodput difference
(+0.014 Mbps, bootstrap 95% interval -0.0319 to +0.0593 Mbps) and a clear
269 ms P95 latency penalty. The compact proxy result is negative. The full
five-seed staged study exposes conditional positive and negative regions, but
none of the cited families survives Holm correction with only five pairs.

The strongest defensible claim is therefore:

> Causal future motion changes packet-scheduling trade-offs, but the value of
> link-lifetime urgency is conditional on deadline, load, channel and traffic
> class; a universal communication gain is not supported.

## Submission gate

Before submission, add and hash official WOMD shards, build true-SDC
scenario-safe splits, run the four learned objectives for at least three seeds,
freeze a compatible checkpoint, repeat held-out motion/link/scheduler
evaluation, then add authors, venue formatting and the final bibliography.
