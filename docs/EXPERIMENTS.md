# Experiment protocol

## Evidence tiers

1. **Exact regression:** deterministic unit and scientific-gate tests.
2. **Controlled benchmark:** 12 independent synthetic episodes.
3. **Diagnostic robustness:** two-seed or two-episode quick sweeps.
4. **Compact WOMD proxy:** three real-motion scenes with proxy ego.
5. **Publication evidence:** official WOMD true-SDC held-out scenes plus frozen
   learned checkpoints and multi-seed inference. This tier is not yet available.

## Reproduction

```bash
make test
make lint
make corrected-quick
```

`corrected-quick` writes only under `artifacts/corrected_v1/` and never reads
old benchmark values. `corrected-full` uses all five staged seeds and all
default ablation episodes.

## Controlled benchmark

The default holds 12 s physical duration, 100 ms slots, five receivers, 1 s
prediction horizon, Poisson load 0.72 and physical packet deadlines of
1.2±0.4 s. Each episode generates one scenario/seed cluster. All policy traffic
and packet-success draws are shared.

## Staged design

`configs/staged_experiments.json` replaces the previous inferential use of a
large Cartesian product. Each study changes one axis around the frozen default:

- load: 0.3, 0.5, 0.7, 0.9, 1.1;
- physical horizon: 0.1, 0.3, 0.5, 1, 2 s;
- slot duration: 0.05, 0.1, 0.2 s;
- receiver count: 3, 5, 10;
- Poisson, periodic, Markov-modulated and saturated traffic;
- packet size: 2,400, 9,600, 12,000 bits;
- deadline: 0.05, 0.1, 0.25, 0.5, 1 s;
- reference SNR offset: −6, −3, 0, +3, +6 dB;
- FoV: 50°, 70°, 90°;
- BER-, PER- and goodput-defined outage;
- perfect, Cartesian and assumed range/bearing sensing.

When slot duration changes, the number of slots and horizon steps are converted
so physical duration, horizon and deadlines remain constant.

## Learned-model ablation

The pre-registered comparison uses identical architecture, data split and
optimizer while changing only the loss:

1. trajectory-only;
2. trajectory + link;
3. trajectory + outage;
4. full trajectory + link + outage.

At least three training seeds are required. Model selection uses development
scenes. Final motion, link and scheduler metrics must be evaluated on held-out
scenario IDs. `scripts/04_run_training_ablation.py --plan-only` validates the
run matrix without importing PyTorch.

## Official WOMD gate

The publication run must record:

- exact WOMD release and shard hashes;
- true `sdc_track_index`;
- validity-mask and actor-eligibility counts;
- scenario-safe train/development/test lists;
- feature schema and checkpoint SHA-256;
- exact command, config, seed, software versions and commit.

The included official adapter conservatively drops any retained track with an
invalid state. It does not interpolate hidden values.

## Statistical plan

Primary policy comparisons use paired scenario/seed differences. Bootstrap,
paired t and Wilcoxon tests operate on independent cluster aggregates.
Metric direction is declared in advance; lower latency/outage/miss is favorable,
while higher goodput/PDR/fairness is favorable. Wilcoxon families receive Holm
correction. Effect sizes, win rates, support and negative regimes are reported.

Two-seed quick runs cannot support p-value claims and are labelled diagnostic.

## Acceptance rule

The hypothesis is supported only in an operating region where:

- the paired communication interval excludes zero in the favorable direction;
- tail latency, PDR, censoring and fairness are reported together;
- the result survives pre-registered channel and sensing sensitivity;
- the learned objective improves link-relevant metrics on held-out official
  scenes without unacceptable trajectory degradation.

A negative or null result is retained.
