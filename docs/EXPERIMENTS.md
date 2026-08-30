# Experiment protocol

## Evidence tiers

1. **Exact regression:** 51 deterministic tests and five scientific gates.
2. **Controlled benchmark:** 12 independent synthetic episodes.
3. **Full staged study:** five independent seeds over 45 one-axis settings.
4. **Compact WOMD proxy:** three real-motion scenes with a proxy ego.
5. **Publication evidence:** official WOMD true-SDC held-out scenes and frozen
   learned checkpoints. This tier is blocked by missing inputs.

## Reproduction

```bash
make test
make lint
make corrected-full
make paper-draft
make reproducibility
```

The completed full output is isolated under `artifacts/corrected_v2/`. The
quick target writes to `artifacts/corrected_v2_quick/` so it cannot overwrite
the five-seed evidence.

## Controlled benchmark

The default holds 12 s physical duration, 100 ms slots, five receivers, 1 s
prediction horizon, Poisson load 0.72 and packet deadlines of 1.2±0.4 s. Every
policy receives identical scenarios, arrivals, deadlines and per-attempt random
draws. Decisions may use predicted links; delivery always uses the current
ground-truth-derived link.

## Staged design

Each study changes one axis around the frozen reference:

- offered load: 0.3, 0.5, 0.7, 0.9, 1.1;
- horizon: 0.1, 0.3, 0.5, 1, 2 s;
- slot duration: 0.05, 0.1, 0.2 s;
- receiver count: 3, 5, 10;
- Poisson, periodic, Markov-modulated and saturated traffic;
- single-class versus urgent/bulk traffic;
- packet size: 2,400, 9,600, 12,000 bits;
- deadline: 0.05, 0.1, 0.25, 0.5, 1 s;
- reference-SNR offset: -6, -3, 0, +3, +6 dB;
- FoV: 50°, 70°, 90°;
- BER-, PER- and goodput-defined outage;
- perfect, Cartesian and range/bearing sensing.

The full run has 45 settings × 5 seeds × 5 policies = 1,125 rows. When slot
duration changes, slot counts and horizon steps are converted so all other
physical durations remain fixed.

## Part-A BER protocol

The LUT reproduces the supplied receiver sequence: PC-FMCW chirp construction,
FFT carrier-bin detection, parabolic sub-bin refinement, rotation compensation
and differential phase decisions. Its SNR axis is explicitly waveform-sample
SNR. Monte Carlo values are converted into a conservative monotone LUT; a
zero-error point uses the one-sided 95% Wilson upper bound.

## Learned-model protocol

The preregistered comparison holds architecture, split and optimizer constant
and changes only the loss: trajectory-only; trajectory+link;
trajectory+outage; and full. At least three training seeds and held-out official
scenario IDs are required. `scripts/04_run_training_ablation.py --plan-only`
validates the run matrix without importing PyTorch.

The executable classical uncertainty baseline uses the first six controlled
scenario IDs only to fit per-horizon isotropic residual variance for CV/CA and
the remaining six IDs only for NLL and 50/90/95% coverage. The split has zero
scenario overlap. It does not claim to reproduce the missing learned K=5 GMM.

## Statistical protocol

Primary comparisons use paired scenario/seed differences. Bootstrap, paired-t
and Wilcoxon calculations operate on independent cluster aggregates. Metric
direction is declared in advance; Wilcoxon families receive Holm correction.
Effect sizes, win fractions and negative regimes are retained.

Five pairs still provide low rank-test resolution: even five same-direction
differences have a minimum two-sided Wilcoxon p-value of 0.0625. Therefore the
full staged study identifies conditional regimes but is not treated as final
official-WOMD inference.

## Acceptance rule

A favorable claim requires a paired interval in the beneficial direction,
acceptable latency/PDR/fairness/censoring, robustness to channel and sensing
assumptions, and learned-model confirmation on held-out official scenes. Null
and negative results remain part of the paper.
