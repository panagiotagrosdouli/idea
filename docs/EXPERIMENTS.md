# Experiment protocol

## Reproducibility gates

1. Run all unit/integration tests.
2. Run `pcfmcw validate` and require every monotonicity and causality gate to
   pass.
3. Generate the DBPSK LUT with a fixed bit count and seed.
4. Keep motion, arrivals, deadlines and channel random numbers identical across
   schedulers.
5. Run the reactive baselines before any learned model.
6. Compare predictive policies to Reactive Greedy with paired scenario-level
   differences.
7. Treat Oracle as an information reference, not a deployable method.

## Main benchmark

The checked-in default uses 12 controlled episodes, 120 scheduling slots,
five connected receivers, 100 ms slots and a 1 s prediction horizon. This is a
fast reproducibility configuration, not a frozen publication configuration.

## Required sweeps

`configs/experiment_matrix.json` defines:

- horizon: 0.3, 0.5, 1.0 and 2.0 s at the 100 ms default;
- connected receivers: 3, 5 and 10;
- normalized offered load: 0.35, 0.55, 0.75 and 0.90;
- scheduler slot: 50, 100 and 200 ms;
- five paired random seeds.

Use:

```bash
pcfmcw matrix --config configs/default.json \
  --matrix configs/experiment_matrix.json \
  --output results/matrix
```

## Communication-aware ablation

Train the same GRU architecture with:

1. `lambda_link=0`, `lambda_outage=0`;
2. link loss only;
3. outage loss only;
4. full objective.

Report ADE, FDE, log-SNR error, outage classification, goodput, PDR, latency and
fairness. Checkpoint selection must use the internal validation split; the test
set is evaluated once after weights are frozen.

## Acceptance rule

The main hypothesis is supported only if paired confidence intervals show a
communication gain in the declared operating region. A gain in controlled
motion but not in the compact WOMD proxy benchmark is evidence that more real
scenes and correct ego geometry are required; it is not permission to generalize
the controlled result.

