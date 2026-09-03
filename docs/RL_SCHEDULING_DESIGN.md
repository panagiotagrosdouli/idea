# Reinforcement-Learning Scheduler Extension

This document defines the research protocol for adding RL scheduling to the existing predictive PC-FMCW/DPSK vehicular communication pipeline.

## Goal

The RL extension does not replace the existing simulator, link model, traffic model, WOMD pipeline, or classical schedulers. It uses the same causal `SchedulerContext` and the same packet-level realization so that comparisons remain paired and scientifically meaningful.

Primary question:

> Can a learned scheduler exploit causal queue, deadline, link and prediction information better than hand-designed reactive and predictive scheduling policies?

Secondary questions:

1. Does predictive information improve RL over a reactive RL agent trained with the same algorithm and simulator?
2. When does prediction uncertainty reduce or reverse the benefit of predictive RL?
3. Which reward terms are responsible for gains in goodput, deadline reliability, latency and fairness?
4. Does an RL policy generalize across WOMD scenarios, traffic loads, deadlines and sensing noise without held-out tuning?

## Causality rule

No future ground-truth vehicle state or future ground-truth link state may enter the RL observation. Predictive features may come only from deployable causal predictors. Future ground truth is reserved for packet/link realization and evaluation, exactly as in the existing simulator.

## Agent families

The confirmatory comparison should include:

- classical reactive and predictive schedulers already present in the repository;
- reactive RL: current queue/deadline/link state only;
- predictive RL: the same state plus causal predicted goodput, outage and link lifetime;
- uncertainty-aware predictive RL: predictive state plus calibrated uncertainty features;
- oracle information policy only as a non-deployable reference.

At least PPO and one value-based baseline should be considered, but algorithm selection and hyperparameters must be frozen on development scenarios before official evaluation.

## Observation

The first implementation uses a per-vehicle feature matrix with normalized causal features:

- eligibility / non-empty queue;
- queue occupancy;
- deadline urgency;
- current goodput;
- current outage;
- delivered-service share;
- previous scheduling decision;
- predicted mean goodput (predictive mode only);
- predicted outage fraction (predictive mode only);
- predicted link-lifetime fraction (predictive mode only).

Later uncertainty-aware variants may add calibrated predictive variance, coverage or outage probability, but only after the deterministic predictive RL baseline is frozen.

## Action

For a single-RSU experiment the discrete action selects one eligible vehicle for the current resource slot. An optional idle action may be introduced only if the packet simulator has a scientifically justified reason to avoid transmission.

Invalid actions must be handled explicitly and identically during training and evaluation; they must never silently become access to future information.

## Reward

The initial packet-level reward is multi-objective:

`delivered information - deadline drops - failed transmissions - scheduled outage - switching cost + fairness gain`

Reward weights are development-set hyperparameters. They must not be optimized on official validation/test scenarios.

Ablations must include at minimum:

- delivery-only reward;
- delivery + deadline reliability;
- delivery + deadline + outage awareness;
- full reward including fairness/switching terms.

## Environment architecture

The packet simulator should be refactored into a reusable single-step transition engine rather than duplicated inside the RL package.

Target architecture:

```text
WOMD MotionScenario
      |
causal predictor
      |
SchedulerContext ---------> observation builder ---------> RL policy
      |                                                |
      +---------------------- action <------------------+
      |
packet/link transition engine
      |
reward + next SchedulerContext + metrics
```

The existing `run_simulation()` should remain a thin episodic runner over the same transition engine so classical and RL policies use identical physical and traffic semantics.

## Experimental protocol

Training, development and official held-out scenario IDs must remain disjoint. Traffic randomness must be paired across policies during evaluation. RL training seeds and traffic seeds must be recorded separately.

Confirmatory evaluation should report the existing packet metrics, including goodput, PDR, deadline miss ratio, P50/P95/P99 latency, outage scheduling fraction, fairness and delivered-before-disconnect metrics.

RL-specific evidence should additionally include learning curves, seed dispersion, inference runtime, model size, action entropy and invalid-action rate.

## Planned implementation gates

1. **RL-0: Interface foundation** — causal observation and reward modules with regression tests.
2. **RL-1: Step environment** — refactor packet simulator into reset/step transition semantics without changing existing outputs.
3. **RL-2: Reactive baseline** — train/evaluate reactive PPO and value-based baseline.
4. **RL-3: Predictive RL** — add causal predicted-link observation features.
5. **RL-4: Uncertainty-aware RL** — add calibrated uncertainty only after Stage 5 predictor calibration exists.
6. **RL-5: Ablations** — observation, reward and algorithm ablations across development regimes.
7. **RL-6: Held-out evaluation** — paired official WOMD comparisons with frozen agents and hyperparameters.
8. **RL-7: Statistics and paper artifacts** — scenario-clustered uncertainty intervals and corrected confirmatory tests.

## Current implementation status

RL-0 has started on branch `feature/rl-scheduling` with `predictive_pc_fmcw.rl.state`, `predictive_pc_fmcw.rl.reward` and regression tests. The next engineering task is RL-1: extracting the slot-level packet transition from `simulation/engine.py` so both classical and learned policies share one environment implementation.
