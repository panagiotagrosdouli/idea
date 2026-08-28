# Predictive PC-FMCW/DPSK Vehicular Communications

This repository implements the new research assignment built on top of the
PC-FMCW/DPSK laser-headlamp work in `reference/assignment1`.

The scientific pipeline is:

```text
real or controlled vehicle motion
    -> causal trajectory forecast
    -> future range/bearing
    -> calibrated PC-FMCW/DPSK link state
    -> packet queues and deadlines
    -> predictive scheduling / proactive transmission
    -> goodput, PDR, outage, latency and fairness
```

The new contribution is communication control. Adaptive Top-K beam selection,
ADB optimization and illumination control are intentionally outside the core
decision problem.

## What is implemented

- Causal motion adapters for controlled trajectories and the supplied compact
  real-WOMD export.
- Last-position, constant-velocity, Kalman-CV, IMM, constant-acceleration and
  perfect-future motion references, plus optional learned GRU forecasts.
- Geometry-to-link mapping: range/bearing, atmospheric loss, pointing loss,
  reference-SNR calibration and fixed optical field of view.
- Analytical DBPSK BER plus a reproducible Monte Carlo BER-vs-SNR LUT.
- BER-to-PER and successfully delivered goodput, rather than nominal rate only.
- Poisson, periodic and Markov-modulated arrivals, bounded queues, deadlines,
  failures and common random traces.
- Ten policies: Random, Round Robin, Reactive Greedy, Proportional Fair,
  CV/Kalman/IMM Predictive, Predictive Utility, Link-Lifetime Prefetch and a
  perfect-future lifetime-aware Oracle reference.
- Communication-aware GRU objective:
  trajectory loss + log-SNR loss + differentiable outage loss.
- Scenario-level train/validation splitting and checkpointed PyTorch training.
- ADE/FDE, range/SNR error, outage F1/AUROC, link-lifetime error, delivered
  before expiry and undelivered-at-disconnection metrics.
- Paired bootstrap confidence intervals, paired t-tests, Wilcoxon tests, effect
  sizes, experiment matrices, ablations, CSV/JSON/LaTeX and paper figures.
- Scientific sanity gates and 28 automated tests.

## Scientific scope

The mobility samples may come from real WOMD trajectories. The optical
PC-FMCW/DPSK communication outcome is model-based. It is therefore correct to
describe the evaluation as:

> real-world mobility + physics-informed PC-FMCW/DPSK communication simulation

It is not a real optical-channel measurement campaign. The compact WOMD export
also omits the SDC identifier, so its included adapter uses a deterministic
medoid proxy ego and labels every result accordingly.

The `oracle` policy receives perfect future motion but uses the same interpretable
utility heuristic. It is an oracle-information reference, not a proof of the
globally optimal offline packet schedule.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,paper]"
```

For communication-aware GRU training:

```bash
pip install -e ".[dev,ml]"
```

## One-command reproduction

Run the complete fast integration pipeline:

```bash
make paper-quick
```

It freezes the dataset manifest, creates the Part-A BER LUT, evaluates motion
and link forecasts, benchmarks all schedulers on controlled and compact WOMD
motion, executes traffic/channel/noise ablations, runs a paired matrix and
generates LaTeX tables and figures under `artifacts/paper_run/`.
It also builds the six-page manuscript draft at
`output/pdf/predictive_pc_fmcw_paper_draft.pdf`.

Run the declared full paper matrix with:

```bash
make paper-full
```

The full run is intentionally compute-intensive. It does not replace the need
for the official WOMD shards and frozen trained checkpoints.

## Individual reproduction commands

Scientific sanity gates:

```bash
pcfmcw validate --config configs/default.json
```

DBPSK Monte Carlo LUT:

```bash
pcfmcw ber-lut --output artifacts/ber/dbpsk_ber_lut.csv
```

Controlled scheduler benchmark:

```bash
pcfmcw benchmark \
  --config configs/default.json \
  --output artifacts/synthetic_benchmark
```

Benchmark on the supplied compact real-WOMD motion export:

```bash
pcfmcw benchmark \
  --config configs/default.json \
  --womd-export data/example/womd_trajectories.json \
  --max-vehicles 5 \
  --output artifacts/womd_proxy_benchmark
```

Motion/link metrics, paper ablations and experiment matrix:

```bash
pcfmcw motion-eval --config configs/default.json \
  --output artifacts/motion_baselines
pcfmcw paper-ablation --config configs/default.json \
  --ber-lut artifacts/ber/dbpsk_ber_lut.csv \
  --output artifacts/paper_ablations
pcfmcw ablation --config configs/default.json --horizons 3 5 10 20
pcfmcw matrix --config configs/default.json \
  --matrix configs/paper_experiment_matrix.json --output results/matrix
```

Use `--quick` with `pcfmcw matrix` for an integration run.

## Communication-aware training

Prepare scenario-grouped relative-motion data:

```bash
pcfmcw prepare-training \
  data/example/womd_trajectories.json \
  data/cache/womd_relative_motion.npz
```

Train the GRU:

```bash
pcfmcw train data/cache/womd_relative_motion.npz checkpoints/comm_aware \
  --lambda-link 0.2 --lambda-outage 0.1
```

Evaluate the resulting checkpoint as an additional scheduler:

```bash
pcfmcw benchmark --config configs/default.json \
  --checkpoint checkpoints/comm_aware/best_comm_aware_gru.pt \
  --output results/learned_benchmark
```

Full Stage-4 training claims require the original WOMD shards and trained
checkpoints; they are not present in the supplied archives. The code does not
invent those results.

## Repository map

```text
src/predictive_pc_fmcw/   core research library
configs/                  default parameters and scripted matrix
scripts/                  one-command runners
tests/                    deterministic unit and integration tests
artifacts/                reproduced BER/benchmark/ablation outputs
data/example/             compact supplied real-motion export
reference/assignment1/    unchanged Assignment 1 notebook/report
reference/research_plan/  requirements and research-plan PDFs
docs/                     methodology, provenance, results and execution notes
```

## Ελληνική σύνοψη

Ο νέος κώδικας χρησιμοποιεί την πρόβλεψη της μελλοντικής κίνησης για να
προβλέψει την ποιότητα και τη διάρκεια του PC-FMCW/DPSK link και να αποφασίσει
ποιο όχημα θα εξυπηρετηθεί σε κάθε slot. Η Εργασία 1 διατηρείται αυτούσια ως
τεχνική αναφορά. Τα αποτελέσματα διαχωρίζουν ρητά την πραγματική κίνηση WOMD από
την προσομοιωμένη οπτική επικοινωνία.

See [Methodology](docs/METHODOLOGY.md),
[Data provenance](docs/DATA_PROVENANCE.md),
[Experiments](docs/EXPERIMENTS.md), [Current results](docs/RESULTS.md),
[PDF traceability](docs/PDF_REQUIREMENTS_TRACEABILITY.md) and
[Paper readiness](docs/PAPER_READINESS.md).
