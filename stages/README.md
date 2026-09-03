# Executable research stages

This directory is the operational workspace for the paper. Each numbered
folder owns one `stage.json` specification and a direct `run.py`. The
specification contains its dependencies, inputs, commands, outputs and gate.
Scientific implementations stay in `src/`; executable entry points stay in
`scripts/`; generated evidence is written under `artifacts/paper_final/`.

The repository deliberately uses a stage-oriented organization similar in
spirit to the earlier ISCAI Part-B work, while keeping this project's own
scientific pipeline and filenames. Existing research code is not duplicated
inside the stage folders: stages orchestrate the canonical implementation.

```text
stages/
├── 00_freeze_and_provenance/          # repository/protocol freeze
├── 01_womd_data_pipeline/             # official WOMD corpora + leakage audit
├── 02_pc_fmcw_dpsk_link/              # Part-A receiver/LUT/link freeze
├── 03_classical_baselines/             # causal motion baselines
├── 04_communication_aware_gru/         # development-only tuning + 4×5 training
├── 05_official_predictor_evaluation/   # untouched official-validation evaluation
├── 06_packet_scheduling/               # predictive/reactive/oracle packet experiments
├── 07_statistics_and_figures/          # joined analysis + confirmatory statistics
└── 08_final_paper/                      # release gate, manuscript, reproduction
```

## Paper work packages

The numbered stages map to four paper-oriented work packages:

```text
WP1 — Freeze data, physics and protocol
      stages 00–02

WP2 — Establish predictor evidence
      stages 03–05

WP3 — Run official end-to-end communication experiments
      stages 06–07

WP4 — Assemble the publication and reproducible release
      stage 08
```

This ordering is intentional. The core paper must close WP1–WP3 before any RL
extension is allowed to become a dependency of the publication claim. RL can
remain an optional extension after the canonical packet-scheduling result.

## Evidence layout

Canonical evidence mirrors the research stages under one root:

```text
artifacts/paper_final/
├── 00_freeze/
├── 01_data/
├── 02_link/
├── 03_baselines/
├── 04_learning/
├── 05_heldout/
├── 06_scheduling/
├── 07_statistics/
└── 08_release/
```

Large/raw WOMD data and local checkpoints are inputs, not Git-tracked evidence.
Only reproducibility metadata, summaries, tables, figures, manifests and other
appropriate versioned artifacts belong in the repository.

## Execution

Run `make stages` for dependency-aware status, `make stage STAGE=stageN
EXECUTE=--execute`, or run a folder directly, for example
`python stages/03_classical_baselines/run.py`.

A stage is not scientifically complete merely because its code executes. Its
`stage.json` acceptance criteria and upstream evidence must also pass. Missing
official data, LUTs, checkpoints or held-out results remain explicit blockers;
synthetic fixtures may test code but cannot satisfy publication gates.

## Submission gate

The core manuscript is not release-ready until the repository contains
versioned evidence for all of the following:

- frozen training/development and untouched official-validation provenance,
  including zero scenario overlap and corpus hashes;
- canonical Part-A receiver-derived BER LUT and verification metadata;
- classical predictor development evidence using the frozen link model;
- four learned objectives across five canonical seeds, with development-only
  lambda/model selection frozen before held-out use;
- one untouched official-validation predictor evaluation;
- official packet-level scheduler comparisons on the same held-out scenario
  population with paired traffic realizations;
- scenario/horizon/load analyses needed by the final claims;
- scenario-clustered uncertainty intervals and multiplicity-adjusted
  confirmatory statistics;
- final figures/tables/manuscript and a reproducibility/release manifest.

Until these exist, `08_final_paper` must remain fail-closed rather than infer
completion from older proxy or staged artifacts.
