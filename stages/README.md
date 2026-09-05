# Executable research stages

This directory is the canonical Stage 0–8 operator workspace. Each numbered folder owns one `stage.json`, a direct `run.py`, and a stage README describing scientific purpose, required inputs, outputs, acceptance criteria, canonical command, forbidden operations and upstream dependencies.

Scientific implementations stay in `src/predictive_pc_fmcw/`; executable experiment entrypoints stay in `scripts/`; generated evidence is written under `artifacts/paper_final/`. Stage folders orchestrate and document the canonical implementation rather than duplicating library code.

The stage-oriented organization borrows only the high-level navigability of the ISCAI Part-B reference repository. This repository does not copy its science, numerical assumptions, large archives, generated manifests or stage-specific implementation duplication.

```text
stages/
├── 00_freeze_and_provenance/
├── 01_womd_data_pipeline/
├── 02_pc_fmcw_dpsk_link/
├── 03_classical_baselines/
├── 04_communication_aware_gru/
├── 05_official_predictor_evaluation/
├── 06_packet_scheduling/
├── 07_statistics_and_figures/
└── 08_final_paper/
```

## Canonical evidence layout

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

Operational `execution_state.json` files describe restart status only. A stage is not scientifically complete because an execution-state file says `completed`; its declared acceptance evidence and verification reports remain authoritative.

## One production execution path

The canonical production path is deliberately obvious:

```bash
make canonical-preflight   # Stage-1-safe; no Stage-2/CUDA dependency
make canonical-stage1     # WOMD provenance and corpus verification
make canonical-full       # frozen Stage 2 + CUDA + downstream Stages 3–7
```

`make stages` provides dependency-aware status. `make stage STAGE=stageN EXECUTE=--execute` or `python stages/<folder>/run.py` performs targeted stage execution without defining a second scientific protocol.

## Scientific sequence

```text
freeze/provenance
      ↓
WOMD data ──→ causal prediction ──→ frozen link mapping
                                  ↓
                           packet scheduling
                                  ↓
                       scenario-level statistics
                                  ↓
                         reproducibility release
```

Stage 1 deliberately does not depend on Stage 2 at production-preflight time. Full downstream execution requires verified frozen Stage-2 evidence. Stage 4 selects hyperparameters on development only and remains exactly four objectives × five frozen seeds. Stage 5 is the untouched official predictor evaluation. Stage 6 remains exactly eight scheduler families × five paired traffic seeds. Stage 7 keeps scenario/episode as the inferential unit. Stage 8 fails closed until real evidence exists.

For the detailed repository map, restart semantics and the deliberate `07_analysis` → `07_statistics` migration policy, see [`docs/REPOSITORY_ARCHITECTURE.md`](../docs/REPOSITORY_ARCHITECTURE.md).
