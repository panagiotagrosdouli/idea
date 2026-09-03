# Executable research stages

This directory is the operational workspace for the paper. Each numbered
folder owns one `stage.json` specification containing its dependencies, required
inputs, commands, outputs and acceptance gate. Scientific implementations stay
in `src/`; executable entry points stay in `scripts/`; generated evidence is
written under `artifacts/paper_final/`.

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

Run `make stages` for the dependency-aware status and
`make stage STAGE=stageN EXECUTE=--execute` to execute one unlocked stage.
