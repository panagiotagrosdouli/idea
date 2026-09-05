# Stage 7 — Statistics and figures

## Scientific purpose
Join held-out forecast fidelity to realized scheduling utility and produce the scenario/episode-level inferential evidence, publication tables and figures used by the paper.

## Required inputs
- Stage-5 held-out per-scenario metrics and provenance
- Stage-6 canonical paired scheduling raw/completion evidence

## Outputs
Canonical Stage-7 evidence is written under `artifacts/paper_final/07_statistics/`, separated into learned-predictor and scheduler analyses.

## Acceptance criteria
The independent inferential unit remains scenario/episode; paired traffic realizations are retained before clustering; scenario-cluster bootstrap intervals, paired Wilcoxon/t-test sensitivity, effect sizes and Holm correction are applied within declared families; trajectory-to-link and ADE-to-realized-goodput joins use one inferential observation per scenario; negative/null operating regimes remain visible.

## Canonical command
```bash
python stages/07_statistics_and_figures/run.py
```

## Forbidden
Do not treat individual timesteps, packets or paired seeds as independent scenarios; do not present synthetic/proxy studies as official WOMD evidence; do not discard negative results or change multiplicity families after seeing outcomes.

## Dependencies
Stage 6 canonical scheduling and Stage 5 held-out predictor evidence.

## Path note
New canonical runs use `07_statistics`. Historical `07_analysis` files are not silently promoted; they must be deliberately regenerated/migrated and pass Stage 8 before release.
