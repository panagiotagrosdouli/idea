# Repository architecture and canonical stage ownership

This repository uses the explicit stage-oriented navigability of the ISCAI Part-B reference as an organizational model only. Scientific implementations, protocols and assumptions remain this repository's own canonical implementation.

## Flow

```text
Stage 0  freeze/provenance
   ↓
Stage 1  WOMD v1.3.1 causal corpora ─────────────┐
   ↓                                             │
Stage 2  frozen PC-FMCW/DPSK link               │
   ↓                                             │
Stage 3  classical development baselines        │
   ↓                                             │
Stage 4  development-only lambda selection      │
         + 4 objectives × 5 seeds               │
   ↓                                             │
Stage 5  untouched official predictor evaluation│
   ↓                                             │
Stage 6  8 schedulers × 5 paired traffic seeds  │
   ↓                                             │
Stage 7  scenario/episode statistics + figures  │
   ↓                                             │
Stage 8  reproducibility/release gate            │
```

Stage 1 can run without Stage 2. The complete downstream path requires the frozen verified Stage-2 evidence; the production full-run preflight also requires CUDA. Official validation is never a source of model-selection decisions.

## Ownership model

`stages/<number_name>/` owns the human and machine contract for that stage: its README, `stage.json`, direct `run.py`, canonical commands, declared inputs/outputs, acceptance criteria and forbidden operations. `src/predictive_pc_fmcw/` owns reusable scientific/software components. `scripts/` provides executable experiment entrypoints called by stage contracts and the canonical orchestrator. `artifacts/paper_final/` owns generated evidence, aligned by stage.

No library implementation is duplicated merely to make a stage folder self-contained. A stage folder explains and orchestrates; the reusable package implements.

## Stage map

| Stage | Contract directory | Scientific purpose | Canonical evidence |
|---:|---|---|---|
| 0 | `stages/00_freeze_and_provenance` | Freeze release, data roles and provenance | `artifacts/paper_final/00_freeze` |
| 1 | `stages/01_womd_data_pipeline` | Materialize and verify frozen WOMD v1.3.1 corpora | `artifacts/paper_final/01_data` |
| 2 | `stages/02_pc_fmcw_dpsk_link` | Freeze/verify Part-A PC-FMCW/DPSK mapping | `artifacts/paper_final/02_link` |
| 3 | `stages/03_classical_baselines` | Development-only causal classical baselines | `artifacts/paper_final/03_baselines` |
| 4 | `stages/04_communication_aware_gru` | Development-only tuning and frozen GRU ablation | `artifacts/paper_final/04_learning` |
| 5 | `stages/05_official_predictor_evaluation` | One untouched official-validation predictor evaluation | `artifacts/paper_final/05_heldout` |
| 6 | `stages/06_packet_scheduling` | Canonical paired packet-level scheduler comparison | `artifacts/paper_final/06_scheduling` |
| 7 | `stages/07_statistics_and_figures` | Scenario-level inference, joins, tables and figures | `artifacts/paper_final/07_statistics` |
| 8 | `stages/08_final_paper` | Release-readiness, manuscript and reproducibility package | `artifacts/paper_final/08_release` |

`artifacts/paper_final/execution_state.json`, `04_learning/lambda_sweep/execution_state.json` and `04_learning/learned_ablation/execution_state.json` are restart/operations reports. They cannot satisfy a scientific completion gate by themselves.

## Canonical operator path

The production path has three obvious commands: `make canonical-preflight`, `make canonical-stage1`, and `make canonical-full`. The stage workspace (`make stages`, `make stage STAGE=stageN`) remains useful for dependency inspection and targeted stage execution, but it does not define a second scientific protocol.

Stage 4 resumes at objective×seed granularity only when the cached `training_result.json` parses, matches the expected objective, seed and dataset hash, points inside the expected run directory, and references a non-empty checkpoint. Otherwise the run is recomputed. Development-only lambda sweep runs use the same validation rule, and a cached lambda choice is reused only if its PASS artifact, development-only scope, dataset hash and recorded checks validate.

WOMD Colab persistence uses local `.part` → completed local file → Drive `.persisting` → validation → atomic rename. Existing Drive finals are reused only after an operational size check against the source object; Stage-1 SHA-256 provenance remains authoritative.

## What was borrowed from the organizational reference

The transferable ideas are explicit named stages, a high-level pipeline map, stage-local operator documentation, and obvious ownership of evidence. These improve navigation without altering scientific content.

The following are deliberately not copied: large ZIP archives, bulky generated manifests/reports committed merely for convenience, duplicated scientific code inside stage folders, reference-repository numerical claims, reference-specific sensing/ADB assumptions, and any stale frozen-result baggage. Those would increase repository weight, blur provenance, or risk importing a different scientific protocol.

## Backward compatibility and Stage-7 path migration

The previous canonical runner and Stage-7 contract used `artifacts/paper_final/07_analysis`, while the repository-level architecture already described the stage as `07_statistics`. New canonical runs write `07_statistics` consistently. Existing historical `07_analysis` artifacts are not silently moved or rewritten; they remain historical evidence. Any release assembled under the new contract must regenerate or deliberately migrate Stage-7 evidence and pass the Stage-8 gate. This avoids presenting old files as newly verified evidence.

## Scientific boundaries

WOMD v1.3.1 stays frozen. Historical 249,137-sample / 24,182-scenario values are fingerprints, not target counts. No future ground truth enters deployable choices. Model selection uses development only. Stage 4 remains exactly 20 checkpoints and Stage 6 remains exactly 8 families × 5 paired seeds. Scenario/episode remains the inferential unit. Negative results remain reportable. A finite Monte Carlo run with zero observed errors is not proof that true BER is zero. Frozen Stage-2 science is not changed for repository aesthetics.
