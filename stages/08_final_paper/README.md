# Stage 8 — Final paper and reproducibility release

## Scientific purpose
Assemble an evidence-consistent manuscript and reproducibility package only after the canonical upstream gates are complete.

## Required inputs
Frozen Stage-1 provenance, verified Stage-2 link evidence, Stage-4 completion manifest, Stage-5 official-validation provenance, Stage-6 scheduling manifest, Stage-7 scenario-level statistics/figures, and the manuscript source.

## Outputs
- `artifacts/paper_final/08_release/release_readiness.json`
- `artifacts/paper_final/08_release/predictive_pc_fmcw_final_paper.pdf`
- `artifacts/paper_final/08_release/reproducibility_manifest.json`

## Acceptance criteria
Release readiness passes; all claims trace to versioned canonical evidence; exactly 20 Stage-4 checkpoints are represented; official Stage-5/6/7 evidence is present; stale proxy results are not presented as official; limitations and negative findings remain visible; a clean reproduction manifest is generated and semantically verified.

## Canonical command
```bash
python stages/08_final_paper/run.py
```

## Forbidden
Do not fabricate PASS, held-out results, checkpoints, scheduling evidence or statistics. Do not weaken an upstream gate to make the release pass. Do not silently relabel historical/proxy artifacts as canonical evidence.

## Dependencies
Stage 7 and, transitively, every scientific gate in Stages 0–6.
