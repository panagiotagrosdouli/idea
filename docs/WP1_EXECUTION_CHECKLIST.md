# WP1 execution checklist

This checklist is the paper-oriented execution contract for Stages 00–02. It does not replace the executable `stage.json` files; it summarizes the evidence that must exist before predictor training/evaluation is treated as publication evidence.

## Stage 00 — freeze and provenance

Status: PASS from the repository audit baseline.

Required evidence remains under `artifacts/audit/` and must not be reinterpreted as proof of later scientific stages.

## Stage 01 — official WOMD corpus freeze

Required real inputs:

- official WOMD v1.3.1 training TFRecords / frozen training NPZ;
- official WOMD v1.3.1 validation TFRecords / frozen validation NPZ;
- true-SDC geometry only;
- history/future shape fixed to 11 observed states and 80 future states.

Canonical export commands:

```bash
PYTHONPATH=src python scripts/01_build_official_womd_samples.py \
  /data/womd/training/*.tfrecord \
  --output data/processed/womd_v131_training.npz \
  --max-vehicles 16

PYTHONPATH=src python scripts/01_build_official_womd_samples.py \
  /data/womd/validation/*.tfrecord \
  --output data/processed/womd_v131_official_validation.npz \
  --fixed-split official_validation \
  --max-vehicles 16
```

Then run the three commands already defined by `stages/01_womd_data_pipeline/stage.json`.

Gate 1 may be marked PASS only if the real corpora produce:

- finite arrays and expected tensor lengths;
- canonical source and coordinate-frame metadata;
- scenario-safe training/development labels;
- validation containing only `official_validation`;
- zero overlap of scenario IDs between training/development and official validation;
- recorded SHA-256 hashes for both frozen corpora;
- `training_audit.json`, `validation_audit.json`, and `corpus_verification.json` under `artifacts/paper_final/01_data/`.

Synthetic/example fixtures may test code paths but cannot close Gate 1.

## Stage 02 — Part-A PC-FMCW/DPSK link freeze

The canonical link artifact must come from the supplied Part-A receiver-derived path and the frozen Part-A configuration. The geometry-dependent optical link remains model-based/calibrated unless measurement evidence is added.

Gate 2 requires:

- the canonical 31-point SNR grid defined by the current Stage-2 implementation;
- 250000 bits per SNR point for the publication LUT;
- receiver mode and SNR semantics recorded;
- finite BER values in the allowed range and monotonicity/statistical sanity checks;
- LUT, config and verification hashes recorded;
- `artifacts/paper_final/02_link/dbpsk_ber_lut.csv` and its verification evidence.

Synthetic/smoke BER runs do not substitute for the publication LUT.

## WP1 exit gate

Do not start a final paper-scale GRU archive or official held-out claims until both Gate 1 and Gate 2 have real, frozen evidence. Code development and tests for later stages may continue, but missing official evidence must remain explicitly BLOCKED/PARTIAL.

After WP1 is closed, continue with:

`Stage 03 classical development baselines -> Stage 04 communication-aware GRU -> Stage 05 untouched official validation`.
