# Official WOMD training-corpus audit

The uploaded paper-scale training corpus was audited independently before any
learned-model claim was accepted. The file contains 249,137 causal true-SDC
samples from 24,182 official WOMD scenarios. Every numeric value is finite.

## Verified contents

| Property | Verified value |
|---|---:|
| File size | 278,168,355 bytes |
| SHA-256 | `b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee` |
| Samples | 249,137 |
| Training samples | 223,786 |
| Internal development samples | 25,351 |
| Unique scenarios | 24,182 |
| History | 11 steps |
| Prediction horizon | 80 steps |
| Median 8 s displacement | 12.73 m |
| P95 8 s displacement | 67.90 m |
| NaN/Inf | 0 |

The internal `development` split is derived from the downloaded training
corpus. It is **not** the official WOMD validation split. Paper-scale held-out
claims therefore remain blocked until the separately downloaded official
validation shards are converted and evaluated.

## Provenance correction

The uploaded NPZ records `real_WOMD_v1.3.0_true_SDC_geometry`, while the Colab
pipeline downloaded from the WOMD v1.3.1 bucket. The builder has been corrected
to label newly generated files as v1.3.1. The original NPZ is retained
unchanged and identified by its SHA-256 digest above.

## Current evidence boundary

This audit establishes that the project now has a real, large-scale WOMD
training corpus rather than only the three-scene proxy. It does not establish
forecasting accuracy, communication utility, or statistical significance.
Those claims require completed checkpoints plus official held-out evaluation.

Reproduce the audit with:

```bash
python scripts/08_audit_womd_dataset.py womd_training_paper.npz \
  --output artifacts/womd_official_dataset/audit.json
python scripts/09_make_womd_dataset_figure.py womd_training_paper.npz \
  --output artifacts/womd_official_dataset
```
