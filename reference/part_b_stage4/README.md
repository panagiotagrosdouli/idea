# Supplied Part-B Stage-4 evidence

These JSON files were copied from the user-supplied Part-B archive. They are
retained as provenance, not as locally reproduced results.

- Source archive SHA-256: `47018e2e9059d5ac1e295af33ebdbdb77ff0fdd5669c1238cf3d3a675b598337`
- Reported dataset release: WOMD/WOMD-LiDAR v1.3.0
- Reported selected scenarios: 116,182 (72,085 training; 44,097 validation)
- Included here: selection configuration, selected-model summary, final result
  report, prediction artifact and reproducibility manifest.
- Missing from every supplied archive: raw WOMD shards and model checkpoints
  (`.pt`, `.pth`, `.ckpt`).

The upstream Stage-4 models consume an eight-feature state interface. The local
research pipeline consumes ego-relative XY histories and requires an explicit
versioned checkpoint schema. Therefore the reported upstream ADE/FDE values
must not be labelled as a local rerun or as downstream scheduler evidence.
