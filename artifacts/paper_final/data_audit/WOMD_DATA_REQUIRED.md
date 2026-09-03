# Canonical WOMD data requirement

**Status: BLOCKED — data payload not present**

The predictive PC-FMCW paper requires real WOMD Motion Dataset input for Stage 1 and all downstream canonical experiments.

Required:

- official WOMD v1.3.1 training TFRecords
- official WOMD v1.3.1 official-validation TFRecords
- scenario-level split ownership
- 11 observed/history steps and 80 future steps
- true-SDC future geometry
- immutable dataset manifests and hashes

Historical v1.3.0 evidence is retained only as provenance and must not be relabelled as v1.3.1. The old Part-B archive contains reports and small diagnostic trajectory exports, but not the raw WOMD payload or canonical NPZ.

Run `python scripts/womd_preflight.py <data-roots> --json womd_preflight.json` before Stage 1 export. The preflight fails closed: it does not synthesize, download, or relabel data.

No downstream Stage 3–8 metric is considered a final scientific result until Stage 1 passes on the canonical dataset.
