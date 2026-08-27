# Assignment 1 to new assignment mapping

| Assignment 1 component | New assignment use |
|---|---|
| PC-FMCW phase-coded waveform | Physical-layer premise; unchanged reference |
| DPSK data at 1 Gbit/s | Nominal PHY rate and DBPSK BER model |
| Range/Doppler sensing | Source of future relative geometry concept |
| MHT tracking | Upstream state-estimation context only |
| ADB illumination | Explicitly outside the new scheduler decision variable |
| Notebook/report | Preserved under `reference/assignment1` |
| New trajectory forecasts | Converted to future link quality and lifetime |
| New packet simulator | Separates nominal rate from successful delivery |
| New scheduler | Decides receiver and transmission time per slot |
| New communication-aware loss | Optimizes downstream link relevance, not ADE alone |

The new code is a separate Python package so Assignment 1 remains reproducible
and auditable. No cell in the original notebook is silently modified.

