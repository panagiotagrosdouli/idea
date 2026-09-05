# Stage 2 — PC-FMCW/DPSK link

## Scientific purpose
Regenerate, verify and freeze the Part-A receiver-derived PC-FMCW/DPSK BER mapping used downstream. This stage owns physical-layer evidence, not prediction or scheduling logic.

## Required inputs
- `configs/part_a_physical_layer.json`
- frozen Stage-1/protocol context through the stage graph

## Outputs
- `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`
- `artifacts/paper_final/02_link/chirp_cluster_diagnostic.json`
- `artifacts/paper_final/02_link/link_verification.json`

## Acceptance criteria
The frozen 31-point −5 to 25 dB waveform-sample SNR grid is used; each point has at least 250,000 simulated bits; zero-error points use a confidence-aware bound; the conservative LUT is monotone; Part-A receiver/SNR semantics verify; chirp-cluster diagnostics show no prohibited instability; model-based absolute optical power is disclosed.

The receiver keeps one continuous unwrapped FFT-frequency alias branch for carrier projection and midpoint compensation. Chirps, not individual within-chirp bits, are the relevant independent unit for the stability diagnostic.

## Canonical command
```bash
python stages/02_pc_fmcw_dpsk_link/run.py
```

## Forbidden
Do not change frozen Stage-2 science to make downstream evidence pass. Do not interpret finite zero observed errors as proof that the true BER is exactly zero. Do not claim measured/calibrated optical power when it is model-based.

## Dependencies
Stage 1 in the formal stage graph. The Stage-1-only production path does not execute or require Stage 2; the full downstream path requires this frozen verified evidence.
