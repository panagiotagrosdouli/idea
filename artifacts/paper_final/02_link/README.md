# Stage 2 canonical Part-A BER evidence

This directory contains the canonical BER lookup table generated from the supplied Part-A FFT/DPSK receiver with the frozen waveform-sample SNR grid from -5 dB to 25 dB inclusive, 250000 evaluated decisions per SNR point, and seed 20260827.

`dbpsk_ber_lut.csv` preserves both the raw Monte Carlo `simulated_ber` and the conservative monotone `ber_for_lut` used downstream. The raw receiver estimates are not monotone at every SNR point; the LUT therefore uses the repository's predefined reverse-cumulative conservative monotonicity rule rather than altering the raw measurements.

`link_verification.json` records the original Stage 2 contract result and the SHA-256 digest of the LUT. The raw estimates contain materially non-monotone spikes and therefore require receiver-level, chirp-cluster-aware diagnosis under the strengthened scientific gate before the artifact is treated as fully validated. Absolute received power remains normalized/model-based; this artifact does not claim a measured optical channel.

`chirp_cluster_diagnostic.json` is a separate receiver-stability diagnostic. It
uses 50 independently seeded one-chirp trials at 5, 7, 8 and 10 dB, with 1,000
decisions per trial and a 10,000-resample cluster bootstrap over chirps. It does
not replace the canonical LUT. The observed error distribution is strongly
bursty: the median trial BER is zero, while a subset of trials reaches BER
0.214. This supports receiver-level investigation before refreezing Stage 2.
