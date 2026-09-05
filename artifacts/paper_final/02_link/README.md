# Stage 2 canonical Part-A BER evidence

This directory contains the canonical BER lookup table generated from the supplied Part-A FFT/DPSK receiver on the frozen waveform-sample SNR grid from -5 dB to 25 dB inclusive, with 250000 evaluated decisions per SNR point and seed 20260827.

`dbpsk_ber_lut.csv` was regenerated after correcting a receiver alias-branch inconsistency. Carrier projection and midpoint phase compensation now use the same continuous unwrapped FFT-frequency branch. The previous implementation projected with wrapped FFT frequencies but compensated with their unwrapped equivalents; with the Part-A 13/14-sample symbol timing this could introduce a spurious pi rotation on half-sample transitions and invert a repeatable subset of DPSK decisions for an entire chirp.

The regenerated raw Monte Carlo `simulated_ber` values satisfy the material-reversal gate. The conservative monotone `ber_for_lut` remains the downstream quantity and still uses finite one-sided confidence bounds at zero-error points. Zero observed errors are finite Monte Carlo outcomes, not claims that the true BER is exactly zero.

`chirp_cluster_diagnostic.json` is the canonical post-fix receiver-stability diagnostic. It uses 50 independently seeded one-chirp trials at 5, 7, 8 and 10 dB, 1000 decisions per trial, and a 10000-resample cluster bootstrap over chirps. All 200 diagnostic chirps have zero observed errors and no catastrophic chirp burst. Bits within a chirp are not treated as independent inferential units.

`link_verification.json` records the Stage 2 contract result and the SHA-256 digest of the canonical LUT. Re-running the current Stage 2 commands additionally verifies the independent-chirp stability artifact and records its digest. The earlier paired 7-to-8 dB diagnostic was part of the pre-fix investigation and is not canonical evidence after the receiver correction.

Absolute received power remains normalized/model-based. These artifacts do not claim a measured optical channel.
