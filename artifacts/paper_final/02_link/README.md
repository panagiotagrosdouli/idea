# Stage 2 canonical Part-A BER evidence

This directory contains the canonical BER lookup table generated from the supplied Part-A FFT/DPSK receiver with the frozen waveform-sample SNR grid from -5 dB to 25 dB inclusive, 250000 evaluated decisions per SNR point, and seed 20260827.

`dbpsk_ber_lut.csv` preserves both the raw Monte Carlo `simulated_ber` and the conservative monotone `ber_for_lut` used downstream. It was regenerated after correcting the FFT alias-branch inconsistency between carrier projection and midpoint phase compensation. The raw estimates now satisfy the material-reversal gate; the predefined reverse-cumulative rule remains applied to `ber_for_lut`, including finite confidence bounds at zero-error points.

`link_verification.json` records the regenerated Stage 2 contract result and the SHA-256 digest of the LUT. All checks pass, including the strengthened raw-BER material-reversal gate. Absolute received power remains normalized/model-based; this artifact does not claim a measured optical channel.

`chirp_cluster_diagnostic.json` is a separate receiver-stability diagnostic. It
uses 50 independently seeded one-chirp trials at 5, 7, 8 and 10 dB, with 1,000
decisions per trial and a 10,000-resample cluster bootstrap over chirps. It does
not replace the canonical LUT. After the alias-branch correction, all 200
diagnostic chirps have zero observed errors. This is a finite diagnostic result,
not a claim that the true BER is exactly zero; the canonical LUT retains
confidence-bound handling for zero-error points.
