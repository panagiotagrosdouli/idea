# Stage 2 canonical Part-A BER evidence

This directory contains the canonical BER lookup table generated from the supplied Part-A FFT/DPSK receiver with the frozen waveform-sample SNR grid from -5 dB to 25 dB inclusive, 250000 evaluated decisions per SNR point, and seed 20260827.

`dbpsk_ber_lut.csv` preserves both the raw Monte Carlo `simulated_ber` and the conservative monotone `ber_for_lut` used downstream. The raw receiver estimates are not monotone at every SNR point; the LUT therefore uses the repository's predefined reverse-cumulative conservative monotonicity rule rather than altering the raw measurements.

`link_verification.json` records the Stage 2 contract result and the SHA-256
digests of the LUT and paired diagnostic. The original 7→8 dB aggregate
increase is not supported when 100 independent chirps are compared using common
payload/noise realizations: mean BER decreases from 0.02360 to 0.01717 and the
95% paired chirp-bootstrap interval for the higher-minus-lower difference is
[-0.01499, -0.00001]. The burst-failure regime remains explicitly visible and
must not be described as independent bit errors. Absolute received power remains
normalized/model-based; this artifact does not claim a measured optical channel.
