# Stage 2 - PC-FMCW/DPSK link

Regenerates and freezes the Part-A receiver-derived BER mapping. It records the
waveform-sample SNR semantics and retains the explicit limitation that absolute
received optical power is not measurement-calibrated.

The canonical receiver uses one continuous unwrapped FFT-frequency alias branch
for both per-symbol carrier projection and midpoint phase compensation. This
avoids the deterministic pi-rotation failure that can occur when wrapped and
unwrapped branches are mixed across the Part-A 13/14-sample symbol timing.

Raw bit decisions remain clustered within chirps because a chirp shares the
receiver tracking process. The canonical stability diagnostic therefore uses
independently seeded one-chirp trials at 5, 7, 8 and 10 dB and treats the chirp,
not each bit, as the inferential unit. The targeted
`scripts/02_diagnose_part_a_paired_reversal.py` utility remains available when a
future regenerated LUT exhibits a material adjacent raw-BER reversal.

Canonical outputs are:

- `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`
- `artifacts/paper_final/02_link/chirp_cluster_diagnostic.json`
- `artifacts/paper_final/02_link/link_verification.json`

```bash
python stages/02_pc_fmcw_dpsk_link/run.py
```
