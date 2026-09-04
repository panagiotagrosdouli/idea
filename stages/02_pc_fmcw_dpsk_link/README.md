# Stage 2 - PC-FMCW/DPSK link

Regenerates and freezes the Part-A receiver-derived BER mapping. It records the
waveform-sample SNR semantics and retains the explicit limitation that absolute
received optical power is not measurement-calibrated.

Raw bit decisions are clustered within chirps because each chirp shares a
carrier estimate. Apparent adjacent reversals must therefore be investigated
with `scripts/02_diagnose_part_a_receiver.py`, which uses independent chirps as
the sampling unit and common payload/noise realizations within each paired SNR
comparison. Aggregate bit-level binomial tests are not valid for this purpose.

Output: `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`.

```bash
python stages/02_pc_fmcw_dpsk_link/run.py
make stage2-diagnostic
```
