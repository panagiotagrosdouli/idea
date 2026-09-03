# Stage 2 - PC-FMCW/DPSK link

Regenerates and freezes the Part-A receiver-derived BER mapping. It records the
waveform-sample SNR semantics and retains the explicit limitation that absolute
received optical power is not measurement-calibrated.

Output: `artifacts/paper_final/02_link/dbpsk_ber_lut.csv`.

```bash
python stages/02_pc_fmcw_dpsk_link/run.py
```
