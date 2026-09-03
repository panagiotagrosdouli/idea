# Stage 6 - Packet scheduling

Runs Reactive, PF, CV, Kalman, IMM, Link-Lifetime, Learned and Oracle-information
schedulers with paired traffic/channel randomness. Packet success is always
realized from the ground-truth-derived link.

Inputs: validation TFRecords and checkpoints. Outputs: `artifacts/paper_final/06_scheduling/`.

```bash
python stages/06_packet_scheduling/run.py
```
