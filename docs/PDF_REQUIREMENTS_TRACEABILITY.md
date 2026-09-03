# Non-Joint PDF requirements traceability

This audit covers all three supplied predictive PC-FMCW/DPSK research plans,
the Part-A PC-FMCW/DPSK paper and notebook-derived code, and the general thesis
topic document. The separate **Joint trajectory-to-beam-and-ADB** proposal is
explicitly outside this repository.

Status meanings:

- **Executed:** implemented, tested, and represented by a current artifact.
- **Implemented:** executable code exists, but the required external input was
  not supplied.
- **Partial:** a defensible subset exists; the complete requested evidence does
  not.
- **Blocked:** the evidence cannot be produced from the supplied files.
- **Optional:** suggested by a plan, but not necessary for the core study.

## Motion, data and causality

| PDF requirement | Evidence | Status |
|---|---|---|
| Strictly causal history through time `t` | predictor API; future-mutation validation gate | Executed |
| Scenario-safe split and provenance hashes | manifest/split/checkpoint schema modules; audited paper NPZ SHA-256 | Executed on official training corpus |
| Official WOMD true SDC | `data/womd_official.py`, true `sdc_track_index`, validity masks; 249,137 samples from 24,182 scenarios | Executed on official training corpus; held-out validation pending |
| Compact supplied WOMD export | proxy dataset manifest, motion/link evaluation | Executed on 3 scene IDs |
| Last/CV/CA/Kalman/IMM baselines | predictor library and forecast artifacts | Executed |
| Existing/GRU model | versioned GRU inference/training code; 12 smoke checkpoints reported by Colab | Partial; paper-scale checkpoints pending |
| Classical probabilistic baseline and calibration | residual Gaussian CV/CA; six calibration and six disjoint evaluation scenarios | Executed, controlled |
| Learned Gaussian/GMM predictor and calibration | development-only residual Gaussian wrapper now reports held-out NLL and 50/90/95% coverage | Implemented; official execution pending |
| Transformer/SOTA comparator | not implemented | Optional, not done |
| ADE/FDE and range/bearing/SNR metrics | forecast evaluator, CSV/JSON summaries | Executed |
| Outage F1/AUROC and link-lifetime error | forecast evaluator and boundary diagnostics | Executed |

The supplied Stage-4 JSON contains 70 causal predicted trajectories, but no
paired future ground truth. It is preserved as provenance and cannot honestly
produce ADE/FDE, calibration, or communication KPIs.

## Part-A physical layer and link mapping

| PDF requirement | Evidence | Status |
|---|---|---|
| Freeze `fc=193.4 THz`, `B=10 GHz`, chirp `10 us`, `Rb=1 Gbit/s` | physical-layer config and provenance | Executed |
| Use Part-A FFT-carrier/DPSK receiver logic | notebook-derived receiver in `ber.py` | Executed |
| BER LUT with Monte Carlo confidence handling | adaptive bits, Wilson upper bound, monotone conservative LUT | Executed |
| Same geometry/link mapping for prediction and truth | shared `geometry.py` and `LinkModel` | Executed |
| Range, pointing/FoV, attenuation and reference SNR | normalized channel model and ablations | Executed, model-based |
| BER → PER → successful goodput | link model and monotonicity tests | Executed |
| BER/PER/goodput outage definitions | config, metrics and staged study | Executed |
| Absolute received power / calibrated optical budget | no transmitter/receiver calibration or measurements supplied | Blocked; not claimed |
| Measured optical-channel validation | no measured channel dataset supplied | Blocked; not claimed |

The Part-A waveform code uses **waveform-sample SNR**, not a silently relabeled
`Eb/N0`. Zero observed errors are reported with an upper confidence bound, not
as proof of zero BER.

## Packets, traffic and schedulers

| PDF requirement | Evidence | Status |
|---|---|---|
| FIFO queues, deadlines, retries and finite capacity | traffic and packet simulator | Executed |
| Poisson and bursty traffic | Poisson, periodic, Markov-modulated, saturated modes | Executed |
| Urgent and bulk packet classes | separate deadlines and class KPIs | Executed |
| Random/RR/Reactive/PF baselines | scheduling policies | Executed |
| CV/Kalman/IMM predictive policies | scheduling policies | Executed |
| Predictive utility, prefetch and link-lifetime urgency | policies and exact unit tests | Executed |
| Perfect-future reference | oracle-information heuristic | Executed |
| Global offline optimal scheduler | not required by the core plans; oracle is not mislabeled as optimal | Optional, not done |
| Common random numbers | identical traces and per-attempt uniforms across policies | Executed |
| Ground-truth-derived packet success | simulator separates decision forecast from realization | Executed |

## Learning, studies, metrics and artifacts

| PDF requirement | Evidence | Status |
|---|---|---|
| Trajectory/link/outage/full communication-aware losses | resumable GRU four-objective runner | Executed as smoke test; paper-scale training pending GPU quota |
| Three or more training seeds | three-seed immutable plan and 12-run smoke result | Executed as smoke test; paper-scale evidence pending |
| Communication-loss weight sweep | resumable one-axis link/outage lambda sweep over five settings and three seeds | Implemented; GPU execution pending |
| Official held-out learned evaluation | validation-only label gate plus per-scenario ADE/FDE/link/outage/goodput/lifetime/NLL/coverage evaluator | Implemented; official-validation NPZ and checkpoints pending |
| Learned objective analysis | objective tables, paired scenario-cluster statistics, ADE-vs-link-fidelity figure, explicit scheduler-utility scope warning | Implemented; execution pending held-out rows |
| ADE versus realized communication utility | official-WOMD eight-scheduler benchmark with five paired traffic seeds, joined by objective/model seed/scenario | Implemented; execution pending checkpoints and GPU |
| Horizon/load/channel/traffic/sensing/FoV sweeps | 12 one-axis studies | Executed |
| Five independent staged seeds | 1,125 policy rows in `corrected_v2` | Executed |
| PDR, goodput, P50/P95/P99, miss/censoring, fairness | machine-readable episode metrics | Executed |
| Scenario bootstrap, paired tests, Holm correction | statistics module and summaries | Executed |
| Architecture, motion, link trace and scheduler timeline | publication figures | Executed |
| BER, ECDF, outage-latency, Pareto and lifetime calibration | publication figures | Executed |
| Failure-case analysis | uncensored optical-boundary error figure | Executed |
| Runtime/complexity | CPU median/P95 for classical code; analytical GRU parameters | Partial: GRU runtime blocked |
| Causality, conservation, monotonicity, perfect prediction, H=0 and reproducibility tests | 57 deterministic tests + 5 scientific gates | Executed |
| Full manuscript and supplement | Markdown manuscript, generated PDF, traceability and manifests | Partial: evidence/metadata blockers remain |

## Exact publication blockers

1. A large official WOMD v1.3.1 training corpus now exists and has been audited,
   but a separately converted official-validation NPZ has not yet been supplied,
   so there is no large true-SDC held-out result.
2. Twelve smoke checkpoints were reported by Colab, but the paper-scale archive
   has not yet been supplied because the free GPU quota stopped training.
3. No optical measurement/calibration dataset was supplied, so absolute-power
   and real-channel claims are prohibited.
4. Authors, affiliations and target-venue template are still placeholders.

“Executed” means the requested code path was actually run and produced a
current artifact. It does not mean the scientific hypothesis was favorable.
