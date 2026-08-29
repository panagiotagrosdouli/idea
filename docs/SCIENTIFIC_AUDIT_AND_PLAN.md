# Scientific repository audit and publication plan

> **Historical audit snapshot.** This document records the initial gap analysis
> before the corrected-v1 implementation pass. Counts and readiness statements
> below are intentionally preserved as audit evidence and must not be read as the
> current repository status. For the current 45-test implementation state, use
> [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md),
> [`RESULTS.md`](RESULTS.md), and [`PAPER_READINESS.md`](PAPER_READINESS.md).

**Audit date:** 2026-08-29  
**Scope:** mobility prediction → PC-FMCW/DPSK link forecasting → link-lifetime-aware packet scheduling  
**Explicitly excluded:** Joint Beam/ADB, Adaptive Top-K beam selection, glare and illumination optimization  
**Audit rule:** source code, automated tests and experiment artifacts are the evidence. A design document alone is not experimental evidence.

## Executive verdict

The repository is a coherent, executable research prototype with a complete causal path from motion to packet-level scheduling. It is **not yet submission-ready empirical evidence**.

What is defensible today:

- the non-Joint system architecture is implemented;
- all 28 available automated tests pass in the audited workspace;
- the scientific sanity gates pass, including causal-future invariance and oracle sensitivity;
- controlled synthetic experiments show a small goodput/PDR benefit from link-lifetime urgency, with an important latency trade-off;
- an existing 3,600-row exploratory matrix suggests that the benefit is operating-region dependent and can become negative at high load or long horizons;
- the compact three-scene WOMD proxy result is negative and is correctly retained;
- no learned-model, measured-channel or full-WOMD result has been fabricated.

What is not defensible today:

- a general claim that prediction improves communication;
- a full-WOMD generalization claim;
- a communication-aware GRU performance claim;
- an absolute received-power or measured optical-channel claim;
- a claim that the current Oracle policy is globally optimal;
- final inferential statistics from the quick or exploratory matrices.

The correct current label is:

> **Paper-oriented research code and an honest mechanism study; not yet a publication-complete empirical paper.**

## 1. Repository audit

### 1.1 Inventory

| Area | What is present | Audited evidence | Verdict |
|---|---|---|---|
| Core package | 32 Python source modules under `src/predictive_pc_fmcw/` | importable CLI and direct execution | IMPLEMENTED |
| Tests | 11 test files, 28 test cases | `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` → 28/28 PASS | TESTED |
| Scientific gates | distance/pointing/BER monotonicity; causal invariance; oracle sensitivity | `pcfmcw validate` → PASS | TESTED |
| Configs | default, legacy exploratory matrix and declared paper matrix | three JSON files under `configs/` | IMPLEMENTED |
| Controlled artifacts | benchmark, BER LUT, horizon and 3,600-row matrix outputs | `artifacts/` | EXPERIMENTALLY VALIDATED, controlled only |
| Quick paper artifacts | 224-row matrix, 12-episode benchmark, proxy-WOMD benchmark, ablations, figures and tables | `artifacts/paper_run/` | EXPERIMENTALLY VALIDATED, diagnostic only |
| WOMD input | 56 actor records, three scenario IDs, 10 past + 10 future samples | `data/example/womd_trajectories.json` and manifest | PARTIAL |
| Official WOMD | upstream selection metadata says WOMD/WOMD-LiDAR v1.3.0 and 116,182 selected scenarios | supplied Part-B archive metadata; raw shards absent | DOCUMENTED ONLY / BLOCKED |
| Learned predictors | local GRU training/inference infrastructure; supplied Stage-4 result reports | no `.pt`, `.pth` or `.ckpt` in supplied archives | PARTIAL / BLOCKED |
| PC-FMCW Part A | supplied notebook/report and declared constants | upstream notebook; local reference-SNR abstraction | PARTIAL |
| Manuscript | six-page draft PDF and Markdown source | `paper/PAPER_DRAFT.md`, `output/pdf/…pdf` | DRAFT, not submission-ready |
| README visuals | two generated project visuals | `docs/assets/readme-hero.webp`, `docs/assets/readme-predictive-scheduler.webp` | ASSETS READY; README integration intentionally paused |
| GitHub state | PR #5 merged into `main` | merge commit `5a3b17ebf51f8e77084fa7a85c6b888c464c9b32` | VERIFIED REMOTELY |
| Local provenance | workspace copy has no normal commit history; stored manifest says `uncommitted-workspace` | `git status`, reproducibility manifest | GAP |

### 1.2 Implemented end-to-end chain

The following path exists and executes:

```text
observed positions through time t
→ causal trajectory forecast
→ ego-relative distance and bearing
→ normalized geometry-dependent optical gain
→ SNR → DBPSK BER → packet PER
→ packet delivery and successful goodput
→ future outage and remaining link lifetime
→ queue/deadline-aware scheduler decision
→ delivery using the ground-truth-derived current link
→ packet, outage, latency and fairness metrics
```

This is the correct non-Joint research scope. The decision is which receiver to serve in the current slot; beam indices and ADB illumination actions are not decision variables.

### 1.3 Existing empirical evidence

#### Controlled 12-episode default benchmark

| Policy | Goodput (Mbps) | PDR | P95 latency (ms) | Delivered before expiry | Jain fairness |
|---|---:|---:|---:|---:|---:|
| Reactive Greedy | 2.293 | 0.6439 | 699.6 | 0.8010 | 0.5355 |
| Generic Predictive Utility | 2.3039 | 0.6469 | 965.4 | 0.8014 | 0.5644 |
| Link Lifetime | 2.3358 | 0.6558 | 916.7 | 0.8136 | 0.5683 |
| Oracle-information reference | 2.3391 | 0.6568 | 916.7 | 0.8148 | 0.5659 |

The link-lifetime method gains approximately 1.9% goodput over Reactive Greedy in this controlled default. It also worsens P95 latency by about 217 ms. The defensible interpretation is a **throughput/expiry/fairness versus tail-latency trade-off**, not universal dominance.

#### Current 224-row quick paper matrix

- only two values from each configured axis and two seeds are used;
- Link Lifetime − Reactive mean goodput difference: **+0.04266 Mbps**;
- diagnostic bootstrap 95% interval: **[+0.02291, +0.06287] Mbps**;
- win fraction: **87.5%**;
- the result is useful for integration testing but not final inference because only two independent seeds contribute.

#### Existing 3,600-row exploratory matrix

This artifact predates the declared paper matrix. It contains five policies and 720 paired Link-Lifetime/Reactive operating points.

| Slice | Mean goodput difference (Mbps) | Win fraction |
|---|---:|---:|
| Overall | +0.00545 | 61.5% |
| Load 0.35 | +0.02303 | 69.4% |
| Load 0.55 | +0.02409 | 71.1% |
| Load 0.75 | +0.00386 | 58.3% |
| Load 0.90 | **−0.02916** | 47.2% |
| Horizon 3 steps | +0.02524 | 72.8% |
| Horizon 20 steps | **−0.02330** | 48.3% |

This is promising evidence for the question “when does prediction help?”, but it is exploratory rather than publication-grade because physical horizon, scenario duration and deadline duration are confounded by the slot-duration sweep. The 720 operating points also share only five underlying seeds; treating all points as independent would be pseudoreplication. Per-seed aggregate gains are negative for three of the five seeds.

#### Compact WOMD proxy benchmark

| Policy | Goodput (Mbps) | PDR | P95 latency (ms) |
|---|---:|---:|---:|
| Reactive Greedy | 1.160 | 0.3288 | 368.3 |
| Link Lifetime | **1.056** | 0.2990 | 400.0 |
| Oracle-information reference | **0.976** | 0.2767 | 400.0 |

This result is negative and must remain in the paper trail. It uses only three scenes, a medoid proxy ego, one second of evaluation and model-based communication. It is an integration/failure-case artifact, not real-world optical validation.

### 1.4 Supplied upstream Part-B evidence

The large supplied Part-B archive contains Stage-4 reports and manifests that describe:

- exact release metadata: `WOMD/WOMD-LiDAR v1.3.0`;
- 116,182 selected paired scenarios: 72,085 training and 44,097 validation;
- deterministic GRU reported ADE/FDE of approximately 0.134/0.267 m;
- Gaussian GRU reported best-ADE result of approximately 0.276/0.440 m on 1,217,244 validation samples;
- GMM K=5 reported minADE/minFDE of approximately 0.494/0.639 m;
- manifest flags reporting causal prediction and no future used as input.

However, the referenced checkpoints and raw WOMD shards are absent. The upstream models use an eight-feature state representation, while the local optional GRU consumes relative position plus internally derived velocity. Therefore the reports are useful provenance, but the local scheduler cannot honestly claim to have rerun or integrated those trained models.

## 2. Claim–evidence matrix

Status vocabulary:

- **IMPLEMENTED:** executable code path exists.
- **TESTED:** a relevant automated test passes.
- **EXPERIMENTALLY VALIDATED:** a stored, reproducible result exercises the claim.
- **PARTIAL:** useful pieces exist but the claim is incomplete.
- **DOCUMENTED ONLY:** text or upstream report exists without a locally reproducible result.
- **MISSING:** no adequate implementation/evidence exists.
- **BLOCKED:** required external data, checkpoint, measurements or compute output is absent.

| Potential paper claim | Implementation | Automated test | Experiment/artifact | Evidence status | Gap before claim |
|---|---|---|---|---|---|
| The system is non-Joint receiver scheduling | scheduling decision selects at most one receiver | scheduler eligibility test | controlled benchmark | IMPLEMENTED + TESTED | retain explicit exclusion in all docs |
| Deployable predictors are causal | `forecast_scenario` slices history through `t` | future-mutation tests for Last/CV/Kalman/IMM | validation report | IMPLEMENTED + TESTED | add learned-checkpoint causality test when a checkpoint exists |
| Oracle responds to hidden future | oracle path reads future samples | oracle-sensitivity gate | validation report | IMPLEMENTED + TESTED | rename remaining “upper bound” wording |
| Last/CV/CA/Kalman/IMM baselines exist | `predictors.py` | linear-motion and leakage tests | motion summaries | IMPLEMENTED + TESTED + VALIDATED | IMM is a lightweight CV/CA approximation, not a full canonical IMM |
| A trained deterministic/uncertain WOMD model is integrated | local learned adapter exists | none with a real checkpoint | upstream Stage-4 reports only | PARTIAL / BLOCKED | checkpoint, matching eight-feature adapter and official shards |
| Communication-aware GRU improves link decisions | loss and training infrastructure exist | NumPy loss unit tests only | no trained checkpoint/result | IMPLEMENTED INFRASTRUCTURE ONLY | coordinate/FoV fixes, real training, multi-seed ablation |
| ADE/FDE are evaluated | forecast evaluation module | oracle zero-error test | synthetic/proxy summaries | IMPLEMENTED + TESTED + VALIDATED | add confidence intervals and scenario slices |
| Similar ADE can imply different link utility | ADE, SNR, outage, lifetime metrics coexist | metric helper tests | small summaries | PARTIAL | paired predictor-ranking study near link boundaries |
| Geometry maps trajectories to range/bearing | `geometry.py` | range/bearing tests | forecast summaries | IMPLEMENTED + TESTED | explicit frame metadata, offsets and stationary-heading fix |
| Link model is PC-FMCW/DPSK-informed | reference-SNR geometry abstraction and DBPSK chain | monotonicity tests | controlled ablations | PARTIAL | no calibrated optical budget or measurements |
| Absolute received power is physically calibrated | normalized 1 µW reference is used | no absolute-budget test | none | MISSING | either calibrate from Part A/measurements or label as normalized proxy |
| Part-A physical constants are frozen | 1 Gbit/s is configured; other constants appear in draft text | no provenance test | supplied notebook | DOCUMENTED/PARTIAL | add fc, B, chirp period and source to immutable config/manifests |
| BER is calibrated by Part A | symbol-level DBPSK Monte Carlo LUT | theory-tracking test | LUT −4…16 dB, 250k bits/point | PARTIAL | required range, confidence bounds and exact relationship to notebook receiver |
| BER→PER→successful goodput is packet based | `link.py` and simulation engine | boundedness and conservation tests | benchmarks | IMPLEMENTED + TESTED | reconcile BER outage threshold with packet usability |
| Future link quality is forecast | SNR/outage trace from predicted geometry | oracle zero-error test | motion/link summaries | IMPLEMENTED + TESTED | add visible-link SNR error and threshold-stratified metrics |
| Link lifetime is predicted | first future outage step | oracle zero-error test | scheduler and motion artifacts | IMPLEMENTED + TESTED | report seconds and handle censored/no-outage horizons explicitly |
| Ground truth evaluates delivery | current actual geometry drives PER draw | end-to-end test | benchmarks | IMPLEMENTED | add direct anti-self-confirmation regression test |
| Common random numbers ensure fair policies | one traffic trace reused per scenario | traffic reproducibility test | paired benchmarks | IMPLEMENTED + TESTED | explicitly test identical traces across all policy runs |
| Queue/deadline packet conservation holds | FIFO queues, failure requeue, drops | conservation test | benchmark artifacts | IMPLEMENTED + TESTED | address right-censoring at short episode end |
| Poisson/periodic/Markov traffic exists | `traffic.py` | reproducibility tests | ablation summary | IMPLEMENTED + TESTED | saturated/bulk traffic missing |
| Random/RR/Reactive/PF baselines exist | scheduler library | one-eligible-vehicle test | benchmark | IMPLEMENTED + TESTED | PF is not a standard exponentially averaged PF implementation |
| Predictive CV/Kalman/IMM/utility policies exist | scheduler library | integration tests | benchmark | IMPLEMENTED + TESTED + VALIDATED | final paired operating-region results missing |
| Link-Lifetime urgency is a distinct mechanism | separate urgency term | integration test only | no-lifetime ablation | IMPLEMENTED + VALIDATED | normalization stability and pre-registered weight ablation needed |
| Oracle is an information reference | perfect future with same heuristic utility | oracle context guard | benchmarks | IMPLEMENTED + TESTED | not an optimal scheduler and not a mathematical upper bound |
| Communication metrics are complete | goodput, PDR, outages, mean/P95, miss, fairness, expiry | boundedness/conservation tests | summaries | PARTIAL | P50/P99 and scheduled BER/PER/SNR/normalized power missing |
| Link-forecast metrics are complete | range MAE, SNR MAE, F1, AUROC, lifetime error | helper/oracle tests | summaries | PARTIAL | bearing error, class support and CI/calibration details |
| Sensing uncertainty is modeled | iid coordinate noise knobs | no covariance test | 0.5/1/2 m stress ablations | PARTIAL | no PC-FMCW-like observation model or covariance-aware filter |
| Prediction helps in a defined operating region | matrix infrastructure exists | matrix smoke tests | 3,600-row exploratory matrix | PARTIAL / EXPLORATORY | deconfounded staged experiment and independent-seed inference |
| Full declared paper matrix is complete | config declares 11,340 rows | quick matrix smoke test | only 224-row quick output | MISSING COMPUTE OUTPUT | execute only after validity fixes |
| Statistical claims are controlled | paired bootstrap, t, Wilcoxon, Cohen dz | no stats regression suite | quick goodput summary | PARTIAL | multi-metric direction, Holm correction, cluster dependence |
| Official WOMD evaluation is strong | compact adapter only | proxy adapter test | three-scene benchmark | BLOCKED | raw official shards, true SDC, scenario-safe split and scale |
| Reproducibility is publication-grade | scripts/configs/manifests exist | pipeline components tested | manifest records environment | PARTIAL | current artifacts say `uncommitted-workspace`; add commands and actual commit |
| Paper figures/tables are final | generators and six figures exist | no render-regression test | quick artifacts | PARTIAL | regenerate after final experiments; add required CDF/slice/trade-off figures |
| Manuscript is ready for submission | six-page draft exists | PDF built previously | draft PDF | DOCUMENTED/DRAFT | final evidence, references, author metadata and complete limitations |

## 3. Scientific gaps

### P0 — required before a paper claim

1. **Fix validity defects before new large runs.** Resolve coordinate-frame mismatch in the communication-aware loss, stationary-heading inconsistency, end-of-record horizon handling and time-unit confounds.
2. **Freeze a valid data source.** Use official WOMD shards with true `sdc_track_index`, validity masks, vehicle filtering, scenario-level train/validation/test splits and exact release hashes.
3. **Integrate or retrain a real predictor.** The upstream checkpoint is missing and its feature interface differs from the local model. No learned result may be reported until a compatible frozen checkpoint exists.
4. **Run the required learned ablation.** Trajectory-only versus link-only/outage-only/full communication-aware objectives, with multiple seeds and link-level validation.
5. **Deconfound the experiment design.** Keep physical duration and deadlines constant while sweeping slot duration; express horizons and lifetimes in seconds.
6. **Repair statistical inference.** Use scenario/seed-level clustered paired analysis, multiple-comparison correction and metric-specific direction.
7. **Clarify link calibration.** Freeze Part-A constants and label received power as normalized unless a real link budget/measurement calibration is supplied.
8. **Regenerate all main results after the fixes.** Existing results must not be silently mixed with corrected results.

### P1 — very important

1. Add scenario-regime labels: straight, closing/receding, lane change, turn/merge, FoV edge and dense traffic.
2. Add P50/P99 latency, scheduled SNR/BER/PER and properly normalized service fairness.
3. Add saturated/bulk traffic, physical deadline grids and packet-size sensitivity.
4. Make link-lifetime utility normalization stable across horizon and slot duration, then test it without tuning on the test set.
5. Add explicit support counts and threshold-stratified link metrics near FoV/outage boundaries.
6. Add adaptive/high-confidence BER estimation over the planned SNR range.
7. Record exact command, config, seed, data split, commit and environment beside every final result.

### P2 — useful extensions

1. PC-FMCW-like synthetic observation model with declared assumptions and covariance `R_t`.
2. Covariance-aware Kalman/learned prediction and uncertainty propagation to link lifetime.
3. Offline finite-horizon scheduling optimization as a separate small-scale upper bound.
4. Gaussian/GMM uncertainty adapter if compatible checkpoints and features become available.
5. Blockage/weather sensitivity as model-based robustness, clearly separated from measurements.

### P3 — future work, not current scope

- handover and multi-transmitter coordination;
- Transformer or large motion-prediction architecture search;
- raw LiDAR ingestion;
- reinforcement learning scheduling;
- Joint Beam/ADB/illumination optimization.

## 4. Bug and validity risks

| Severity | Risk | Why it matters | Required resolution |
|---|---|---|---|
| P0 | Communication-aware loss uses relative world axes while scheduling uses ego-heading-relative bearing | link/outage gradients can optimize the wrong optical direction | rotate samples into the same ego frame or supply heading explicitly |
| P0 | Differentiable training objective omits the hard FoV behavior of the evaluated link | the model is not trained on the boundary central to the paper | consistent smooth FoV surrogate plus exact evaluation and boundary ablation |
| P0 | `_current_heading` resets to zero when the last displacement is near zero | actual and forecast links use inconsistent headings for stopped vehicles | reuse the last valid heading convention |
| P0 | Oracle future indices are clamped to the last sample while the horizon length is retained | repeated tail states can distort oracle decisions near record end | mask/truncate unavailable future uniformly for all policies |
| P0 | Slot-duration sweeps change total scenario time and physical packet deadlines | effects attributed to slot duration are confounded | configure physical duration/deadline seconds and resample consistently |
| P0 | Quick and exploratory statistics treat many shared-seed operating points as independent | nominal p-values can be overconfident | cluster by independent scenario/seed or use hierarchical paired bootstrap |
| P0 | BER outage threshold 1e−3 implies almost unit PER for a 12,000-bit packet | “not outage” can still mean practically zero packet success | define/justify outage at BER, PER or minimum goodput; sweep thresholds |
| P0 | Received power is normalized to 1 µW at the reference geometry | absolute watt values can be mistaken for a calibrated optical budget | rename normalized power or calibrate from an auditable link budget |
| P0 | Compact WOMD episode is only one second; many packets are right-censored at episode end | PDR/deadline-miss comparisons can be misleading | longer official windows or explicit censoring/drain policy |
| P0 | No compatible trained checkpoint is present | learned and communication-aware claims cannot be evaluated | obtain/retrain and freeze a compatible checkpoint |
| P1 | Link lifetime is reported in steps | not comparable across slot durations | store both steps and seconds |
| P1 | Lifetime urgency contains a step-dependent term | scheduler behavior changes with horizon/slot units | dimensionless normalization and pre-registered ablation |
| P1 | Current “Proportional Fair” uses cumulative delivered bits rather than a standard throughput average | baseline name may overstate fidelity | implement conventional PF or label it as a simplified fairness heuristic |
| P1 | Jain fairness uses raw delivered packets under unequal generated demand | demand asymmetry can look like unfair service | also report demand-normalized service fairness |
| P1 | SNR MAE assigns floor values outside FoV | small boundary errors produce huge values and dominate averages | report all-sample and visible-only errors plus boundary slices |
| P1 | F1=1 when both truth and prediction contain no outage | easy all-negative windows inflate macro summaries | report support, micro/macro definitions and confidence intervals |
| P1 | Zero BER observations are clipped to 1e−15 in the LUT | high-SNR BER is not statistically identified by 250k bits | adaptive bit counts and upper confidence bounds |
| P1 | `deadline_miss_ratio` excludes packets still queued at simulation termination | shorter runs can appear to have few misses | report censored/remaining separately and standardize endpoint policy |
| P1 | Latency includes delivered packets only | a policy with worse PDR may look artificially fast | interpret jointly with PDR or use deadline-aware censored analysis |
| P1 | Local compact manifest says unknown release although upstream metadata says v1.3.0 | the exact lineage of the compact subset is not proved | freeze a verifiable export manifest from official shards |
| P1 | PyTorch path is untested in the audited runtime | infrastructure presence is not end-to-end verification | CI job with ML extra and tiny deterministic train/inference test |
| P2 | Synthetic forecast noise is iid by time step | not representative of correlated trajectory uncertainty | label as stress noise or use a declared temporal covariance model |

## 5. Prioritized implementation plan

### Phase 0 — validity freeze

1. Add regression tests for coordinate frames, stationary headings, horizon masking, deadline units and common-random comparisons.
2. Fix only the failures exposed by those tests.
3. Version the corrected metric schema so old and new artifacts cannot be mixed.
4. Freeze Part-A assumptions and upstream Stage-4 provenance in machine-readable manifests.

**Exit criterion:** all old tests plus new validity tests pass; no empirical result is regenerated yet.

### Phase 1 — data and predictor evidence

1. Add a true-SDC official-WOMD adapter with validity masks and scenario-safe splits.
2. Decide between obtaining the exact upstream checkpoint or retraining a compatible model.
3. Implement the four pre-registered objectives and multi-seed checkpoint manifest.
4. Evaluate motion and link forecasts before any scheduler comparison.

**Exit criterion:** frozen predictors have scenario-level ADE/FDE and link-forecast metrics on held-out scenes; deployable predictors pass future-mutation tests.

### Phase 2 — scheduling experiments

1. Freeze baseline definitions and utility weights using development data only.
2. Run the scheduler comparison with common random numbers.
3. Run staged operating-region sweeps instead of one uncontrolled Cartesian search.
4. Preserve negative regimes and failure cases.

**Exit criterion:** every primary claim has paired scenario/seed evidence and uncertainty intervals.

### Phase 3 — publication artifacts

1. Regenerate tables/figures from immutable result files.
2. Rewrite the beginner-friendly README using the prepared visuals.
3. Update methodology, provenance, results and limitations.
4. Replace the draft manuscript with a complete paper and supplementary reproducibility record.
5. Open a GitHub PR, verify CI and merge only after review.

## 6. Experiment plan

### E1. Validity regression experiment

- **Hypothesis:** corrected frame/time/horizon handling changes only the cases affected by identified defects and makes all policies comparable in physical units.
- **Method:** deterministic micro-scenarios for stationary heading, FoV crossing, truncated future and equal physical deadlines across slot sizes.
- **Baselines:** current behavior recorded as a legacy fixture; corrected implementation.
- **Variables:** heading state, horizon availability, slot duration, deadline seconds.
- **Metrics:** exact geometry, SNR/outage trace, lifetime seconds, packet conservation.
- **Statistics:** none; these are exact invariants.
- **Interpretation:** any invariant failure blocks all larger experiments.

### E2. Predictor and link-forecast benchmark

- **Hypothesis:** trajectory ranking and communication ranking differ near optical boundaries.
- **Method:** held-out official WOMD scenes evaluated at causal anchors; report whole-set and boundary-slice results.
- **Baselines:** Last, CV, CA, Kalman, IMM, frozen deterministic GRU, Gaussian/GMM if available, oracle-information trajectory.
- **Variables:** horizon, mobility slice and boundary distance.
- **Metrics:** ADE, FDE, range/bearing error, SNR error, outage F1/AUROC with support, lifetime MAE in seconds.
- **Statistics:** scenario-clustered paired bootstrap; corrected paired tests for pre-registered comparisons.
- **Interpretation:** establish whether better motion prediction actually yields better link prediction.

### E3. Communication-aware training ablation

- **Hypothesis:** link-aware terms improve boundary-sensitive link forecasting without unacceptable trajectory degradation.
- **Method:** identical data split, architecture and optimizer; vary only objective terms; 3–5 training seeds.
- **Baselines:** trajectory-only; trajectory+link; trajectory+outage; full trajectory+link+outage.
- **Variables:** objective, seed and horizon.
- **Metrics:** ADE/FDE, SNR error, outage F1/AUROC, lifetime error, calibration and downstream scheduling KPIs.
- **Statistics:** seed- and scenario-clustered paired intervals; Holm correction across objective variants.
- **Interpretation:** a model is communication-aware only if a real trained checkpoint improves declared link metrics; no improvement is an acceptable result.

### E4. Scheduler benchmark

- **Hypothesis:** link-lifetime urgency changes allocation in closing-window cases and may improve delivery before disconnection.
- **Method:** fixed scenes, traffic and transmission random numbers shared across policies.
- **Baselines:** Random, RR, Reactive Greedy, conventional PF, CV/Kalman/IMM Predictive, generic Predictive Utility, Link Lifetime and Oracle-information reference.
- **Variables:** predictor and scheduler; weights frozen on development scenes.
- **Metrics:** goodput, PDR, scheduled outage, mean/P50/P95/P99 latency, miss ratio, fairness, delivered-before-disconnection and queue at disconnection.
- **Statistics:** scenario/seed-level paired bootstrap, paired t/Wilcoxon as sensitivity, effect size, win rate and Holm correction.
- **Interpretation:** report trade-offs; do not declare a winner from one metric alone.

### E5. Operating-region study

- **Hypothesis:** prediction helps at low/moderate load with scheduling flexibility, and loses value under saturation, excessive horizon error or unsuitable slot timing.
- **Method:** staged one- or two-axis sweeps around a frozen reference point; physical duration/deadlines held constant.
- **Baselines:** Reactive, best causal predictive policy, Link Lifetime, Oracle-information reference.
- **Variables:** offered load, vehicles, horizon seconds, slot seconds, mobility complexity and queue pressure.
- **Metrics:** paired differences for all primary communication KPIs.
- **Statistics:** cluster bootstrap by independent scene/seed; interaction models only if sample size supports them.
- **Interpretation:** identify sign changes and confidence bands, including regimes where Reactive remains preferable.

### E6. Traffic/deadline sensitivity

- **Hypothesis:** proactive service is most useful for bursty or deadline-tight traffic when capacity remains sufficient.
- **Method:** Poisson, Markov, periodic and saturated/bulk modes under fixed mean load where applicable.
- **Baselines:** Reactive, PF, Link Lifetime and Oracle-information reference.
- **Variables:** packet size 300/1200/1500 B; deadlines 20/50/100/250/500/1000 ms; traffic model and load.
- **Metrics:** PDR, goodput, deadline miss, tail latency, expiry delivery and fairness.
- **Statistics:** paired scenario/seed intervals with family correction.
- **Interpretation:** distinguish deadline benefit from raw-capacity benefit.

### E7. Channel and calibration sensitivity

- **Hypothesis:** scheduler ranking is robust only inside a bounded set of link assumptions.
- **Method:** pre-registered SNR offsets, attenuation scales, FoV values, outage definitions and BER sources.
- **Baselines:** analytical DBPSK and audited Monte Carlo LUT; Reactive and Link Lifetime.
- **Variables:** reference SNR ±3/±6 dB, attenuation ±10/20/30%, BER/PER thresholds and FoV.
- **Metrics:** link availability, scheduled outage, goodput, PDR, lifetime error and policy gain.
- **Statistics:** paired intervals by channel setting; no tuning after observing the test set.
- **Interpretation:** a fragile sign reversal is a limitation, not a parameter-search opportunity.

### E8. Sensing uncertainty propagation

- **Hypothesis:** covariance-aware state estimation reduces downstream link/lifetime error relative to treating noisy observations as exact.
- **Method:** declared synthetic observation model only; never call it measured sensor performance.
- **Baselines:** perfect state; noisy state; noisy+covariance-aware Kalman/GRU.
- **Variables:** observation variance, temporal correlation and range/bearing dependence.
- **Metrics:** state error → geometry error → link error → lifetime error → scheduling KPI change.
- **Statistics:** paired scenario/seed intervals.
- **Interpretation:** quantify sensitivity to assumed sensing quality and clearly separate measurement uncertainty from future predictive uncertainty.

### E9. Small-scale optimization reference

- **Hypothesis:** perfect future information with a heuristic utility can be worse than Reactive, whereas an exact small-instance schedule provides a genuine upper reference.
- **Method:** offline dynamic/integer optimization only for small scenes and horizons.
- **Baselines:** Reactive, Link Lifetime, current Oracle-information heuristic and exact offline solution.
- **Variables:** small queue counts, horizons and deadlines.
- **Metrics:** objective gap and packet KPIs.
- **Statistics:** exact per-instance gaps plus bootstrap across instances if sufficient.
- **Interpretation:** separate information value from heuristic quality.

## 7. Proposed code changes

| File/module | Proposed change | Reason |
|---|---|---|
| `config.py` | add immutable Part-A provenance, physical-duration/deadline fields, outage definition and sensitivity axes | eliminate hidden units and source ambiguity |
| `geometry.py` | explicit world/ego frame transforms, persistent heading and optional TX/RX offsets | consistent optical bearing |
| `link.py` | separate normalized power from calibrated power; explicit BER/PER/goodput outage modes; lifetime seconds/censoring | align semantics with packet delivery |
| `ber.py` | adaptive errors/bits, confidence bounds and planned −5…25 dB grid | scientifically auditable LUT |
| `data/womd_official.py` (new) | true-SDC adapter, valid-state masks, actor eligibility and fixed scene splits | publication-scale WOMD evaluation |
| `data/manifest.py` | dataset release/export lineage, split hashes and scenario lists | reproducibility |
| `learning/torch_model.py` | frame-consistent link surrogate, smooth FoV term and configurable dropout | valid communication-aware learning |
| `learning/train.py` | objective modes, multi-seed metadata and link-level validation | required learned ablation |
| `learning/inference.py` | feature-schema and checkpoint compatibility checks | prevent silent upstream mismatch |
| `simulation/engine.py` | horizon masks, stationary heading fix, endpoint/censoring policy and extra scheduled-link metrics | validity and complete KPIs |
| `traffic.py` | saturated model and deadlines in seconds | required traffic regimes and fair slot sweeps |
| `scheduling/policies.py` | conventional PF; unit-stable lifetime normalization; rename oracle comments | defensible baselines |
| `forecast_evaluation.py` | bearing error, lifetime seconds, support counts and boundary slices | answer the ADE-versus-utility question |
| `metrics.py` | P50/P99, direction-aware paired statistics, clustered bootstrap and Holm correction | publication-grade inference |
| `scenario_slices.py` (new) | deterministic mobility/link-regime classification | critical-case analysis |
| `experiment_matrix.py` | staged experiment definitions and immutable run IDs | avoid uncontrolled Cartesian searches |
| `paper_artifacts.py` | confidence/error bars and required operating-region/failure-case figures | publication figures |
| `tests/` | exact regression tests for every P0 validity issue and an ML smoke test | prevent recurrence |
| `README.md` | beginner-first explanation, diagrams, verified commands and honest readiness status | user-facing clarity after scientific outputs stabilize |
| `docs/` and `paper/` | regenerated claim traceability, methods, results, limitations and supplementary log | align every sentence with evidence |

## 8. Strongest paper story today

Based only on existing evidence, the strongest honest story is:

> A causal mobility-to-link-to-packet pipeline makes it possible to study when predicted communication opportunities should affect scheduling. In controlled motion, generic trajectory prediction alone gives little benefit, while an explicit link-lifetime urgency term yields a modest goodput/PDR/expiry gain with worse tail latency. A broader exploratory matrix suggests the gain disappears or reverses under high load and long horizons, and a three-scene proxy-WOMD evaluation is negative. Therefore prediction is not universally beneficial; its value depends on scheduling flexibility, forecast horizon and mobility-induced link closure.

This is scientifically more interesting than “prediction always wins”, but the operating-region conclusion remains provisional until the P0 confounds and independence issues are corrected.

## 9. Claims that could be supported after successful completion

The following are **future conditional claims**, not established results:

1. Causal mobility prediction improves packet delivery only in identifiable operating regions with predictable link closures and sufficient scheduling flexibility.
2. Link-lifetime-aware urgency explains more downstream gain than generic future-rate utility.
3. Trajectory ADE/FDE ranking does not reliably determine SNR/outage/lifetime or scheduling ranking.
4. A communication-aware predictor improves boundary-sensitive link forecasting, if the pre-registered multi-seed ablation confirms it.
5. Prediction gain decreases or becomes negative under saturation, excessive horizon error, severe sensing uncertainty or link-model mismatch.
6. Reactive scheduling remains preferable in measurable regimes.
7. Real WOMD mobility plus declared model-based PC-FMCW/DPSK communication supports the conclusions, while remaining distinct from measured optical-channel validation.

## Approval gate

No major implementation or new large experiment should start before this audit is accepted. After approval, the correct order is:

```text
P0 regression tests
→ P0 validity fixes
→ data/checkpoint integration
→ predictor/link evaluation
→ scheduler experiments
→ statistical aggregation
→ figures/tables
→ README and reproducibility docs
→ final paper
→ GitHub PR and CI
```

Old artifacts must remain identifiable as pre-fix evidence and must not be overwritten or mixed with corrected results.
