# Predictive PC-FMCW/DPSK Vehicular Communications

**Αιτιακή πρόβλεψη τροχιάς για deadline-aware optical vehicular scheduling**

[English](README.md) | **Ελληνικά** · [Εκτελέσιμα stages](stages) · [Paper draft](paper/PAPER_DRAFT.md)

![Επισκόπηση Predictive PC-FMCW/DPSK](docs/assets/readme-hero.webp)

> Μπορεί η πρόβλεψη κίνησης να βοηθήσει έναν scheduler να παραδώσει πακέτα
> πριν χαθεί ένα κατευθυντικό optical link; Και σημαίνει πράγματι ότι μικρότερο
> trajectory error οδηγεί σε καλύτερη επικοινωνία;

Το repository συνδέει πραγματικές τροχιές από το Waymo Open Motion Dataset
(WOMD) με model-based PC-FMCW/DPSK optical link και packet simulator με ουρές,
deadlines, retries και fairness. Αποτελεί συνέχεια της supplied Εργασίας Α και
**δεν** είναι το ξεχωριστό Joint beam/ADB project.

## Ερευνητικά ερωτήματα

1. Σε ποια operating regimes βελτιώνει η πρόβλεψη το scheduling;
2. Συνεπάγεται καλύτερο ADE/FDE καλύτερο SNR, outage, link lifetime και goodput;
3. Βελτιώνει το communication-aware GRU training την πραγματική παράδοση πακέτων;

```mermaid
flowchart LR
    A["Παρατηρημένο WOMD history"] --> B["Causal predictor"]
    B --> C["Μελλοντική γεωμετρία"]
    C --> D["PC-FMCW/DPSK link"]
    D --> E["Ουρές και deadlines"]
    E --> F["Predictive scheduler"]
    F --> G["Packet KPIs και statistics"]
```

Το future ground truth χρησιμοποιείται αποκλειστικά για την πραγματοποίηση και
αξιολόγηση του link. Κανένας deployable predictor ή scheduler δεν το βλέπει.

## Πραγματικό workflow Stage 0-8

Η εργασία έχει οργανωθεί σε εννέα gated φακέλους μέσα στο [`stages/`](stages).
Κάθε φάκελος διαθέτει δικό του `stage.json`, άμεσο `run.py`, dependencies,
εντολές, outputs και acceptance criteria.

| Stage | Φάκελος | Εργασία | Completion gate |
|---:|---|---|---|
| 0 | [`00_freeze_and_provenance`](stages/00_freeze_and_provenance) | Freeze πρωτοκόλλου και splits | Dataset hashes και μηδενικό scenario overlap |
| 1 | [`01_womd_data_pipeline`](stages/01_womd_data_pipeline) | Audit των causal true-SDC samples | Έγκυρα arrays και split labels |
| 2 | [`02_pc_fmcw_dpsk_link`](stages/02_pc_fmcw_dpsk_link) | Freeze του Part-A link | Confidence-aware BER LUT 31 σημείων |
| 3 | [`03_classical_baselines`](stages/03_classical_baselines) | Last/CV/CA/Kalman/IMM | Αναπαραγώγιμα trajectory/link metrics |
| 4 | [`04_communication_aware_gru`](stages/04_communication_aware_gru) | Lambda selection και GRU training | 4 objectives × 5 seeds = 20 checkpoints |
| 5 | [`05_official_predictor_evaluation`](stages/05_official_predictor_evaluation) | Untouched validation | ADE/FDE, link fidelity, NLL και coverage |
| 6 | [`06_packet_scheduling`](stages/06_packet_scheduling) | Paired packet experiments | 8 schedulers × 5 traffic seeds |
| 7 | [`07_statistics_and_figures`](stages/07_statistics_and_figures) | Operating-region analysis | Cluster CI, Wilcoxon, Holm, ADE-goodput join |
| 8 | [`08_final_paper`](stages/08_final_paper) | Final release | Figures, tables, paper PDF και manifest |

Η δομή των αποτελεσμάτων ακολουθεί ακριβώς τα stages:

```text
artifacts/paper_final/
├── 00_freeze/       ├── 01_data/        ├── 02_link/
├── 03_baselines/    ├── 04_learning/    ├── 05_heldout/
├── 06_scheduling/   ├── 07_analysis/    └── 08_release/
```

Έλεγχος, preview και εκτέλεση:

```bash
make stages
make stage STAGE=stage0
make stage STAGE=stage0 EXECUTE=--execute
```

## Τι έχει γίνει πραγματικά

| Evidence | Κατάσταση |
|---|---|
| Trajectory → link → packet simulation | Υλοποιημένο και ελεγμένο |
| Scientific regression suite | **63/63 tests επιτυχή** |
| Part-A receiver-derived LUT | Εκτελεσμένο σε 31 SNR σημεία |
| Controlled scheduling study | 1.125 rows: 45 settings × 5 seeds × 5 policies |
| Official WOMD training corpus | **249.137 samples, 24.182 scenarios** |
| Training/development leakage | **0 overlapping scenario IDs** |
| Προηγούμενο 3-seed training | 7/12 results και 8/12 checkpoints διατηρούνται |
| Canonical learned archive | Εκκρεμεί: απαιτούνται 20 verified checkpoints |
| Untouched official validation | Υλοποιημένο export, εκκρεμεί το artifact |
| Official learned scheduling | Εκκρεμεί validation dataset και checkpoints |
| Measured optical channel | Δεν υπάρχει και δεν ισχυριζόμαστε ότι υπάρχει |

SHA-256 του training corpus:
`b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee`.

## Γιατί δεν αρκεί το ADE

Ένα μικρό Cartesian error κοντά στο FoV boundary μπορεί να προκαλέσει μεγάλη
μεταβολή στο pointing gain ή στο outage. Γι' αυτό αξιολογείται ολόκληρη η αλυσίδα:

\[
\mathrm{ADE/FDE}\rightarrow\{r,\theta\}\rightarrow
\{\mathrm{SNR},\mathrm{BER},\mathrm{PER},T_{link}\}\rightarrow
\{\mathrm{goodput},\mathrm{deadline\ misses},\mathrm{latency}\}.
\]

Το learned objective είναι:

\[
\mathcal{L}=\lambda_{traj}\mathcal{L}_{traj}
+\lambda_{link}\mathcal{L}_{link}
+\lambda_{out}\mathcal{L}_{outage}.
\]

Το Stage 4 εκπαιδεύει χωριστά trajectory-only, trajectory+link,
trajectory+outage και full communication-aware GRU.

## Μέθοδοι

Predictors:

- Last Position, Constant Velocity και Constant Acceleration,
- position-only Kalman CV και causal CV/CA IMM,
- deterministic GRU με τέσσερα objectives,
- development-fitted residual Gaussian calibration για held-out NLL και
  coverage 50/90/95%,
- perfect-future information reference.

Schedulers:

- Random, Round Robin, Reactive Greedy και Proportional Fair,
- CV, Kalman, IMM και Learned Predictive,
- Predictive Utility και Link-Lifetime urgency,
- information-oracle heuristic.

Το oracle δεν παρουσιάζεται ως global optimum. Διαθέτει τέλεια πληροφορία
μέλλοντος, αλλά χρησιμοποιεί την ίδια heuristic οικογένεια scheduling.

## Υπάρχον controlled αποτέλεσμα

Το controlled benchmark δεν αποδεικνύει universal gain. Η διαφορά Link-Lifetime
μείον Reactive είναι μόλις `+0,014 Mbps`, με 95% bootstrap interval
`[-0,0319, +0,0593] Mbps`, ενώ το P95 latency είναι περίπου 269 ms χειρότερο.
Άλλα loads και deadlines αλλάζουν το πρόσημο του αποτελέσματος.

![Controlled benchmark](artifacts/corrected_v2/figures/corrected_benchmark_tradeoff.png)

Άρα το πραγματικό ερώτημα είναι: **πότε βοηθά η πρόβλεψη και ποια prediction
errors έχουν σημασία για την επικοινωνία;**

## Εγκατάσταση και έλεγχος

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,ml,paper]"

make test
make lint
make validate
```

Το PyTorch απαιτείται μόνο στα learned-model stages.

## Εξωτερικές είσοδοι

Αντιγράφουμε το [`stages/.env.example`](stages/.env.example) και ορίζουμε:

```bash
export TRAIN_NPZ=/data/womd/womd_v131_training.npz
export VALIDATION_NPZ=/data/womd/womd_v131_official_validation.npz
export VALIDATION_TFRECORD='/data/womd/validation/*.tfrecord'
export CHECKPOINT_GLOB='artifacts/paper_final/04_learning/learned_ablation/*/seed_*/best_comm_aware_gru.pt'
export LAMBDA_LINK=0.2
export LAMBDA_OUTAGE=0.1
```

Τα loss weights επιλέγονται μόνο στο development set και παγώνουν πριν ανοίξει
το official validation.

## Δομή repository

```text
stages/                        gated work packages
src/predictive_pc_fmcw/        κοινή επιστημονική βιβλιοθήκη
├── data/                      WOMD adapters, exports και audits
├── learning/                  GRU, losses, calibration και evaluation
├── scheduling/                reactive και predictive policies
├── simulation/                packet-level realization
├── ber.py / link.py           PC-FMCW/DPSK abstraction
└── research_stages.py         dependency/gate engine
scripts/                       εκτελέσιμα experiments
configs/                       frozen assumptions
tests/                         regression και scientific gates
artifacts/                     evidence και αποτελέσματα
paper/                         manuscript source
notebooks/                     Colab GPU workflow
reference/                     provenance της supplied εργασίας
```

## Scientific guardrails

- Κανένα future state δεν εισέρχεται σε deployable decision.
- Απορρίπτεται overlap scenarios μεταξύ data partitions.
- Τα hyperparameters παγώνουν πριν από official validation.
- Οι schedulers χρησιμοποιούν paired randomness.
- Η επιτυχία πακέτου προκύπτει από το ground-truth-derived link.
- Ανεξάρτητο statistical cluster είναι το WOMD scenario.
- Τα confirmatory comparisons χρησιμοποιούν CI και Holm correction.
- Η optical ισχύς παραμένει model-based χωρίς εξωτερικά measurements.
- Τα αρνητικά αποτελέσματα και trade-offs παραμένουν ορατά.

## Ξεκίνα από εδώ

- [Executable stage workspace](stages)
- [Οδηγός εκτέλεσης](docs/STAGED_EXECUTION_GR.md)
- [Πρόοδος — 2026-09-03](docs/PROGRESS_2026-09-03.md)
- [Επιστημονικό audit και πλάνο](docs/SCIENTIFIC_AUDIT_AND_PLAN.md)
- [WOMD audit](docs/WOMD_DATASET_AUDIT.md)
- [Προέλευση δεδομένων](docs/DATA_PROVENANCE.md)
- [Οδικός χάρτης υλοποίησης 2026](docs/ROADMAP_IMPLEMENTATION_2026.md)

## Publication status

Το repository αποτελεί ελεγμένη ερευνητική υλοποίηση και σαφές publication
protocol, αλλά όχι ακόμη submission-ready empirical paper. Για να κλείσει το
Stage 8 απαιτούνται 20 checkpoints, untouched official-validation evaluation,
paired packet experiments, scenario-clustered statistics και clean release.
