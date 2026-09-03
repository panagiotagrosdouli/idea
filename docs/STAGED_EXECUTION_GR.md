# Σταδιακή υλοποίηση του Predictive PC-FMCW/DPSK project

Το Joint trajectory-to-beam-and-ADB PDF χρησιμοποιείται μόνο ως πρότυπο
οργάνωσης. Το αντικείμενο αυτής της εργασίας παραμένει αποκλειστικά:

> causal trajectory prediction → PC-FMCW/DPSK link prediction → queues και
> deadlines → predictive vehicular communication scheduling.

Δεν μεταφέρονται στο project ADB, beam control, raw LiDAR ή κοινός
communication/illumination controller.

## Βασικός κανόνας

Κάθε στάδιο έχει συγκεκριμένες εισόδους, αναπαραγώγιμες εντολές, αναμενόμενα
artifacts, acceptance criteria και εξάρτηση από το προηγούμενο στάδιο. Το
επόμενο στάδιο παραμένει `blocked` μέχρι να κλείσει το προηγούμενο. Έτσι δεν
χρησιμοποιούμε το official validation για tuning και δεν περνάμε preliminary
αποτελέσματα στο paper ως τελικά.

## Τα εννέα stages

| Stage | Περιεχόμενο | Completion gate |
|---|---|---|
| 0 | Freeze πρωτοκόλλου, hashes και split audit | training/validation hashes και μηδενικό scenario overlap |
| 1 | WOMD causal sample audit | true-SDC, 11 history/80 future states, finite arrays, σωστά labels |
| 2 | Part-A PC-FMCW/DPSK link freeze | 31-point confidence-aware και monotone BER LUT |
| 3 | Classical predictors | Last/CV/CA/Kalman/IMM με ADE/FDE και link metrics |
| 4 | Communication-aware learning | development lambda selection και 4 objectives × 5 seeds = 20 checkpoints |
| 5 | Official held-out predictor evaluation | ADE/FDE, link fidelity, outage/lifetime, NLL και coverage |
| 6 | Official packet scheduling | οκτώ schedulers, πέντε paired traffic seeds, per-scenario metrics |
| 7 | Scientific analysis | horizon/N/load/slices, scenario-cluster CI, Wilcoxon, Holm, ADE-vs-goodput |
| 8 | Paper και reproduction | τελικοί πίνακες/figures, manuscript PDF και clean manifest |

## Χρήση

Η συνολική κατάσταση εμφανίζεται με:

```bash
make stages
```

Οι καταστάσεις είναι `complete`, `ready` και `blocked`. Οι εντολές ενός stage
εμφανίζονται χωρίς εκτέλεση με:

```bash
make stage STAGE=stage0
```

Η πραγματική εκτέλεση γίνεται μόνο με ρητό `EXECUTE`:

```bash
make stage STAGE=stage0 EXECUTE=--execute
```

## Μεταβλητές εισόδων

```bash
export TRAIN_NPZ=/data/womd/womd_v131_training.npz
export VALIDATION_NPZ=/data/womd/womd_v131_official_validation.npz
export VALIDATION_TFRECORD='/data/womd/validation/*.tfrecord'
export CHECKPOINT_GLOB='artifacts/paper_final/learned_ablation/*/seed_*/best_comm_aware_gru.pt'
export LAMBDA_LINK=0.2
export LAMBDA_OUTAGE=0.1
```

Οι lambda τιμές είναι προσωρινές μέχρι να ολοκληρωθεί το development-only
sweep. Μετά την επιλογή παγώνουν και δεν αλλάζουν βάσει official validation.

## Σημερινή πραγματική κατάσταση

- Το training corpus έχει 249.137 samples από 24.182 scenarios.
- Ο εσωτερικός έλεγχος έδειξε μηδενική επικάλυψη training/development scenarios.
- Υπάρχουν 7/12 αποτελέσματα και 8/12 checkpoint files από το προηγούμενο
  τριών-seed πλάνο.
- Το canonical πρωτόκολλο απαιτεί 20 checkpoints: τέσσερα objectives επί πέντε
  seeds.
- Το official-validation NPZ δεν έχει ακόμη παραχθεί, άρα το Stage 0 δεν μπορεί
  να κλείσει πλήρως.
- Official held-out και packet-level αριθμοί δεν αντικαθίστανται με proxy ή
  synthetic αποτελέσματα.

Η μηχανή των stages βρίσκεται στο `research_stages.py`, κάθε εκτελέσιμο
specification στον αντίστοιχο υποφάκελο του `stages/` και το CLI στο
`scripts/run_research_stage.py`.
