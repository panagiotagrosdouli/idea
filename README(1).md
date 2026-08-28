# Predictive PC-FMCW/DPSK Vehicular Communications

## Πρόβλεψη τροχιάς, πρόβλεψη μελλοντικού optical link και proactive scheduling οχημάτων

> **Κεντρική ιδέα:** Δεν προβλέπουμε την τροχιά ενός οχήματος απλώς για να ξέρουμε πού θα βρίσκεται. Χρησιμοποιούμε την προβλεπόμενη κίνηση για να εκτιμήσουμε πώς θα εξελιχθεί το PC-FMCW/DPSK communication link και να αποφασίσουμε **ποιο όχημα πρέπει να εξυπηρετηθεί τώρα, πριν χαθεί μια μελλοντική ευκαιρία επικοινωνίας**.

---

# Περιεχόμενα

1. [Τι κάνει αυτή η εργασία](#1-τι-κάνει-αυτή-η-εργασία)
2. [Το βασικό ερευνητικό ερώτημα](#2-το-βασικό-ερευνητικό-ερώτημα)
3. [Η συνολική pipeline](#3-η-συνολική-pipeline)
4. [Δεδομένα και τροχιές οχημάτων](#4-δεδομένα-και-τροχιές-οχημάτων)
5. [Causal trajectory prediction](#5-causal-trajectory-prediction)
6. [Έλεγχος future leakage](#6-έλεγχος-future-leakage)
7. [Από την τροχιά στη γεωμετρία του link](#7-από-την-τροχιά-στη-γεωμετρία-του-link)
8. [PC-FMCW/DPSK link model](#8-pc-fmcwdpsk-link-model)
9. [Από SNR σε BER, PER και goodput](#9-από-snr-σε-ber-per-και-goodput)
10. [Traffic, queues και deadlines](#10-traffic-queues-και-deadlines)
11. [Το scheduling problem](#11-το-scheduling-problem)
12. [Scheduling policies](#12-scheduling-policies)
13. [Η βασική ιδέα του Link-Lifetime scheduler](#13-η-βασική-ιδέα-του-link-lifetime-scheduler)
14. [Communication-aware GRU](#14-communication-aware-gru)
15. [Μετρικές αξιολόγησης](#15-μετρικές-αξιολόγησης)
16. [Πειραματική μεθοδολογία](#16-πειραματική-μεθοδολογία)
17. [Synthetic αποτελέσματα](#17-synthetic-αποτελέσματα)
18. [Πλήρες experiment matrix](#18-πλήρες-experiment-matrix)
19. [WOMD proxy benchmark](#19-womd-proxy-benchmark)
20. [Τι έχουμε αποδείξει και τι όχι](#20-τι-έχουμε-αποδείξει-και-τι-όχι)
21. [Τι δεν έχει υλοποιηθεί ακόμη](#21-τι-δεν-έχει-υλοποιηθεί-ακόμη)
22. [Πώς συνδέεται με το Joint Beam/ADB project](#22-πώς-συνδέεται-με-το-joint-beamadb-project)
23. [Προτεινόμενα επόμενα βήματα](#23-προτεινόμενα-επόμενα-βήματα)
24. [Η εργασία σε μία πρόταση](#24-η-εργασία-σε-μία-πρόταση)

---

# 1. Τι κάνει αυτή η εργασία

Η αρχική εργασία PC-FMCW/DPSK έδινε τη βάση για optical sensing και communication.

Η νέα εργασία προσθέτει κάτι διαφορετικό:

**πρόβλεψη του μέλλοντος και χρήση αυτής της πρόβλεψης για communication control.**

Η λογική είναι:

```text
Τι έκανε το αρχικό σύστημα:

PC-FMCW / DPSK
      ↓
τρέχουσα κατάσταση sensing / communication
      ↓
reactive λειτουργία


Τι προσθέτει αυτή η εργασία:

ιστορικό κίνησης
      ↓
πρόβλεψη τροχιάς
      ↓
πρόβλεψη μελλοντικής γεωμετρίας
      ↓
πρόβλεψη μελλοντικού communication link
      ↓
πρόβλεψη outage / link lifetime
      ↓
proactive scheduling
```

Το σημαντικό είναι ότι η trajectory prediction **δεν είναι ο τελικός στόχος**.

Είναι το μέσο για να προβλέψουμε το communication opportunity.

---

# 2. Το βασικό ερευνητικό ερώτημα

Το βασικό ερώτημα είναι:

> **Αν προβλέψουμε πού θα βρίσκεται κάθε όχημα στα επόμενα χρονικά βήματα, μπορούμε να προβλέψουμε πώς θα εξελιχθεί η ποιότητα του PC-FMCW/DPSK link και να μεταδώσουμε προληπτικά packets πριν το link υποβαθμιστεί ή χαθεί;**

Αυτό μπορεί να γραφτεί ως:

```text
Future Mobility
      ↓
Future Communication Geometry
      ↓
Future Link Quality
      ↓
Future Communication Opportunity
      ↓
Scheduling Decision
```

Άρα η εργασία βρίσκεται στη διασταύρωση:

- trajectory prediction,
- optical vehicular communication,
- packet scheduling,
- proactive network control.

---

# 3. Η συνολική pipeline

Η πλήρης λογική του framework είναι:

```text
┌──────────────────────────────┐
│ Real / Controlled Trajectory │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Causal Trajectory Prediction │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Future Range + Bearing       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ PC-FMCW/DPSK Link Model      │
│ SNR → BER → PER → Goodput    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Outage + Link Lifetime       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Packet Queues + Deadlines    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Predictive Scheduler         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Goodput / PDR / Latency /    │
│ Deadline Miss / Fairness     │
└──────────────────────────────┘
```

Για να καταλάβουμε την εργασία, πρέπει να καταλάβουμε **γιατί υπάρχει κάθε βήμα**.

---

# 4. Δεδομένα και τροχιές οχημάτων

Κάθε scenario περιγράφει την κίνηση του ego vehicle και των άλλων οχημάτων.

Για ένα όχημα \(i\), η θέση στο χρονικό βήμα \(t\) είναι:

$$
\mathbf{p}_{i,t}
=
\begin{bmatrix}
x_{i,t} \\
y_{i,t}
\end{bmatrix}
$$

όπου:

- \(x_{i,t}\): θέση στον άξονα \(x\),
- \(y_{i,t}\): θέση στον άξονα \(y\),
- \(i\): actor/vehicle,
- \(t\): χρονικό βήμα.

Αντίστοιχα, για το ego vehicle:

$$
\mathbf{p}_{e,t}
=
\begin{bmatrix}
x_{e,t} \\
y_{e,t}
\end{bmatrix}
$$

## 4.1 Γιατί χρειαζόμαστε το ego vehicle;

Η ποιότητα του communication link εξαρτάται από τη **σχετική** γεωμετρία.

Δεν μας αρκεί να ξέρουμε ότι ένα όχημα βρίσκεται στο \((100,50)\).

Θέλουμε να ξέρουμε:

- πόσο απέχει από το ego,
- σε ποια γωνία βρίσκεται,
- αν κινείται προς ή μακριά από το ego,
- αν πρόκειται να βγει από το Field of View.

Άρα αργότερα θα χρησιμοποιήσουμε:

$$
\Delta\mathbf{p}_{i,t}
=
\mathbf{p}_{i,t}-\mathbf{p}_{e,t}
$$

και όχι μόνο την absolute position.

## 4.2 Controlled trajectories

Χρησιμοποιούνται ελεγχόμενες synthetic trajectories.

Αυτές είναι χρήσιμες επειδή μπορούμε να δημιουργήσουμε συγκεκριμένες καταστάσεις:

- όχημα που απομακρύνεται,
- όχημα που πλησιάζει,
- acceleration,
- braking,
- όχημα που βγαίνει από το FoV,
- διαφορετικές vehicle densities.

Έτσι μπορούμε να ελέγξουμε αν ο scheduler συμπεριφέρεται όπως περιμένουμε.

## 4.3 Compact WOMD

Χρησιμοποιείται επίσης compact export πραγματικών trajectories από το Waymo Open Motion Dataset (WOMD).

Το πλεονέκτημα είναι ότι η κινηματική προέρχεται από πραγματικές traffic scenes και όχι μόνο από χειροποίητες trajectories.

### Σημαντικός περιορισμός

Στο compact WOMD export που χρησιμοποιήθηκε δεν υπήρχε το πραγματικό `sdc_track_index`.

Για αυτό χρησιμοποιήθηκε **deterministic medoid proxy ego**.

Δηλαδή επιλέγεται ως proxy ego ένας actor που είναι γεωμετρικά κεντρικός στη σκηνή.

Αυτό είναι αποδεκτό για:

- software testing,
- preliminary experiments,
- pipeline validation.

Δεν είναι αρκετό για ισχυρό claim πλήρους real-world ego-aware αξιολόγησης.

---

# 5. Causal trajectory prediction

## 5.1 Τι σημαίνει causal;

Στη χρονική στιγμή \(t\), ένα πραγματικό σύστημα γνωρίζει μόνο το παρελθόν και το παρόν.

Άρα ο predictor επιτρέπεται να δει:

$$
\mathbf{p}_{i,0:t}
$$

και πρέπει να προβλέψει:

$$
\hat{\mathbf{p}}_{i,t+1:t+H}
$$

όπου \(H\) είναι το prediction horizon.

Δεν επιτρέπεται να χρησιμοποιήσει τις πραγματικές θέσεις \(t+1,t+2,\ldots\).

Αν τις χρησιμοποιούσε, θα υπήρχε **future leakage**.

### Default χρονική ρύθμιση

- slot duration: \(100\,\text{ms}\),
- horizon: 10 slots,
- συνολική πρόβλεψη: \(1\,\text{s}\).

Άρα σε κάθε scheduling instant ρωτάμε περίπου:

> «Πού προβλέπεται να βρίσκεται αυτό το όχημα μέσα στο επόμενο ένα δευτερόλεπτο;»

---

## 5.2 Constant Velocity (CV)

Το Constant Velocity είναι ένα απλό αλλά σημαντικό baseline.

### Βασική υπόθεση

Υποθέτουμε ότι η τρέχουσα ταχύτητα θα παραμείνει περίπου σταθερή μέσα στον μικρό prediction horizon.

Η εκτιμώμενη ταχύτητα είναι:

$$
\hat{\mathbf{v}}_t
=
\frac{\mathbf{p}_t-\mathbf{p}_{t-1}}{\Delta t}
$$

και η predicted position \(k\) slots μπροστά είναι:

$$
\hat{\mathbf{p}}_{t+k}
=
\mathbf{p}_t
+
k\Delta t\,\hat{\mathbf{v}}_t
$$

### Τι σημαίνει κάθε όρος;

- \(\mathbf{p}_t\): πού βρίσκεται το όχημα τώρα.
- \(\hat{\mathbf{v}}_t\): πόσο γρήγορα και προς ποια κατεύθυνση εκτιμάμε ότι κινείται.
- \(\Delta t\): χρόνος ανά slot.
- \(k\): πόσα slots μπροστά κοιτάμε.
- \(k\Delta t\): πόσος πραγματικός χρόνος έχει περάσει.

Ο τύπος λέει απλά:

$$
\text{μελλοντική θέση}
=
\text{σημερινή θέση}
+
\text{ταχύτητα}\times\text{χρόνος}
$$

### Παράδειγμα

Αν ένα όχημα βρίσκεται στα \(20\,m\), κινείται με \(10\,m/s\), και θέλουμε πρόβλεψη \(0.5\,s\) μπροστά:

$$
x_{\text{future}}
=
20 + 10(0.5)
=
25\,m
$$

### Γιατί είναι σημαντικό communication-wise;

Η νέα θέση θα αλλάξει:

- την απόσταση από το ego,
- το bearing,
- το pointing loss,
- το SNR,
- το πιθανό outage.

Άρα ακόμη και ένα απλό CV model μπορεί να δώσει στον scheduler πληροφορία για το **μέλλον του link**.

### Περιορισμός

Αν το όχημα φρενάρει, επιταχύνει ή στρίβει, η υπόθεση constant velocity γίνεται λιγότερο ακριβής.

---

## 5.3 Constant Acceleration (CA)

Το Constant Acceleration προσθέτει την επιτάχυνση.

Χρησιμοποιούμε:

$$
\hat{\mathbf{p}}_{t+k}
=
\mathbf{p}_t
+
k\Delta t\,\hat{\mathbf{v}}_t
+
\frac{1}{2}(k\Delta t)^2\hat{\mathbf{a}}_t
$$

### Τι σημαίνει πρακτικά;

Ο τύπος αποτελείται από τρία κομμάτια:

$$
\underbrace{\mathbf{p}_t}_{\text{πού είμαι τώρα}}
+
\underbrace{k\Delta t\,\hat{\mathbf{v}}_t}_{\text{κίνηση λόγω ταχύτητας}}
+
\underbrace{\frac{1}{2}(k\Delta t)^2\hat{\mathbf{a}}_t}_{\text{διόρθωση λόγω επιτάχυνσης}}
$$

Αν:

$$
\hat{\mathbf{a}}_t=0
$$

τότε ο τρίτος όρος μηδενίζεται και το CA ουσιαστικά γίνεται CV.

### Παράδειγμα

Έστω:

- \(v=10\,m/s\),
- \(a=2\,m/s^2\),
- horizon \(=1\,s\).

Το CV προβλέπει μετατόπιση:

$$
\Delta x_{\text{CV}}
=
10(1)
=
10\,m
$$

Το CA προβλέπει:

$$
\Delta x_{\text{CA}}
=
10(1)
+
\frac{1}{2}(2)(1)^2
=
11\,m
$$

Η διαφορά είναι \(1\,m\).

Αυτό φαίνεται μικρό, αλλά communication-wise μπορεί να είναι σημαντικό αν το όχημα βρίσκεται κοντά:

- στο FoV boundary,
- σε απότομη περιοχή του pointing-loss model,
- σε outage threshold.

Άρα ένα μικρό geometric error μπορεί να προκαλέσει μεγάλο communication error.

---

## 5.4 Kalman-CV

Το Kalman filter δεν κάνει απλώς extrapolation από τις δύο τελευταίες θέσεις.

Διατηρεί μία εσωτερική εκτίμηση της κατάστασης, π.χ.:

$$
\mathbf{x}_t
=
\begin{bmatrix}
x_t & y_t & v_{x,t} & v_{y,t}
\end{bmatrix}^{T}
$$

και εκτελεί επαναληπτικά:

```text
prediction
    ↓
νέα observation
    ↓
correction/update
    ↓
βελτιωμένη state estimate
```

Η χρησιμότητά του είναι ότι μπορεί να εξομαλύνει noisy observations και να δώσει πιο σταθερή velocity estimate.

Στο communication framework αποτελεί ισχυρότερο classical baseline από το απλό CV.

---

## 5.5 IMM

Το Interacting Multiple Model (IMM) χρησιμοποιεί περισσότερα από ένα motion models και εκτιμά ποιο είναι πιθανότερο κάθε στιγμή.

Για παράδειγμα:

```text
Model 1: σχεδόν σταθερή ταχύτητα
Model 2: acceleration / maneuver
```

Το IMM μπορεί να συνδυάζει τις εκτιμήσεις αντί να δεσμεύεται μόνιμα σε μία μόνο υπόθεση κίνησης.

Αυτό είναι χρήσιμο σε vehicular trajectories όπου η συμπεριφορά αλλάζει.

---

## 5.6 Oracle

Ο Oracle χρησιμοποιεί το πραγματικό future trajectory.

Άρα:

$$
\hat{\mathbf{p}}_{t+k}^{\text{Oracle}}
=
\mathbf{p}_{t+k}^{\text{GT}}
$$

Δεν μπορεί να χρησιμοποιηθεί σε πραγματικό deployment.

Υπάρχει για να απαντήσει σε ερώτηση τύπου:

> «Τι θα μπορούσε να κάνει η policy αν είχε τέλεια πληροφορία για τη future motion;»

### Πολύ σημαντικό

**Perfect future information δεν σημαίνει globally optimal scheduler.**

Ο Oracle εξακολουθεί να χρησιμοποιεί heuristic scheduling utility.

Άρα είναι δυνατόν σε κάποιο experiment μία άλλη heuristic policy να εμφανίσει μεγαλύτερο goodput.

Αυτό δεν αποτελεί μαθηματική αντίφαση.

---

# 6. Έλεγχος future leakage

Για να ελέγξουμε ότι οι causal predictors δεν βλέπουν κρυφά το μέλλον χρησιμοποιούμε **future-mutation test**.

Η διαδικασία είναι:

```text
1. Τρέχουμε τον predictor.
2. Αποθηκεύουμε την prediction.
3. Αλλάζουμε το ground-truth μέλλον.
4. Τρέχουμε ξανά τον predictor.
5. Συγκρίνουμε τις predictions.
```

Για causal predictor πρέπει:

$$
\hat{\mathbf{p}}^{(1)}_{t+1:t+H}
=
\hat{\mathbf{p}}^{(2)}_{t+1:t+H}
$$

Για Oracle περιμένουμε το αντίθετο, επειδή χρησιμοποιεί το ground-truth future.

Το validation έδωσε:

```text
causal_forecast_invariant = true
oracle_forecast_sensitive = true
```

Αυτό αποτελεί σημαντικό sanity check της πειραματικής μεθοδολογίας.

---

# 7. Από την τροχιά στη γεωμετρία του link

Μέχρι εδώ γνωρίζουμε predicted Cartesian positions.

Το communication model όμως χρειάζεται **σχετική απόσταση και γωνία**.

## 7.1 Relative position

Για vehicle \(i\):

$$
\Delta\mathbf{p}_{i,k}
=
\hat{\mathbf{p}}_{i,t+k}
-
\hat{\mathbf{p}}_{e,t+k}
$$

## 7.2 Range

Η predicted απόσταση είναι:

$$
d_{i,k}
=
\left\|
\Delta\mathbf{p}_{i,k}
\right\|_2
$$

ή σε 2D:

$$
d_{i,k}
=
\sqrt{
(\Delta x_{i,k})^2
+
(\Delta y_{i,k})^2
}
$$

### Γιατί έχει σημασία;

Μεγαλύτερη απόσταση σημαίνει γενικά μεγαλύτερη propagation/geometric loss.

Άρα:

```text
trajectory
→ future distance
→ future received quality
```

## 7.3 Bearing

Η relative angle υπολογίζεται ως:

$$
\phi_{i,k}
=
\operatorname{wrap}
\left[
\operatorname{atan2}
(\Delta y_{i,k},\Delta x_{i,k})
-
\psi_{e,k}
\right]
$$

όπου \(\psi_e\) είναι το heading του ego.

### Γιατί αφαιρούμε το ego heading;

Επειδή μας ενδιαφέρει η γωνία του άλλου οχήματος **ως προς την κατεύθυνση που κοιτά το ego**, όχι ως προς έναν παγκόσμιο άξονα.

### Παράδειγμα

Αν το ego κοιτά ευθεία και το άλλο όχημα είναι λίγο δεξιά:

```text
small |φ|
→ μικρό pointing error
→ πιθανώς καλό link
```

Αν βρίσκεται πολύ πλάγια:

```text
large |φ|
→ μεγαλύτερο pointing loss
→ πιθανό FoV exit
→ πιθανό outage
```

---

# 8. PC-FMCW/DPSK link model

## 8.1 Γιατί χρειαζόμαστε link model;

Η trajectory prediction από μόνη της δεν μας λέει πόσα packets μπορούν να μεταδοθούν.

Χρειαζόμαστε mapping:

```text
distance + angle
      ↓
link gain
      ↓
SNR
      ↓
BER
      ↓
PER
      ↓
goodput
```

## 8.2 Relative link gain

Το μοντέλο χρησιμοποιεί:

$$
h(d,\phi)
\propto
\frac{1}
{\pi[d\tan(\theta_b)]^2}
\,
e^{-\kappa d}
\,
e^{-\frac{1}{2}\left(\frac{\phi}{\sigma_\phi}\right)^2}
\,
\mathbb{1}
\left(
|\phi|
\le
\frac{\Phi_{\mathrm{FOV}}}{2}
\right)
$$

Ο τύπος φαίνεται σύνθετος, αλλά αποτελείται από απλά φυσικά κομμάτια.

### A. Geometric spreading

$$
\frac{1}{[d\tan(\theta_b)]^2}
$$

Όσο αυξάνεται η απόσταση, το beam footprint μεγαλώνει και το διαθέσιμο optical power density μειώνεται.

Σε απλοποιημένη μορφή η συμπεριφορά μοιάζει με:

$$
h\propto\frac{1}{d^2}
$$

### B. Atmospheric attenuation

$$
e^{-\kappa d}
$$

Όσο μεγαλώνει το \(d\), αυξάνεται η modelled atmospheric loss.

### C. Pointing loss

$$
e^{-\frac{1}{2}
\left(\frac{\phi}{\sigma_\phi}\right)^2}
$$

Όταν \(\phi=0\), το angular alignment είναι καλύτερο.

Όσο αυξάνεται το \(|\phi|\), το gain μειώνεται.

### D. Field of View

$$
\mathbb{1}
\left(
|\phi|
\le
\frac{\Phi_{\mathrm{FOV}}}{2}
\right)
$$

Αυτός είναι indicator.

Είναι:

$$
1
$$

όταν το όχημα βρίσκεται εντός FoV και:

$$
0
$$

όταν βρίσκεται εκτός.

Άρα το modelled link μπορεί να κοπεί όταν το vehicle φύγει από το receiver angular region.

---

## 8.3 Default παράμετροι

| Παράμετρος | Τιμή | Τι σημαίνει |
|---|---:|---|
| Reference distance | 40 m | σημείο calibration |
| Reference SNR | 18 dB | SNR στο reference point |
| Beam half-angle | 5° | angular spread του beam |
| Field of View | 70° | receiver angular acceptance |
| Pointing std. dev. | 18° | sensitivity σε angular mismatch |
| Atmospheric attenuation | 0.004/m | modelled distance-dependent loss |
| Data rate | 1 Gbit/s | nominal PHY rate |
| Packet size | 12,000 bits | μήκος packet |

---

## 8.4 Reference-SNR calibration

Το SNR υπολογίζεται σχετικά με ένα reference point:

$$
\gamma(d,\phi)
=
\gamma_{\mathrm{ref}}
\frac{
h(d,\phi)
}{
h(d_{\mathrm{ref}},0)
}
$$

### Γιατί το κάνουμε έτσι;

Δεν έχουμε πλήρες measured absolute optical link budget για όλες τις πραγματικές συνθήκες.

Άρα λέμε:

> «Στο reference geometry θεωρούμε συγκεκριμένο SNR και υπολογίζουμε τις υπόλοιπες καταστάσεις σχετικά με αυτό.»

Αυτό είναι **model-based calibration**.

Δεν πρέπει να παρουσιαστεί ως πραγματική measurement campaign.

---

# 9. Από SNR σε BER, PER και goodput

## 9.1 Γιατί δεν αρκεί το SNR;

Το SNR μας λέει πόσο καθαρό είναι το received signal.

Ο network scheduler όμως ενδιαφέρεται τελικά για:

> «Πόσα packets θα παραδοθούν επιτυχώς;»

Για αυτό πρέπει να περάσουμε από πολλά επίπεδα:

```text
SNR
 ↓
BER
 ↓
PER
 ↓
successful payload
 ↓
goodput
```

---

## 9.2 DBPSK BER

Για differential DBPSK detection σε AWGN:

$$
P_b
=
\frac{1}{2}
e^{-E_b/N_0}
$$

όπου \(P_b\) είναι η πιθανότητα bit error.

Όσο αυξάνεται το \(E_b/N_0\):

$$
P_b\rightarrow0
$$

και όταν το signal γίνεται πολύ κακό:

$$
P_b\rightarrow0.5
$$

---

## 9.3 Monte Carlo validation

Υπάρχει επίσης Monte Carlo DBPSK simulation.

Η διαδικασία είναι:

```text
random bits
    ↓
differential encoding
    ↓
AWGN
    ↓
differential detection
    ↓
bit comparison
    ↓
empirical BER
```

Ο σκοπός δεν είναι να αντικαταστήσει το analytical model.

Ο σκοπός είναι να λειτουργεί ως independent sanity check.

---

## 9.4 Από BER σε PER

Για packet μήκους \(L\) bits και independent bit errors:

$$
PER
=
1-(1-P_b)^L
$$

Για:

$$
L=12000
$$

και:

$$
P_b=10^{-4}
$$

παίρνουμε περίπου:

$$
PER
=
1-(1-10^{-4})^{12000}
\approx0.70
$$

### Γιατί είναι σημαντικό;

Το \(10^{-4}\) μπορεί να φαίνεται μικρό BER.

Αλλά σε packet 12,000 bits υπάρχουν πολλές ευκαιρίες να εμφανιστεί error.

Άρα:

> **Μικρό BER δεν σημαίνει απαραίτητα μικρό packet loss.**

---

## 9.5 Goodput

Το nominal bitrate δεν είναι το ίδιο με το useful delivered bitrate.

Χρησιμοποιούμε:

$$
G
=
R_b(1-PER)
$$

όπου:

- \(R_b\): nominal rate,
- \(PER\): packet error probability,
- \(G\): expected useful rate.

Στην simulation υπάρχει επίσης:

```text
resource_fraction = 0.005
```

ώστε να μη θεωρείται ότι ολόκληρο το θεωρητικό \(1\,Gbit/s\) είναι διαθέσιμο ως payload σε κάθε scheduling slot.

---

# 10. Traffic, queues και deadlines

## 10.1 Γιατί χρειάζεται network traffic;

Αν κάθε όχημα είχε πάντα ένα packet και δεν υπήρχαν deadlines ή queues, το scheduling problem θα ήταν υπερβολικά απλό.

Το project δημιουργεί packet-level demand.

Κάθε όχημα έχει δική του FIFO queue.

## 10.2 Packet arrivals

Τα packets μπορούν να δημιουργούνται σύμφωνα με traffic processes όπως Poisson arrivals.

Κάθε packet έχει:

- arrival time,
- deadline,
- θέση στην queue,
- transmission outcome.

## 10.3 Τι συμβαίνει σε failed transmission;

Ανάλογα με τη simulation logic, ένα packet που αποτυγχάνει μπορεί να επιστρέψει στην queue για επόμενη προσπάθεια.

## 10.4 Τι συμβαίνει όταν λήξει το deadline;

Αν ένα packet δεν έχει παραδοθεί εγκαίρως, θεωρείται expired/deadline miss.

Άρα δεν αρκεί να παραδοθεί «κάποτε».

Πρέπει να παραδοθεί **έγκαιρα**.

## 10.5 Queue overflow

Αν η queue φτάσει στο maximum capacity, νέα packets μπορούν να απορριφθούν.

### Default configuration

- offered load: 0.72,
- nominal deadline: 12 slots,
- deadline jitter: 4 slots,
- max queue: 2,000 packets.

---

# 11. Το scheduling problem

Σε κάθε slot μπορεί να εξυπηρετηθεί το πολύ ένα vehicle.

Για \(N\) οχήματα:

$$
a_t
\in
\{1,2,\ldots,N\}
$$

όπου \(a_t\) είναι η scheduling action.

Το πρόβλημα είναι:

> **Ποιο όχημα πρέπει να πάρει τον διαθέσιμο communication resource στο slot \(t\);**

Ο scheduler μπορεί να λάβει υπόψη:

- current goodput,
- future goodput,
- current queue,
- deadlines,
- fairness,
- future outage,
- switching penalty,
- remaining link lifetime.

---

# 12. Scheduling policies

## 12.1 Random

Επιλέγει τυχαία.

Είναι lower baseline.

Αν μία σύνθετη policy δεν μπορεί να ξεπεράσει σταθερά ένα random baseline, υπάρχει σοβαρό πρόβλημα.

## 12.2 Round Robin

Εξυπηρετεί διαδοχικά όλους τους χρήστες.

Πλεονέκτημα:

- απλή fairness.

Μειονέκτημα:

- δεν βλέπει channel quality,
- δεν βλέπει queues,
- δεν βλέπει το μέλλον.

## 12.3 Reactive Greedy

Χρησιμοποιεί current information.

Μπορεί να κοιτά:

- current goodput,
- queue pressure,
- deadline urgency.

Η βασική αδυναμία του είναι:

> **δεν γνωρίζει ότι ένα link που είναι καλό τώρα μπορεί να χαθεί σε λίγα slots.**

## 12.4 Proportional Fair

Προσπαθεί να ισορροπήσει efficiency και fairness.

Ένα όχημα με καλό link έχει υψηλή αξία, αλλά η policy λαμβάνει υπόψη και το πόση εξυπηρέτηση έχει ήδη λάβει.

## 12.5 CV Predictive

Χρησιμοποιεί CV trajectory forecast.

Η λογική είναι:

```text
past motion
→ CV future trajectory
→ future link
→ predictive utility
```

## 12.6 Predictive Utility

Χρησιμοποιεί finite-horizon information.

Η utility μπορεί να συνδυάζει:

- discounted expected future goodput,
- outage penalty,
- queue pressure,
- deadline urgency,
- fairness,
- switching cost,
- opportunity loss.

Άρα δεν επιλέγει απλώς το καλύτερο future SNR.

Προσπαθεί να αξιολογήσει **την αξία της εξυπηρέτησης**.

---

# 13. Η βασική ιδέα του Link-Lifetime scheduler

Αυτό είναι το πιο χαρακτηριστικό communication-control concept της εργασίας.

## 13.1 Το πρόβλημα του Reactive Greedy

Έστω δύο vehicles:

| | Vehicle A | Vehicle B |
|---|---:|---:|
| Current link | εξαιρετικό | καλό |
| Remaining predicted link | μεγάλο | πολύ μικρό |
| Queue | μέτρια | μεγάλη |
| Deadline pressure | μικρή | μεγάλη |

Ο Reactive Greedy μπορεί να πει:

```text
A έχει καλύτερο link τώρα
→ serve A
```

Ο predictive scheduler όμως βλέπει:

```text
A θα είναι διαθέσιμο και αργότερα
B θα χαθεί σύντομα
→ ίσως πρέπει να serve B τώρα
```

## 13.2 Link lifetime

Το \(T_i^{link}\) περιγράφει πόσο ακόμη προβλέπεται να παραμείνει διαθέσιμο το link του vehicle \(i\).

Μία lifetime pressure term είναι:

$$
U_i^{\mathrm{life}}
\propto
Q_i
\left(
1-\frac{T_i^{\mathrm{link}}}{H}
\right)
+
\frac{Q_i}{1+T_i^{\mathrm{link}}}
$$

όπου:

- \(Q_i\): packets στην queue,
- \(T_i^{link}\): remaining predicted link lifetime,
- \(H\): prediction horizon.

### Πώς διαβάζεται ο τύπος;

Αν \(Q_i\) είναι μεγάλο:

```text
πολλά packets περιμένουν
→ μεγαλύτερη πίεση
```

Αν \(T_i^{link}\) είναι μικρό:

```text
λίγος χρόνος σύνδεσης απομένει
→ μεγαλύτερη urgency
```

Άρα ο scheduler προσπαθεί να εξυπηρετήσει packets πριν κλείσει το communication window.

---

## 13.3 Παράδειγμα με χρόνο

Έστω:

```text
Vehicle A
current SNR: πολύ καλό
link lifetime: 900 ms

Vehicle B
current SNR: καλό
link lifetime: 300 ms
```

Ο Reactive Greedy πιθανόν επιλέγει A.

Ο Link-Lifetime scheduler μπορεί να επιλέξει B:

```text
t = 0 ms      serve B
t = 100 ms    πιθανώς serve B
t = 200 ms    τελευταία ευκαιρία
t = 300 ms    B disconnects / outage

A παραμένει διαθέσιμο μέχρι περίπου 900 ms.
```

Έτσι χρησιμοποιούμε τη future information για **temporal prioritization**.

---

# 14. Communication-aware GRU

## 14.1 Γιατί GRU;

Τα classical models κάνουν συγκεκριμένες υποθέσεις:

- CV → σταθερή ταχύτητα,
- CA → σταθερή επιτάχυνση.

Ένα learned recurrent model μπορεί να μάθει πιο σύνθετα temporal patterns.

Η GRU διαβάζει history sequence και παράγει multi-step future trajectory.

## 14.2 Γιατί δεν αρκεί trajectory-only training;

Έστω δύο predictions με position error \(1\,m\).

### Περίπτωση A

Το vehicle βρίσκεται στο κέντρο του FoV.

Το \(1\,m\) error μπορεί να αλλάξει ελάχιστα το communication outcome.

### Περίπτωση B

Το vehicle βρίσκεται ακριβώς κοντά στο FoV boundary.

Το ίδιο \(1\,m\) error μπορεί να αλλάξει:

```text
inside FoV
→ outside FoV
```

και επομένως:

```text
usable link
→ outage
```

Άρα το ίδιο geometric error δεν έχει πάντα την ίδια communication σημασία.

## 14.3 Communication-aware loss

Χρησιμοποιείται objective της μορφής:

$$
\mathcal{L}
=
\mathcal{L}_{traj}
+
\lambda_{link}\mathcal{L}_{link}
+
\lambda_{out}\mathcal{L}_{out}
$$

### Trajectory loss

$$
\mathcal{L}_{traj}
$$

τιμωρεί position prediction errors.

### Link loss

$$
\mathcal{L}_{link}
$$

τιμωρεί errors σε communication-relevant quantity, όπως predicted log-SNR.

### Outage loss

$$
\mathcal{L}_{out}
$$

τιμωρεί λανθασμένη πρόβλεψη του link/outage state.

### Τι προσπαθούμε να πετύχουμε;

Όχι απλώς:

> «να προβλέπω όσο γίνεται καλύτερα τη θέση».

Αλλά:

> **«να προβλέπω τη θέση με τρόπο που να διατηρεί σωστή την communication-relevant future information».**

## 14.4 Τρέχων περιορισμός

Το training pipeline είναι υλοποιημένο, αλλά δεν υπάρχει ακόμη πλήρες trained GRU checkpoint πάνω στο πλήρες WOMD dataset.

Άρα δεν πρέπει να παρουσιαστούν ανύπαρκτα learned WOMD results.

---

# 15. Μετρικές αξιολόγησης

Δεν υπάρχει μία μόνο metric.

Η εργασία αξιολογείται σε διαφορετικά επίπεδα.

## 15.1 ADE / FDE

Μετρούν trajectory accuracy.

Το ADE μετρά μέση απόσταση prediction-ground truth κατά μήκος του horizon.

Το FDE κοιτά το error στο τελευταίο predicted step.

Αυτά απαντούν:

> «Πόσο καλά προβλέπω την κίνηση;»

Δεν απαντούν από μόνα τους:

> «Πόσο καλά προβλέπω την επικοινωνία;»

## 15.2 SNR error

Μετρά πόσο απέχει το predicted SNR από το ground-truth/model-evaluated future SNR.

## 15.3 Outage F1 / AUROC

Αξιολογούν αν ο predictor αναγνωρίζει σωστά πότε το future link θα είναι σε outage.

## 15.4 Link-lifetime error

Μετρά πόσο σωστά προβλέπεται ο remaining usable link time.

## 15.5 Goodput

$$
\text{Goodput}
=
\frac{
\text{successfully delivered payload bits}
}{
\text{total time}
}
$$

Αυτό είναι από τα βασικά end-to-end communication KPIs.

## 15.6 PDR

$$
PDR
=
\frac{
\text{delivered packets}
}{
\text{generated packets}
}
$$

## 15.7 Scheduled outage

Μετρά πόσο συχνά ο scheduler επιλέγει vehicle που βρίσκεται σε outage.

Αυτό αξιολογεί **την απόφαση**.

## 15.8 Availability outage

Μετρά πόσο συχνά το link είναι γενικά unavailable.

Αυτό αξιολογεί **το environment/link**, όχι απαραίτητα τον scheduler.

## 15.9 Latency

Μετρώνται:

- mean latency,
- P95 latency.

Το P95 είναι χρήσιμο επειδή δείχνει την ουρά της latency distribution και όχι μόνο τον μέσο όρο.

## 15.10 Deadline-miss ratio

$$
\text{Deadline Miss Ratio}
=
\frac{
\text{expired packets}
}{
\text{generated packets}
}
$$

## 15.11 Jain fairness

$$
J
=
\frac{
\left(\sum_i x_i\right)^2
}{
N\sum_i x_i^2
}
$$

Αν:

$$
J=1
$$

έχουμε τέλεια ισότητα.

Χαμηλότερο \(J\) σημαίνει πιο άνιση κατανομή service.

---

# 16. Πειραματική μεθοδολογία

## 16.1 Γιατί πρέπει όλοι οι schedulers να βλέπουν τα ίδια random events;

Έστω ότι συγκρίνουμε δύο policies.

Αν η πρώτη πάρει τυχαία ευκολότερα packet arrivals και η δεύτερη δυσκολότερα, η σύγκριση δεν είναι δίκαιη.

Για αυτό χρησιμοποιούνται common random numbers.

Όλοι παίρνουν:

- ίδιες trajectories,
- ίδια arrivals,
- ίδια deadlines,
- ίδια random success/failure traces,
- ίδιο seed.

Έτσι η διαφορά στο αποτέλεσμα αποδίδεται περισσότερο στην policy και λιγότερο στην τύχη.

## 16.2 Paired comparisons

Συγκρίνουμε policies στο **ίδιο operating point**.

Παράδειγμα:

```text
ίδιο scenario
ίδιο load
ίδιο horizon
ίδιο slot duration
ίδιο seed
```

και μετά μετράμε:

$$
\Delta G
=
G_{\text{predictive}}
-
G_{\text{reactive}}
$$

Αυτό είναι ισχυρότερο από το να συγκρίνουμε απλώς δύο ανεξάρτητους συνολικούς μέσους όρους.

## 16.3 Statistical tools

Η πειραματική υποδομή υποστηρίζει:

- paired comparisons,
- bootstrap confidence intervals,
- paired t-tests,
- Wilcoxon tests,
- effect sizes.

---

# 17. Synthetic αποτελέσματα

Σε controlled experiments:

| Policy | Goodput | PDR | Deadline miss | Fairness |
|---|---:|---:|---:|---:|
| Reactive Greedy | 2.293 Mbps | 0.644 | 0.308 | 0.536 |
| CV Predictive | 2.303 Mbps | 0.647 | 0.298 | 0.564 |
| Predictive Utility | 2.304 Mbps | 0.647 | 0.298 | 0.564 |
| **Link-Lifetime** | **2.336 Mbps** | **0.656** | **0.292** | **0.568** |
| Oracle | 2.307 Mbps | 0.648 | 0.298 | 0.564 |

Η σχετική διαφορά goodput είναι:

$$
\frac{2.336-2.293}{2.293}\times100
\approx1.9\%
$$

### Πώς πρέπει να ερμηνευτεί;

Στο συγκεκριμένο controlled configuration, το Link-Lifetime concept κατάφερε να χρησιμοποιήσει καλύτερα τις communication opportunities.

Παράλληλα βελτιώθηκαν:

- PDR,
- deadline misses,
- fairness.

### Τι ΔΕΝ μπορούμε να πούμε;

Δεν μπορούμε να πούμε:

> «Το Link-Lifetime δίνει γενικά +1.9% σε πραγματικά vehicular optical networks.»

Το αποτέλεσμα αφορά τις συγκεκριμένες simulated conditions.

---

# 18. Πλήρες experiment matrix

Το matrix περιλαμβάνει:

$$
4
\times
3
\times
4
\times
3
\times
5
\times
5
=
3600
$$

policy results από συνδυασμούς:

- 4 horizons,
- 3 vehicle counts,
- 4 offered loads,
- 3 slot durations,
- 5 seeds,
- 5 policies.

Για Link-Lifetime έναντι Reactive Greedy:

- mean difference: \(+0.0055\,Mbps\),
- win rate: \(61.5\%\).

## Ανάλυση ανά offered load

| Load | Mean difference |
|---:|---:|
| 0.35 | +0.0230 Mbps |
| 0.55 | +0.0241 Mbps |
| 0.75 | +0.0039 Mbps |
| 0.90 | −0.0292 Mbps |

### Το σημαντικό επιστημονικό αποτέλεσμα

Το predictive scheduling **δεν είναι πάντα καλύτερο**.

Σε χαμηλό/μεσαίο load υπάρχει χώρος για έξυπνη αναδιάταξη του service.

Σε πολύ υψηλό load:

```text
πολλά arrivals
      ↓
μεγάλες queues
      ↓
πολλά urgent packets
      ↓
λίγος διαθέσιμος communication resource
      ↓
queue-dominated regime
```

Η prediction δεν μπορεί να δημιουργήσει capacity που δεν υπάρχει.

Άρα το proactive pressure μπορεί ακόμη και να οδηγήσει σε χειρότερες αποφάσεις.

Το σωστό συμπέρασμα είναι:

> **Η αξία του predictive scheduling εξαρτάται από το operating region.**

---

# 19. WOMD proxy benchmark

Στο μικρό WOMD proxy benchmark:

| Policy | Goodput | PDR |
|---|---:|---:|
| **Reactive Greedy** | **1.160 Mbps** | **0.329** |
| CV Predictive | 1.056 Mbps | 0.299 |
| Predictive Utility | 0.892 Mbps | 0.253 |
| Link-Lifetime | 1.056 Mbps | 0.299 |
| Oracle | 0.852 Mbps | 0.242 |

Εδώ ο Reactive Greedy είναι καλύτερος.

Αυτό **δεν πρέπει να κρυφτεί**.

Είναι σημαντικό negative/non-conclusive result.

## Γιατί δεν αρκεί αυτό το benchmark;

Υπάρχουν σημαντικοί περιορισμοί:

- μόνο 3 scenarios,
- μόνο 1 s evaluation,
- proxy ego,
- απουσία πραγματικού SDC identifier,
- κανένα πλήρες learned checkpoint,
- πολύ μικρό sample.

Άρα δεν υπάρχει ακόμη αρκετό evidence για claim:

> «predictive scheduling improves communication on real WOMD mobility.»

Χρειάζεται μεγαλύτερη αξιολόγηση.

---

# 20. Τι έχουμε αποδείξει και τι όχι

## 20.1 Τι υποστηρίζεται από την υπάρχουσα εργασία

Έχουμε evidence ότι:

1. causal trajectory forecasts μπορούν να μετατραπούν σε future communication geometry,
2. future geometry μπορεί να μετατραπεί σε modelled future SNR/BER/PER/goodput,
3. ο scheduler μπορεί να χρησιμοποιήσει την πληροφορία χωρίς future leakage,
4. link lifetime μπορεί να χρησιμοποιηθεί ως proactive scheduling signal,
5. σε controlled conditions μπορεί να βελτιώσει communication KPIs,
6. το gain εξαρτάται από το offered load και το operating regime,
7. σε πολύ υψηλό load η predictive policy μπορεί να χάσει από Reactive Greedy.

## 20.2 Τι ΔΕΝ έχει αποδειχθεί

Δεν έχουμε αποδείξει ότι:

- predictive scheduling είναι πάντα καλύτερο,
- το \(1.9\%\) είναι universal gain,
- το WOMD είναι PC-FMCW measurement dataset,
- έχουμε πραγματικές optical channel measurements,
- έχουμε full-WOMD learned validation,
- ο Oracle είναι globally optimal,
- το υπάρχον μικρό WOMD proxy result γενικεύεται.

---

# 21. Τι δεν έχει υλοποιηθεί ακόμη

Δεν έχει ολοκληρωθεί:

- full WOMD-LiDAR processing,
- raw LiDAR point-cloud processing,
- calibrated probabilistic trajectory posterior,
- πλήρης common observation-interface σύγκριση Kalman/IMM/MHT,
- adaptive Top-K beam selection,
- predictive ADB,
- joint communication-illumination control,
- DeepSense validation,
- πραγματική experimental PC-FMCW optical measurement campaign,
- full-WOMD trained GRU checkpoint.

Αυτό είναι σημαντικό να αναφέρεται καθαρά ώστε το repository να μη δημιουργεί μεγαλύτερα claims από όσα υποστηρίζουν τα experiments.

---

# 22. Πώς συνδέεται με το Joint Beam/ADB project

Τα δύο projects έχουν κοινή αρχή:

```text
WOMD / mobility
      ↓
trajectory prediction
      ↓
future geometry
      ↓
PC-FMCW-related reasoning
```

Μετά χωρίζουν.

## Αυτό το repository

```text
future geometry
      ↓
future link quality
      ↓
outage / link lifetime
      ↓
queues / deadlines
      ↓
predictive scheduling
```

Το ερώτημα είναι:

> **ΠΟΙΟΝ και ΠΟΤΕ να εξυπηρετήσω;**

## Joint Beam/ADB direction

```text
future trajectory uncertainty
      ↓
future angular posterior
      ↓
beam selection
      +
predictive ADB
```

Το ερώτημα είναι:

> **ΠΟΥ να κατευθύνω το beam και ΠΩΣ να ελέγξω τον φωτισμό;**

Άρα δεν χρειάζεται να βάλουμε όλο το Joint project μέσα σε αυτό το repository.

### Χρήσιμα μελλοντικά στοιχεία από το Joint

Θα μπορούσαν να προστεθούν:

- PC-FMCW-like noisy observations,
- measurement covariance \(R_t\),
- SNR/CRLB uncertainty,
- uncertainty-aware predictor,
- receiver-aware geometry,
- predicted LoS/blockage probability.

Τότε η pipeline θα μπορούσε να γίνει:

```text
WOMD motion
   ↓
PC-FMCW-like noisy observation z_t
+ measurement covariance R_t
   ↓
uncertainty-aware predictor
   ↓
future geometry distribution
   ↓
future link distribution
   ↓
probabilistic link lifetime
   ↓
risk-aware scheduling
```

Αυτό είναι φυσική επέκταση χωρίς να αλλάξει το βασικό research question.

---

# 23. Προτεινόμενα επόμενα βήματα

## 23.1 Full WOMD evaluation

Πρώτη προτεραιότητα:

- περισσότερα scenarios,
- σωστό ego/SDC,
- μεγαλύτερα evaluation windows,
- scenario-level splits,
- πολλαπλά traffic regimes.

## 23.2 Train το communication-aware GRU

Να συγκριθούν:

```text
CV
CA
Kalman
IMM
trajectory-only GRU
communication-aware GRU
Oracle-information reference
```

και όχι μόνο ως προς ADE/FDE.

Πρέπει να μετρηθεί και:

```text
trajectory accuracy
      ↓
link prediction accuracy
      ↓
outage/lifetime accuracy
      ↓
scheduler performance
```

## 23.3 Προσθήκη sensing uncertainty

Ισχυρό ablation:

```text
A. Perfect WOMD state
B. Noisy PC-FMCW-like observation
C. Noisy observation + covariance R_t
```

Έτσι μπορούμε να μετρήσουμε αν το uncertainty-aware sensing/prediction βελτιώνει το downstream communication control.

## 23.4 Probabilistic link lifetime

Αντί για μόνο deterministic:

$$
T_i^{link}
$$

μπορούμε να προβλέπουμε:

$$
P(T_i^{link}\le\tau)
$$

δηλαδή την πιθανότητα το link να χαθεί πριν από χρόνο \(\tau\).

Αυτό επιτρέπει risk-aware scheduling.

## 23.5 Blockage prediction

Με multi-agent trajectories μπορούμε μελλοντικά να εκτιμήσουμε:

$$
P_{\mathrm{LOS}}(t+k)
$$

και:

$$
P_{\mathrm{block}}(t+k)
$$

και να τα εισάγουμε στην communication utility.

## 23.6 Το πιο ενδιαφέρον ερευνητικό ερώτημα για συνέχεια

Τα υπάρχοντα αποτελέσματα δείχνουν ότι η prediction δεν βοηθά παντού.

Άρα ένα ισχυρό επόμενο ερώτημα είναι:

> **Πότε ακριβώς αξίζει να χρησιμοποιούμε trajectory prediction για communication scheduling;**

Μπορούμε να χαρτογραφήσουμε:

```text
traffic load
× vehicle density
× prediction horizon
× slot duration
× mobility complexity
× link severity
× prediction uncertainty
```

και να εντοπίσουμε τα operating regions όπου η predictive information έχει πραγματική αξία.

---

# 24. Η εργασία σε μία πρόταση

Η επιστημονικά ασφαλής περιγραφή είναι:

> **Αναπτύσσουμε και αξιολογούμε ένα causal trajectory-predictive scheduling framework στο οποίο πραγματική ή ελεγχόμενη vehicle mobility μετατρέπεται μέσω physics-informed PC-FMCW/DPSK link model σε πρόβλεψη future goodput, outage και link lifetime, ώστε να γίνεται proactive packet scheduling πριν από προβλεπόμενη απώλεια της communication opportunity.**

---

# TL;DR — Αν θέλεις να θυμάσαι μόνο την κεντρική ιδέα

Ένα reactive σύστημα ρωτά:

> **«Ποιος έχει το καλύτερο link τώρα;»**

Το δικό μας predictive framework ρωτά:

> **«Ποιος έχει ανάγκη να εξυπηρετηθεί τώρα, αν λάβω υπόψη και το πώς θα εξελιχθούν τα links στα επόμενα slots;»**

Η πλήρης λογική είναι:

```text
Βλέπω πώς κινείται το όχημα
        ↓
Προβλέπω πού θα πάει
        ↓
Υπολογίζω μελλοντική απόσταση και γωνία
        ↓
Προβλέπω SNR
        ↓
Προβλέπω BER / PER / goodput
        ↓
Εκτιμώ πότε θα χαθεί το link
        ↓
Κοιτάζω queues και deadlines
        ↓
Επιλέγω ποιο όχημα πρέπει να εξυπηρετήσω τώρα
```

## Scientific scope

Το framework πρέπει να περιγράφεται ως:

> **real-world mobility + physics-informed PC-FMCW/DPSK communication simulation**

και όχι ως πραγματική optical measurement campaign.

Το WOMD παρέχει πραγματική mobility πληροφορία. Τα optical communication outcomes είναι model-based.

---

## Κεντρικό μήνυμα

**Δεν προβλέπουμε την τροχιά επειδή η πρόβλεψη τροχιάς είναι από μόνη της ο στόχος.**

**Προβλέπουμε την τροχιά επειδή το μέλλον της κίνησης καθορίζει το μέλλον της communication opportunity.**

Και χρησιμοποιούμε αυτή την πληροφορία για να πάρουμε καλύτερες scheduling αποφάσεις **πριν να είναι αργά**.
