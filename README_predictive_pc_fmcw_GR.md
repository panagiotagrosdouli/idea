# Predictive PC-FMCW/DPSK Vehicular Communications

## Causal Trajectory Prediction, Future Link Forecasting και Predictive Scheduling για Οπτική Επικοινωνία Οχημάτων

> **Σκοπός του repository:** ανάπτυξη και αξιολόγηση ενός causal
> trajectory-predictive communication framework, στο οποίο η
> προβλεπόμενη κίνηση των οχημάτων μετατρέπεται μέσω physics-informed
> PC-FMCW/DPSK link model σε πρόβλεψη μελλοντικού goodput, outage και
> link lifetime, ώστε ο scheduler να μπορεί να αποφασίζει προληπτικά
> **ποιο όχημα πρέπει να εξυπηρετηθεί και πότε**.

------------------------------------------------------------------------

## Περιεχόμενα

1.  [Ερευνητική ιδέα](#1-ερευνητική-ιδέα)
2.  [Τι αλλάζει σε σχέση με την Εργασία
    1](#2-τι-αλλάζει-σε-σχέση-με-την-εργασία-1)
3.  [Συνολική αρχιτεκτονική](#3-συνολική-αρχιτεκτονική)
4.  [Τροχιές και σενάρια](#4-τροχιές-και-σενάρια)
5.  [Causal trajectory forecasting](#5-causal-trajectory-forecasting)
6.  [Έλεγχος future leakage](#6-έλεγχος-future-leakage)
7.  [Trajectory-to-link geometry](#7-trajectory-to-link-geometry)
8.  [Physics-informed PC-FMCW/DPSK link
    model](#8-physics-informed-pc-fmcwdpsk-link-model)
9.  [SNR → BER → PER → Goodput](#9-snr--ber--per--goodput)
10. [Traffic, queues και deadlines](#10-traffic-queues-και-deadlines)
11. [Predictive scheduling problem](#11-predictive-scheduling-problem)
12. [Scheduling policies](#12-scheduling-policies)
13. [Link-Lifetime proactive
    scheduling](#13-link-lifetime-proactive-scheduling)
14. [Communication-aware GRU](#14-communication-aware-gru)
15. [Μετρικές αξιολόγησης](#15-μετρικές-αξιολόγησης)
16. [Πειραματικός σχεδιασμός](#16-πειραματικός-σχεδιασμός)
17. [Αποτελέσματα ελεγχόμενων
    σεναρίων](#17-αποτελέσματα-ελεγχόμενων-σεναρίων)
18. [Πλήρες experiment matrix](#18-πλήρες-experiment-matrix)
19. [WOMD proxy benchmark](#19-womd-proxy-benchmark)
20. [Τι έχει αποδειχθεί](#20-τι-έχει-αποδειχθεί)
21. [Τι δεν έχει αποδειχθεί](#21-τι-δεν-έχει-αποδειχθεί)
22. [Scientific scope και σωστή διατύπωση
    claims](#22-scientific-scope-και-σωστή-διατύπωση-claims)
23. [Τι δεν περιλαμβάνει το παρόν
    project](#23-τι-δεν-περιλαμβάνει-το-παρόν-project)
24. [Σχέση με το Joint Beam/ADB research
    direction](#24-σχέση-με-το-joint-beamadb-research-direction)
25. [Προτεινόμενα επόμενα βήματα](#25-προτεινόμενα-επόμενα-βήματα)
26. [Συμπέρασμα](#26-συμπέρασμα)

------------------------------------------------------------------------

# 1. Ερευνητική ιδέα

Η αρχική PC-FMCW/DPSK εργασία περιείχε το βασικό φυσικό και
επικοινωνιακό σύστημα:

``` text
PC-FMCW waveform
        ↓
DPSK communication
        ↓
sensing / tracking
        ↓
reactive operation
```

Η παρούσα εργασία προσθέτει ένα νέο επίπεδο **πρόβλεψης και ελέγχου**.

Το βασικό ερευνητικό ερώτημα είναι:

> **Αν μπορούμε να προβλέψουμε πού θα βρίσκεται κάθε όχημα στο άμεσο
> μέλλον, μπορούμε να μετατρέψουμε αυτή την πρόβλεψη σε πρόβλεψη της
> ποιότητας και της διάρκειας του PC-FMCW/DPSK link και να
> εξυπηρετήσουμε προληπτικά το κατάλληλο όχημα πριν χαθεί η σύνδεσή
> του;**

Επομένως, το contribution δεν είναι απλώς ένα καλύτερο trajectory
predictor. Η πρόβλεψη τροχιάς χρησιμοποιείται ως **πληροφορία ελέγχου
της επικοινωνίας**.

------------------------------------------------------------------------

# 2. Τι αλλάζει σε σχέση με την Εργασία 1

Η Εργασία 1 αποτελεί την τεχνική βάση για PC-FMCW/DPSK sensing και
communication.

Η νέα εργασία μετακινεί το research question από:

``` text
Τι συμβαίνει στο link τώρα;
```

σε:

``` text
Τι θα συμβεί στο link στα επόμενα slots
και πώς πρέπει να δράσει ο scheduler από τώρα;
```

Η νέα λογική είναι:

``` text
παρελθοντική κίνηση
        ↓
πρόβλεψη μελλοντικής κίνησης
        ↓
πρόβλεψη μελλοντικής γεωμετρίας
        ↓
πρόβλεψη μελλοντικού link
        ↓
πρόβλεψη link lifetime / outage
        ↓
proactive scheduling
```

Αυτό διαφοροποιεί την εργασία από μία καθαρά reactive προσέγγιση.

------------------------------------------------------------------------

# 3. Συνολική αρχιτεκτονική

Η επιστημονική pipeline του project είναι:

``` text
Real / Controlled Vehicle Motion
              ↓
      Causal Trajectory Forecast
              ↓
       Future Range / Bearing
              ↓
Physics-Informed PC-FMCW/DPSK Link Model
              ↓
 SNR → BER → PER → Expected Goodput
              ↓
     Future Outage / Link Lifetime
              ↓
       Packet Queues + Deadlines
              ↓
Predictive / Proactive Scheduling
              ↓
Goodput • PDR • Outage • Latency • Fairness
```

Υπάρχουν επομένως τέσσερα λογικά επίπεδα:

### A. Mobility layer

Περιγράφει την πραγματική ή ελεγχόμενη κίνηση των οχημάτων.

### B. Prediction layer

Προβλέπει αιτιακά τη μελλοντική τροχιά χωρίς πρόσβαση στο ground-truth
μέλλον.

### C. Communication layer

Μετατρέπει την προβλεπόμενη γεωμετρία σε PC-FMCW/DPSK link quantities.

### D. Control / scheduling layer

Χρησιμοποιεί την πρόβλεψη του link μαζί με queues και deadlines για
proactive resource allocation.

------------------------------------------------------------------------

# 4. Τροχιές και σενάρια

Κάθε scenario περιέχει πληροφορίες όπως:

-   θέση ego vehicle,
-   θέσεις connected vehicles,
-   timestamps,
-   actor IDs,
-   χρονικό σημείο έναρξης αξιολόγησης.

Για όχημα (i):

\[ `\mathbf{p}`{=tex}\_{i,t}=\[x\_{i,t},y\_{i,t}\] \]

και για το ego:

\[ `\mathbf{p}`{=tex}\_{e,t}=\[x\_{e,t},y\_{e,t}\]. \]

## 4.1 Controlled trajectories

Τα ελεγχόμενα σενάρια χρησιμοποιούνται για:

-   debugging,
-   deterministic validation,
-   controlled comparisons,
-   sensitivity analysis,
-   reproducible experiments.

Είναι ιδιαίτερα χρήσιμα επειδή επιτρέπουν να γνωρίζουμε με ακρίβεια
ποιος μηχανισμός προκαλεί μία μεταβολή στην απόδοση.

## 4.2 Compact WOMD trajectories

Το project μπορεί επίσης να χρησιμοποιήσει πραγματική κίνηση από compact
export του **Waymo Open Motion Dataset (WOMD)**.

Αυτό προσφέρει πιο ρεαλιστική κινηματική από ένα αποκλειστικά synthetic
benchmark.

### Σημαντικός περιορισμός

Στο compact export που χρησιμοποιήθηκε δεν υπήρχε το πραγματικό
`sdc_track_index`.

Για αυτό χρησιμοποιήθηκε **deterministic medoid proxy ego**: επιλέγεται
ως ego ο actor που βρίσκεται γεωμετρικά πιο κεντρικά στη σκηνή.

Αυτό είναι χρήσιμο για:

-   software validation,
-   pipeline testing,
-   preliminary WOMD experiments.

Δεν είναι όμως αρκετό για ισχυρό τελικό claim ότι έχει γίνει πλήρης
real-WOMD ego-aware αξιολόγηση.

------------------------------------------------------------------------

# 5. Causal trajectory forecasting

Στο slot (t), ο predictor επιτρέπεται να χρησιμοποιήσει μόνο:

\[ `\mathbf{p}`{=tex}\_{i,0:t}. \]

Παράγει:

\[ `\hat{\mathbf{p}}`{=tex}\_{i,t+1:t+H}. \]

Η default πειραματική ρύθμιση είναι:

-   slot duration: **100 ms**,
-   prediction horizon: **10 slots**,
-   συνολικός horizon: **1 s**.

## 5.1 Last-position baseline

Η απλούστερη δυνατή αναφορά θεωρεί ότι το όχημα θα παραμείνει στην
τελευταία γνωστή θέση.

Χρησιμεύει ως πολύ χαμηλό motion baseline.

## 5.2 Constant Velocity

Η CV πρόβλεψη είναι:

\[ `\hat{\mathbf{p}}`{=tex}\_{t+k} =
`\mathbf{p}`{=tex}\_t+k`\Delta `{=tex}t,`\hat{\mathbf{v}}`{=tex}\_t \]

με:

\[ `\hat{\mathbf{v}}`{=tex}\_t =
`\frac{\mathbf{p}_t-\mathbf{p}_{t-1}}{\Delta t}`{=tex}. \]

Η μέθοδος είναι causal και ιδιαίτερα ερμηνεύσιμη.

## 5.3 Constant Acceleration

Για acceleration/braking χρησιμοποιείται:

\[ `\hat{\mathbf{p}}`{=tex}\_{t+k} = `\mathbf{p}`{=tex}\_t+
k`\Delta `{=tex}t,`\hat{\mathbf{v}}`{=tex}\_t+
`\frac{1}{2}`{=tex}(k`\Delta `{=tex}t)\^2`\hat{\mathbf{a}}`{=tex}\_t. \]

Η CA μπορεί να αποδώσει καλύτερα από CV όταν η ταχύτητα μεταβάλλεται
ομαλά.

## 5.4 Kalman / IMM και πρόσθετα baselines

Η ευρύτερη υλοποίηση προβλέπει/περιλαμβάνει κλασικές causal αναφορές
όπως Kalman-CV και IMM, ώστε τα learned ή predictive approaches να μη
συγκρίνονται μόνο με αδύναμα baselines.

## 5.5 Oracle

Ο Oracle χρησιμοποιεί την πραγματική μελλοντική κίνηση.

Δεν είναι deployable predictor.

Χρησιμοποιείται αποκλειστικά ως **information reference**.

Επιπλέον, δεν πρέπει να περιγράφεται ως globally optimal offline
scheduler: ακόμη και όταν έχει perfect future motion, χρησιμοποιεί την
ίδια ερμηνεύσιμη heuristic utility λογική με τις υπόλοιπες policies.

------------------------------------------------------------------------

# 6. Έλεγχος future leakage

Η causal εγκυρότητα είναι κρίσιμη.

Για αυτό υπάρχει **future-mutation test**:

1.  εκτελείται ο causal predictor,
2.  αλλάζουν οι πραγματικές future positions,
3.  εκτελείται ξανά ο predictor,
4.  η causal prediction πρέπει να παραμείνει ακριβώς ίδια.

Αν αλλάξει, υπάρχει future leakage.

Το validation έδωσε:

``` text
causal_forecast_invariant = true
oracle_forecast_sensitive = true
```

Άρα:

-   οι κανονικοί causal predictors δεν χρησιμοποιούν κρυφά το μέλλον,
-   ο Oracle αντιδρά όπως αναμένεται όταν μεταβάλλεται το future ground
    truth.

Αυτό είναι σημαντικό επειδή μία predictive policy μπορεί να φαίνεται
τεχνητά εξαιρετική αν, έστω και έμμεσα, έχει πρόσβαση σε future
information.

------------------------------------------------------------------------

# 7. Trajectory-to-link geometry

Για κάθε predicted vehicle position υπολογίζεται η σχετική γεωμετρία ως
προς το ego.

## 7.1 Απόσταση

\[ d\_{i,k} = `\left`{=tex}\| `\hat{\mathbf{p}}`{=tex}*{i,t+k} -
`\hat{\mathbf{p}}`{=tex}*{e,t+k} `\right`{=tex}\|\_2. \]

## 7.2 Bearing / relative angle

\[ `\phi`{=tex}\_{i,k} = `\operatorname{wrap}`{=tex} `\left[
\operatorname{atan2}(\Delta y,\Delta x)-\psi_{e,k}
\right]`{=tex}. \]

Για κάθε future slot μπορούμε έτσι να εκτιμήσουμε:

-   predicted range,
-   predicted bearing,
-   αν ο receiver παραμένει εντός Field of View,
-   pointing error,
-   πιθανή μείωση link quality,
-   πιθανό future outage.

Η trajectory prediction αποκτά επομένως άμεση communication σημασία.

------------------------------------------------------------------------

# 8. Physics-informed PC-FMCW/DPSK link model

Η αρχική εργασία παρείχε PC-FMCW waveform/DPSK τεχνικό υπόβαθρο, αλλά το
παρόν framework δεν ισχυρίζεται ότι διαθέτει πλήρως μετρημένο absolute
optical link budget.

Χρησιμοποιείται **reference-SNR calibration**.

Η σχετική link gain μοντελοποιείται ως:

\[ h(d,`\phi`{=tex}) `\propto`{=tex}
`\frac{1}{\pi[d\tan(\theta_b)]^2}`{=tex} e\^{-`\kappa `{=tex}d}
e^{-`\frac{1}{2}`{=tex}`\left`{=tex}(`\frac{\phi}{\sigma_\phi}`{=tex}`\right`{=tex})^2}
`\mathbb{1}`{=tex} `\left`{=tex}(
\|`\phi`{=tex}\|`\leq`{=tex}`\frac{\Phi_{\mathrm{FOV}}}{2}`{=tex}
`\right`{=tex}). \]

Οι όροι αντιστοιχούν σε:

-   (1/d\^2): geometric spreading,
-   (e\^{-`\kappa `{=tex}d}): atmospheric attenuation,
-   Gaussian angular factor: pointing loss,
-   FoV indicator: link availability.

### Βασικές default παράμετροι

  Παράμετρος                      Default τιμή
  ----------------------------- --------------
  Reference distance                      40 m
  Reference SNR                          18 dB
  Beam half-angle                           5°
  Field of View                            70°
  Pointing standard deviation              18°
  Atmospheric attenuation            0.004 / m
  Nominal data rate                   1 Gbit/s
  Packet size                      12,000 bits

Το SNR υπολογίζεται σχετικά με reference operating point:

\[ `\gamma`{=tex}(d,`\phi`{=tex}) =
`\gamma`{=tex}*{`\mathrm{ref}`{=tex}} `\frac{h(d,\phi)}`{=tex}
{h(d*{`\mathrm{ref}`{=tex}},0)}. \]

### Επιστημονική ερμηνεία

Τα αποτελέσματα πρέπει να περιγράφονται ως **conditional πάνω στο
δηλωμένο calibration model**.

Δεν πρέπει να παρουσιάζονται ως measured absolute optical received
power.

------------------------------------------------------------------------

# 9. SNR → BER → PER → Goodput

## 9.1 DBPSK BER

Για differential DBPSK detection σε AWGN:

\[ P_b=`\frac{1}{2}`{=tex}e\^{-E_b/N_0}. \]

Όταν αυξάνεται το SNR:

\[ P_b`\rightarrow 0`{=tex}. \]

Όταν μειώνεται έντονα:

\[ P_b`\rightarrow 0.5`{=tex}. \]

## 9.2 Monte Carlo validation

Υπάρχει ανεξάρτητη Monte Carlo DBPSK διαδικασία:

1.  random bit generation,
2.  differential encoding,
3.  complex AWGN,
4.  adjacent-symbol differential detection,
5.  error counting.

Ο στόχος είναι να ελεγχθεί ότι το analytical BER model ακολουθεί τη
σωστή συμπεριφορά.

## 9.3 Packet Error Rate

Για packet (L=12{,}000) bits:

\[ PER=1-(1-P_b)\^L. \]

Παράδειγμα:

\[ P_b=10\^{-4} \]

δίνει περίπου:

\[ PER = 1-(1-10^{-4})^{12000} `\approx 0.70`{=tex}. \]

Άρα ένα BER που φαίνεται αριθμητικά μικρό μπορεί να είναι πολύ σημαντικό
σε packet level.

## 9.4 Goodput

Το nominal bitrate δεν είναι το ίδιο με successfully delivered
throughput.

Χρησιμοποιείται:

\[ G=R_b(1-PER). \]

Το goodput μετρά μόνο τα payload bits που παραδόθηκαν επιτυχώς.

Η simulation περιλαμβάνει επίσης:

``` text
resource_fraction = 0.005
```

ώστε ένα slot να μη θεωρείται ότι διαθέτει ολόκληρο το θεωρητικό 1
Gbit/s αποκλειστικά για payload.

------------------------------------------------------------------------

# 10. Traffic, queues και deadlines

Κάθε connected vehicle διαθέτει δική του FIFO queue.

Τα packets:

-   φτάνουν σύμφωνα με traffic process,
-   έχουν arrival time,
-   έχουν deadline,
-   μπορούν να αποτύχουν,
-   μπορούν να επανατοποθετηθούν στην ουρά,
-   λήγουν όταν περάσει το deadline,
-   μπορούν να απορριφθούν όταν γεμίσει η queue.

Η βασική traffic υλοποίηση υποστηρίζει διαφορετικά arrival models,
συμπεριλαμβανομένων Poisson και πιο σύνθετων/ελεγχόμενων processes.

### Default queue configuration

-   offered load: **0.72**,
-   nominal deadline: **12 slots**,
-   deadline jitter: **4 slots**,
-   maximum queue: **2,000 packets**.

Σε κάθε slot επιτρέπεται να εξυπηρετηθεί το πολύ **ένα όχημα**.

Αυτό δημιουργεί πραγματικό resource-allocation conflict.

------------------------------------------------------------------------

# 11. Predictive scheduling problem

Για (N) connected vehicles, ο scheduler επιλέγει σε κάθε slot:

\[ a_t`\in`{=tex}{1,`\ldots`{=tex},N}. \]

Για πέντε οχήματα:

\[ a_t`\in`{=tex}{1,2,3,4,5}. \]

Η επιλογή δεν βασίζεται μόνο στο current SNR.

Ο scheduler μπορεί να χρησιμοποιεί:

-   current link quality,
-   predicted future link quality,
-   expected future goodput,
-   queue length,
-   deadline urgency,
-   fairness,
-   predicted outage,
-   switching cost,
-   remaining link lifetime.

Έτσι το πρόβλημα μετατρέπεται από **reactive channel selection** σε
**future-aware communication control**.

------------------------------------------------------------------------

# 12. Scheduling policies

Το benchmark περιλαμβάνει πολλαπλές policies ώστε να συγκρίνεται η
predictive λογική με απλές και ισχυρότερες αναφορές.

## Random

Επιλέγει τυχαία όχημα.

Χρησιμοποιείται ως lower baseline.

## Round Robin

Εξυπηρετεί κυκλικά τα οχήματα.

Προσφέρει απλή fairness αλλά αγνοεί channel quality και future
information.

## Reactive Greedy

Χρησιμοποιεί μόνο τρέχουσα πληροφορία, όπως:

-   current goodput,
-   queue length,
-   deadline urgency.

Δεν γνωρίζει ότι ένα link μπορεί να χαθεί μετά από λίγα slots.

## Proportional Fair

Συνδυάζει current link quality με ιστορική εξυπηρέτηση ώστε να
περιορίζει ακραία άνιση κατανομή πόρων.

## CV Predictive

Χρησιμοποιεί Constant Velocity trajectory forecast και μετατρέπει τη
future geometry σε future link information.

## Kalman / IMM Predictive

Όπου ενεργοποιούνται, χρησιμοποιούν κλασικά state-estimation /
motion-model forecasts ως causal predictive references.

## Predictive Utility

Χρησιμοποιεί finite-horizon utility με στοιχεία όπως:

-   discounted expected goodput,
-   predicted outage penalty,
-   queue pressure,
-   deadline urgency,
-   fairness,
-   switching penalty,
-   opportunity loss.

## Link-Lifetime Prefetch / Proactive Policy

Προσθέτει explicit pressure όταν το link ενός οχήματος προβλέπεται να
λήξει σύντομα.

## Oracle

Χρησιμοποιεί perfect future motion αλλά την ίδια heuristic decision
logic.

Είναι **oracle-information reference**, όχι globally optimal scheduler.

## Optional learned policy

Μπορεί να χρησιμοποιηθεί όταν υπάρχει κατάλληλο εκπαιδευμένο checkpoint.

Δεν πρέπει να παρουσιάζονται learned results όταν checkpoint/full
training evidence δεν υπάρχει.

------------------------------------------------------------------------

# 13. Link-Lifetime proactive scheduling

Αυτό είναι ένα από τα πιο χαρακτηριστικά concepts της εργασίας.

Η βασική ιδέα είναι:

> **Αν ένα όχημα έχει packets προς μετάδοση και προβλέπεται ότι το link
> του θα χαθεί σύντομα, μπορεί να είναι καλύτερο να εξυπηρετηθεί τώρα,
> ακόμη και αν κάποιο άλλο όχημα έχει ελαφρώς καλύτερο instantaneous
> link.**

Μία μορφή lifetime pressure είναι:

\[ U_i\^{`\mathrm{life}`{=tex}} `\propto`{=tex} Q_i `\left`{=tex}(
1-`\frac{T_i^{\mathrm{link}}}{H}`{=tex} `\right`{=tex}) +
`\frac{Q_i}{1+T_i^{\mathrm{link}}}`{=tex}. \]

όπου:

-   (Q_i): queue occupancy,
-   (T_i\^{`\mathrm{link}`{=tex}}): predicted remaining link lifetime,
-   (H): prediction horizon.

## Παράδειγμα

Έστω:

-   Vehicle A: πολύ καλό link τώρα και αναμένεται να παραμείνει
    διαθέσιμο.
-   Vehicle B: ελαφρώς χειρότερο link τώρα, αλλά προβλέπεται να βγει από
    το FoV σε 300 ms.

Ένας Reactive Greedy scheduler πιθανότατα επιλέγει A.

Ο Link-Lifetime scheduler μπορεί να επιλέξει B:

``` text
B θα χάσει σύντομα το link
        ↓
υπάρχει μικρό remaining service window
        ↓
μεταδίδουμε προληπτικά packets του B
        ↓
εξυπηρετούμε το A αργότερα
```

Αυτό είναι η ουσία του **proactive transmission / prefetching before
disconnection**.

------------------------------------------------------------------------

# 14. Communication-aware GRU

Εκτός από classical predictors, έχει υλοποιηθεί trainable GRU trajectory
predictor.

Η αρχιτεκτονική χρησιμοποιεί GRU encoder και multi-step prediction.

Η βασική communication-aware objective είναι:

\[ `\mathcal{L}`{=tex} = `\mathcal{L}`{=tex}*{traj} +
`\lambda`{=tex}*{`\mathrm{link}`{=tex}}`\mathcal{L}`{=tex}*{link} +
`\lambda`{=tex}*{`\mathrm{out}`{=tex}}`\mathcal{L}`{=tex}\_{out}. \]

## Trajectory loss

Μετρά τη γεωμετρική απόκλιση της predicted trajectory από το ground
truth.

## Link loss

Μετρά αν η predicted trajectory οδηγεί σε σωστή πρόβλεψη
communication-relevant quantity, όπως log-SNR.

## Outage loss

Ενθαρρύνει σωστή πρόβλεψη του πότε το link θα γίνει μη διαθέσιμο.

## Γιατί communication-aware training;

Δύο trajectory errors ίδιου μεγέθους δεν έχουν απαραίτητα την ίδια
communication σημασία.

Παράδειγμα:

``` text
1 m error μακριά από FoV boundary
→ πιθανώς μικρή communication επίδραση

1 m error κοντά στο FoV boundary
→ μπορεί να αλλάξει link ↔ outage
```

Άρα η trajectory accuracy από μόνη της δεν αρκεί πάντα.

Το project επιχειρεί να βελτιστοποιήσει prediction features που έχουν
άμεση επίδραση στην επικοινωνία.

### Τρέχουσα κατάσταση

Το training pipeline είναι διαθέσιμο, αλλά δεν πρέπει να ισχυριστούμε
ότι υπάρχουν πλήρη learned WOMD αποτελέσματα χωρίς πραγματικό trained
checkpoint πάνω στο πλήρες dataset.

------------------------------------------------------------------------

# 15. Μετρικές αξιολόγησης

Η αξιολόγηση γίνεται σε πολλαπλά επίπεδα.

## 15.1 Trajectory metrics

Όπου χρησιμοποιείται learned/forecasting evaluation:

-   ADE,
-   FDE,
-   range error,
-   communication-relevant forecast error.

## 15.2 Link-prediction metrics

Μπορούν να περιλαμβάνουν:

-   SNR prediction error,
-   outage classification,
-   outage F1,
-   AUROC,
-   link-lifetime error.

## 15.3 Goodput

\[ `\text{Goodput}`{=tex} =
`\frac{\text{successfully delivered payload bits}}`{=tex}
{`\text{total simulation time}`{=tex}}. \]

## 15.4 Packet Delivery Ratio

\[ PDR= `\frac{\text{delivered packets}}`{=tex}
{`\text{generated packets}`{=tex}}. \]

## 15.5 Scheduled outage

Μετρά πόσο συχνά ο scheduler επιλέγει όχημα όταν το συγκεκριμένο link
βρίσκεται σε outage.

## 15.6 Availability outage

Μετρά πόσο συχνά το φυσικό/modelled link είναι μη διαθέσιμο ανεξάρτητα
από την απόφαση scheduling.

Η διάκριση είναι σημαντική:

``` text
availability outage = κατάσταση του link
scheduled outage    = ποιότητα απόφασης scheduler
```

## 15.7 Latency

Αξιολογούνται, μεταξύ άλλων:

-   mean latency,
-   P95 latency.

## 15.8 Deadline-miss ratio

\[ `\text{Deadline Miss Ratio}`{=tex} =
`\frac{\text{packets που έχασαν το deadline}}`{=tex}
{`\text{generated packets}`{=tex}}. \]

## 15.9 Jain fairness

\[ J= `\frac{\left(\sum_i x_i\right)^2}`{=tex} {N`\sum`{=tex}\_i
x_i\^2}. \]

-   (J=1): τέλεια ισότητα,
-   μικρότερο (J): μεγαλύτερη ανισότητα.

## 15.10 Disconnection-aware metrics

Η ευρύτερη αξιολόγηση μπορεί να εξετάζει επίσης:

-   delivered-before-expiry,
-   undelivered-at-disconnection,
-   predicted link-lifetime error.

Αυτές οι μετρικές είναι ιδιαίτερα κατάλληλες για το proactive scheduling
research question.

------------------------------------------------------------------------

# 16. Πειραματικός σχεδιασμός

## 16.1 Common random numbers

Για δίκαιη σύγκριση, οι policies λαμβάνουν:

-   ίδιες trajectories,
-   ίδια packet arrivals,
-   ίδια deadlines,
-   ίδιες random transmission success/failure realizations,
-   ίδιο seed.

Αυτό είναι **common-random-number experimental design**.

Χωρίς αυτή την πρακτική, μία policy θα μπορούσε τυχαία να λάβει
ευκολότερο traffic ή ευνοϊκότερες channel realizations.

## 16.2 Paired evaluation

Οι policies συγκρίνονται πάνω στα ίδια operating points και seeds.

Αυτό επιτρέπει paired statistical comparisons.

## 16.3 Statistical analysis

Η αξιολόγηση περιλαμβάνει/υποστηρίζει:

-   paired comparisons,
-   nonparametric bootstrap,
-   confidence intervals,
-   paired t-tests,
-   Wilcoxon tests,
-   effect sizes.

Στόχος δεν είναι απλώς να παρουσιαστεί ένας μέσος όρος, αλλά να
εξεταστεί αν η διαφορά είναι συνεπής στα ίδια experimental conditions.

------------------------------------------------------------------------

# 17. Αποτελέσματα ελεγχόμενων σεναρίων

Σε ένα σύνολο synthetic/controlled experiments καταγράφηκαν:

  -------------------------------------------------------------------------------
  Policy                     Goodput            PDR  Deadline miss       Fairness
  ------------------- -------------- -------------- -------------- --------------
  Reactive Greedy         2.293 Mbps          0.644          0.308          0.536

  CV Predictive           2.303 Mbps          0.647          0.298          0.564

  Predictive Utility      2.304 Mbps          0.647          0.298          0.564

  **Link-Lifetime**   **2.336 Mbps**      **0.656**      **0.292**      **0.568**

  Oracle                  2.307 Mbps          0.648          0.298          0.564
  -------------------------------------------------------------------------------

Η σχετική διαφορά goodput Link-Lifetime έναντι Reactive Greedy είναι:

\[ `\frac{2.336-2.293}{2.293}`{=tex}`\times100`{=tex}
`\approx1.9`{=tex}%. \]

Στο συγκεκριμένο controlled setting, ο Link-Lifetime scheduler εμφάνισε
επίσης:

-   καλύτερο PDR,
-   λιγότερα deadline misses,
-   καλύτερη fairness.

### Τι σημαίνει αυτό;

Δείχνει ότι ο μηχανισμός **μπορεί να λειτουργήσει** υπό ελεγχόμενες
συνθήκες.

### Τι ΔΕΝ σημαίνει;

Δεν αποδεικνύει ότι υπάρχει γενικό +1.9% κέρδος σε πραγματικό optical
vehicular network.

Το αποτέλεσμα είναι scenario/model/configuration dependent.

------------------------------------------------------------------------

# 18. Πλήρες experiment matrix

Το experiment matrix εξετάζει πολλαπλές διαστάσεις:

\[ 4 `\text{horizons}`{=tex} `\times
3`{=tex} `\text{vehicle counts}`{=tex} `\times
4`{=tex} `\text{loads}`{=tex} `\times
3`{=tex} `\text{slot durations}`{=tex} `\times
5`{=tex} `\text{seeds}`{=tex} `\times
5`{=tex} `\text{policies}`{=tex} = 3600 \]

policy results.

Για κάθε predictive policy υπάρχουν 720 paired operating points.

Για τον Link-Lifetime scheduler καταγράφηκε:

-   mean difference: **+0.0055 Mbps** έναντι Reactive Greedy,
-   win rate: **61.5%**.

### Ανά offered load

    Offered load   Mean difference vs Reactive
  -------------- -----------------------------
            0.35                  +0.0230 Mbps
            0.55                  +0.0241 Mbps
            0.75                  +0.0039 Mbps
            0.90                  −0.0292 Mbps

## Κρίσιμο συμπέρασμα

Το predictive scheduling **δεν είναι καθολικά καλύτερο**.

Σε χαμηλό και μεσαίο offered load υπάρχει μεγαλύτερος χώρος για έξυπνη
χρονική ανακατανομή των transmissions.

Σε πολύ υψηλό load:

``` text
queues γεμίζουν
      ↓
σύστημα γίνεται queue-dominated
      ↓
οι επιλογές περιορίζονται
      ↓
η predictive πληροφορία δεν αρκεί
      ↓
το επιπλέον proactive pressure μπορεί να βλάψει
```

Άρα το όφελος εξαρτάται από το **operating region**.

Αυτό είναι σημαντικότερο επιστημονικά από έναν απλό ισχυρισμό «η
proposed method είναι πάντα καλύτερη».

------------------------------------------------------------------------

# 19. WOMD proxy benchmark

Στο μικρό WOMD proxy benchmark καταγράφηκαν:

  Policy                         Goodput         PDR
  --------------------- ---------------- -----------
  **Reactive Greedy**     **1.160 Mbps**   **0.329**
  CV Predictive               1.056 Mbps       0.299
  Predictive Utility          0.892 Mbps       0.253
  Link-Lifetime               1.056 Mbps       0.299
  Oracle                      0.852 Mbps       0.242

Εδώ ο Reactive Greedy είναι καλύτερος.

Αυτό είναι **αρνητικό / μη καταληκτικό αποτέλεσμα**, όχι κάτι που πρέπει
να αποκρυφτεί.

Οι βασικοί περιορισμοί του συγκεκριμένου benchmark είναι:

-   μόνο 3 scenarios,
-   μόνο 1 s evaluation,
-   απουσία πραγματικού SDC identifier,
-   proxy ego geometry,
-   απουσία trained learned checkpoint,
-   πολύ μικρό dataset.

Επομένως:

> **Το υπάρχον WOMD proxy experiment δεν παρέχει ακόμη επαρκές evidence
> για ισχυρό real-world generalization claim.**

------------------------------------------------------------------------

# 20. Τι έχει αποδειχθεί

Με βάση την υπάρχουσα υλοποίηση και τα experiments μπορούμε να
υποστηρίξουμε ότι:

1.  Η causal trajectory prediction μπορεί να μετατραπεί σε future
    link-quality prediction.
2.  Ο scheduler μπορεί να χρησιμοποιήσει future predictions χωρίς future
    leakage.
3.  Το PC-FMCW/DPSK model επιτρέπει trajectory → geometry → SNR → BER →
    PER → goodput mapping.
4.  Packet queues και deadlines μπορούν να συνδεθούν με future link
    information.
5.  Το link-lifetime concept μπορεί να οδηγήσει σε proactive service
    πριν από predicted disconnection.
6.  Σε controlled scenarios, η συγκεκριμένη Link-Lifetime policy μπορεί
    να βελτιώσει goodput, PDR, deadline misses και fairness.
7.  Η βελτίωση δεν είναι καθολική.
8.  Το offered load επηρεάζει έντονα το αν η predictive scheduling
    πληροφορία είναι χρήσιμη.
9.  Σε πολύ υψηλό load, το σύστημα μπορεί να γίνει queue-dominated.
10. Το μικρό WOMD proxy benchmark δεν επαρκεί για real-world γενίκευση.

------------------------------------------------------------------------

# 21. Τι δεν έχει αποδειχθεί

Δεν πρέπει να συναχθούν από τα υπάρχοντα αποτελέσματα claims όπως:

-   ότι η proposed policy είναι πάντα καλύτερη από reactive scheduling,
-   ότι το +1.9% είναι γενικό real-world gain,
-   ότι το optical channel έχει μετρηθεί σε πραγματικό vehicular
    deployment,
-   ότι τα WOMD trajectories αποτελούν PC-FMCW measurements,
-   ότι υπάρχει πλήρης real-WOMD learned evaluation,
-   ότι υπάρχει globally optimal oracle scheduling,
-   ότι έχει αποδειχθεί superiority σε πραγματικό optical vehicular
    network.

Η εργασία πρέπει να διαχωρίζει καθαρά:

``` text
ΤΙ ΕΙΝΑΙ REAL
-------------
vehicle mobility / WOMD trajectories

ΤΙ ΕΙΝΑΙ MODEL-BASED
--------------------
PC-FMCW/DPSK link outcome
SNR
BER/PER
goodput
outage
link lifetime
packet transmission success
```

------------------------------------------------------------------------

# 22. Scientific scope και σωστή διατύπωση claims

Η σωστή περιγραφή είναι:

> **Real-world mobility + physics-informed PC-FMCW/DPSK communication
> simulation.**

Δεν είναι σωστό να περιγράφεται ως:

> real optical-channel measurement campaign.

Επίσης το WOMD δεν πρέπει να παρουσιάζεται ως dataset πραγματικών
PC-FMCW measurements.

Το WOMD παρέχει mobility/scene information.

Το communication layer είναι model-based.

### Προτεινόμενη ακαδημαϊκή διατύπωση

> We develop a causal trajectory-predictive vehicular communication
> framework in which real-world or controlled vehicle motion is mapped
> through a physics-informed PC-FMCW/DPSK link model to future
> communication quality, outage, and link-lifetime estimates. These
> predictions are then used for proactive packet scheduling under queue,
> deadline, and fairness constraints.

------------------------------------------------------------------------

# 23. Τι δεν περιλαμβάνει το παρόν project

Το παρόν communication-focused project **δεν ισχυρίζεται ότι έχει
ολοκληρώσει**:

-   full WOMD-LiDAR processing,
-   raw LiDAR point-cloud pipeline,
-   calibrated probabilistic trajectory posterior,
-   πλήρη Kalman/IMM/MHT comparison στο ίδιο sensing observation
    interface,
-   adaptive Top-K beam selection,
-   predictive Adaptive Driving Beam (ADB),
-   joint communication--illumination control,
-   DeepSense measured beam validation,
-   πραγματική experimental PC-FMCW optical measurement campaign,
-   πλήρως εκπαιδευμένο GRU πάνω στο πλήρες WOMD dataset.

Αυτά αποτελούν διαφορετικές ή μελλοντικές research extensions.

------------------------------------------------------------------------

# 24. Σχέση με το Joint Beam/ADB research direction

Υπάρχει κοινή βάση μεταξύ του παρόντος repository και του μεγαλύτερου
Joint uncertainty-aware Beam/ADB research direction.

Η κοινή βάση είναι:

``` text
WOMD / vehicle motion
        ↓
trajectory prediction
        ↓
future geometry
        ↓
PC-FMCW-related communication reasoning
```

Μετά όμως οι δύο κατευθύνσεις χωρίζουν.

## Παρόν repository

``` text
future trajectory
      ↓
future range / bearing
      ↓
future link quality
      ↓
future outage / link lifetime
      ↓
queues + deadlines
      ↓
predictive scheduling
```

Το ερώτημα είναι:

> **ΠΟΤΕ και ΠΟΙΟΝ πρέπει να εξυπηρετήσω;**

## Joint Beam/ADB direction

Η άλλη κατεύθυνση εξετάζει περισσότερο:

``` text
future trajectory distribution
        ↓
future angular posterior
        ↓
adaptive beam probing / selection
        +
predictive illumination / ADB
```

Το ερώτημα εκεί είναι περισσότερο:

> **ΠΟΥ πρέπει να κατευθύνω το beam και ΠΩΣ πρέπει να ελέγξω τον
> φωτισμό;**

### Τι θα μπορούσε να μεταφερθεί στο μέλλον

Χωρίς να αλλάξει το core scheduling research question, το παρόν project
θα μπορούσε να ενισχυθεί με:

1.  PC-FMCW-like noisy observation interface,
2.  measurement covariance (R_t),
3.  SNR/CRLB-based sensing uncertainty,
4.  uncertainty-aware trajectory predictor,
5.  receiver-aware geometry,
6.  future LoS/blockage probability.

Μία ισχυρότερη μελλοντική pipeline θα μπορούσε να είναι:

``` text
WOMD ground-truth motion
        ↓
PC-FMCW-like noisy observations z_t + covariance R_t
        ↓
uncertainty-aware causal predictor
        ↓
future receiver geometry distribution
        ↓
future SNR / BER / PER / outage distribution
        ↓
probabilistic link lifetime
        ↓
queues + deadlines
        ↓
risk-aware predictive scheduling
```

Αυτό θα συνδέσει ακόμη πιο στενά το sensing uncertainty με το
communication control χωρίς να απαιτεί να μεταφερθεί ολόκληρο το
ADB/Top-K beam-control project.

------------------------------------------------------------------------

# 25. Προτεινόμενα επόμενα βήματα

## Priority 1 --- Πλήρες WOMD dataset

Το σημαντικότερο βήμα είναι να αντικατασταθεί το μικρό proxy benchmark
με μεγαλύτερη και σωστά διαχωρισμένη WOMD αξιολόγηση.

Χρειάζονται:

-   περισσότερα scenarios,
-   σωστό ego/SDC information όπου είναι διαθέσιμο,
-   scenario-level train/validation/test split,
-   μεγαλύτερα evaluation windows,
-   πολλαπλά traffic/load conditions.

## Priority 2 --- Train communication-aware GRU

Να εκπαιδευτεί πραγματικό checkpoint και να συγκριθεί με:

-   CV,
-   CA,
-   Kalman,
-   IMM,
-   trajectory-only GRU,
-   communication-aware GRU,
-   perfect-future reference.

Το κρίσιμο ερώτημα δεν είναι μόνο:

> ποιος έχει χαμηλότερο ADE;

αλλά:

> ποιος δίνει καλύτερο downstream scheduling;

## Priority 3 --- Observation uncertainty

Να μη θεωρείται ότι ο predictor λαμβάνει πάντα perfect WOMD state.

Προτεινόμενο ablation:

``` text
A. Perfect state
B. Noisy PC-FMCW-like observation
C. Noisy observation + covariance-aware predictor
```

και να μετρηθεί η επίδραση σε:

``` text
trajectory error
    ↓
SNR error
    ↓
outage prediction
    ↓
link-lifetime error
    ↓
scheduling performance
```

## Priority 4 --- Robust link-lifetime prediction

Αντί για μόνο deterministic link lifetime, μπορεί να χρησιμοποιηθεί:

\[ P(T_i\^{link} `\leq `{=tex}`\tau`{=tex}) \]

ή γενικότερα probabilistic remaining-link distribution.

Έτσι ο scheduler μπορεί να γίνει uncertainty/risk aware.

## Priority 5 --- Blockage / LoS

Με multi-agent trajectories μπορεί να προβλέπεται μελλοντικό blockage:

\[ P\_{`\mathrm{LOS}`{=tex}}(t+k),`\qquad`{=tex}
P\_{`\mathrm{block}`{=tex}}(t+k). \]

Αυτό μπορεί να εισαχθεί στο future link utility.

## Priority 6 --- Operating-region characterization

Επειδή τα υπάρχοντα αποτελέσματα δείχνουν ότι το predictive scheduling
δεν είναι πάντα καλύτερο, αξίζει να χαρτογραφηθεί συστηματικά:

``` text
vehicle density
× traffic load
× prediction horizon
× slot duration
× mobility regime
× link severity
```

και να εντοπιστεί:

> **πότε ακριβώς αξίζει να χρησιμοποιούμε prediction;**

Αυτό μπορεί να είναι από μόνο του ισχυρό paper contribution.

------------------------------------------------------------------------

# 26. Συμπέρασμα

Το repository υλοποιεί ένα **causal trajectory-predictive PC-FMCW/DPSK
vehicular communication framework**.

Η βασική καινοτομία δεν είναι απλώς η πρόβλεψη τροχιάς.

Η ουσία είναι η αλυσίδα:

\[ `\boxed{
\text{Motion}
\rightarrow
\text{Prediction}
\rightarrow
\text{Future Geometry}
\rightarrow
\text{Future Link}
\rightarrow
\text{Link Lifetime}
\rightarrow
\text{Scheduling}
}`{=tex} \]

Η κεντρική ιδέα είναι ότι η κίνηση ενός οχήματος περιέχει πληροφορία για
το **πόσο χρόνο επικοινωνιακής ευκαιρίας απομένει**.

Αν αυτή η πληροφορία χρησιμοποιηθεί σωστά, ο scheduler μπορεί να
μεταδώσει packets **πριν** από μία προβλεπόμενη αποσύνδεση, αντί να
αντιδρά μόνο όταν το link έχει ήδη υποβαθμιστεί.

Τα controlled experiments δείχνουν ότι αυτή η λογική μπορεί να βελτιώσει
communication KPIs σε συγκεκριμένα operating regions. Ταυτόχρονα, τα
αποτελέσματα δείχνουν ότι η predictive scheduling δεν είναι universally
superior, ιδιαίτερα σε πολύ υψηλό offered load.

Το μικρό WOMD proxy benchmark δεν επαρκεί ακόμη για ισχυρή real-world
γενίκευση. Για αυτό η σωστή επόμενη φάση είναι η πλήρης WOMD αξιολόγηση,
η εκπαίδευση του communication-aware GRU και η εισαγωγή
sensing/prediction uncertainty.

------------------------------------------------------------------------

## TL;DR

``` text
Δεν προβλέπουμε την τροχιά απλώς για να ξέρουμε πού θα πάει το όχημα.

Την προβλέπουμε για να ξέρουμε:

• πώς θα αλλάξει η απόσταση και η γωνία,
• πώς θα αλλάξει το PC-FMCW/DPSK link,
• πότε θα εμφανιστεί outage,
• πόσο link lifetime απομένει,
• ποια packets κινδυνεύουν να μη μεταδοθούν,
• και ποιο όχημα πρέπει να εξυπηρετηθεί ΤΩΡΑ.

Αυτό είναι predictive / proactive vehicular communication scheduling.
```

------------------------------------------------------------------------

## Scientific status

**Υλοποιημένο / αξιολογημένο σε controlled framework**

-   causal motion prediction,
-   trajectory-to-link mapping,
-   PC-FMCW/DPSK link abstraction,
-   BER/PER/goodput pipeline,
-   packet queues και deadlines,
-   reactive και predictive schedulers,
-   link-lifetime proactive scheduling,
-   common-random-number evaluation,
-   statistical comparison,
-   communication-aware GRU training pipeline.

**Preliminary / περιορισμένο evidence**

-   compact WOMD proxy evaluation.

**Δεν αποτελεί ακόμη ολοκληρωμένο claim**

-   full-WOMD learned validation,
-   real PC-FMCW optical measurement validation,
-   full WOMD-LiDAR sensing,
-   calibrated probabilistic sensing/trajectory uncertainty,
-   Beam + ADB joint control.

------------------------------------------------------------------------

### Μία πρόταση για paper / παρουσίαση

> **Αναπτύσσουμε ένα causal trajectory-predictive scheduling framework
> για PC-FMCW/DPSK vehicular communications, στο οποίο η προβλεπόμενη
> κίνηση μετατρέπεται σε μελλοντική ποιότητα και διάρκεια του optical
> link και χρησιμοποιείται για proactive packet transmission πριν από
> προβλεπόμενη αποσύνδεση.**
