# Exact implementation status

## What was changed in corrected-v1

| Area | Concrete change | Verification |
|---|---|---|
| Coordinate frame | Explicit ego heading in link-aware loss | Rotation-invariance test |
| Heading | Last valid direction persists while stationary | Regression test |
| Horizon | All forecasts truncate at record end | Tail-horizon test |
| Time units | Duration, deadlines and horizon represented in seconds | Slot-invariance tests |
| Power | Relative power separated from calibration flag | Link tests |
| Outage | BER/PER/goodput modes stored and selectable | Semantic tests |
| BER LUT | −5…25 dB adaptive bits and upper confidence bound | Theory tracking test |
| Traffic | Saturated mode and physical deadlines | Reproducibility tests |
| PF baseline | Normalized current rate / past service | Policy integration tests |
| Lifetime utility | Dimensionless horizon normalization | Scale-invariance test |
| Metrics | P50/P99, censoring, scheduled link state, demand fairness | End-to-end tests |
| Statistics | Cluster aggregation, metric direction and Holm | Exact stats tests |
| Slices | Motion/FoV regime classifier | Actor-coverage test |
| Sensing | Cartesian and range/bearing AR(1) assumptions | Reproducibility test |
| Official WOMD | True-SDC, vehicle and valid-state adapter | Schema fixture test |
| GRU | Four objectives, smooth FoV, frame-correct loss, dropout | Loss tests |
| Checkpoint | Feature schema and dataset hash enforcement | Invalid schema rejected by design |
| Experiments | One-axis staged design and corrected run isolation | 430 rows generated |
| Documentation | Beginner README, methods, results, traceability, paper | Files regenerated |

## Executed now

- 45/45 automated tests;
- lint and compile checks;
- five scientific validation gates;
- 31-point adaptive DBPSK LUT;
- 120 controlled policy episodes across 12 scenarios and 10 policies;
- synthetic and compact-proxy forecast evaluation;
- 42 quick ablation episodes;
- 430 staged diagnostic policy episodes;
- exact CSV/JSON/LaTeX and ten corrected result figures.

## Implemented but not executable with supplied inputs

- official-WOMD TFRecord export and true-SDC training samples;
- 4 objectives × 3+ seeds GRU training;
- learned checkpoint evaluation in the scheduler.

These require external WOMD shards and PyTorch. The supplied Stage-4 JSON files
are reports only and cannot replace the missing checkpoint.

## Not done because evidence does not exist

- measured PC-FMCW vehicular channel validation;
- calibrated absolute received power;
- official full-WOMD held-out results;
- successful communication-aware learned result;
- exact offline scheduling upper bound;
- final venue-formatted, author-complete submission.

No item in this final section is claimed elsewhere in the repository.
