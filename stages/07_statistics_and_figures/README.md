# Stage 7 - Statistics and figures

Joins forecast accuracy with realized communication utility and performs
scenario-cluster inference, bootstrap confidence intervals, Wilcoxon/t-test
sensitivity, effect sizes and Holm correction. It creates publication tables
and figures without hiding negative regimes.

Input: Stage-5 and Stage-6 raw results. Outputs: `artifacts/paper_final/07_analysis/`.

```bash
python stages/07_statistics_and_figures/run.py
```
