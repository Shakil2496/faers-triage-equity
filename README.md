# FAERS Equity Audit — Machine-Learning Seriousness Triage

Reproducible code for an equity audit of a leakage-controlled machine-learning model that triages FDA Adverse Event Reporting System (FAERS) reports by seriousness, focused on reports with missing patient demographics.

The study characterizes under-documented reports, tests candidate mechanisms for the reliability gap they exhibit, and derives a calibrated conformal-prediction abstention boundary that separates reliably-scoreable reports from principled abstentions.

- **Registration (analysis protocol):** [OSF, doi:10.17605/OSF.IO/3EF4W](https://doi.org/10.17605/OSF.IO/3EF4W)
- **Manuscript:** under review (2026)
- **Author:** Shakil Mahmud (sole author)
- **License:** MIT

## Data

Data are public and are **not** stored in this repository; they regenerate from the openFDA drug-event API (FAERS, 2025 Q1–Q4; accessed April 2026). The analysis uses a temporal split: training on 2025 Q1–Q3 and a held-out temporal test set of 2025 Q4 (n = 24,801). The feature pipeline (deduplication, leakage-controlled feature construction, a training-quarter-only vocabulary of 800 reaction preferred terms and 300 suspect ingredients) is documented in the companion seriousness-triage study.

Each script reads two parquet files, `faers_temporal_train.parquet` and `faers_temporal_test.parquet`; set `DATA_DIR` at the top of each script to their location.

## Scripts

| Script | Purpose |
|---|---|
| `10_part1_content_density.py` | Part 1 — characterizes under-documented reports: content density by subgroup with effect sizes, correlated missingness, and a documentation-completeness gradient. Model-free. |
| `11_reliability_mechanism.py` | Part 2a — reproduces the clinical model, quantifies the subgroup reliability gap, and tests two mechanisms (content mediation and feature/age mediation), both refuted; plus a reporter-stream refinement. |
| `12_part2b_conformal_boundary.py` | Part 2b — split-conformal prediction (marginal and conditional/Mondrian), reporting coverage and abstention by subgroup over repeated calibration splits. |
| `13_figures.py` | Generates the three manuscript figures (completeness gradient; three-diagnostics mechanism spine; calibrated boundary). |
| `14_tables.py` | Generates the three manuscript tables (cohort characteristics; content density with effect sizes; conformal coverage/abstention) and writes CSVs. |

## Reproduce

```bash
pip install -r requirements.txt
# set DATA_DIR at the top of each script, then:
python 10_part1_content_density.py
python 11_reliability_mechanism.py
python 12_part2b_conformal_boundary.py
python 13_figures.py
python 14_tables.py
```

Model training (scripts 11, 13, 14) takes a few minutes per run; the analyses use a fixed random seed for reproducibility.

## Leakage control

The feature set excludes every field downstream of the seriousness determination (seriousness sub-criteria, expedited-reporting flag, reaction outcome, action taken) and outcome-equivalent reaction terms (e.g., death, hospitalisation), enforced by a build-time assertion in the feature pipeline. This is the same leakage-controlled design used in the companion triage study.

## Citation

If you use this code, please cite the registration (doi:10.17605/OSF.IO/3EF4W) and the manuscript once available.
