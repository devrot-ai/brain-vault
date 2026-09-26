# BrainVuln results dashboard

*Generated 2026-09-26T15:04:15+00:00 from evaluation artifacts. Every number below is measured;
nothing is projected. Research use only — not a clinical diagnostic system.*

## Model performance (canonical seed-42 baseline, held-out test)

| Metric | Value | 95% CI (subject bootstrap) |
| --- | ---: | ---: |
| ROC-AUC | **0.817** | [0.618, 0.972] |
| PR-AUC | 0.814 | [0.604, 0.984] |
| Accuracy | 0.667 | — |
| Balanced accuracy | 0.642 | [0.466, 0.810] |
| Sensitivity | 0.867 | [0.667, 1.000] |
| Specificity | 0.417 | [0.125, 0.700] |
| Precision | 0.650 | — |
| Recall | 0.867 | [0.667, 1.000] |
| F1 | 0.743 | — |
| Brier | 0.192 | [0.080, 0.314] |

Threshold **0.07** (frozen, validation-only
Youden). Confusion matrix:

| | pred CN | pred AD |
| --- | ---: | ---: |
| **true CN** | 5 | 7 |
| **true AD** | 2 | 13 |

## Model robustness (multi-seed distribution, seeds 42/1/2/3/4)

| Metric | Mean ± SD | Median [IQR] | Min–Max |
| --- | --- | --- | --- |
| ROC-AUC | 0.800 ± 0.022 | 0.794 [0.789, 0.817] | 0.772–0.828 |
| PR-AUC | 0.850 ± 0.023 | 0.855 [0.844, 0.859] | 0.814–0.875 |
| Accuracy | 0.689 ± 0.033 | 0.667 [0.667, 0.704] | 0.667–0.741 |
| Balanced Acc | 0.678 ± 0.032 | 0.683 [0.658, 0.683] | 0.642–0.725 |
| Sensitivity | 0.773 ± 0.146 | 0.867 [0.733, 0.867] | 0.533–0.867 |
| Specificity | 0.583 ± 0.156 | 0.583 [0.500, 0.583] | 0.417–0.833 |
| Precision | 0.709 ± 0.057 | 0.688 [0.684, 0.722] | 0.650–0.800 |
| Recall | 0.773 ± 0.146 | 0.867 [0.733, 0.867] | 0.533–0.867 |
| F1 | 0.729 ± 0.058 | 0.743 [0.710, 0.765] | 0.640–0.788 |
| Brier | 0.277 ± 0.094 | 0.283 [0.193, 0.297] | 0.192–0.421 |

Reproducibility (phase-19 repeat): * Same seed + same config, run twice: predictions bitwise identical = **True** (max |Δp| = 0.00e+00); ROC-AUC 0.789 vs 0.383; checkpoint bytes identical = False.

## Test subjects

* 27 test subjects: 15 AD / 12 CN (one session per subject)

## Error analysis

* False positives: **7** (OAS1_0013, OAS1_0169, OAS1_0199, OAS1_0221, OAS1_0330, OAS1_0337, OAS1_0341)
* False negatives: **2** (OAS1_0082, OAS1_0142)
* Details: `results/ml/evaluation/error_analysis.md`

## Explainability

* Grad-CAM coverage: **27/27** test subjects (regional CSVs: 27)
* Population top region (seed 42): `caudalanteriorcingulate_L` (M_CNN 0.631)
* Grad-CAM vs occlusion agreement (seed 42): Spearman 0.294

* Regions in the population top-10 for all seeds: **0**
* Regions in top-10 for ≥ 4 seeds: 2/87
* Table: `results/ml/multiseed/regional_stability.csv`

## Calibration

* Brier 0.192; calibration slope **0.34** (intercept 0.06) — slope < 1 = overconfident
* Reliability bins (predicted → observed): 0.02 → 0.29 · 0.48 → 0.29 · 0.86 → 0.83 · 0.98 → 0.86

## Validation status

* External validation (OASIS-3 / ADNI): **NOT YET PERFORMED**.
* Genomics integration (GWAS/AHBA bridge): code complete and audited; the
  CNN side of the bridge awaits the multi-seed gate.
* Research use only. Not a clinical diagnostic system.
