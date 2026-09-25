# BrainVuln results dashboard

*Generated 2026-09-25T12:59:25+00:00 from evaluation artifacts. Every number below is measured;
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

*Multi-seed campaign **in progress** — see `audit_artifacts/multiseed_status.md` for per-seed state. This section fills automatically when `scripts/multiseed_analysis.py` completes.*

Reproducibility (phase-19 repeat): *Campaign in progress — the seed-4 repeat has not finished.*

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

*Campaign in progress — regional-stability table fills post-campaign.*

## Calibration

* Brier 0.192; calibration slope **0.34** (intercept 0.06) — slope < 1 = overconfident
* Reliability bins (predicted → observed): 0.02 → 0.29 · 0.48 → 0.29 · 0.86 → 0.83 · 0.98 → 0.86

## Validation status

* External validation (OASIS-3 / ADNI): **NOT YET PERFORMED**.
* Genomics integration (GWAS/AHBA bridge): code complete and audited; the
  CNN side of the bridge awaits the multi-seed gate.
* Research use only. Not a clinical diagnostic system.
