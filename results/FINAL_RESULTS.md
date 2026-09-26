# BrainVuln — Final Results (measured only)

*Generated 2026-09-26T15:21:04+00:00. This file contains measured results and their limitations.
No marketing claims. Research use only — not a clinical diagnostic system.*

## Primary model

* Checkpoint: `results/ml/resnet_seed42/checkpoints/best.pt`
* SHA256: `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`
* Training seed: 42 (canonical baseline; never replaced by the robustness seeds)
* Architecture: ResNet18Binary (MONAI ResNet-18, 1 input channel, single-logit head)
* Preprocessing: `t88_masked_gfc-2mm-128cube-brainz-v1`

## Dataset

OASIS-1 cross-sectional, age ≥ 60, CDR-based labels.

* 27 test subjects: 15 AD / 12 CN (one session per subject)

Full cohort: 182 subjects (100 AD / 82 CN), deterministic 1:1 matching,
70/15/15 subject-level split (seed 42). The threshold was selected on the
27 validation subjects only.

## Test results (27 held-out subjects, unit = subject)

| Metric | Value | 95% CI (subject bootstrap, 2000 draws) |
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

## Multi-seed results (robustness distribution — seeds 42, 1, 2, 3, 4)

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

* Same seed + same config, run twice: predictions bitwise identical = **True** (max |Δp| = 0.00e+00); ROC-AUC 0.789 vs 0.789; checkpoint bytes identical = False.

## Prediction stability

* Median across-seed SD of predicted probability: 0.252 (max 0.472)
* Subjects with the same label from all 5 seeds: **17/27**
* Per-subject table: `results/ml/multiseed/prediction_stability.csv`

## Error analysis

* False positives: **7** (OAS1_0013, OAS1_0169, OAS1_0199, OAS1_0221, OAS1_0330, OAS1_0337, OAS1_0341)
* False negatives: **2** (OAS1_0082, OAS1_0142)
* Details: `results/ml/evaluation/error_analysis.md`

False-positive profile (seed 42): oldest group (mean age ~80.6), MMSE 27–30,
probabilities 0.225–0.999. False-negative profile: two CDR-0.5 subjects,
probabilities 0.027/0.042. Per-seed persistence:
`results/ml/multiseed/error_stability.csv`.

## Age confounding

* Spearman(probability, age): 0.368 (p = 0.059) (across seeds: 0.368 to 0.425 — sign-consistent)
* Pearson: 0.396 (p = 0.041)
* Age+sex baseline (train-fitted): test ROC-AUC 0.700 vs CNN 0.817
* Reading: association, not causation; the CNN exceeds the demographic baseline but a demographic component cannot be excluded at n=27 (FPs skew old: mean age 80.6)

## Calibration

* Brier 0.192; calibration slope **0.34** (intercept 0.06) — slope < 1 = overconfident
* Reliability bins (predicted → observed): 0.02 → 0.29 · 0.48 → 0.29 · 0.86 → 0.83 · 0.98 → 0.86

Probabilities are usable for ranking, not as absolute risk estimates; the
model was **not** recalibrated (measurement only).

## Explainability

* Grad-CAM coverage: **27/27** test subjects (regional CSVs: 27)
* Population top region (seed 42): `caudalanteriorcingulate_L` (M_CNN 0.631)
* Grad-CAM vs occlusion agreement (seed 42): Spearman 0.294

Explanations are model-derived relevance, not biological truth; a region
highlighted by a single seed is reported as such.

## Current limitations

* n = 27 test subjects; all CIs are wide; single site, single scanner.
* CDR-based "probable AD" is not biomarker-confirmed AD.
* Class-level sex imbalance in the cohort (AD 59% F vs CN 70% F).
* Overconfident calibration (slope ≈ 0.34) measured, not corrected.
* External validation (OASIS-3, ADNI): **not performed**.
* Grad-CAM vs occlusion agreement is weak-to-moderate; regional claims stay coarse.
* Research prototype only — must not inform any clinical decision.
