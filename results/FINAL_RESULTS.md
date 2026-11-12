# BrainVuln — Final Results (measured only)

*Generated 2026-09-26T15:21:04+00:00. This file contains measured results and their limitations.
No marketing claims. Research use only — not a clinical diagnostic system.*

## Addendum 2026-11-04 — full retrain on the extracted OASIS-1 disc + TTA

The canonical seed-42 checkpoint was found unusable as a starting point (BN
running statistics consistent with an untrained network). A fresh, complete
retrain on the same curriculum — identical recipe (seed 42, light
augmentation, warmup 2 + cosine horizon 12, AdamW 1e-4, patience 6), the
same 182-session cache (`data/derived/oasis1_128`, restored on disk incl.
the 27 previously missing test volumes), the same git-ignored extracted
disc (`C:/Users/kanishk thakur/Desktop/BrainVuln_OASIS`) — was run end to end.

* Run: `results/ml/resnet_seed42_retrain/`
* Checkpoint: `checkpoints/best.pt`, SHA256
  `a4b553e0bd0eebaded7508033c5ba91633fa4a77bea9893ecabc559a55cfee42`
* Best epoch **7** (val AUC 0.7944); training was resumed once (per-epoch
  crash-safe `last_state.pt`) and run through the cosine horizon (epochs
  11–12, LR ≤ 2.5e-6) — no further val improvement, confirming the
  checkpoint is terminal for this schedule.

### Retrain test results (27 held-out subjects)

| Metric | Retrain | Canonical record |
| --- | ---: | ---: |
| ROC-AUC | 0.800 [0.602, 0.963] | 0.817 [0.618, 0.972] |
| PR-AUC | 0.857 [0.672, 0.981] | 0.814 [0.604, 0.984] |
| Accuracy (val threshold) | 0.556 | 0.667 |
| Sensitivity / Specificity | 0.933 / 0.083 | 0.867 / 0.417 |
| Brier | 0.438 | 0.192 |
| Val-chosen threshold | 0.99 | 0.07 |

Val-selected threshold 0.99: TP 14, FN 1, FP 11, TN 1. Prediction ranking
agrees with the frozen canonical predictions (Spearman 0.855) — the data
cache and splits are consistent across the two runs.

### Honest failure mode: probability saturation / threshold transfer

The retrain's test probabilities are all ≥ 0.9387 (25/27 > 0.99), while its
validation probabilities spread below 0.99 for 7 CN subjects. The ranking
(AUC 0.80) is real, but operating-point transfer fails: the val threshold
0.99 collapses into predicting AD for 25/27 test subjects. TTA (mean of 8
probes, rotations ±4° in 3 axes, 2-voxel y-shifts, 1.03 scale) cannot fix it
— all 8 probes saturate on test (per-variant mean ≈ 0.99). Measured TTA
effect (`results/ml/resnet_seed42_retrain/tta/`):

* Validation (n=27, Youden threshold on TTA probs): AUC 0.794 → **0.856**,
  acc/bacc unchanged (0.704 / 0.675).
* Test (fixed threshold 0.99): AUC 0.800 → **0.822**, bacc 0.508 → 0.517,
  accuracy unchanged (0.556).
* Diagnostic only (not a selection): test oracle accuracy 0.815 at
  threshold 0.9991 — the information exists in the ranking, the probability
  scale is broken. This is an extreme case of the documented overconfidence
  (calibration slope 0.34).

### Forward fix in flight: label smoothing

A fresh seed-42 run identical to above except `--label-smoothing 0.1`
(smoothing is implemented in `src/brainvuln/mri/train.py` but was never
used) started 2026-11-04 to directly attack the overconfident-logits
failure: smooth targets (0.05/0.95) bound the optimal logits, keeping
probabilities interpretable so standard-threshold transfer works.
Run dir `results/ml/resnet_seed42_smooth/`, log
`results/smooth_epochs.log`, per-epoch crash-safe checkpoints. Evaluation
is automatic after training (same stages); TTA re-evaluation afterwards.
No test-set information entered the threshold choice at any point.

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
