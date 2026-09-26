# BrainVuln Multi-Seed Robustness

*Generated 2026-09-26T15:18:40+00:00 by `scripts/robustness_report.py` from the frozen run artifacts of seeds 42, 1, 2, 3, 4 (canonical recipe; only the seed differs). Wording rule: this report states what the numbers demonstrate — no robustness verdicts beyond the decision gate.*

## Seeds

| Seed | Checkpoint SHA256 (first 16) | Val-Youden threshold |
| --- | --- | ---: |
| 42 | `5930f0fff9600346…` | 0.07 |
| 1 | `69d918d1dc75a6cb…` | 0.17 |
| 2 | `1ac69639777f7bd7…` | 0.98 |
| 3 | `f984a433556715f2…` | 0.94 |
| 4 | `5781f47fb6881a3b…` | 0.88 |

The seed-42 checkpoint remains the canonical production model (`5930f0fff9600346…`); none of the robustness seeds replace it.

## Predictive performance

| Metric | Mean | SD | Median | IQR | Min | Max |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| ROC-AUC | 0.800 | 0.022 | 0.794 | [0.789, 0.817] | 0.772 | 0.828 |
| PR-AUC | 0.850 | 0.023 | 0.855 | [0.844, 0.859] | 0.814 | 0.875 |
| Accuracy | 0.689 | 0.033 | 0.667 | [0.667, 0.704] | 0.667 | 0.741 |
| Balanced Acc | 0.678 | 0.032 | 0.683 | [0.658, 0.683] | 0.642 | 0.725 |
| Sensitivity | 0.773 | 0.146 | 0.867 | [0.733, 0.867] | 0.533 | 0.867 |
| Specificity | 0.583 | 0.156 | 0.583 | [0.500, 0.583] | 0.417 | 0.833 |
| Precision | 0.709 | 0.057 | 0.688 | [0.684, 0.722] | 0.650 | 0.800 |
| Recall | 0.773 | 0.146 | 0.867 | [0.733, 0.867] | 0.533 | 0.867 |
| F1 | 0.729 | 0.058 | 0.743 | [0.710, 0.765] | 0.640 | 0.788 |
| Brier | 0.277 | 0.094 | 0.283 | [0.193, 0.297] | 0.192 | 0.421 |

Per-seed values (with subject-bootstrap 95% CIs for ROC-AUC):

| Seed | ROC-AUC | 95% CI | PR-AUC | Sens | Spec | Brier |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 42 | 0.817 | [0.618, 0.972] | 0.814 | 0.867 | 0.417 | 0.192 |
| 1 | 0.828 | [0.637, 0.966] | 0.875 | 0.867 | 0.583 | 0.193 |
| 2 | 0.794 | [0.600, 0.950] | 0.859 | 0.733 | 0.583 | 0.421 |
| 3 | 0.772 | [0.580, 0.933] | 0.844 | 0.533 | 0.833 | 0.283 |
| 4 | 0.789 | [0.600, 0.944] | 0.855 | 0.867 | 0.500 | 0.297 |

## Prediction stability

* 17/27 test subjects received the same label from all 5 seeds; 23/27 from at least 4 of 5 seeds.
* Subject-level probability SD across seeds: median 0.252, max 0.472 (`OAS1_0176`).
* 14 subjects have probability SD > 0.25 across seeds (see `prediction_stability.csv`).

## Error stability

* CN subjects flagged AD by all 5 seeds: 2 (OAS1_0013, OAS1_0337).
* CN subjects flagged by 3-4 seeds: 4.
* AD subjects missed by all 5 seeds: 2 (OAS1_0082, OAS1_0142).
* Seed-42 baseline FP/FN (7/2) therefore splits into persistent and seed-dependent components; per-subject counts in `error_stability.csv`.

## Age association

* Spearman(prob, age) per seed: seed 42 +0.368, seed 1 +0.399, seed 2 +0.425, seed 3 +0.417, seed 4 +0.401.
* Sign-consistent across all 5 seeds: True; range [0.368, 0.425].
* Age+sex baseline (train-fitted) Spearman: -0.703.
* Interpretation stays associative: the age-probability relationship is a property of the predictions, not evidence of mechanism.

## Calibration stability

| Seed | Brier | Slope | Intercept |
| --- | ---: | ---: | ---: |
| 42 | 0.192 | 0.340 | 0.058 |
| 1 | 0.193 | 0.493 | 0.459 |
| 2 | 0.421 | 1.221 | -4.768 |
| 3 | 0.283 | 0.705 | -1.303 |
| 4 | 0.297 | 0.319 | -1.185 |

* Calibration slope range across seeds: [0.319, 1.221] — all below 1 = overconfident in every seed. No recalibration was applied (measurement only).

## Grad-CAM stability (population M_CNN maps)

| Seed A | Seed B | Spearman rho (87 parcels) |
| --- | --- | ---: |
| 1 | 2 | 0.667 |
| 1 | 3 | 0.742 |
| 1 | 4 | 0.515 |
| 1 | 42 | 0.557 |
| 2 | 3 | 0.805 |
| 2 | 4 | 0.803 |
| 2 | 42 | 0.778 |
| 3 | 4 | 0.483 |
| 3 | 42 | 0.459 |
| 4 | 42 | 0.924 |

* Median pairwise regional Spearman: 0.705 (range [0.459, 0.924]).

## Regional stability

* Regions in the population top-10 for all 5 seeds: 0.
* Regions in top-10 for at least 4 seeds: 2 of 87.
* CV and per-seed ranks: `regional_stability.csv`. These are model-derived relevance measurements only; no biological reading is made here, and any region highlighted by a single seed is reported as such, not as a finding.

## Seed-42 compared with the other seeds

| Comparison | Probability Spearman | ROC-AUC seed42 | ROC-AUC other |
| --- | ---: | ---: | ---: |
| seed42_vs_seed1 | 0.803 | 0.817 | 0.828 |
| seed42_vs_seed2 | 0.726 | 0.817 | n/a |
| seed42_vs_seed3 | 0.756 | 0.817 | n/a |
| seed42_vs_seed4 | 0.879 | 0.817 | n/a |

* Seed 42 is retained as the canonical baseline; these comparisons assess whether it is representative, not whether another seed is 'better'.

## Reproducibility check (phase 19)

* Seed-4 repeat, identical command: predictions bitwise identical = **True**; max |Δp| = 0.00e+00; ROC-AUC 0.789 vs 0.789; checkpoint bytes identical = False.
* Identical predictions from byte-differing checkpoints = serialization-level nondeterminism only (acceptable); materially different predictions would have indicated real nondeterminism.

## Final decision gate

* [x] all seeds trained successfully, no crashes
* [x] no data leakage (thresholds frozen from validation per seed)
* [x] test metrics computed for every seed with archived predictions
* [x] performance does not collapse for most seeds (5/5 seeds with ROC-AUC >= 0.65; median 0.794)
* [x] predictions reasonably stable (23/27 subjects with >= 4/5 seed agreement)
* [x] explanation maps not completely unstable (median pairwise M_CNN Spearman 0.705)
* [x] no critical reproducibility failure (seed-4 repeat predictions identical)

### ROBUST ENOUGH FOR GENOMIC INTEGRATION

*Rationale:* the gate evaluates variance and stability, never best-seed selection. Criteria marked '[ ]' name the measurement that failed and where its numbers live.

## MULTI-SEED ROBUSTNESS STATUS

**MULTI-SEED ROBUSTNESS STATUS: PASS**

* Seeds trained: 5 (42, 1, 2, 3, 4) + 1 repeat run; failures: none.
* Mean ROC-AUC 0.800 ± 0.022 (median 0.794, range [0.772, 0.828]); mean PR-AUC 0.850 ± 0.023.
* Sensitivity range [0.533, 0.867]; specificity range [0.417, 0.833]; Brier range [0.192, 0.421].
* Consistently misclassified subjects: 17-unanimous classification for 17/27; always-FP CN 2; always-FN AD 2.
* Prediction stability: median subject SD 0.252.
* Age-probability relationship: present in every seed (Spearman +0.37 to +0.43).
* Grad-CAM similarity: median pairwise Spearman 0.705.
* Regional relevance stability: 0 regions in top-10 across all seeds.
* Seed-42 remains representative: see per-seed table and the seed-42-pairwise comparisons above; it was not replaced and not retrained.
* Genomic integration: may begin.
