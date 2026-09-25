# Error analysis (frozen checkpoint, test subjects)

Threshold 0.07 (frozen, validation-selected). Probabilities are mean predicted AD-probabilities. Age/Sex from data/splits/matched_cohort.csv.

| Group | n | mean prob | prob range | mean age | age range | F | M |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| TP | 13 | 0.872 | [0.384, 1.000] | 79.0 | [70, 89] | 5 | 8 |
| TN | 5 | 0.021 | [0.012, 0.026] | 76.4 | [65, 88] | 5 | 0 |
| FP | 7 | 0.577 | [0.225, 0.999] | 80.6 | [69, 94] | 5 | 2 |
| FN | 2 | 0.034 | [0.027, 0.042] | 72.5 | [70, 75] | 1 | 1 |

## False positives (CN called AD by the model)

* `OAS1_0013` — prob 0.431, age 81, sex F, MMSE 30.0
* `OAS1_0169` — prob 0.501, age 88, sex F, MMSE 30.0
* `OAS1_0199` — prob 0.225, age 69, sex M, MMSE 30.0
* `OAS1_0221` — prob 0.835, age 94, sex F, MMSE 29.0
* `OAS1_0330` — prob 0.659, age 80, sex F, MMSE 27.0
* `OAS1_0337` — prob 0.999, age 81, sex M, MMSE 28.0
* `OAS1_0341` — prob 0.391, age 71, sex F, MMSE 30.0

## False negatives (AD called CN by the model)

* `OAS1_0082` — prob 0.027, age 75, sex F, MMSE 28.0
* `OAS1_0142` — prob 0.042, age 70, sex M, MMSE 27.0

The test set exists to reveal these errors; no subject was re-labeled, removed, or 'fixed'.
