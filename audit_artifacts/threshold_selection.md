# Threshold selection (REAL MODEL EVALUATION, phases 1-2)

*Created 2026-09-25T09:28:47+00:00 — validation subjects only; no test contact (`test_contact: false` in threshold_selection.json).*

## Method

* **Method:** Youden's J — maximize sensitivity + specificity − 1 over a 197-point grid on [0.01, 0.99] (the tested `select_threshold(..., "youden")`).
* **Why:** symmetric treatment of sensitivity/specificity; already fixed as `threshold.method: youden` in `configs/oasis1_resnet3d.yaml` before this phase, so the re-derivation changes no pre-registered choice.
* **Data:** validation subjects only (n = 27). The threshold was frozen before the test loader was ever constructed in this evaluation.
* **Checkpoint:** `best.pt` (sha256 `5930f0fff9600346…`), preprocessing `t88_masked_gfc-2mm-128cube-brainz-v1`.

## Selected operating point

* **Threshold = 0.07**
* Validation ROC-AUC 0.800 (PR-AUC 0.853) — discrimination context only.

| Validation metric @ threshold | Value |
| --- | ---: |
| Sensitivity | 0.933 |
| Specificity | 0.583 |
| Balanced accuracy | 0.758 |
| Accuracy | 0.778 |
| F1 | 0.824 |

## Bootstrap stability (validation subjects only)

* 2000 resamples of the 27 validation subjects (unit = subject, seed 42); Youden re-selected per resample.
* **Median** 0.070 — **95% interval** [0.015, 0.760]
* Interpretation: with n=27 validation subjects the operating point is only moderately stable; the interval quantifies that uncertainty and is NOT a recommendation to re-tune per dataset.

## Freeze

* The value 0.07 is frozen as `inference.threshold.value` in config/analysis.yaml; serving code resolves it through `brainvuln.config.resolve_threshold()` (tests/test_threshold_frozen.py pins the value).
* Agreement with the original run's frozen threshold: 0.07 (match = True).

## Test-set integrity

No test label, prediction, metric, ROC curve, or confusion matrix was used in any step above. Test evaluation happens only in phase B, after this file was written.