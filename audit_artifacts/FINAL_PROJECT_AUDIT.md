# BrainVuln Final Project Audit

*Generated 2026-09-25T13:57:17+00:00. Every claim below is backed by an artifact path; anything without artifact support is marked not-completed.*

## Training Completion

* Canonical seed-42 run: COMPLETE (best val ROC-AUC 0.800 @ epoch 11; early stop; `results/ml/resnet_seed42/metrics/train_history.json`).
* Multi-seed campaign: ALL RUNS COMPLETE (`audit_artifacts/multiseed_status.md`).

## Model Reliability

* Threshold integrity: frozen 0.07 (validation Youden), enforced by tests/test_threshold_frozen.py.
* Reproducibility (same-seed repeat): pending.
* Prediction stability across seeds: pending.
* Explainability stability: pending (campaign in progress).

## Test Performance (canonical seed-42, n=27 subjects)

| Metric | Value | 95% CI |
| --- | ---: | ---: |
| ROC-AUC | 0.817 | [0.618, 0.972] |
| PR-AUC | 0.814 | [0.604, 0.984] |
| Accuracy | 0.667 | — |
| Balanced accuracy | 0.642 | [0.466, 0.810] |
| Sensitivity | 0.867 | [0.667, 1.000] |
| Specificity | 0.417 | [0.125, 0.700] |
| F1 | 0.743 | — |
| Brier | 0.192 | [0.080, 0.314] |

Calibration: slope 0.34 (overconfident); age association Spearman 0.368; confound details in results/ml/evaluation/age_sex_confounds.json.

## Multi-Seed Robustness

*Campaign in progress — distribution pending; seed 1 first result: ROC-AUC 0.828 [0.644, 0.963] (frozen val threshold 0.18). No robustness claims are made until `audit_artifacts/multiseed_robustness.md` exists.*

## Explainability

* Grad-CAM coverage: 27/27 test subjects (seed 42), NIfTI + planes + 87-parcel regional tables.
* Grad-CAM vs occlusion agreement (seed 42): Spearman 0.294 — weak-to-moderate; regional relevance is model-derived, not biology.

## Genetic Integration & Spatial Statistics

* Full-scale bridge executed (5000 matched-gene + 5000 spatial nulls, `results/bridge/`): the **matched-gene null was not passed** (expression-matched random sets correlate at least as high), so the CNN–gene spatial association is not demonstrated to be AD-genetics-specific. Spatial-structure nulls passed. This is recorded as a negative specificity result, not as success.
* Independent disease maps: imported with provenance (data/external/disease_maps/); cross-disease negative controls: code verified, results pending final CNN-side confirmation.

## Product / Inference

* CLI: `predict.py` prints Prediction / Probability / Threshold / Model / Checkpoint sha256 and writes outputs/prediction.json (verified by live runs in audit phase_09/phase_21 and by tests).
* Web demo: `app.py` shows MRI + Grad-CAM visualizations, atlas-derived regional relevance, supported-input statement, and the research-only warning.
* Dashboard / results pages: `results/dashboard.md`, `results/FINAL_RESULTS.md` (measured values only).

## Limitations

* n=27 test subjects; single site; wide CIs; CDR labels are not biomarker-confirmed AD; class-level sex imbalance.
* Overconfident calibration (slope 0.34) measured, not corrected.
* Age–probability association (+0.37 Spearman) — demographic component not excluded at this sample size.
* External validation not performed (OASIS-3/ADNI stagers ready).
* Matched-gene null not passed — genetic specificity not demonstrated.

## Final Scientific Status

| Capability | Completed | Evidence |
| --- | --- | --- |
| MRI model | YES | frozen checkpoint + training history + tests |
| Held-out evaluation | YES | results/ml/evaluation/test_metrics_subject_level.json |
| Multi-seed robustness | NO (campaign in progress) | results/ml/multiseed/ + audit_artifacts/multiseed_robustness.md |
| Grad-CAM | YES | 27/27 subject CAMs + regional tables |
| AHBA | YES | results/ahba/expression parquet + phase_13/14 artifacts |
| GWAS | YES | results/gene_sets/alzheimer_gwascat.json + phase_12 metadata |
| Matched genetic null | YES (executed; NOT passed) | results/bridge/cnn_gene_spatial_null_r.json |
| Spatial null | YES | results/bridge/ (Moran + Burt-2020 pass) |
| Independent disease map | YES | data/external/disease_maps/* + provenance |
| External cohort | NO (not started) | scripts/stage_oasis3.py, stage_adni.py ready |
| Cross-disease | NO (pending final CNN-side run) | src/brainvuln/cross_disorder.py verified |

Anything marked NO must not be presented as completed functionality.

## Final Readiness

**NOT READY FOR PORTFOLIO DEMONSTRATION (yet)** — the multi-seed campaign and its reliability gates must complete first (see audit_artifacts/multiseed_status.md). All other components are complete and artifact-backed; this verdict flips automatically when `scripts/robustness_report.py` and this generator are re-run after the campaign.
