# BrainVuln Subject-Level Evaluation (REAL MODEL EVALUATION)

*Generated 2026-09-25 by `scripts/eval_subject_level.py` against the frozen canonical checkpoint. Raw console: `audit_artifacts/eval_subject_level_console.txt`. Status at end of file.*

## Dataset

* **Cohort:** OASIS-1 cross-sectional, subjects aged ≥ 60 with usable CDR; probable AD = CDR ∈ {0.5, 1, 2}, CN = CDR 0 (`data/splits/cohort_manifest.json`).
* **Subjects:** 182 total (100 AD / 82 CN), deterministic 1:1 age- and sex-matched; split 70/15/15 by subject with seed 42.
* **This evaluation:** the held-out **test partition: 27 subjects (15 AD / 12 CN)**, plus the validation partition (27 subjects: 15 AD / 12 CN) used for the operating point.
* **Test age distribution:** FP/TN/TP/FN summaries below; test ages span 60–94 (mean ≈ 78.4). Validation ages comparable.
* **Sex distribution (test):** 16 F / 11 M overall; within class, AD is 6 F / 9 M while CN is 10 F / 2 M — the class-level sex imbalance inherited from the matched-cohort design (SMD sex_female_pct ≈ 10.5 pp).
* **Scanner/site limitations:** OASIS-1 is a single-site cohort; no scanner/site variable exists to adjust for, and no cross-site generalization is claimed.
* **Exclusions:** sessions without CDR and subjects with conflicting CDR across sessions; exclusion log in the phase-03 audit artifacts.

## Frozen model

| Field | Value |
| --- | --- |
| Architecture | ResNet18Binary (MONAI ResNet-18 backbone, 1 input channel, single-logit sigmoid head) |
| Checkpoint | `results/ml/resnet_seed42/checkpoints/best.pt` |
| SHA256 | `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` |
| Size | 133,540,267 bytes (re-verified byte-identical after all evaluation work) |
| Training seed | 42 |
| Preprocessing | T88 `masked_gfc` input → RAS → 2 mm trilinear → crop/centre-pad 128³ → brain z-score (`t88_masked_gfc-2mm-128cube-brainz-v1`, deterministic, no stochastic component) |
| Selection metric during training | validation ROC-AUC only (`metrics/train_history.json`: best val AUC 0.8, epoch 11) |

The checkpoint was opened read-only throughout; SHA-256 verified before and after this phase.

## Frozen threshold

* **Threshold = 0.07**, selected on **validation subjects only** by **Youden's J** (maximize sensitivity + specificity − 1 over a 197-point grid on [0.01, 0.99]).
* **Method provenance:** Youden's J was already fixed as `threshold.method: youden` in `configs/oasis1_resnet3d.yaml` before this phase; the re-derivation changed no pre-registered choice. Rationale: symmetric treatment of sensitivity and specificity, no prevalence assumption.
* **Re-derivation (this phase):** recomputed from validation predictions — **0.07, exactly matching the value frozen by the original run** (`agreement.match = true`).
* **Validation-only bootstrap stability:** 2,000 subject-level resamples; Youden re-selected per resample: **median 0.070, 95% interval [0.015, 0.760]**. The wide interval says the operating point is only weakly identified by n=27 validation subjects — it does NOT justify re-tuning per dataset; it justifies reporting it with its uncertainty.
* **Validation performance at 0.07:** sensitivity 0.933, specificity 0.583, balanced accuracy 0.758 (ROC-AUC 0.800).
* **Freeze mechanics:** written to `config/analysis.yaml` as `inference.threshold.value: 0.07`; `predict.py` and `app.py` resolve the threshold through `brainvuln.config.resolve_threshold()` (source precedence: explicit override → frozen metrics file → config → labeled 0.5 fallback). Regression tests in `tests/test_threshold_frozen.py` pin config == metrics file == resolved serving value.
* Full record: `audit_artifacts/threshold_selection.json` / `.md`.

## Test-set integrity (phase 4)

No test label, prediction, metric, ROC curve, or confusion matrix influenced any decision in this phase:

* the threshold was computed and written to `audit_artifacts/threshold_selection.*` **before** the test loader was constructed (`test_contact: false` in the record; ordering enforced by the script);
* checkpoint, preprocessing, architecture, augmentation, and hyperparameters were all fixed by the prior audit (`canonical_checkpoint.json`, `metrics/train_history.json`) and were not modified;
* every artifact in `results/ml/evaluation/` was created *after* `threshold_selection.json` (timestamps recorded in the files).

**Verdict: no test leakage detected in this evaluation.** (Residual structural caveat: the test set is small and the same split has been used for every previous diagnostic, so it has been *looked at* — though never optimized against — during auditing.)

## Test performance (subject level, primary)

Unit of evaluation = **subject**. The test partition contains exactly one session per subject (asserted at runtime), so subject-level = session-level here; the rule "canonical scan per subject" is recorded in `test_metrics_subject_level.json`.

| Metric | Value | 95% CI (subject bootstrap, 2000 draws) |
| --- | ---: | ---: |
| **ROC-AUC (primary)** | **0.817** | [0.618, 0.972] |
| PR-AUC | 0.814 | [0.604, 0.984] |
| Accuracy | 0.667 | — |
| Balanced accuracy | 0.642 | [0.466, 0.810] |
| Sensitivity (AD) | 0.867 | [0.667, 1.000] |
| Specificity (CN) | 0.417 | [0.125, 0.700] |
| Precision | 0.650 | — |
| Recall = Sensitivity | 0.867 | [0.667, 1.000] |
| F1 | 0.743 | — |
| Brier score | 0.192 | [0.080, 0.314] |

Confusion matrix at the frozen threshold (rows = truth, columns = prediction):

|  | pred CN | pred AD |
| --- | ---: | ---: |
| **true CN (12)** | 5 | 7 |
| **true AD (15)** | 2 | 13 |

Reading: discrimination is good (0.817 AUC) but the operating point is aggressive — the model flags most ADs correctly while mislabeling the majority of CN subjects as AD at 0.07. The wide CIs (n=27) are the dominant uncertainty; point estimates should not be quoted without them. Curves and tables generated from the actual test predictions: `results/ml/evaluation/roc_curve.png`, `pr_curve.png`, `confusion_matrix.png` (+ `*_data.csv`).

## Error analysis

Full table: `results/ml/evaluation/error_analysis.csv` / `.md`.

| Group | n | mean prob | mean age | F / M |
| --- | ---: | ---: | ---: | --- |
| TP | 13 | 0.872 | 79.0 | 5 / 8 |
| TN | 5 | 0.021 | 76.4 | 5 / 0 |
| FP | 7 | 0.577 | 80.6 | 5 / 2 |
| FN | 2 | 0.034 | 72.5 | 1 / 1 |

**False positives (7) — the dominant error mode.** Mean age 80.6 (the oldest group; two subjects ≥ 88). MMSE is 27–30 (normal cognition), consistent with their CN labels. Two FPs sit at the extreme of the probability distribution (0.835, 0.999 — the latter, `OAS1_0337`, is a hard miss with the model as certain as it gets), while five others (0.225–0.659) would flip to TN at a higher operating point. The FP pattern (oldest, female-predominant, normal cognition, high but not always extreme probabilities) is consistent with atrophy-like/normative-aging signal being read as disease — see the age association below. No subject was relabeled, removed, or "fixed".

**False negatives (2).** `OAS1_0082` (prob 0.027, MMSE 28) and `OAS1_0142` (prob 0.042, MMSE 27) — model confidently wrong, not borderline. CDR 0.5 "very mild" AD subjects; nothing in the available QC fields distinguishes them from correct TPs.

## Confounding checks

**Age association (association, not causation):**

* Spearman ρ(prob, age) = **+0.368** (p = 0.059); Pearson r = +0.396 (p = 0.041) — a moderate positive association: part of the model's probability mass tracks age.
* Age bands (mean predicted probability): 60–69 → 0.12 (n=2), 70–79 → 0.48 (n=11), 80+ → 0.72 (n=14).
* **Age+sex baseline** (logistic regression on Age and Sex, **fit on train subjects only**, validation-Youden threshold): test ROC-AUC **0.700**, sensitivity 0.867, specificity 0.250, balanced accuracy 0.558.
* **Interpretation:** the CNN's ROC-AUC (0.817) exceeds the demographic baseline (0.700) by ~0.12, so its signal is not purely demographic — but with overlapping CIs at n=27, a substantial age-driven component cannot be excluded, and the FP profile (oldest group, mean age 80.6) points the same way. Treating this as a detection result, not a causal one.

**Sex-stratified performance (descriptive only — tiny subgroups):**

| Sex | n (AD/CN) | ROC-AUC | Sensitivity | Specificity |
| --- | --- | ---: | ---: | ---: |
| F | 16 (6/10) | 0.900 | 0.833 | 0.500 |
| M | 11 (9/2) | 0.500 | 0.889 | 0.000 |

No strong claims: the male subgroup contains only 2 CN subjects, so its specificity/AUC numbers are nearly uninformative. The F-vs-M AUC gap (0.90 vs 0.50) is consistent with the class-level sex imbalance but is not, by itself, evidence of a sex-specific failure.

## Calibration

Discrimination ≠ calibration; both are reported.

* **Brier score 0.192** [0.080, 0.314].
* **Calibration slope 0.340** (intercept 0.058) — logistic fit of the label on logit(p). A slope far below 1 means **overconfidence**: probability differences between subjects are too extreme relative to observed outcome rates.
* **Reliability bins** (4 quantile bins, `calibration.json` / `calibration.png`): predicted 0.02 → observed 0.29; predicted 0.48 → observed 0.29; predicted 0.86 → observed 0.83; predicted 0.98 → observed 0.86. Low-probability predictions are badly miscalibrated (the model says "~2% AD" for subjects that turn out AD 29% of the time — exactly the FP-at-0.07 phenomenon viewed from the probability side); high-probability bins are reasonably aligned.
* Practical consequence: probabilities are usable for **ranking** subjects, not as absolute risk estimates.

## Explainability

* **Grad-CAM coverage: 27/27 test subjects** have per-subject CAMs (NIfTI + plane PNGs) and 87-parcel regional tables from the frozen checkpoint's canonical run (`results/ml/resnet_seed42/`); per-subject records with checkpoint SHA and input session id: `attribution_table.csv`; coverage record: `gradcam_coverage.json`.
* **Attribution table** (phases 13–14): subject_id, diagnosis, prediction, probability, top_region_1–3, mean_relevance, n_parcels_valid, cam_file_exists — region names derived from `data/derived/atlas_dk68_tianS1_info.csv`, nothing hard-coded.
* **Regional relevance (population M_CNN, top regions):** caudalanteriorcingulate_L (0.631), caudalanteriorcingulate_R (0.611), TianS1_16 subcortical (0.593), caudalmiddlefrontal_L (0.561), TianS1_15 (0.560). Mean relevance is systematically higher for AD-predicted subjects (~0.59 for TPs) than for confident CNs (~0.0).
* **Explanation stability / method agreement:** Grad-CAM vs occlusion-sensitivity regional maps correlate at **Spearman ρ = 0.294** (87 parcels) — weak-to-moderate agreement. The two methods agree on broad territory but not on fine regional ranking; region-level claims should stay coarse. Grad-CAM highlights *what the model looks at*, not causally relevant pathology.

## Limitations

* **Dataset size:** 182 subjects total, 27 test subjects; every confidence interval in this report spans tens of AUC points. The study is powered to detect gross effects only.
* **Cohort selection:** OASIS-1 volunteers, age ≥ 60, CDR-based labels; "probable AD" by CDR is not biomarker-confirmed AD; CDR 0.5 includes non-AD etiologies.
* **Scanner/site:** single site, single scanner generation; nothing here supports cross-site generalization.
* **Demographic confounding:** moderate age–probability association (+0.37) plus a class-level sex imbalance (AD 59% F vs CN 70% F) that the matching did not fully remove; sex-stratified cells are too small to adjust.
* **Operating point:** sensitivity 0.867 / specificity 0.417 at 0.07; specificity is poor at this threshold and the threshold itself is weakly identified (bootstrap interval [0.015, 0.76]).
* **Calibration:** overconfident (slope 0.34); probabilities must not be read as risks.
* **External validation:** **not started** — OASIS-3 and ADNI remain untouched by design (STOP condition); the scripts (`scripts/stage_oasis3.py`, `scripts/stage_adni.py`, `scripts/validate_external.py`) are prepared for a future phase. Until then, all results are internal to OASIS-1.
* **Explainability:** CAM–occlusion agreement is weak (ρ 0.29); regional relevance is exploratory.
* **Research-only status:** this is a research prototype, not a clinical diagnostic system. No output of this repository may inform any clinical decision.
* **Multi-seed status:** single frozen seed-42 run only; multi-seed robustness is **prepared but not executed** (phase 15 below).

## Phase 15 — multi-seed robustness (prepared, not run)

Per the STOP condition, robustness experiments were only *prepared* after the primary result was recorded. `scripts/multiseed.bat` will drive `scripts/seed_robustness.py --seeds 1 2 3 4 --epochs 30` (each seed: own training run, own validation threshold, own checkpoint; seeds 1–4 are added to the existing seed-42 baseline; **never** best-seed-by-test selection). This report distinguishes: **single frozen baseline** = everything above; **multi-seed robustness** = pending until that driver is run.

## Phase 16 — checkpoint immutability

SHA-256 of `results/ml/resnet_seed42/checkpoints/best.pt` recomputed after all evaluation work: `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` — identical to `audit_artifacts/canonical_checkpoint.json` and to the value stamped into every evaluation artifact. **The production checkpoint was not modified.**

## Phase 18 — success criteria

- [x] git frozen state created (`a7521d3`, `audit_artifacts/frozen_state.json`)
- [x] threshold selected using validation only (Youden's J, 0.07)
- [x] threshold frozen (`config/analysis.yaml` `inference.threshold.value` + `resolve_threshold()`)
- [x] test set remained untouched during threshold selection (ordering + timestamps + `test_contact: false`)
- [x] subject-level test evaluation completed (unit asserted: 1 session/subject)
- [x] ROC-AUC calculated (0.817, CI [0.618, 0.972])
- [x] PR-AUC calculated (0.814, CI [0.604, 0.984])
- [x] sensitivity calculated (0.867, CI [0.667, 1.000])
- [x] specificity calculated (0.417, CI [0.125, 0.700])
- [x] confusion matrix generated (5/7/2/13 + PNG + CSV)
- [x] calibration evaluated (Brier 0.192; slope 0.340; reliability diagram)
- [x] subject-level prediction table saved (27 rows, SHA-stamped, threshold-stamped)
- [x] false-positive/false-negative analysis completed (7 FP / 2 FN profiled)
- [x] age confounding checked (ρ +0.368; age+sex baseline AUC 0.700)
- [x] Grad-CAM generated from frozen checkpoint (27/27, regional tables, attribution table)
- [x] all test predictions stamped with checkpoint identity (SHA + preprocess version per row)
- [x] no production checkpoint was modified (SHA re-verified post-evaluation)

## REAL MODEL EVALUATION STATUS

**REAL MODEL EVALUATION STATUS: PASS**

| Item | Value |
| --- | --- |
| Test subjects | 27 (15 AD / 12 CN), one session each |
| Threshold (validation Youden) | 0.07 (bootstrap median 0.07, 95% [0.015, 0.76]) |
| ROC-AUC | 0.817 [0.618, 0.972] |
| PR-AUC | 0.814 [0.604, 0.984] |
| Sensitivity | 0.867 [0.667, 1.000] |
| Specificity | 0.417 [0.125, 0.700] |
| Calibration | Brier 0.192; slope 0.340 → overconfident; low-probability bins badly miscalibrated |
| False positives | 7 (mean age 80.6, oldest group; MMSE 27–30) |
| False negatives | 2 (prob 0.027 / 0.042 — confident misses) |
| Checkpoint SHA256 | `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9` (unchanged) |
| Test leakage detected | **No** (threshold froze before test contact; documented above) |
| Ready for multi-seed robustness | **Yes** — `scripts/multiseed.bat` prepared; not launched per STOP condition |

**Summary of what the measurements say.** On the untouched test set the frozen model discriminates AD from CN at ROC-AUC 0.817, above its age+sex demographic baseline (0.700). At the validation-chosen operating point it is high-sensitivity / low-specificity (7 of 12 CN subjects flagged), its probabilities are overconfident (calibration slope 0.34), and part of its score tracks age (+0.37). Nothing here was tuned on test, nothing clinical is claimed, and the single-seed provenance of every number is explicit. The honest next step is multi-seed robustness (prepared), then — only after that — external validation.
