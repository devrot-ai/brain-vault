# BrainVuln

**Research question.**

> Are the neuroanatomical features learned by a deep-learning MRI classifier
> of Alzheimer's disease spatially aligned with the regional expression of
> genetically implicated Alzheimer's-associated genes and independently
> observed disease vulnerability?

> **Research use only. Not a clinical diagnostic system.** This is a research
> prototype trained on a small research cohort. Nothing here supports
> clinical decisions or causal claims.

## Problem

Alzheimer's disease damages brain regions unevenly — "selective vulnerability."
If a classifier learns disease-relevant anatomy, and that anatomy overlaps
with where AD-risk genes are expressed and where independent disease maps
show damage, that convergence is a testable, quantitative statement about
vulnerability. BrainVuln makes that statement falsifiable: it reports the
alignment **and** the nulls that could refute it.

## What BrainVuln does

One pipeline, three maps, one bridge:

```
 OASIS-1 T1 MRI ──► 3D ResNet-18 ──► Grad-CAM ──► M_CNN (regional relevance)
                                                      │
 GWAS Catalog (EFO, explicit) ──► AD genes ──► AHBA ──► M_GENE (expression)
                                                      │        │
        independent disease maps (La Joie 2020) ─────► M_DISEASE
                                                      │
                     matched-gene nulls · Moran/Burt/spin nulls
                                                      ▼
              rho_CNN_GENE, rho_CNN_DIS, rho_GENE_DIS  (spatial inference)
```

* **MRI classifier** — 3D ResNet-18 on OASIS-1 T88 volumes (128³, 2 mm,
  brain z-score), trained with early stopping on validation ROC-AUC only.
* **Grad-CAM** — per-subject 3D relevance mapped onto a fixed 87-parcel atlas.
* **GWAS + AHBA** — AD gene set from the GWAS Catalog (EFO_0000249, explicit
  retrieval metadata) intersected with Allen Human Brain Atlas expression
  (abagen, differential-stability probes, 6 donors, 87 parcels).
* **Spatial statistics** — the maps are compared and tested against matched
  gene nulls and three spatial-null families (Moran spectral, Burt-2020,
  surface spins) with concordance required.

## Dataset

OASIS-1 cross-sectional MRI, subjects ≥ 60: **100 AD (CDR 0.5–2) / 82 CN
(CDR 0)**, deterministically matched 1:1 by age and sex, split 70/15/15 by
subject (seed 42). Held-out test set: **27 subjects (15 AD / 12 CN)**.

## Model

ResNet18Binary (MONAI ResNet-18 backbone, single-logit head, ~33.4 M
parameters), AdamW + warmup/cosine, light augmentation, CPU-trainable.
Canonical checkpoint: `results/ml/resnet_seed42/checkpoints/best.pt`,
SHA256 `5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9`
(pinned by `tests/test_checkpoint_integrity.py`; never searched by mtime).

## Evaluation (measured, held-out test, unit = subject)

| Metric | Value | 95% CI |
| --- | ---: | ---: |
| ROC-AUC | 0.817 | [0.618, 0.972] |
| PR-AUC | 0.814 | [0.604, 0.984] |
| Sensitivity | 0.867 | [0.667, 1.000] |
| Specificity | 0.417 | [0.125, 0.700] |
| F1 | 0.743 | — |
| Brier | 0.192 | [0.080, 0.314] |

Threshold **0.07**, Youden-selected on validation subjects only, frozen
before any test use (enforced by `tests/test_threshold_frozen.py`).
Confusion: TP 13, TN 5, **FP 7** (oldest subjects, mean age 80.6),
**FN 2**. The model is high-sensitivity / low-specificity at this operating
point, and its probabilities are overconfident (calibration slope 0.34) —
documented, not hidden. Age–probability association: Spearman +0.37
(vs +0.70 baseline-free demographic comparator AUC 0.700), so signal exceeds
the age+sex baseline but a demographic component cannot be excluded at n=27.
Full detail: `results/FINAL_RESULTS.md`, `audit_artifacts/subject_level_evaluation.md`.

## Robustness (multi-seed)

A 5-seed campaign (42/1/2/3/4, identical recipe, only the seed differs) plus
a same-seed repeat measures how much of the above is seed luck:
per-seed test metrics with subject-bootstrap CIs, prediction/classification/
error stability, calibration and age-association robustness, and pairwise
Grad-CAM regional-map similarity. Campaign state:
`audit_artifacts/multiseed_status.md`; results land in
`results/ml/multiseed/` and `audit_artifacts/multiseed_robustness.md`
when complete. Seed 42 is never replaced by another seed, and no seed is
ever selected by test AUC.

## Explainability

Grad-CAM (target layer `layer4`) for **27/27 test subjects**, saved into
T88 space with the corrected preprocessed-grid affine, parcellated onto the
atlas. Population map M_CNN's top regions (seed 42) and per-subject top-3
are in `results/ml/evaluation/attribution_table.csv`. Independent-method
check: Grad-CAM vs occlusion-sensitivity regional agreement is
**Spearman 0.294** — weak-to-moderate, so regional claims stay coarse.
Explanations are *model-derived relevance*, not biological truth.

## Genetics

AD gene sets are fetched from the GWAS Catalog REST v2 with explicit EFO
ids and recorded retrieval metadata (live counts, overlap with stored sets).
Expression comes from the Allen Human Brain Atlas via abagen; parcels
without samples stay missing (never imputed).

**Measured headline (full-scale bridge, 5000 nulls):** rho_CNN_GENE = 0.383;
spatial nulls pass (Moran p = 0.029, Burt-2020 p = 0.0014) but the
**matched-gene null does not** (p = 1.0 — expression-matched random sets
correlate higher): the association is generic spatial structure of
brain-expression maps, **not** AD-genetic specificity. The pre-registered
design correctly refuses the over-claim. See `results/bridge/`.

## Spatial statistics

Three null families (Moran spectral on parcel centroids, Burt-2020
volumetric, DK68 surface spins) with BH-FDR and two-method concordance;
1:1 expression-matched gene nulls for genetic specificity; donor
leave-one-out for AHBA robustness; independent disease maps
(La Joie 2020 tau/atrophy) with full provenance.

## Limitations

* n = 27 test subjects; single site, era-specific scanners; wide CIs.
* CDR-based "probable AD" is not biomarker-confirmed AD.
* Class-level sex imbalance (AD 59% F vs CN 70% F); per-sex cells too small.
* Overconfident probabilities (slope 0.34) — measured, not corrected.
* Weak Grad-CAM↔occlusion agreement — regional relevance is exploratory.
* Matched-gene null not passed — the genetic association is not AD-specific.
* External validation (OASIS-3/ADNI): **not yet performed** (stagers ready).
* Six AHBA donors bound all transcriptomic conclusions.

## Demo

```bash
python predict.py --image patient_T88.nii.gz   # prints Prediction / Probability /
                                               # Threshold / Model / Checkpoint sha256
python app.py                                  # Gradio demo (research-use banner)
```

Supported input: a **3D NIfTI T1 volume** (`.nii` / `.nii.gz`) or an
OASIS-1 ANALYZE `.img/.hdr` pair, ideally the `*_t88_masked_gfc` derivative
(skull-stripped, T88-registered) or a conversion of it. The same canonical preprocessing runs on everything:
reorient to RAS+, 2 mm trilinear resample, tight brain crop, 128³ centre
pad/crop, z-normalise inside nonzero voxels. `predict.py` validates the
format and volume before inference and exits with a clear message (code 2)
on unsupported input, empty/non-finite volumes, or a missing file — DICOM
series must be converted first (e.g. `dcm2niix`). Note the model was trained on skull-stripped volumes; a raw
(unstripped) head scan is out of distribution because nonzero voxels are
used as the brain mask. Outputs: probability, classification, Grad-CAM
planes + NIfTI, atlas-derived top regions, and `outputs/prediction.json`
(checkpoint SHA256, preprocessing version, frozen threshold, input QC
stamped in).

### Minimum-accuracy gate (≥ 0.85)

`scripts/eval_validation.py` is the acceptance gate for an 85% accuracy
floor. It evaluates the canonical checkpoint on the **validation** split
only, sweeps every distinct validation probability as a candidate cut, and
exits 0 (PASS) or 1 (FAIL) depending on whether any validation-selected
threshold reaches the target; 2 means it could not evaluate. It writes
`results/ml/evaluation/accuracy_gate.json` plus validation predictions. It
never selects a threshold on test and never changes the frozen serving
threshold — the gate measures the model, it does not tune it.

```bash
python scripts/eval_validation.py                  # default gate: >= 0.85
python scripts/eval_validation.py --min-accuracy 0.9
```

Past experience here says the current checkpoint does **not** clear 0.85:
the canonical model's frozen-threshold test accuracy is 0.667, and even the
best possible threshold on the frozen test predictions only reaches 22/27
(0.815) — so 85% requires a genuinely better model (retraining/ensembling
with validation-only selection), not a threshold choice.


## Reproducibility

* Configuration lives in `configs/oasis1_resnet3d.yaml` +
  `config/analysis.yaml`; every run manifest records the config hash,
  package versions, seeds, and checkpoint identity.
* Model selection reads validation only; the training code has no test
  loader by construction. Thresholds are frozen before test.
* Bootstrap CIs resample subjects, never scans.
* The canonical checkpoint's SHA256 is pinned in
  `audit_artifacts/canonical_checkpoint.json` and re-verified by tests;
  serving resolves it from config — no filesystem search, no mtime picking.
* Re-run: `python scripts/train_oasis1.py --model resnet --seed 42`
  (recipe: `--epochs 40 --patience 8 --warmup 2 --cos-epochs 12
  --aug-strength light`); multi-seed: `scripts/multiseed.bat`.

Tests: `python -m pytest -q` (112 passing at last run, including checkpoint
integrity, threshold freeze, and inference-path regressions).
