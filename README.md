# BrainVuln

**Research question.**

> Are the neuroanatomical features learned by a deep-learning MRI classifier
> of Alzheimer's disease spatially aligned with the regional expression of
> genetically implicated Alzheimer's-associated genes and independently
> observed disease vulnerability?

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

The project combines 3D MRI deep learning, explainable AI, GWAS genetics,
Allen Human Brain Atlas transcriptomics, spatial statistics, and independent
disease-vulnerability validation.

> **Research use only.** This is a research prototype. Nothing here is a
> clinical diagnostic system, and nothing in it supports causal claims.

---

## What is implemented

| Part | Status | Where |
|---|---|---|
| A/B/C OASIS-1 cohort, age/sex matching, subject splits | **verified on real data** (100 AD / 82 CN, age SMD 0.04) | `scripts/build_oasis1_cohort.py` → `data/splits/` |
| D 3D preprocessing (T88 derivatives → 2 mm 128³, brain-z) | **verified** (full-brain coverage, deterministic) | `src/brainvuln/mri/preprocess.py` |
| E augmentation (no LR flip — documented rationale) | implemented, unit-tested | `src/brainvuln/mri/augment.py` |
| F/G models: ResNet-18 vs SimpleCNN vs age+sex baseline | implemented, unit-tested | `src/brainvuln/mri/models.py` |
| H training (AdamW, warmup+cosine, early stop on **val** AUC) | **smoke-verified on real data** | `src/brainvuln/mri/train.py`, `scripts/train_oasis1.py` |
| I/J frozen test eval + subject-bootstrap CIs + Youden threshold on val | **smoke-verified** | `src/brainvuln/mri/evaluate.py` |
| K/L Grad-CAM + occlusion → regional M_CNN (full 87-parcel atlas) | **verified on real scan** (non-degenerate CAMs) | `src/brainvuln/mri/gradcam.py`, `regional.py` |
| M GWAS Catalog (EFO, child-trait + mapping policy recorded) | **verified** | `src/brainvuln/gwas.py`, `scripts/fetch_gene_sets.py` |
| N AHBA via abagen (differential stability, donor-aware) | **verified** | `src/brainvuln/ahba.py`, `scripts/build_expression.py` |
| O M_GENE (mean-z primary, rank sensitivity) | **verified** | `src/brainvuln/scoring.py` |
| P/Q/R/T bridge: rho_CNN_GENE + matched nulls + spatial nulls + three-way | implemented against verified APIs; run after training | `scripts/bridge_cnn_gene.py` |
| S independent disease maps (La Joie 2020, imported + provenance) | **verified** | `data/external/disease_maps/` |
| U donor leave-one-out | **verified** (existing pipeline) | `src/brainvuln/` |
| V multi-seed (≥5 seeds) + attribution agreement | implemented (`--aggregate-only` tested) | `scripts/seed_robustness.py` |
| W Grad-CAM vs occlusion agreement | implemented | `regional.py:method_agreement` |
| X external validation (OASIS-3 / ADNI stagers, no tuning) | implemented + documented access path | `scripts/stage_oasis3.py`, `stage_adni.py` |
| AA inference CLI | **verified on real scan** | `predict.py` |
| AB web demo (research-use banner) | UI builds; needs checkpoint | `app.py` |
| AC configs | **complete** | `configs/*.yaml` |
| AE figures (10 main figures) | implemented | `scripts/make_figures.py` |

Tests: `python -m pytest tests -q` (offline suite; MRI-stack tests use
synthetic volumes and skip torch-dependent parts gracefully).

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[atlas,mri,dev]"
# 1. data (deterministic, resumable)
python scripts/download_oasis1.py            # 12 official OASIS-1 discs (~15.5 GB)
python scripts/build_oasis1_cohort.py        # matched cohort + splits -> data/splits/
# 2. genomics (real APIs; cached under results/gene_sets)
python scripts/verify_diseases.py            # check EFO ids resolve to intended traits
python scripts/fetch_gene_sets.py
python scripts/build_expression.py           # AHBA via abagen (first run: ~2-4 GB)
python scripts/import_disease_map.py --help  # independent maps + provenance
# 3. ML + bridge
python scripts/train_oasis1.py --model resnet --seed 42
python scripts/seed_robustness.py --seeds 42 1 2 3 4
python scripts/bridge_cnn_gene.py --n-matched 5000 --n-spatial 5000
# 4. products
python predict.py --image patient.nii.gz
python app.py                                # research-use-only demo
python scripts/make_figures.py
```

Everything is CPU-runnable (the primary config is calibrated for a 6-thread
CPU; raise `training.batch_size` on a GPU). Full-scale inference values
(≥5000 nulls) are set in `configs/spatial.yaml`; `--quick` flags run reduced
wiring checks only and are labeled as such in every output.

## Current production status

The primary model (`resnet`, seed 42) was retrained after a controlled
diagnosis of an epoch-1 validation peak in the first production run
(v1, archived at `results/ml/resnet_lr1e4_fullaug_v1`):

* **Diagnosis** (`scripts/compare_lr_sweep.py`): full-strength augmentation
  (±7°, prob 0.9) held the train loss at a ~0.6 plateau — the network could
  not fit even the training set, and val AUC peaked at epoch 1 by luck of
  initialization. Raising LR (3e-4, 1e-3) did not help; removing augmentation
  broke the plateau (loss 0.30) but overfit immediately (val → 0.52).
* **v2 recipe**: LR 1e-4, 2-epoch warmup, cosine annealed over 12 epochs
  (`--cos-epochs`) so LR actually decays inside the realistic early-stopping
  window, and light augmentation (`--aug-strength light`) so loss can descend
  without a memorization collapse. Epoch-level crash resume
  (`--resume-epochs`) makes long runs robust to interruptions.
* **Outcome**: best val ROC-AUC **0.800** at epoch 11 (loss 0.21, well below
  the old plateau); frozen test **ROC-AUC 0.817 [0.618, 0.972]**, PR-AUC
  0.814 (v1: 0.583 / chance). Grad-CAM↔occlusion regional agreement 0.997.
  The 27-session test CI is wide — the number is honest, not definitive.
* **Full-scale bridge on the v2 model** (5000 matched-gene + 5000 spatial
  nulls, `results/bridge/`): rho_CNN_GENE = **0.383**. Spatial nulls pass
  (Moran p = 0.029, Burt-2020 p = 0.0014, concordance robust) but the
  **matched-gene null does not** (p = 1.0; expression-matched random sets
  correlate *higher*, null mean ≈ 0.47): the association is generic spatial
  structure shared by all brain-expression maps, not AD-genetic specificity.
  Three-way: CNN↔La Joie atrophy −0.145, GENE↔atrophy −0.323 (both ns under
  Moran) — no endorsement at current model/atlas resolution. This is the
  two-null design correctly refusing an over-claimed result.

## Statistical honesty

* Model selection sees **validation only**; the test partition is read once,
  after freezing (the training code has no test loader by construction).
* Thresholds are chosen on validation (Youden or F1, documented), frozen, and
  only then applied to test.
* Bootstrap CIs resample **subjects**, never scans.
* H1 (genetic enrichment) uses 1:1 matched gene nulls (expression mean +
  variance); H2/H3 use three spatial-null families (Moran spectral,
  Burt-2020, surface spins) with concordance required.
* AHBA parcels without sampling stay missing — never imputed.
* Negative controls are pre-registered in `paper/analysis_plan.md`.

## Claims policy

This project never claims: clinical diagnosis, causal mechanism, causal GWAS
genes, that Grad-CAM proves mechanism, or that internal AUC guarantees
clinical generalization. It claims, at most, that model-derived regional
relevance **is spatially associated with** the expression of genetically
implicated Alzheimer's-associated genes and with independently observed
disease vulnerability, insofar as the pre-registered nulls and robustness
checks hold. Every result file carries the configuration and seed that
produced it.

## Known limitations

* OASIS-1 is a single-site, mostly cross-sectional cohort with era-specific
  scanners; OASIS-3/ADNI external validation is staged but requires their
  data-access agreements (see `scripts/stage_oasis3.py` header).
* The matched cohort is 1:1 by design (18 AD subjects had no same-sex CN
  within ±5 y and are recorded as unmatched in the cohort manifest).
* Occlusion sensitivity is coarse (32-voxel patches) and runs on a documented
  subset of test subjects; Grad-CAM runs on all.
* Six AHBA donors bound all transcriptomic conclusions; donor-LOO quantifies
  but cannot remove that dependence.
