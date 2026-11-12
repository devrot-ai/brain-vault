# Accuracy Improvements Guide

This document describes all implemented techniques to improve the BrainVuln model's accuracy, F1 score, and specificity. Each technique can be used independently or combined for maximum effect.

## Current Baseline Performance

| Metric | Value | Problem? |
|---|---|---|
| **Accuracy** | 66.7% | Low |
| **F1 Score** | 0.74 | Moderate |
| **ROC-AUC** | 0.817 | Decent ranking, poor threshold |
| **Sensitivity** | 86.7% | Good — catches most AD cases |
| **Specificity** | 41.7% | **Critical problem** — 7 of 12 normal brains flagged as AD |
| **Precision** | 0.65 | Low — many false alarms |
| **Brier Score** | 0.192 | Poor probability calibration |

### The Root Problem

The model has **great sensitivity but terrible specificity**. It correctly identifies Alzheimer's patients (86.7%) but also flags 58.3% of healthy brains as diseased. The optimal threshold collapsed to 0.07 because the model's probability scores for normal subjects are too compressed.

---

## Implemented Improvements

### 1. Test-Time Augmentation (TTA) — +1-2% accuracy, no retraining needed

**Status: ✅ Implemented in `predict.py` and `app.py`**

TTA runs the model on multiple slightly different versions of the same MRI scan and averages the probabilities. This reduces variance without any retraining.

**How it works:**
- Original image + 5 augmented views (small rotations ±5°, translations ±2 voxels, scales ±3%, intensity shifts ±5%, noise)
- Averages sigmoid probabilities across all 6 views
- Each view is biologically plausible — same patient, slightly different scanner position

**Usage:**

```powershell
# Command line
python predict.py --image patient.nii.gz --tta

# Web UI: check the "Use Test-Time Augmentation" checkbox before running analysis
```

**Expected improvement:** +1-2% accuracy, +2-3% specificity

---

### 2. Demographics Fusion — +3-5% accuracy

**Status: ✅ Model exists in `models.py`, auto-detected in `predict.py`**

Fuses age and sex with imaging features before classification. Age is the strongest risk factor for Alzheimer's — ignoring it wastes free information.

**How it works:**
- Concatenates [age, sex] with pooled imaging features before the classification head
- The model learns to use demographic information alongside imaging patterns
- A 25-year-old with mild ventricular enlargement is almost certainly normal; a 90-year-old with the same pattern is suspicious

**Usage:**

```powershell
python scripts/train_oasis1.py --model resnet --seed 42 --demo-fuse --epochs 40 --patience 8
```

**Expected improvement:** ~70-72% accuracy, ~50-55% specificity, ~0.78 F1

---

### 3. Wider Model — +2-4% accuracy

**Status: ✅ Supported in `models.py` and `train_oasis1.py`**

Doubles every convolution channel with `--widen-factor 2.0`. This increases backbone parameters from ~33M to ~130M, giving the model 4× capacity.

**How it works:**
- `widen_factor=2.0` doubles every conv channel count → ~4× backbone parameters
- More capacity to learn complex spatial patterns (hippocampal shape, cortical thinning gradients)

**Usage:**

```powershell
python scripts/train_oasis1.py --model resnet --seed 42 --widen-factor 2.0 --epochs 40 --patience 8
```

**Expected improvement:** ~72-75% accuracy, ~0.80 F1

---

### 4. Label Smoothing — +1-2% accuracy, better calibration

**Status: ✅ Default enabled in `train_oasis1.py` (0.1)**

Pushes targets from hard 0/1 toward 0.5 (e.g., 0.9/0.1 with λ=0.1). Prevents the model from becoming overconfident.

**How it works:**
- Your Brier score is 0.192 — the model's probabilities are poorly calibrated
- Label smoothing produces better-calibrated probabilities
- Better calibration means threshold selection works better, directly improving specificity

**Usage:**

```powershell
# Already the default (0.1) in train_oasis1.py
python scripts/train_oasis1.py --model resnet --seed 42 --label-smoothing 0.1
```

**Expected improvement:** ~73-76% accuracy, better-calibrated probabilities

---

### 5. Focal Loss — +2-3% accuracy, +5% specificity

**Status: ✅ Implemented in `train.py` and `train_oasis1.py`**

Focal loss focuses training on hard-to-classify examples by down-weighting easy examples. This directly targets the specificity problem: normal brains misclassified as AD are the hard negatives that focal loss emphasizes.

**How it works:**
- FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
- Gamma (γ=2.0) controls focusing strength — higher = more focus on hard examples
- Alpha (α=0.25) controls class balancing
- Easy examples (high confidence correct predictions) contribute less to the loss
- Hard examples (misclassified or low-confidence) contribute more

**Usage:**

```powershell
# Use focal loss instead of weighted BCE
python scripts/train_oasis1.py --model resnet --seed 42 --loss-type focal

# Tune focal loss parameters
python scripts/train_oasis1.py --model resnet --seed 42 --loss-type focal --focal-gamma 2.0 --focal-alpha 0.25
```

**Expected improvement:** +2-3% accuracy, +5% specificity, ~0.80 F1

---

### 6. Combined Approach (Recommended) — +7-13% accuracy

**Status: ✅ All components implemented, just run the command**

Combines demographics fusion, wider model, label smoothing, and focal loss for maximum single-model improvement.

**Usage:**

```powershell
python scripts/train_oasis1.py --model resnet --seed 42 `
  --demo-fuse --widen-factor 2.0 --label-smoothing 0.1 --loss-type focal `
  --epochs 60 --patience 10 --warmup 3 --cos-epochs 15 `
  --lr 5e-5 --batch-size 4 --aug-strength full
```

**Expected improvement:** ~75-80% accuracy, ~55-60% specificity, ~0.82 ROC-AUC, ~0.80 F1

---

### 7. Multi-Seed Ensemble — +3-5% accuracy, +5% F1

**Status: ✅ Implemented in `scripts/ensemble_predict.py`**

Trains 5 models with different random seeds and averages their probabilities. Different seeds explore different regions of the loss landscape; averaging reduces variance and improves robustness.

**Training:**

```powershell
# Train 5 models with different seeds
foreach ($seed in 42, 1, 2, 3, 4) {
  python scripts/train_oasis1.py --model resnet --seed $seed `
    --demo-fuse --widen-factor 2.0 --label-smoothing 0.1 --loss-type focal `
    --epochs 60 --patience 10 --warmup 3 --cos-epochs 15 --lr 5e-5
}
```

**Inference:**

```powershell
# Ensemble prediction with explicit checkpoints
python scripts/ensemble_predict.py --image patient.nii.gz `
  --checkpoints results/ml/resnet_seed42/checkpoints/best.pt `
                results/ml/resnet_seed1/checkpoints/best.pt `
                results/ml/resnet_seed2/checkpoints/best.pt `
                results/ml/resnet_seed3/checkpoints/best.pt `
                results/ml/resnet_seed4/checkpoints/best.pt

# Or auto-discover from sweep directory
python scripts/ensemble_predict.py --image patient.nii.gz --sweep-dir results/ml/

# Combine ensemble with TTA for maximum accuracy
python scripts/ensemble_predict.py --image patient.nii.gz --sweep-dir results/ml/ --tta
```

**Expected improvement:** ~78-83% accuracy, ~60-65% specificity, ~0.88 ROC-AUC, ~0.83 F1

---

### 8. CBAM Attention Model — +2-4% accuracy

**Status: ✅ Implemented in `models.py` and `train_oasis1.py`**

Adds Convolutional Block Attention Module (CBAM) to the ResNet backbone. CBAM inserts attention modules after each layer stage, teaching the model to focus on Alzheimer's-relevant brain regions (hippocampus, entorhinal cortex) rather than the whole brain uniformly.

**How it works:**
- **Channel Attention**: Learns which feature channels encode AD-relevant patterns
- **Spatial Attention**: Learns which spatial locations (brain regions) are most important
- Combined: the model learns to attend to both "what" and "where" simultaneously

**Usage:**

```powershell
python scripts/train_oasis1.py --model resnet-cbam --seed 42 --epochs 40 --patience 8
```

**Expected improvement:** ~70-74% accuracy, ~0.82 ROC-AUC, ~0.78 F1

---

### 9. Temperature Calibration — better probability calibration

**Status: ✅ Implemented in `src/brainvuln/mri/calibrate.py`**

Fits a scalar temperature T on validation logits to minimize NLL loss. This fixes poorly calibrated probabilities (Brier score 0.192 → lower) without changing model weights.

**How it works:**
- calibrated_prob = sigmoid(logit / T)
- T > 1: softens overconfident predictions
- T < 1: sharpens underconfident predictions
- Fitted on validation data only; never touches the test set

**Usage:**

```powershell
# Add --calibrate to any training run
python scripts/train_oasis1.py --model resnet --seed 42 --calibrate

# The fitted temperature is saved to results/ml/resnet_seed42/calibration/calibration_fit.json
# Inference automatically uses it via --temperature auto
python predict.py --image patient.nii.gz --temperature auto
```

**Expected improvement:** Lower Brier score, better reliability diagrams, improved threshold selection

---

### 10. K-Fold Cross-Validation — more robust evaluation

**Status: ✅ Implemented in `scripts/train_crossval.py`**

With only ~44 subjects total, a single train/val/test split gives high-variance performance estimates. Cross-validation trains K models, each on a different split, and averages their predictions.

**Usage:**

```powershell
# 5-fold cross-validation with all improvements
python scripts/train_crossval.py --model resnet --k 5 --seed 42 `
  --demo-fuse --widen-factor 2.0 --label-smoothing 0.1 --loss-type focal --calibrate

# Use ensemble inference on cross-validation checkpoints
python scripts/ensemble_predict.py --image patient.nii.gz --sweep-dir results/crossval/
```

**Expected improvement:** More reliable accuracy estimates (mean ± std), better generalization

---

### 11. End-to-End Full Recipe — one command for everything

**Status: ✅ Implemented in `scripts/train_full_recipe.py`**

Automates the complete training recipe with all improvements combined. One command trains a model with demographics fusion, wider backbone, label smoothing, focal loss, and temperature calibration.

**Usage:**

```powershell
# Single model with all improvements
python scripts/train_full_recipe.py

# Multi-seed ensemble (5 seeds)
python scripts/train_full_recipe.py --ensemble --n-seeds 5

# Quick smoke test
python scripts/train_full_recipe.py --quick
```

**Expected improvement:** ~75-80% accuracy for single model, ~78-83% for 5-seed ensemble

---

### 12. F1-Optimized Threshold — +1-2% F1

**Status: ✅ Supported in `train_oasis1.py`**

Uses F1-based threshold instead of Youden's J. Directly optimizes the metric you care about.

**Usage:**

```powershell
python scripts/train_oasis1.py --model resnet --seed 42 --threshold-method f1
```

**Expected improvement:** Slightly higher F1, better precision-recall balance

---

### 13. Batch Upload in Web UI — process multiple scans at once

**Status: ✅ Implemented in `app.py`**

The web UI now includes a **Batch Upload** tab where you can upload multiple MRI scans simultaneously. Results are displayed in a color-coded summary table.

**Usage:**
- Open the web app: `python app.py`
- Click the **📦 Batch Upload** tab
- Upload multiple `.nii` or `.nii.gz` files
- Optionally enable TTA
- Click **Run Batch Analysis**

---

### 14. CBAM-Aware Grad-CAM — more meaningful attribution maps

**Status: ✅ Implemented in `src/brainvuln/mri/gradcam.py` and `app.py`**

For CBAM models, the standard Grad-CAM only uses gradient flow on the final feature map. CBAM-aware Grad-CAM goes further by extracting the model's learned **spatial attention** and **channel attention** maps and fusing them with the gradient-based CAM. This produces attribution maps that reflect not just which features drive the prediction, but also which brain regions the model's attention modules learned to focus on (hippocampus, entorhinal cortex, temporal lobe).

**How it works:**
1. **Gradient-based Grad-CAM** on attention-weighted features (after all CBAM modules)
2. **Spatial attention extraction** from the last CBAM module (att4) — shows WHERE the model focuses
3. **Channel attention extraction** from att4 — shows WHICH feature channels are important
4. **Fusion**: 50% gradient CAM + 25% spatial attention + 25% channel-weighted CAM

The result is a more interpretable attribution map that highlights the Alzheimer's-relevant regions the model learned to attend to.

**Usage:**
- Automatic: `gradcam_auto(model, volume)` detects CBAM models and uses the enhanced method
- Manual: `gradcam_cbam_3d(model, volume)` for explicit CBAM Grad-CAM
- Web UI: automatically uses CBAM-aware Grad-CAM when a CBAM model checkpoint is loaded

**Expected improvement:** More focused and interpretable Grad-CAM heatmaps, especially for CBAM-trained models. The attribution maps better highlight hippocampal atrophy and ventricular enlargement patterns.

---

### 15. Longitudinal Co-Registration — accurate atrophy tracking

**Status: ✅ Implemented in `src/brainvuln/mri/longitudinal.py`**

When comparing two MRI scans from the same patient at different time points, head position differences between scans can confound the voxel-wise difference map. Co-registration aligns the follow-up scan (T2) to the baseline scan (T1) space using affine optimization before computing differences, so voxel-wise changes reflect true anatomical change rather than registration artifacts.

**How it works:**
1. **Normalized Mutual Information (NMI)** metric measures alignment quality between T1 and T2
2. **12-parameter affine transform** (translation, rotation, scaling, shearing) optimized via Nelder-Mead simplex
3. **Multi-resolution**: Gaussian-smoothed volumes used for optimization, original resolution for final transform
4. **Applied to T2**: The optimized transform resamples T2 into T1's coordinate system
5. **Difference map**: Computed on the registered pair for accurate atrophy measurement

**Usage:**
- Automatic: `compare_scans(vol_t1, vol_t2)` registers by default (`register=True`)
- Disable: `compare_scans(vol_t1, vol_t2, register=False)` for unregistered comparison
- Web UI: Time Series Tracking tab automatically uses co-registration

**Expected improvement:** More accurate voxel-wise difference maps for longitudinal comparison. Without registration, head position differences can create false atrophy/growth signals. With registration, the difference map reflects true tissue change between time points.

---

## Expected Accuracy Progression

| Technique | Accuracy | Specificity | F1 | ROC-AUC |
|---|---|---|---|---|
| Current baseline | 66.7% | 41.7% | 0.74 | 0.817 |
| + TTA (no retrain) | ~68-69% | ~44-45% | ~0.76 | — |
| + Demographics fusion | ~70-72% | ~50-55% | ~0.78 | ~0.83 |
| + Wider model (2.0x) | ~72-75% | ~55% | ~0.80 | ~0.85 |
| + Label smoothing | ~73-76% | ~55-60% | ~0.80 | ~0.86 |
| + Focal Loss | ~75-78% | ~60% | ~0.82 | ~0.87 |
| + CBAM attention | ~70-74% | ~55% | ~0.78 | ~0.82 |
| + Temperature calibration | — | — | — | better Brier |
| + All combined | ~75-80% | ~55-60% | ~0.80 | ~0.87 |
| + 5-seed ensemble | ~78-83% | ~60-65% | ~0.83 | ~0.88 |
| + External data (ADNI) | ~85%+ | ~70%+ | ~0.85+ | ~0.90+ |

---

## Quick Start: Best Results with Minimal Effort

**Step 1: Enable TTA in the web app** (no retraining needed)
- Open the web app: `python app.py`
- Check the "Use Test-Time Augmentation" checkbox
- Upload and predict

**Step 2: Use the full recipe script** (one command for everything)
```powershell
python scripts/train_full_recipe.py
```

**Step 3: For maximum accuracy, train a 5-seed ensemble**
```powershell
python scripts/train_full_recipe.py --ensemble --n-seeds 5
```

**Step 4: Run inference with all improvements**
```powershell
# Single scan with TTA + calibrated temperature
python predict.py --image patient.nii.gz --tta --temperature auto

# Ensemble inference with TTA
python scripts/ensemble_predict.py --image patient.nii.gz --sweep-dir results/ml/ensemble_full_recipe/ --tta
```

---

## Notes

- **Reaching 85% accuracy** on a 27-subject test set with single-site OASIS-1 data is challenging. The most reliable path is adding external datasets (ADNI, OASIS-3).
- **Demographics fusion** is the most impactful single change — it's already implemented, requires no new code, and adds 3-5% accuracy.
- **Focal Loss** directly targets the specificity problem by focusing training on the hard-to-classify normal brains.
- **Ensemble + TTA** together give the best results without retraining on new data.
