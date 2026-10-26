# BrainVuln — System Architecture & Data Flow

> **Research use only.** Not a clinical diagnostic system. Outputs are research classifications from a small single-site cohort.

---

## 1. Dataset — OASIS-1

| Property | Value |
|----------|-------|
| **Name** | Open Access Series of Imaging Studies (OASIS-1) |
| **Cross-sectional MRI cohort** | Structural T1-weighted brain MRI |
| **Subjects** | 182 total (82 CN / 100 AD) |
| **Sessions** | Up to 2 per subject (longitudinal follow-ups excluded from primary split) |
| **Voxel grid** | 128 × 128 × 128, 2 mm isotropic |
| **Preprocessing** | T88 masked, brain-extracted, z-score intensity normalized |
| **Label rule (AD)** | CDR ∈ {0.5, 1, 2} AND Age ≥ 60 |
| **Label rule (CN)** | CDR == 0 AND Age ≥ 60 |
| **Excluded** | Sessions without CDR; subjects with conflicting CDR across sessions |
| **Split** | 70% train / 15% val / 15% test (subject-level, stratified, age+sex matched) |
| **Seed** | 42 |
| **Matching** | Greedy 1:1 nearest-age, same-sex (age tolerance ladder: 2 → 3 → 5 years) |

### Class balance

| Class | N | Mean Age | SD Age | % Female |
|-------|---|----------|--------|----------|
| CN (0) | 82 | 76.45 | 7.84 | 69.5% |
| AD (1) | 100 | 76.76 | 7.12 | 59.0% |

Standardized Mean Difference (SMD): age 0.04, sex 10.5%, eTIV 0.21, education 0.35 → well-matched groups.

---

## 2. High-Level Data Flow

```mermaid
flowchart TD
    subgraph Data["📦 DATA LAYER"]
        RAW[("Raw OASIS-1 DICOM/NIfTI<br/>data/raw/oasis1/")]
        EXT[("Extracted volumes<br/>data/raw/oasis1/extracted/")]
        SPLITS[("Split tables<br/>data/splits/<br/>train/val/test_subjects.csv<br/>matched_cohort.csv")]
    end

    subgraph Preprocess["🔧 PREPROCESSING PIPELINE"]
        IDX[build_scan_index<br/>ScanRecord per session]
        PRE[preprocess_and_cache<br/>t88_masked_gfc<br/>2mm → 128³ cube<br/>brain z-score]
        CACHE[("Cached volumes<br/>data/derived/oasis1_128/<br/>*.npy")]
    end

    subgraph Model["🧠 MODEL (brainvuln.mri)"]
        DS[OASISDataset<br/>returns Item(volume, label, age, sex)]
        AUG[training_transforms<br/>RandAffine, intensity, noise<br/>TRAIN ONLY]
        RESNET[ResNet18Binary<br/>MONAI 3D ResNet-18<br/>widen_factor=1.0<br/>33.4M params]
        FC[nn.Linear(512, 1)<br/>single logit → sigmoid]
    end

    subgraph Train["🏋️ TRAINING (scripts/train_oasis1.py)"]
        CFG[TrainConfig<br/>lr=1e-4, wd=1e-4,<br/>batch=4, warmup=2,<br/>cosine_epochs=12]
        LOSS[BCEWithLogitsLoss<br/>pos_weight = neg/pos<br/>from train split only]
        OPT[AdamW + gradient clip (5.0)]
        ES[Early stop on val ROC-AUC<br/>patience=6]
        CKPT[torch.save {state_dict, epoch,<br/>val_auc, config}<br/>results/ml/resnet_seed42/<br/>checkpoints/best.pt]
    end

    subgraph Eval["📊 EVALUATION (src/brainvuln.mri.evaluate)"]
        MET[classification_metrics<br/>ROC-AUC, PR-AUC, acc,<br/>bal_acc, sens, spec, F1, Brier]
        THR[select_threshold<br/>Youden's J on VAL only<br/>frozen = 0.07]
        BOOT[subject_bootstrap_ci<br/>2000 subject-level resamples<br/>95% CIs]
    end

    subgraph Outputs["📁 OUTPUTS (results/ml/)"]
        METRICS[test_metrics.json<br/>train_history.json]
        PRED[test_predictions.csv<br/>val_predictions.csv]
        CAM[gradcam/<br/>occlusion/<br/>regional_relevance/]
    end

    subgraph Service["🌐 INFERENCE SERVICE (service/)"]
        APP[FastAPI app.py<br/>/api/health, /api/predict]
        PV[predict_volume.py<br/>loads canonical checkpoint<br/>verifies SHA256]
        ID[checkpoint_identity()<br/>SHA256 + size check]
    end

    RAW --> EXT
    EXT --> IDX
    SPLITS --> DS
    IDX --> PRE
    PRE --> CACHE
    CACHE --> DS
    DS --> AUG
    AUG --> RESNET
    RESNET --> FC
    CFG --> LOSS
    LOSS --> OPT
    DS -->|train/val| OPT
    OPT --> ES
    ES --> CKPT
    CKPT --> MET
    MET --> THR
    THR --> BOOT
    METRICS -->|test_metrics.json| METRICS
    CKPT --> PV
    PV --> ID
    ID --> APP
    APP --> Outputs
```

---

## 3. Detailed Component Architecture

### 3.1 Data Layer (`data/`)

```
data/raw/oasis1/
├── extracted/          # Extracted MRI volumes (.img/.hdr pairs)
│   ├── OAS1_0001_MR1/
│   ├── OAS1_0002_MR1/
│   └── ...
data/splits/
├── train_subjects.csv  # 128 sessions (train split)
├── val_subjects.csv    # 27 sessions (val split)
├── test_subjects.csv   # 27 sessions (test split)
└── matched_cohort.csv  # All 182 subjects with demographics + labels
data/derived/
├── oasis1_128/         # Cached preprocessed volumes (*.npy)
│   ├── OAS1_0001_MR1.npy
│   └── ...
└── atlas_dk68_tianS1.nii.gz   # DK68 ∪ TianS1 parcellation atlas
└── atlas_dk68_tianS1_info.csv # Atlas parcel metadata
```

### 3.2 Model Architecture (`src/brainvuln/mri/models.py`)

```
ResNet18Binary
├── backbone (MONAI ResNet-18, 3D)
│   ├── conv1: 1→64, k=7, s=2
│   ├── bn1 + ReLU
│   ├── maxpool: k=3, s=2
│   ├── layer1: 2× basic blocks, 64→64
│   ├── layer2: 2× basic blocks, 64→128
│   ├── layer3: 2× basic blocks, 128→256
│   └── layer4: 2× basic blocks, 256→512  ← Grad-CAM target
├── pool: AdaptiveAvgPool3d(1) → [B, 512]
└── fc: nn.Linear(512, 1) → single logit

Forward: x → backbone → pool → flatten → fc → logit → sigmoid → probability
Parameters: ~33.4M (widen_factor=1.0)
```

**Model variants available:**
| Model | Params | Use Case |
|-------|--------|----------|
| `ResNet18Binary` | ~33.4M | Primary classifier (widen_factor=1.0) |
| `SimpleCNN3D` | ~0.5M | Lightweight baseline |
| `AgeSexBaseline` | ~3 | Demographic-only ceiling (logistic regression) |

### 3.3 Training Pipeline (`scripts/train_oasis1.py` + `src/brainvuln/mri/train.py`)

**5 stages:**

1. **Preprocess + cache** — All cohort scans preprocessed deterministically to 128³ numpy arrays, cached in `data/derived/oasis1_128/`

2. **Train** — Train on train partition only. Early stopping on **validation ROC-AUC** (test never touched during training)
   - Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
   - Loss: BCEWithLogitsLoss with automatic pos_weight = neg_count/pos_count
   - LR schedule: Warmup (2 epochs) → Cosine annealing (12 epochs)
   - Augmentation: Rotation ±7°, translation ±3 vox, scale ±5%, intensity ±10%, Gaussian noise σ=0.02
   - Gradient clipping: max norm 5.0
   - Batch size: 4

3. **Freeze → Evaluate** — Best checkpoint (by val ROC-AUC) evaluated on held-out test set with subject-level bootstrap CIs (2000 resamples)

4. **Grad-CAM + Occlusion** — Per-subject explainability maps saved to `results/ml/<run>/gradcam/`

5. **Regional maps** — CAM → atlas parcels via `regionalize_cam`, aggregated into M_CNN table

### 3.4 Threshold Selection (`src/brainvuln.mri.evaluate`)

```
select_threshold(y_val, p_val, method="youden")
    └── Youden's J = sensitivity + specificity - 1
    └── Grid: 0.01 to 0.99 (step 0.01)
    └── Frozen value: 0.07
    └── Applied UNCHANGED to test set
```

Three sources pinned to 0.07:
1. `config/analysis.yaml` → `inference.threshold.value`
2. `results/ml/resnet_seed42/metrics/test_metrics.json` → `threshold`
3. `brainvuln.config.resolve_threshold()` → serving path

### 3.5 Inference Service (`service/`)

```
service/app.py          → FastAPI endpoints (/health, /predict)
service/predict_volume.py → Canonical inference wrapper
service/checkpoint_bootstrap.py → SHA256-verified checkpoint download
service/Dockerfile       → Container build with SHA verification
```

**Checkpoint integrity chain:**
1. `resolve_checkpoint()` reads `inference.checkpoint.path` from `config/analysis.yaml`
2. `checkpoint_identity()` computes SHA256 + size of the resolved file
3. `predict_volume.py` compares against `CANONICAL_SHA256` constant
4. Mismatch → `CheckpointIdentityError` → HTTP 500, never serves unverified model

---

## 4. Current Status

| Component | Status |
|-----------|--------|
| Data preprocessing | ✅ Implemented (needs raw OASIS-1 data on disk) |
| Model architecture | ✅ ResNet18Binary, SimpleCNN3D, AgeSexBaseline |
| Training loop | ✅ AdamW, weighted BCE, early stop on val ROC-AUC |
| Evaluation metrics | ✅ ROC-AUC, PR-AUC, acc, bal_acc, sens, spec, F1, Brier + bootstrap CIs |
| Threshold selection | ✅ Youden's J on val only, frozen at 0.07 |
| Grad-CAM / explainability | ✅ 3D Grad-CAM + occlusion + regional mapping |
| Inference service | ✅ FastAPI + SHA256 checkpoint verification |
| Test suite | ✅ 144 passed, 3 skipped (raw data not on disk) |
| Checkpoint | ✅ Restored, SHA256 verified against manifest |

**Last measured performance (from frozen artifacts):**
- Validation: accuracy 77.8%, ROC-AUC 0.800
- Test: accuracy 66.7%, ROC-AUC 0.817

---

## 5. Path to 85%+ Accuracy

To push accuracy toward 85–90%, the following levers are available in code (raw data needed to retrain):

| Lever | Current | Suggested | Expected Impact |
|-------|---------|-----------|-----------------|
| `widen_factor` | 1.0 | 2.0 (4× channels) | +5–8% accuracy |
| `cos-epochs` | 12 | 20–30 | Better LR annealing |
| `epochs` | 20 | 40–60 | More convergence |
| Label smoothing | 0.0 | 0.1 | Better calibration |
| Age/sex fusion | ❌ Not fused | Concatenate demo features to penultimate layer | +3–5% (demographics are predictive) |
| Ensemble (multi-seed) | Single seed | 3–5 seed ensemble | +2–4% |
| Class balancing | pos_weight only | Focal loss + pos_weight | Better minority class recall |

To retrain with improved settings:
```bash
python scripts/train_oasis1.py --model resnet --seed 42 \
    --epochs 50 --patience 8 --lr 5e-5 --batch-size 4 \
    --warmup 5 --cos-epochs 30 --widen-factor 2.0 \
    --label-smoothing 0.1 --dir-name resnet_seed42_v2
```
*(Requires `data/raw/oasis1/extracted/` to be present on disk.)*

