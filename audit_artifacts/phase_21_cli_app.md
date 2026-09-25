=== train_oasis1.py --help ===
usage: train_oasis1.py [-h] [--model {resnet,simplecnn,agesex}] [--seed SEED]
                       [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
                       [--warmup WARMUP] [--cos-epochs COS_EPOCHS] [--tag TAG]
                       [--patience PATIENCE] [--extracted-root EXTRACTED_ROOT]
                       [--splits-dir SPLITS_DIR] [--cache-dir CACHE_DIR]
                       [--out-dir OUT_DIR] [--no-augment]
                       [--aug-strength {full,light}]
                       [--normalization {brain_z,global_minmax}]
EXIT:0

=== bridge_cnn_gene.py --help ===
usage: bridge_cnn_gene.py [-h] [--ml-dir ML_DIR] [--disease DISEASE]
                          [--n-matched N_MATCHED] [--n-spatial N_SPATIAL]
                          [--seed SEED] [--quick] [--no-ckpt]
                          [--out-dir OUT_DIR]

Three-way bridge analysis (Parts P, Q, R, S, T). Core statistic — the central
hypothesis of the project: rho_CNN_GENE = Spearman(M_CNN, M_GENE) where M_CNN
is the model-derived regional relevance map (Part L) and M_GENE the regional
EXIT:0

=== predict.py --help ===
usage: predict.py [-h] --image IMAGE [--checkpoint CHECKPOINT]
                  [--out-dir OUT_DIR] [--threshold THRESHOLD]

Single-image inference (Part AA).

    python predict.py --image patient.nii.gz

Prints the prediction in the documented format and saves:
EXIT:0

=== import_disease_map.py --help ===
usage: import_disease_map.py [-h] --map MAP [--format {nifti,csv}] --disease
                             DISEASE --name NAME --modality MODALITY
                             --sign-convention SIGN_CONVENTION --citation
                             CITATION [--source-url SOURCE_URL]
                             [--persistent-id PERSISTENT_ID]
                             [--license-note LICENSE_NOTE]
EXIT:0

=== verify_diseases.py --help ===
usage: verify_diseases.py [-h] [--disease DISEASE]
                          [--diseases-config DISEASES_CONFIG]

Verify the EFO / MONDO ids configured for each disease before any retrieval.
Queries the GWAS Catalog REST v2 for each configured efo_id and the Open
Targets GraphQL API for each mondo_id, and prints the resolved trait labels so
EXIT:0

=== make_figures.py --help ===
usage: make_figures.py [-h] [--ml-dir ML_DIR] [--fig-dir FIG_DIR]

Generate the ten main figures (Part AE) from on-disk artifacts. python
scripts/make_figures.py --ml-dir results/ml/resnet_seed42 Figure 10 (external
validation) renders a documented status panel until OASIS-3/ADNI data access
is granted; every other figure is produced from real artifacts in results/ and
EXIT:0

=== smallest valid inference (real checkpoint, real volume) ===
Prediction:
  Alzheimer's disease
Probability:
  0.18
Threshold:
  0.07
Model:
  BrainVuln-ResNet18Binary

saved audit_artifacts\phase_21_out\phase_05_example_preprocessed_gradcam.nii.gz and audit_artifacts\phase_21_out\phase_05_example_preprocessed_gradcam.png
EXIT:0

=== web app: import + handler smoke (no server) ===
app.py imports OK; public names: ['build_ui']
EXIT:0
--- predict_upload output ---
### Prediction: cognitively normal (model prediction)

**Probability:** 0.16  
**Threshold:** 0.50  
**Model:** BrainVuln-ResNet18Binary (`results\ml\resnet_seed42_lightaug\checkpoints\best.pt`)

Top regions by model relevance are in `outputs/phase_05_example_preprocessed_gradcam.nii.gz` (parcellate with `brainvuln.mri.regional.regionalize_cam`).

**Research use only. Not a clinical diagnostic sys
CAM image type: str
NIfTI output: outputs\phase_05_example_preprocessed_gradcam.nii.gz
CAM finite: True | nonzero frac: 0.0
--- predict_upload (canonical checkpoint) ---
### Prediction: cognitively normal (model prediction)

**Probability:** 0.16  
**Threshold:** 0.50  
**Model:** BrainVuln-ResNet18Binary (`results\ml\INTERRUPTED_partial_resnet_seed42_lightaug\checkpoints\best.pt`)

Top regions by model relevance are in `outputs/phase_05_example_preprocessed_gradcam
CAM finite: True | nonzero frac: 0.0
--- predict_upload (production checkpoint, attempt 3) ---
### Prediction: cognitively normal (model prediction)

**Probability:** 0.16  
**Threshold:** 0.50  
**Model:** BrainVuln-ResNet18Binary (`results\ml\zz_archived_interrupted_lightaug\checkpoints\best.pt`)

Top regions by model relevance are in `outputs/phase_05_example_preprocess
CAM finite: True | nonzero frac: 0.0 | max: 0.0
--- predict_upload (production checkpoint) ---
### Prediction: Alzheimer's disease (model prediction)

**Probability:** 0.18  
**Threshold:** 0.07  
**Model:** BrainVuln-ResNet18Binary (`results\ml\resnet_seed42\checkpoints\best.pt`)

Top regions by model relevance are in `outputs/phase_05_example_preprocessed_gradcam.nii.gz` (parcellate with `b
CAM finite: True | nonzero frac: 0.4375 | max: 1.0
