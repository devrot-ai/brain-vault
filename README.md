# BrainVuln

A research project that uses AI to look at brain scans and try to understand which parts of the brain are most affected by Alzheimer's disease.

This is research only. It is not a medical tool and should not be used for diagnosis.

## The Question

Alzheimer's does not damage the brain evenly. Some areas are hit harder than others. If an AI model learns to spot Alzheimer's from brain scans, the areas it focuses on should match the areas scientists already know are vulnerable. This project tests whether that is true.

## How It Works

Three pieces come together:

1. Brain MRI scans are fed into a 3D neural network that learns to classify Alzheimer's cases.
2. Grad-CAM highlights which brain regions the model pays attention to.
3. Those regions are compared against known Alzheimer's genes and independent disease maps.

If the AI looks at the same areas that genetics and prior research point to, that is meaningful. If not, the project reports that too.

## The Data

182 brain scans from the OASIS-1 dataset: 100 from people with Alzheimer's and 82 from healthy controls, matched by age and sex. Split into training, validation, and test sets. The test set has 27 subjects.

## Results

The model achieves an ROC-AUC of 0.817 on the test set. It is good at catching Alzheimer's cases (86.7% sensitivity) but also raises some false alarms (specificity 41.7%). The probabilities it outputs tend to be overconfident. Full numbers and confidence intervals are in `results/FINAL_RESULTS.md`.

## Key Finding

The model's attention maps correlate with gene expression (rho = 0.383), but the correlation is not specific to Alzheimer's genes — it reflects the general spatial structure of brain gene expression maps. The study design correctly flags this limitation rather than overstating the result.

## Limitations

Small test set (27 subjects). Single scanning site. Probable Alzheimer's based on clinical rating, not biomarkers. The model is overconfident. External validation has not been done yet.

## Try It

Run a prediction on a brain scan:

```bash
python predict.py --image patient_T88.nii.gz
```

Or start the web demo:

```bash
python app.py
```

The input should be a skull-stripped 3D NIfTI T1 brain scan.

## Reproducibility

Everything is documented. Config files, seeds, checkpoint hashes, and package versions are all recorded. You can retrain the model with:

```bash
python scripts/train_oasis1.py --model resnet --seed 42
```

Run the test suite with `python -m pytest -q` (112 tests).
