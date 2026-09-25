#!/usr/bin/env bash
# Phase 9: inference CLI test - valid prediction + graceful invalid inputs.
cd "$(dirname "$0")/../.."
PY=./.venv/Scripts/python.exe

CKPT=results/ml/resnet_seed42/checkpoints/best.pt
EXAMPLE_NII=audit_artifacts/phase_05_example_preprocessed.nii.gz

echo "=== valid prediction (preprocessed example NIfTI, production checkpoint) ==="
PYTHONIOENCODING=utf-8 $PY predict.py --image "$EXAMPLE_NII" --checkpoint "$CKPT" \
  --out-dir audit_artifacts/phase_09_out 2>&1
echo "EXIT:$?"

echo; echo "=== invalid: nonexistent file ==="
PYTHONIOENCODING=utf-8 $PY predict.py --image does_not_exist.nii.gz --checkpoint "$CKPT" 2>&1 | tail -3
echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== invalid: corrupted NIfTI ==="
echo "not a nifti file at all" > audit_artifacts/phase_09_corrupt.nii.gz
PYTHONIOENCODING=utf-8 $PY predict.py --image audit_artifacts/phase_09_corrupt.nii.gz --checkpoint "$CKPT" 2>&1 | tail -3
echo "EXIT:${PIPESTATUS[0]}"

echo; echo "=== invalid: wrong dimensionality (2D numpy saved as nii) ==="
PYTHONIOENCODING=utf-8 $PY - <<'EOF'
import numpy as np, nibabel as nib
nib.save(nib.Nifti1Image(np.zeros((64, 64), dtype=np.float32), np.eye(4)),
         "audit_artifacts/phase_09_2d.nii.gz")
EOF
PYTHONIOENCODING=utf-8 $PY predict.py --image audit_artifacts/phase_09_2d.nii.gz --checkpoint "$CKPT" 2>&1 | tail -3
echo "EXIT:${PIPESTATUS[0]}"
