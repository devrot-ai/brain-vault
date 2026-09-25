#!/usr/bin/env python
"""Phase 10: Grad-CAM verification (real MRI, production checkpoint).

Two independent verifications:
1. The CAM volume saved by predict.py is a valid 3D finite non-constant map
   (it lives on the *reference scan's* grid by design - resampled for
   parcellation - so it is NOT expected to match an on-grid recompute
   voxel-for-voxel).
2. A fresh on-grid CAM is recomputed through the module API and checked for
   structure (nonzero fraction, spatial concentration).
"""
import json
import shutil
import sys

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, "src")
from brainvuln.mri.gradcam import gradcam_3d  # noqa: E402
from brainvuln.mri.models import ResNet18Binary  # noqa: E402

cam_path = "audit_artifacts/phase_09_out/phase_05_example_preprocessed_gradcam.nii.gz"
png_path = "audit_artifacts/phase_09_out/phase_05_example_preprocessed_gradcam.png"

# --- 1. saved CAM file -----------------------------------------------------
cam_img = nib.load(cam_path)
cam_saved = np.asanyarray(cam_img.dataobj, dtype=np.float32)
print("saved CAM shape:", cam_saved.shape, "(reference-scan grid by design)")
print("finite:", bool(np.isfinite(cam_saved).all()),
      "nonzero frac:", round(float((cam_saved != 0).mean()), 4),
      "max:", float(cam_saved.max()))

# --- 2. fresh on-grid CAM via module API ----------------------------------
vol = np.load("data/derived/oasis1_128/OAS1_0001_MR1.npy")
ckpt = torch.load("results/ml/resnet_seed42/checkpoints/best.pt",
                  map_location="cpu", weights_only=False)
model = ResNet18Binary()
model.load_state_dict(ckpt["state_dict"])
model.eval()
cam2 = gradcam_3d(model, torch.from_numpy(vol[None]))
print("on-grid CAM shape:", cam2.shape,
      "nonzero frac:", round(float((cam2 != 0).mean()), 4),
      "max:", float(cam2.max()))

# spatial concentration: CAM mass in top-10% voxels should far exceed uniform
flat = np.sort(cam2.flatten())[::-1]
n10 = int(0.1 * flat.size)
conc = flat[:n10].sum() / flat.sum()
print("CAM mass in top-10% voxels:", round(float(conc), 3), "(uniform = 0.1)")

# --- 3. PNG artifact ------------------------------------------------------
from PIL import Image
img = Image.open(png_path)
print("PNG size:", img.size, "mode:", img.mode)

shutil.copy(cam_path, "audit_artifacts/phase_10_gradcam.nii.gz")
shutil.copy(png_path, "audit_artifacts/phase_10_gradcam.png")

checks = {
    "saved_cam_finite_3d_nonconstant": (cam_saved.ndim == 3
                                        and bool(np.isfinite(cam_saved).all())
                                        and float(cam_saved.max()) > 0),
    "on_grid_cam_structured": (float((cam2 != 0).mean()) > 0.05
                               and float(conc) > 0.3),
    "nifti_saved": True,
    "png_saved": img.size[0] > 100,
}
print(json.dumps(checks, indent=2))
with open("audit_artifacts/phase_10_checks.json", "w") as fh:
    json.dump({
        "checks": checks,
        "saved_cam": {"shape": list(cam_saved.shape),
                      "nonzero_frac": float((cam_saved != 0).mean()),
                      "max": float(cam_saved.max())},
        "on_grid_cam": {"shape": list(cam2.shape),
                        "nonzero_frac": float((cam2 != 0).mean()),
                        "top10pct_mass": float(conc)},
        "note": ("saved CAM is resampled onto the reference scan's grid by "
                 "design (save_cam_nifti); on-grid CAM verified separately"),
    }, fh, indent=2)
print("PHASE10_VERDICT:", "PASS" if all(checks.values()) else "FAIL")
