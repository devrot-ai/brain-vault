#!/usr/bin/env python
"""Phase 11: atlas/regional mapping audit after the affine fix.

1. Re-save one production CAM with the corrected affine and parcellate it.
2. Require BOTH hemispheres + subcortex to carry CAM mass.
3. Write the region-mapping TSV artifact.
"""
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln.mri.gradcam import save_cam_nifti  # noqa: E402
from brainvuln.mri.preprocess import find_scan_file  # noqa: E402
from brainvuln.mri.regional import regionalize_cam  # noqa: E402

atlas = "data/derived/atlas_dk68_tianS1.nii.gz"
info = "data/derived/atlas_dk68_tianS1_info.csv"

# recompute the CAM fresh from the production checkpoint on the 128^3 grid
import torch
from brainvuln.mri.gradcam import gradcam_3d  # noqa: E402
from brainvuln.mri.models import ResNet18Binary  # noqa: E402

sid = "OAS1_0013_MR1"
vol = np.load(f"data/derived/oasis1_128/{sid}.npy")
ckpt = torch.load("results/ml/resnet_seed42/checkpoints/best.pt",
                  map_location="cpu", weights_only=False)
model = ResNet18Binary()
model.load_state_dict(ckpt["state_dict"])
model.eval()
cam128 = gradcam_3d(model, torch.from_numpy(vol[None]))

sess_dirs = sorted(glob.glob(f"data/raw/oasis1/extracted/disc*/disc*/{sid}"))
ref_scan = find_scan_file(Path(sess_dirs[0]))

fixed = Path("audit_artifacts/phase_11_cam_fixed.nii.gz")
save_cam_nifti(cam128, ref_scan, fixed)

s, qc, _ = __import__("brainvuln.disease_maps",
                      fromlist=["parcellate_nifti"]).parcellate_nifti(
    fixed, atlas, info, mask_zero=False)
tab = pd.DataFrame({
    "region": s.index,
    "mean_relevance": s.values,
    "voxel_count": [int(qc["n_voxels_used"].iloc[i]) for i in range(len(s))],
})
tab.to_csv("audit_artifacts/phase_11_region_mapping.tsv", sep="\t", index=False)

nonzero = s.dropna()
n_pos = int((nonzero > 0).sum())
dk = [r for r in s.index if not r.startswith("TianS1")]
tian = [r for r in s.index if r.startswith("TianS1")]
left = [r for r in dk if r.endswith("_L")]
right = [r for r in dk if r.endswith("_R")]
print(f"parcels with data: {len(nonzero)}/87; nonzero: {n_pos}")
print(f"  DK left-hemisphere nonzero: {int((s[left] > 0).sum())}/{len(left)}")
print(f"  DK right-hemisphere nonzero: {int((s[right] > 0).sum())}/{len(right)}")
print(f"  Tian subcortical nonzero: {int((s[tian] > 0).sum())}/{len(tian)}")
print("top-8 regions:", dict(nonzero.nlargest(8).round(4)))

ok = (int((s[left] > 0).sum()) >= 5 and int((s[right] > 0).sum()) >= 5
      and int((s[tian] > 0).sum()) >= 3)
print("PHASE11_AFFINE_VERDICT:", "PASS" if ok else "FAIL")
