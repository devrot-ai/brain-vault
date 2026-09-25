#!/usr/bin/env python
"""Affine-offset probe: is the saved CAM correctly aligned to the atlas?

The CAM is computed on the 128^3 preprocessed grid, which is a crop/pad of
the T88-registered 176x208x176 volume, so the 128^3 grid is NOT the same
physical frame as the T88 affine. save_cam_nifti currently writes the CAM
with np.eye(4) and then resamples to the reference scan's affine - if the
reference affine has a nonzero origin/scale relative to the 128^3 grid,
the saved map is spatially shifted. This probe measures the shift's effect
by parcellating a real production CAM two ways.
"""
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from brainvuln.mri.preprocess import find_scan_file  # noqa: E402
from brainvuln.disease_maps import parcellate_nifti  # noqa: E402
from pathlib import Path  # noqa: E402

cam_file = Path("results/ml/resnet_seed42/gradcam/OAS1_0002_gradcam.nii.gz")
if not cam_file.exists():
    import glob
    cands = sorted(glob.glob("results/ml/resnet_seed42/gradcam/*_gradcam.nii.gz"))
    print("fallback CAM:", cands[:2])
    cam_file = Path(cands[0])
# locate the session dir for the CAM's subject across all discs
import glob
sid = cam_file.name.split("_")[2] if False else cam_file.stem.split("_gradcam")[0]
subj = sid.rsplit("_MR", 1)[0]
sess_dirs = sorted(glob.glob(
    f"data/raw/oasis1/extracted/disc*/disc*/{sid}_MR*"))
assert sess_dirs, f"no session dir for {sid}"
ref_scan = find_scan_file(Path(sess_dirs[0]))
print("CAM subject:", sid, "ref scan:", ref_scan)

atlas = "data/derived/atlas_dk68_tianS1.nii.gz"
info = "data/derived/atlas_dk68_tianS1_info.csv"

# A) production path: saved CAM (identity affine) parcellated as-is
s_prod, qc_a, _ = parcellate_nifti(cam_file, atlas, info, mask_zero=False)

# B) affine-corrected: treat the CAM's grid as the preprocessed grid. We
# rebuild that grid's affine from the reference scan by mimicking
# preprocess_session's resample+crop: resample the T88 scan to 2mm, then
# find its brain bbox and center-pad to 128^3 - the affine of that result
# is the CAM's true frame.
from nibabel.processing import resample_from_to  # noqa: E402

ref = nib.load(str(ref_scan))
data = np.asanyarray(ref.dataobj, dtype=np.float32)
if data.ndim == 4 and data.shape[-1] == 1:
    data = data[..., 0]
img = nib.Nifti1Image(data, ref.affine)
zoom = np.asarray(ref.header.get_zooms()[:3], dtype=float)
scale = zoom / 2.0
new_shape = np.maximum(
    1, np.round(np.asarray(data.shape[:3]) * scale)).astype(int)
target_affine = ref.affine.copy()
target_affine[:3, :3] = (target_affine[:3, :3] @ np.diag(1.0 / zoom)) * 2.0
r = resample_from_to(img, (new_shape.tolist(), target_affine), order=1, cval=0.0)
vol = np.asanyarray(r.dataobj, dtype=np.float32)
idx = np.argwhere(vol > 0)
lo, hi = idx.min(0), idx.max(0) + 1
# affine of the 128^3 grid: origin at voxel lo of the 2mm grid, shifted by pad
pads = [(128 - (hi[k] - lo[k])) // 2 for k in range(3)]
aff128 = target_affine.copy()
off = target_affine[:3, :3] @ (lo - np.array(pads))
aff128[:3, 3] = target_affine[:3, 3] + off
cam128 = nib.Nifti1Image(
    np.asanyarray(nib.load(str(cam_file)).dataobj), aff128)

import tempfile  # noqa: E402
with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as fh:
    tmp = fh.name
nib.save(cam128, tmp)
s_corr, qc_b, _ = parcellate_nifti(tmp, atlas, info, mask_zero=False)

j = pd.concat([s_prod.rename("prod"), s_corr.rename("corrected")], axis=1).dropna()
rho = j["prod"].corr(j["corr".replace("corr", "corrected")], method="spearman")
print(f"n parcels both: {len(j)}")
print(f"spearman(prod, corrected): {rho:.3f}")
print(f"prod mean CAM: {s_prod.mean():.5f}  corrected mean CAM: {s_corr.mean():.5f}")
print("top-5 prod:", dict(s_prod.dropna().nlargest(5).round(4)))
print("top-5 corrected:", dict(s_corr.dropna().nlargest(5).round(4)))
