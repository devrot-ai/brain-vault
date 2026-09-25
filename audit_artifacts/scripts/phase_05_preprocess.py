#!/usr/bin/env python
"""Phase 5: preprocessing smoke test on real raw OASIS sessions."""
import glob
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from brainvuln.mri.preprocess import (  # noqa: E402
    build_scan_index, find_scan_file, preprocess_session)

# pick 3 real sessions from different discs
sessions = []
for disc in ("disc1", "disc2", "disc3"):
    for rec in build_scan_index(Path(f"data/raw/oasis1/extracted/{disc}")):
        sessions.append(rec)
        break

results = []
for rec in sessions:
    vol = preprocess_session(rec.scan_path)
    # NOTE: after z-scoring, brain voxels span negative values; the output
    # brain mask is vol != 0 (non-brain is exactly 0), NOT vol > 0.
    brain = vol[vol != 0]
    entry = {
        "session": rec.session_id,
        "source": str(rec.scan_path),
        "shape": list(vol.shape),
        "dtype": str(vol.dtype),
        "min": float(vol.min()),
        "max": float(vol.max()),
        "mean_brain": float(brain.mean()),
        "std_brain": float(brain.std()),
        "n_nan": int(np.isnan(vol).sum()),
        "n_inf": int(np.isinf(vol).sum()),
        "n_zero_voxels": int((vol == 0).sum()),
        "n_brain_voxels": int((vol != 0).sum()),
        "mean_brain_nonzero_mask": float(vol[vol != 0].mean()),
        "std_brain_nonzero_mask": float(vol[vol != 0].std()),
    }
    # save one example as NIfTI for the artifact requirement
    if len(results) == 0:
        img = nib.Nifti1Image(vol[0], np.eye(4))
        nib.save(img, "audit_artifacts/phase_05_example_preprocessed.nii.gz")
    results.append(entry)

    # cache bit-identity check: rerun and compare with cached .npy if present
    cached = Path(f"data/derived/oasis1_128/{rec.session_id}.npy")
    if cached.exists():
        fresh = preprocess_session(rec.scan_path)
        same = np.array_equal(fresh, np.load(cached))
        entry["cache_bit_identical"] = bool(same)

print(json.dumps(results, indent=2))
ok = all(
    r["shape"] == [1, 128, 128, 128]
    and r["dtype"] == "float32"
    and r["n_nan"] == 0 and r["n_inf"] == 0
    and r["n_brain_voxels"] > 100000
    and abs(r["mean_brain_nonzero_mask"]) < 1e-4
    and abs(r["std_brain_nonzero_mask"] - 1.0) < 1e-3
    and r.get("cache_bit_identical", True)
    for r in results
)
print("PHASE5_VERDICT:", "PASS" if ok else "FAIL")
