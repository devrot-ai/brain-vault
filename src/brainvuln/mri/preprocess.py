"""Deterministic 3D MRI preprocessing for OASIS-1 (Part D).

Input selection
---------------
OASIS-1 ships, per session, four anonymized raw scans (``RAW/mpr-1..4``) and
published derivatives under ``PROCESSED/MPRAGE/T88_111``. We use the
``*_t88_masked_gfc`` volume: gain-field corrected, **affinely registered to
the Colloridi/T88 atlas template** and **skull-stripped** by the OASIS
preprocessing pipeline (Buckner et al., J. Neurosci. Methods 2004). Using the
dataset's own documented derivative makes steps 5 (brain extraction) and 6
(affine registration) of the preprocessing specification deterministic and
identical for every subject — no per-image algorithmic choices of ours —
and it is the same input grid for all subjects (176×208×176 at 1 mm).

Pipeline (identical for train/val/test; no stochastic component):

1. load ANALYZE pair (nibabel), drop a trailing length-1 axis
2. reorient to RAS+ via ``nib.as_closest_canonical``
3. resample trilinearly to 2 mm isotropic (order=1, cval=0)
4. crop tight to the brain bounding box (nonzero voxels), then centre-pad /
   centre-crop to a fixed 128×128×128 grid (256 mm coverage)
5. z-normalize inside the brain (nonzero voxels); zero elsewhere
6. return ``float32[1,128,128,128]``

2 mm isotropic is the standard 3D AD-classification choice: it keeps the
whole brain inside a 128³ lattice with margin, and it is coarse enough that
CPU training is tractable. Preprocessed volumes are cached as
``.npy`` under ``--cache-dir``; a cache hit is bit-identical to a fresh
computation (same code path, no augmentation here).

Never any JPEG/PIL: volumetric data stays float32 arrays end to end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

TARGET_SHAPE = (128, 128, 128)
VOXEL_MM = 2.0
INPUT_SUFFIX = "_t88_masked_gfc.img"  # OASIS T88 skull-stripped registered

# Bump whenever preprocessing semantics change in a way that could alter
# model inputs; predictions must always record which version produced them.
PREPROCESS_VERSION = "t88_masked_gfc-2mm-128cube-brainz-v1"


def find_scan_file(session_dir: Path) -> Path:
    """Locate the T88 skull-stripped registered volume of one OASIS session."""
    cands = list(session_dir.glob(f"**/*{INPUT_SUFFIX}"))
    if not cands:
        # older discs sometimes use .hdr/.img split naming already covered;
        # fall back to any masked_gfc header
        cands = list(session_dir.glob("**/*masked_gfc.hdr"))
    if not cands:
        raise FileNotFoundError(f"no T88 masked_gfc volume under {session_dir}")
    return sorted(cands)[0]


def session_id_from_dir(session_dir: Path) -> str:
    return session_dir.name


def _load_image(path: Path):
    """Load an image as a canonical 3D Nifti1Image (never PIL, never JPEG)."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected 3D volume, got {data.shape} for {path}")
    img = nib.Nifti1Image(data, img.affine)
    return nib.as_closest_canonical(img)  # 2. reorient to RAS+


def preprocess_session(scan_path: str | Path) -> np.ndarray:
    """Full deterministic pipeline; returns float32[1,128,128,128]."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    path = Path(scan_path)
    img = _load_image(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)

    # 3. resample to 2 mm isotropic, trilinear, zero fill
    zoom = np.asarray(img.header.get_zooms()[:3], dtype=float)
    target_affine = img.affine.copy()
    scale = zoom / VOXEL_MM
    new_shape = np.maximum(1, np.round(np.asarray(data.shape) * scale)).astype(int)
    target_affine[:3, :3] = (target_affine[:3, :3] @ np.diag(1.0 / zoom)) * VOXEL_MM
    resampled = resample_from_to(
        img, (new_shape.tolist(), target_affine), order=1, cval=0.0)
    vol = np.asanyarray(resampled.dataobj, dtype=np.float32)

    # 4. tight crop to brain bounding box, then centre-pad/crop to 128³
    idx = np.argwhere(vol > 0)
    if len(idx) == 0:
        raise ValueError(f"empty (fully masked) volume: {path}")
    lo, hi = idx.min(0), idx.max(0) + 1
    vol = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    out = np.zeros(TARGET_SHAPE, dtype=np.float32)
    pads = []
    for axis in range(3):
        free = TARGET_SHAPE[axis] - vol.shape[axis]
        if free < 0:  # crop centrally (near-certain only at extreme outliers)
            start = (vol.shape[axis] - TARGET_SHAPE[axis]) // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(start, start + TARGET_SHAPE[axis])
            vol = vol[tuple(sl)]
            free = 0
        pads.append((free // 2, free - free // 2))
    out = np.pad(vol, pads).astype(np.float32)

    # 5. z-normalize inside the brain
    brain = out > 0
    if brain.sum() == 0:
        raise ValueError(f"no in-brain voxels after padding: {path}")
    mu, sd = out[brain].mean(), out[brain].std()
    sd = sd if sd > 1e-6 else 1.0
    out[brain] = (out[brain] - mu) / sd

    return out[None, ...]  # 1×128×128×128


# ---------------------------------------------------------------------------
# scan index: every session dir on disk with its label-bearing session id
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanRecord:
    session_id: str          # e.g. OAS1_0001_MR1
    subject_id: str          # e.g. OAS1_0001
    scan_path: Path
    n_sessions: int = 1      # sessions available for this subject


def build_scan_index(extracted_root: Path, min_mpr: int = 1) -> Iterator[ScanRecord]:
    """Yield one record per session directory that contains a T88 volume."""
    root = Path(extracted_root)
    session_dirs = sorted(p for p in root.rglob("OAS1_*_MR*")
                          if p.is_dir() and p.parent != root)
    for d in session_dirs:
        try:
            scan = find_scan_file(d)
        except FileNotFoundError:
            continue
        m = re.fullmatch(r"(OAS1_\d+)_MR(\d+)", d.name)
        if not m:
            continue
        subject = m.group(1)
        n_sessions = len([q for q in d.parent.glob(f"{subject}_MR*") if q.is_dir()])
        yield ScanRecord(d.name, subject, scan, n_sessions)


def preprocess_and_cache(
    scan: ScanRecord, cache_dir: Path, overwrite: bool = False
) -> Path:
    """Preprocess one session and cache as ``.npy``; idempotent."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{scan.session_id}.npy"
    if out.exists() and not overwrite:
        return out
    vol = preprocess_session(scan.scan_path)
    np.save(out, vol)
    return out
