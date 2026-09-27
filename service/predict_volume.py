"""Canonical BrainVuln inference for web requests (research use only).

This module is a THIN wrapper: every scientific step is imported unchanged
from the validated BrainVuln pipeline.

  preprocessing .... brainvuln.mri.preprocess.preprocess_session
                     (canonical t88_masked_gfc-2mm-128cube-brainz-v1)
  checkpoint ....... brainvuln.config.resolve_checkpoint / checkpoint_identity
                     (config-driven; never glob, never mtime, never newest)
  threshold ........ brainvuln.config.resolve_threshold
                     (frozen validation-selected operating point)
  model ............ brainvuln.mri.models.ResNet18Binary
  Grad-CAM ......... brainvuln.mri.gradcam.gradcam_3d / save_planes_png
  regions .......... brainvuln.mri.regional.regionalize_cam on the existing
                     DK68∪TianS1 atlas

The canonical checkpoint identity (SHA256) is verified before anything is
served. A checkpoint that does not match the frozen seed-42 artifact is
refused loudly — the backend never serves a different seed silently.

RESEARCH USE ONLY. Probabilities come from a model trained on a small
research cohort (OASIS-1, n=27 held-out subjects) and must not be used for
any clinical decision.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# --- make the canonical package importable when run outside an install -----
ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

CANONICAL_CHECKPOINT_RELPATH = "results/ml/resnet_seed42/checkpoints/best.pt"
CANONICAL_SHA256 = (
    "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
)
ALLOWED_SUFFIXES = (".nii", ".nii.gz")
MAX_UPLOAD_MB = int(os.environ.get("BRAINVULN_MAX_UPLOAD_MB", "200"))
ATLAS_IMG = Path(os.environ.get(
    "BRAINVULN_ATLAS_IMG", str(ROOT / "data/derived/atlas_dk68_tianS1.nii.gz")))
ATLAS_INFO = Path(os.environ.get(
    "BRAINVULN_ATLAS_INFO", str(ROOT / "data/derived/atlas_dk68_tianS1_info.csv")))
MODEL_NAME = "BrainVuln 3D ResNet"
MODEL_VERSION = "seed42-canonical-1.0.0"
DISCLAIMER = (
    "Research use only. Not a clinical diagnostic system. This model has not "
    "been externally validated for clinical use; outputs are research "
    "classifications from a small single-site cohort, not medical diagnoses."
)


class InvalidVolumeError(ValueError):
    """Uploaded file is not a usable 3D NIfTI volume (maps to HTTP 422)."""


class CheckpointIdentityError(RuntimeError):
    """Resolved checkpoint is not the canonical frozen artifact (HTTP 500)."""


# ---------------------------------------------------------------------------
# upload validation (extension / size / loadability)
# ---------------------------------------------------------------------------

def validate_upload(filename: str, size_bytes: int | None) -> None:
    name = (filename or "").strip()
    lower = name.lower()
    if not name:
        raise InvalidVolumeError("No filename provided.")
    if not lower.endswith(ALLOWED_SUFFIXES):
        raise InvalidVolumeError(
            f"Unsupported file type {name!r}. Only .nii and .nii.gz NIfTI "
            "volumes are supported.")
    if size_bytes is not None and size_bytes <= 0:
        raise InvalidVolumeError("Uploaded file is empty.")
    if size_bytes is not None and size_bytes > MAX_UPLOAD_MB * (1 << 20):
        raise InvalidVolumeError(
            f"File too large ({size_bytes / (1 << 20):.0f} MB). The limit is "
            f"{MAX_UPLOAD_MB} MB.")


def load_nifti_or_raise(path: Path):
    """Load with nibabel and enforce structural sanity (3D, finite)."""
    import nibabel as nib

    try:
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj, dtype=np.float32)
    except Exception as exc:  # nibabel raises assorted types on bad files
        raise InvalidVolumeError(
            f"Could not read the file as a NIfTI volume ({type(exc).__name__}: "
            f"{exc}).") from exc
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise InvalidVolumeError(
            f"Expected a 3D volume, got shape {tuple(data.shape)}.")
    if data.size == 0:
        raise InvalidVolumeError("Volume contains no voxels.")
    if not np.isfinite(data).all():
        raise InvalidVolumeError(
            "Volume contains non-finite voxel values (NaN/inf).")
    return img


# ---------------------------------------------------------------------------
# canonical model (lazy, shared, thread-safe)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model_cache: dict = {}


def _load_model():
    import torch
    from brainvuln.config import resolve_checkpoint
    from brainvuln.mri.models import ResNet18Binary

    with _model_lock:
        ckpt_path = resolve_checkpoint(None)  # config-driven, never a search
        identity = _identity_checked(ckpt_path)
        cached = _model_cache.get("path")
        if cached == str(ckpt_path):
            return _model_cache["model"], ckpt_path, identity
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ResNet18Binary()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _model_cache.update(model=model, path=str(ckpt_path),
                            seed=ckpt.get("seed"))
        return model, ckpt_path, identity


def _identity_checked(ckpt_path: Path) -> dict:
    from brainvuln.config import checkpoint_identity

    identity = checkpoint_identity(ckpt_path)
    if identity["sha256"] != CANONICAL_SHA256:
        raise CheckpointIdentityError(
            "Resolved checkpoint is NOT the canonical frozen seed-42 artifact "
            f"(got SHA256 {identity['sha256']}, expected {CANONICAL_SHA256}). "
            "Serving refused; fix inference.checkpoint.path and restore the "
            "frozen checkpoint.")
    return identity


# ---------------------------------------------------------------------------
# full canonical inference
# ---------------------------------------------------------------------------

def run_inference(scan_path: str | Path, *, top_k: int = 10,
                  original_name: str | None = None) -> dict:
    """Upload → canonical preprocessing → frozen model → Grad-CAM → regions.

    Returns the structured prediction record used by the web API. Every
    number is produced by the validated BrainVuln code paths.
    """
    import torch
    from brainvuln.config import resolve_threshold
    from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti, save_planes_png
    from brainvuln.mri.preprocess import PREPROCESS_VERSION, preprocess_session
    from brainvuln.mri.regional import regionalize_cam

    t0 = time.perf_counter()
    path = Path(scan_path)
    if not path.is_file():
        raise InvalidVolumeError(f"File not found: {path.name}")

    load_nifti_or_raise(path)

    # canonical preprocessing (identical to training/evaluation)
    t1 = time.perf_counter()
    volume = preprocess_session(path)                     # [1,128,128,128]
    t2 = time.perf_counter()

    model, ckpt_path, identity = _load_model()
    threshold, threshold_source = resolve_threshold(ckpt_path=ckpt_path)

    x = torch.from_numpy(volume[None].astype(np.float32))  # [1,1,D,H,W]
    with torch.no_grad():
        prob = float(torch.sigmoid(model(x)).item())
    t3 = time.perf_counter()

    # real Grad-CAM from the actual model prediction
    cam = gradcam_3d(model, x)
    t4 = time.perf_counter()

    # regional relevance on the SAME atlas as the gene-expression work
    regions, cam_png_b64 = _regionalize(cam, volume, top_k=top_k)
    t5 = time.perf_counter()

    seed_raw = _model_cache.get("seed")
    seed = int(seed_raw) if seed_raw is not None else 42
    label = "AD" if prob >= threshold else "CN"
    label_text = ("classified: Alzheimer's disease pattern"
                  if label == "AD" else
                  "classified: cognitively normal pattern")

    return {
        "success": True,
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "seed": seed,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": identity["sha256"],
        },
        "prediction": {
            "label": label,
            "label_text": label_text,
            "probability": prob,
            "threshold": float(threshold),
            "threshold_source": threshold_source,
        },
        "explainability": {
            "gradcam_available": True,
            "gradcam_method": "3D Grad-CAM (layer4), positive-class logit",
            "regions": regions,
            "gradcam_png_base64": cam_png_b64,
        },
        "input": {
            "filename": original_name or path.name,
            "size_bytes": int(path.stat().st_size),
        },
        "pipeline": {
            "preprocess_version": PREPROCESS_VERSION,
            "atlas": ATLAS_IMG.name,
            "atlas_regions": len(regions),
        },
        "timings_s": {
            "preprocess": round(t2 - t1, 2),
            "inference": round(t3 - t2, 2),
            "gradcam": round(t4 - t3, 2),
            "regional": round(t5 - t4, 2),
            "total": round(t5 - t0, 2),
        },
        "disclaimer": DISCLAIMER,
    }


def _regionalize(cam: np.ndarray, volume: np.ndarray, *, top_k: int):
    """CAM → atlas parcels (existing parcellation code) + base64 plane PNG."""
    import pandas as pd
    from brainvuln.mri.gradcam import save_cam_nifti, save_planes_png
    from brainvuln.mri.regional import regionalize_cam

    with tempfile.TemporaryDirectory(prefix="brainvuln_cam_") as td:
        cam_nii = Path(td) / "cam.nii.gz"
        png = Path(td) / "cam.png"
        save_cam_nifti(cam, _reference_for_cam(), cam_nii)
        series = regionalize_cam(cam_nii, ATLAS_IMG, ATLAS_INFO)
        save_planes_png(cam, volume, png)
        png_b64 = base64.b64encode(png.read_bytes()).decode("ascii")

    info = pd.read_csv(ATLAS_INFO)
    meta = info.set_index("label")[
        ["hemisphere", "structure", "source"]].to_dict("index")
    valid = series.dropna().sort_values(ascending=False)
    regions = []
    for label, val in valid.head(top_k).items():
        m = meta.get(label, {})
        regions.append({
            "name": str(label),
            "structure": str(m.get("structure", "")),
            "hemisphere": str(m.get("hemisphere", "")),
            "source": str(m.get("source", "")),
            "relevance": round(float(val), 4),
        })
    return regions, png_b64


def _reference_for_cam() -> Path:
    """A reference NIfTI whose affine matches the CAM's MNI-ish grid.

    save_cam_nifti writes the CAM with a reference image's affine; the
    canonical 128³ preprocessing grid carries the input's own affine. For
    parcellation purposes the atlas grid is authoritative, so any 128³
    reference with the input affine works. We cache one from the model's
    own evaluation artifacts when available.
    """
    ref = ROOT / "results/gradcam/OAS1_0013_gradcam.nii.gz"
    if ref.is_file():
        return ref
    raise InvalidVolumeError(
        "Internal error: no CAM reference template available.")


def model_info() -> dict:
    """Identity of the serving model without loading any weights."""
    from brainvuln.config import resolve_checkpoint, resolve_threshold

    ckpt_path = resolve_checkpoint(None)
    identity = _identity_checked(ckpt_path)
    threshold, threshold_source = resolve_threshold(ckpt_path=ckpt_path)
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "seed": 42,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": identity["sha256"],
        "checkpoint_size_bytes": identity["size_bytes"],
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "atlas_img": str(ATLAS_IMG),
        "atlas_info": str(ATLAS_INFO),
        "disclaimer": DISCLAIMER,
    }
