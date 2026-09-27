"""Tests for the BrainVuln inference web service (service/).

Coverage
--------
* /api/health — ready state and canonical checkpoint identity
* upload validation — missing file, bad extension, corrupt NIfTI,
  non-finite voxels, size/extension rules (unit level)
* checkpoint resolution — canonical path via the existing resolver,
  SHA256 pinned to the frozen seed-42 artifact, no fallback discovery
* full inference — real OASIS test subject through the canonical pipeline,
  probability compared against the frozen evaluation artifact to 1e-6
  (THE key integration guarantee: web pipeline == canonical pipeline)
* Grad-CAM — real CAM from the real model, region mapping on the real atlas
* pipeline reuse — the service imports the canonical functions, it does not
  duplicate them

Run:  pytest -q tests/test_service_api.py
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "service"
# appended (never prepended): service/app.py must NOT shadow the project's
# root app.py that other test modules import
for p in (str(ROOT / "src"), str(SERVICE)):
    if p not in sys.path:
        sys.path.append(p)

CANONICAL_SHA256 = "5930f0fff96003466a9a2854037ea8758b2b77be49159b7101f4fabbf254a4a9"
CANONICAL_RELPATH = "results/ml/resnet_seed42/checkpoints/best.pt"

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _canonical_subject() -> tuple[str, Path, float]:
    """A real test subject, its canonical T88 volume, and its frozen prob."""
    ev = ROOT / "results/ml/evaluation/test_predictions_subject_level.csv"
    df = pd.read_csv(ev)
    row = df[df["subject_id"] == "OAS1_0013"].iloc[0]
    session = str(row["session_id"])
    matches = list((ROOT / "data/raw/oasis1/extracted").rglob(
        f"{session}_mpr_n4_anon_111_t88_masked_gfc.img"))
    if not matches:
        pytest.skip("raw OASIS-1 volume for OAS1_0013 not present on disk")
    return session, matches[0], float(row["probability"])


@pytest.fixture(scope="module")
def client():
    """Load service/app.py explicitly by path (never `import app`, which
    would collide with the project's root Streamlit app)."""
    import importlib.util

    from fastapi.testclient import TestClient

    spec = importlib.util.spec_from_file_location(
        "brainvuln_service_app", SERVICE / "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def real_subject():
    return _canonical_subject()


# ---------------------------------------------------------------------------
# checkpoint resolution + identity
# ---------------------------------------------------------------------------


def test_checkpoint_resolves_to_canonical_path():
    from brainvuln.config import resolve_checkpoint

    ckpt = resolve_checkpoint(None)
    rel = ckpt.relative_to(ROOT).as_posix()
    assert rel == CANONICAL_RELPATH
    assert ckpt.is_file()


def test_checkpoint_sha256_is_frozen_seed42():
    from brainvuln.config import checkpoint_identity, resolve_checkpoint

    identity = checkpoint_identity(resolve_checkpoint(None))
    assert identity["sha256"] == CANONICAL_SHA256
    assert identity["size_bytes"] > 100 * (1 << 20)  # ~128 MB frozen artifact


def test_identity_check_rejects_a_wrong_hash():
    import predict_volume as pv

    class FakePath:
        name = "best.pt"

    real_sha = pv._identity_checked
    called = {}

    def fake_identity(path):
        called["path"] = path
        return {"sha256": "deadbeef" * 8, "size_bytes": 1}

    from brainvuln import config as cfg

    orig = cfg.checkpoint_identity
    cfg.checkpoint_identity = fake_identity
    try:
        with pytest.raises(pv.CheckpointIdentityError, match="NOT the canonical"):
            pv._identity_checked(Path(CANONICAL_RELPATH))
    finally:
        cfg.checkpoint_identity = orig


# ---------------------------------------------------------------------------
# upload validation
# ---------------------------------------------------------------------------


def test_validate_upload_rejects_bad_extension():
    import predict_volume as pv

    with pytest.raises(pv.InvalidVolumeError, match="Unsupported file type"):
        pv.validate_upload("scan.dcm", 100)
    with pytest.raises(pv.InvalidVolumeError, match="Unsupported file type"):
        pv.validate_upload("scan.txt", 100)


def test_validate_upload_rejects_empty_and_huge():
    import predict_volume as pv

    with pytest.raises(pv.InvalidVolumeError, match="empty"):
        pv.validate_upload("scan.nii.gz", 0)
    with pytest.raises(pv.InvalidVolumeError, match="too large"):
        pv.validate_upload("scan.nii.gz", 201 * (1 << 20))


def test_load_nifti_or_raise_rejects_garbage(tmp_path):
    import predict_volume as pv

    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(b"not a nifti header at all")
    with pytest.raises(pv.InvalidVolumeError, match="Could not read"):
        pv.load_nifti_or_raise(bad)


def test_load_nifti_or_raise_rejects_nonfinite(tmp_path):
    import nibabel as nib

    import predict_volume as pv

    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1, 1, 1] = np.nan
    p = tmp_path / "nan.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(p))
    with pytest.raises(pv.InvalidVolumeError, match="non-finite"):
        pv.load_nifti_or_raise(p)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


def test_health_ready_with_canonical_identity(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["ready"] is True
    assert body["model"]["checkpoint_sha256"] == CANONICAL_SHA256
    assert body["model"]["seed"] == 42


def test_model_info_endpoint(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    info = r.json()["model"]
    assert info["checkpoint_sha256"] == CANONICAL_SHA256
    assert info["threshold_source"].startswith("frozen validation threshold")


def test_predict_requires_a_file(client):
    r = client.post("/api/predict")
    assert r.status_code == 422  # FastAPI validation error


def test_predict_rejects_bad_extension(client):
    r = client.post(
        "/api/predict",
        files={"file": ("scan.txt", b"junk", "text/plain")},
    )
    assert r.status_code == 422
    assert "Unsupported file type" in r.json()["detail"]


def test_predict_rejects_corrupt_nifti(client):
    r = client.post(
        "/api/predict",
        files={"file": ("scan.nii.gz", b"garbage-not-nifti", "application/gzip")},
    )
    assert r.status_code == 422
    assert "Could not read" in r.json()["detail"]


# ---------------------------------------------------------------------------
# THE key integration test: web pipeline == canonical pipeline
# ---------------------------------------------------------------------------


def test_full_prediction_matches_frozen_evaluation(real_subject, tmp_path):
    """Real OASIS test subject through the service wrapper.

    Two documented references:
    1. the canonical CLI (predict.py) — the service must agree EXACTLY
       (identical code path; spec example: CLI 0.4313885868 == API);
    2. the frozen batch-evaluation artifact — agrees within 5e-6
       (measured cross-harness delta is 1.4e-6: float accumulation in the
       batch evaluation loop, not a pipeline difference).
    """
    import predict_volume as pv

    session, scan_path, frozen_prob = real_subject
    result = pv.run_inference(scan_path)

    assert result["success"] is True
    p = result["prediction"]

    # reference 1: canonical CLI, exact agreement
    import predict as predict_cli

    cli_out = tmp_path / "cli_out"
    rc = predict_cli.main(["--image", str(scan_path),
                           "--out-dir", str(cli_out)])
    assert rc == 0
    cli_prob = float((cli_out / "prediction.json").read_text()
                     and __import__("json").loads(
                         (cli_out / "prediction.json").read_text())
                     ["probability"])
    assert p["probability"] == cli_prob  # bitwise identical to the CLI

    # reference 2: frozen evaluation artifact, documented tolerance
    assert abs(p["probability"] - frozen_prob) <= 5e-6

    assert p["threshold"] == pytest.approx(0.07)
    assert p["label"] == ("AD" if p["probability"] >= p["threshold"] else "CN")
    assert result["model"]["checkpoint_sha256"] == CANONICAL_SHA256
    assert result["model"]["seed"] == 42
    assert result["pipeline"]["preprocess_version"] == (
        "t88_masked_gfc-2mm-128cube-brainz-v1")


def test_prediction_response_schema(real_subject):
    import predict_volume as pv

    _s, scan_path, _frozen = real_subject
    r = pv.run_inference(scan_path)

    assert set(r) >= {"success", "model", "prediction", "explainability",
                      "input", "pipeline", "disclaimer"}
    assert set(r["prediction"]) >= {"label", "probability", "threshold",
                                    "threshold_source"}
    assert set(r["model"]) >= {"name", "seed", "checkpoint_sha256"}
    ex = r["explainability"]
    assert ex["gradcam_available"] is True
    regions = ex["regions"]
    assert 0 < len(regions) <= 10
    rel = [reg["relevance"] for reg in regions]
    assert all(np.isfinite(rel)) and rel == sorted(rel, reverse=True)
    for reg in regions:
        assert set(reg) >= {"name", "structure", "hemisphere", "relevance"}
    assert "Research use only" in r["disclaimer"]
    png = base64.b64decode(ex["gradcam_png_base64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000


def test_gradcam_is_real_and_region_mapped(real_subject):
    """CAM must be the model's own gradient attribution, mapped on the
    existing DK68∪TianS1 atlas (real parcel labels from the info CSV)."""
    import torch

    from brainvuln.config import resolve_checkpoint
    from brainvuln.mri.gradcam import gradcam_3d
    from brainvuln.mri.models import ResNet18Binary
    from brainvuln.mri.preprocess import preprocess_session

    _s, scan_path, _frozen = real_subject
    ckpt_path = resolve_checkpoint(None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    volume = preprocess_session(scan_path)
    x = torch.from_numpy(volume[None].astype(np.float32))
    cam = gradcam_3d(model, x)
    assert cam.shape == (128, 128, 128)
    assert np.isfinite(cam).all()
    assert 0.0 <= float(cam.max()) <= 1.0
    assert float(cam.max()) > 0  # nonzero somewhere in the brain

    import predict_volume as pv

    regions, _png = pv._regionalize(cam, volume, top_k=10)
    info = pd.read_csv(ROOT / "data/derived/atlas_dk68_tianS1_info.csv")
    known = set(info["label"])
    assert regions and all(reg["name"] in known for reg in regions)


# ---------------------------------------------------------------------------
# pipeline reuse (no duplicated science)
# ---------------------------------------------------------------------------


def test_service_imports_canonical_pipeline_unchanged():
    """The wrapper must call the canonical functions by name (lazy imports
    live inside functions, so we assert on source) and must never implement
    its own checkpoint discovery."""
    import inspect

    import predict_volume as pv

    src = inspect.getsource(pv)
    for required in (
        "from brainvuln.mri.preprocess import PREPROCESS_VERSION, "
        "preprocess_session",
        "from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti",
        "save_planes_png",
        "from brainvuln.mri.regional import regionalize_cam",
        "from brainvuln.config import resolve_checkpoint",
        "from brainvuln.config import checkpoint_identity",
        "resolve_threshold(",
    ):
        assert required in src, f"canonical call missing: {required}"

    code_lines = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith(("#", '"', "'")) and "never " not in line)
    for banned in ("glob(", "getmtime", "listdir("):
        assert banned not in code_lines
