"""BrainVuln inference API (research use only).

Endpoints
---------
POST /api/predict     multipart upload of one .nii / .nii.gz T1 volume;
                      canonical preprocessing → frozen seed-42 checkpoint →
                      probability → real Grad-CAM → atlas regions.
GET  /api/health      liveness + checkpoint identity (no weights loaded).
GET  /api/model-info  frozen checkpoint / threshold / atlas details.

The scientific pipeline is imported unchanged from the validated BrainVuln
package (see service/predict_volume.py). Nothing here re-implements
preprocessing, inference, thresholding, or Grad-CAM.

CORS is restricted to the configured origins (BRAINVULN_CORS_ORIGINS,
comma-separated; localhost development origins always allowed).
Uploaded volumes live in temporary files only and are deleted after each
request. Nothing about a request is persisted.

RESEARCH USE ONLY — not a clinical diagnostic system.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from predict_volume import (  # local module (same directory)
    CANONICAL_SHA256,
    DISCLAIMER,
    InvalidVolumeError,
    MAX_UPLOAD_MB,
    CheckpointIdentityError,
    load_nifti_or_raise,
    model_info,
    run_inference,
    validate_upload,
)

# deployed hosts: fetch the frozen checkpoint before serving (no-op locally
# where the artifact already exists; aborts loudly on hash mismatch)
try:
    from checkpoint_bootstrap import download_checkpoint_if_needed
    download_checkpoint_if_needed()
except Exception as _exc:  # noqa: BLE001 - startup must fail loudly but cleanly
    print(f"[startup] checkpoint bootstrap failed: {_exc}", file=__import__("sys").stderr)


app = FastAPI(
    title="BrainVuln inference API",
    version="1.0.0",
    description="Canonical BrainVuln model inference. Research use only — "
                "not a clinical diagnostic system.",
)


def _cors_origins() -> list[str]:
    raw = os.environ.get("BRAINVULN_CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


app.add_middleware(
    CORSMiddleware,
    # production origins come from the environment; localhost on any port is
    # always allowed for development
    allow_origins=_cors_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "BrainVuln inference API",
        "endpoints": ["/api/health", "/api/model-info", "/api/predict"],
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/health")
async def health():
    try:
        info = model_info()
        return {"status": "ok", "ready": True, "model": info}
    except (CheckpointIdentityError, FileNotFoundError) as exc:
        # the service is up but refuses to serve a non-canonical checkpoint
        return JSONResponse(status_code=503, content={
            "status": "degraded", "ready": False,
            "reason": str(exc),
            "expected_sha256": CANONICAL_SHA256,
        })


@app.get("/api/model-info")
async def model_info_endpoint():
    try:
        return {"success": True, "model": model_info()}
    except (CheckpointIdentityError, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """One MRI in, one structured research prediction out."""
    contents = await file.read()
    try:
        validate_upload(file.filename, len(contents))
    except InvalidVolumeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(contents) > MAX_UPLOAD_MB * (1 << 20):
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.")

    suffix = Path(file.filename or "upload.nii.gz").suffix or ".nii"
    if file.filename and file.filename.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"

    with tempfile.TemporaryDirectory(prefix="brainvuln_upload_") as td:
        tmp = Path(td) / f"upload{suffix}"
        tmp.write_bytes(contents)
        # structural validation happens here so malformed files never reach
        # the scientific pipeline and get a precise error instead
        try:
            load_nifti_or_raise(tmp)
            result = run_inference(tmp, original_name=file.filename)
        except InvalidVolumeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CheckpointIdentityError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=(
                f"Server configuration error: {exc}")) from exc
    return result
