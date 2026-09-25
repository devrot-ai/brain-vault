#!/usr/bin/env python
"""Web demo (Part AB): upload an MRI, get a probability + Grad-CAM.

    python app.py          # launches Gradio on the default port

SUPPORTED INPUT: the OASIS-1-style T1 volume the production pipeline was
validated on — ideally the OASIS-1 ``*_t88_masked_gfc`` derivative
(skull-stripped, T88-registered ANALYZE pair). A single .nii/.nii.gz T1
that is close to T88/MNI space is also accepted and is preprocessed by
the SAME canonical path (RAS reorient -> 2 mm -> crop/pad 128^3 ->
brain z-score), but results for non-OASIS images are unvalidated.
The UI states this explicitly; the backend does not pretend to support
arbitrary raw MRI formats.

RESEARCH USE ONLY. Not a clinical diagnostic system — the banner below is
part of the specification, not a formality.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import gradio as gr

SUPPORTED_INPUT_NOTE = (
    "**Supported input:** an OASIS-1-style T1 volume — a single `.nii` or "
    "`.nii.gz` T1 in T88/MNI-like space, or the `.img` file of an OASIS T88 "
    "ANALYZE pair when its `.hdr` sits in the same folder (local use). "
    "Everything goes through the identical canonical preprocessing (RAS "
    "reorient -> 2 mm -> crop/pad 128³ -> brain z-score). Images far from "
    "this acquisition/registration (other scanners, non-brain-stripped raw "
    "MRI) are **not validated inputs**; the model will return a number, but "
    "its research metrics were measured only on OASIS-1 T88 volumes.")

DISCLAIMER = ("**Research use only. Not a clinical diagnostic system.** "
              "Prototype trained on a small research cohort (OASIS-1); "
              "outputs must not inform any clinical decision.")


def _get_model():
    from brainvuln.config import (
        checkpoint_identity, resolve_checkpoint, resolve_threshold)
    from brainvuln.mri.models import ResNet18Binary
    ckpt_path = resolve_checkpoint()  # canonical configured checkpoint
    identity = checkpoint_identity()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold, _source = resolve_threshold(ckpt_path=ckpt_path)
    return model, ckpt_path, threshold, identity


def _mri_png(volume: np.ndarray, out: Path):
    """Axial/sagittal/coronal slices of the input volume (plain MRI view)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vol = volume[0]  # [D,H,W]
    dz, dy, dx = (int(v) for v in np.array(vol.shape) // 2)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, (img, title) in zip(axes, [
            (vol[dz], "axial"), (vol[:, dy, :], "sagittal"),
            (vol[:, :, dx], "coronal")]):
        ax.imshow(np.flipud(img.T), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    png = out / "mri_planes.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png


def _top_regions(cam_nii: Path, n=5):
    """Atlas-derived regional relevance for one CAM (NaN parcels dropped)."""
    from brainvuln.mri.regional import regionalize_cam
    atlas_img = Path("data/derived/atlas_dk68_tianS1.nii.gz")
    atlas_info = Path("data/derived/atlas_dk68_tianS1_info.csv")
    if not atlas_img.exists():
        return None
    s = regionalize_cam(cam_nii, atlas_img, atlas_info).dropna()
    if s.empty:
        return None
    return s.sort_values(ascending=False).head(n)


def predict_upload(file):
    if file is None:
        return DISCLAIMER, SUPPORTED_INPUT_NOTE, None, None, None
    from brainvuln.mri.preprocess import (
        PREPROCESS_VERSION, preprocess_session)
    from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti, save_planes_png

    model, ckpt_path, threshold, identity = _get_model()
    sha = identity["sha256"]
    path = Path(file)
    try:
        volume = preprocess_session(path)
    except Exception as exc:  # noqa: BLE001
        return (f"Could not process {path.name}: {exc}\n\n{DISCLAIMER}",
                SUPPORTED_INPUT_NOTE, None, None, None)
    x = torch.from_numpy(volume[None].astype(np.float32))  # [1,1,D,H,W]
    with torch.no_grad():
        prob = float(torch.sigmoid(model(x)).item())
    label = ("Alzheimer's disease (model prediction)" if prob >= threshold
             else "cognitively normal (model prediction)")
    cam = gradcam_3d(model, x)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    stem = path.stem.replace(".nii", "")
    save_cam_nifti(cam, path, out / f"{stem}_gradcam.nii.gz")
    png = save_planes_png(cam, volume, out / f"{stem}_gradcam.png")
    mri_png = _mri_png(volume, out)

    top = _top_regions(out / f"{stem}_gradcam.nii.gz")
    if top is not None:
        atlas = None
        try:
            import pandas as pd
            atlas = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv")
            struct = dict(zip(atlas["label"].astype(str),
                              atlas["structure"].astype(str)))
        except Exception:  # noqa: BLE001
            struct = {}
        regions_md = "\n".join(
            f"* `{k}` ({struct.get(str(k), 'atlas parcel')}): relevance {v:.3f}"
            for k, v in top.items())
        regions_md = ("**Top regions by model-derived relevance** "
                      "(atlas-derived, research explanation — not biology):\n"
                      + regions_md)
    else:
        regions_md = ("Regional relevance unavailable (atlas files missing "
                      "in this checkout); the CAM volume is still saved.")

    text = (f"### Prediction: {label}\n\n"
            f"**Probability:** {prob:.2f}  \n"
            f"**Threshold:** {threshold:.2f} (frozen, validation-selected)  \n"
            f"**Model:** BrainVuln 3D ResNet (ResNet18Binary)  \n"
            f"**Checkpoint SHA256:** `{sha}`  \n"
            f"**Preprocessing:** {PREPROCESS_VERSION}\n\n"
            f"{regions_md}\n\n{SUPPORTED_INPUT_NOTE}\n\n{DISCLAIMER}")

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image": str(path),
        "probability": prob,
        "label": label,
        "threshold": threshold,
        "threshold_source": "frozen validation threshold",
        "model": "BrainVuln-ResNet18Binary",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha,
        "preprocess_version": PREPROCESS_VERSION,
        "top_regions": ({} if top is None else
                        {str(k): float(v) for k, v in top.items()}),
        "disclaimer": ("Research use only. Not a clinical diagnostic system. "
                       "Trained on a small research cohort (OASIS-1); no "
                       "causal or clinical claims."),
    }
    (out / "prediction.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return text, str(mri_png), str(png), str(out / "prediction.json"), \
        str(out / f"{stem}_gradcam.nii.gz")


def build_ui():
    with gr.Blocks(title="BrainVuln demo") as ui:
        gr.Markdown("# BrainVuln — MRI classification demo\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.File(label="T1 MRI volume (.nii / .nii.gz, or OASIS .img)",
                          file_types=[".nii", ".gz", ".img"])
            with gr.Column():
                out_md = gr.Markdown()
        with gr.Row():
            with gr.Column():
                out_mri = gr.Image(label="MRI visualization "
                                         "(axial/sagittal/coronal)")
            with gr.Column():
                out_img = gr.Image(label="Grad-CAM visualization "
                                         "(model-derived relevance)")
        with gr.Row():
            out_json = gr.File(label="prediction.json")
            out_cam = gr.File(label="Grad-CAM NIfTI")
        btn = gr.Button("Run model")
        btn.click(predict_upload, inputs=inp,
                  outputs=[out_md, out_mri, out_img, out_json, out_cam])
        gr.Markdown(SUPPORTED_INPUT_NOTE)
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7861,
                      show_error=True)
