#!/usr/bin/env python
"""Web demo (Part AB): upload an MRI, get a probability + Grad-CAM.

    python app.py          # launches Gradio on the default port

RESEARCH USE ONLY. Not a clinical diagnostic system — the banner below is
part of the specification, not a formality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import gradio as gr

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


def predict_upload(file):
    if file is None:
        return DISCLAIMER, None, None
    from brainvuln.mri.preprocess import preprocess_session
    from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti, save_planes_png

    model, ckpt_path, threshold, identity = _get_model()
    sha = identity["sha256"]
    path = Path(file)
    try:
        volume = preprocess_session(path)
    except Exception as exc:  # noqa: BLE001
        return (f"Could not process {path.name}: {exc}\n\n{DISCLAIMER}",
                None, None)
    x = torch.from_numpy(volume[None].astype(np.float32))  # [1,1,D,H,W]
    with torch.no_grad():
        prob = float(torch.sigmoid(model(x)).item())
    label = ("Alzheimer's disease (model prediction)" if prob >= threshold
             else "cognitively normal (model prediction)")
    cam = gradcam_3d(model, x)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    stem = path.stem.replace(".nii", "")
    save_cam_nifti(cam, path, out / f"{stem}_gradcam.nii.gz")
    png = save_planes_png(cam, volume, out / f"{stem}_gradcam.png")
    text = (f"### Prediction: {label}\n\n"
            f"**Probability:** {prob:.2f}  \n"
            f"**Threshold:** {threshold:.2f}  \n"
            f"**Model:** BrainVuln-ResNet18Binary ({ckpt_path})\n"
            f"**Checkpoint sha256:** {sha}\n\n"
            "Top regions by model relevance are in "
            f"`outputs/{stem}_gradcam.nii.gz` (parcellate with "
            f"`brainvuln.mri.regional.regionalize_cam`).\n\n{DISCLAIMER}")
    return text, str(png), str(out / f"{stem}_gradcam.nii.gz")


def build_ui():
    with gr.Blocks(title="BrainVuln demo") as ui:
        gr.Markdown("# BrainVuln — MRI classification demo\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.File(label="T1 MRI volume (.nii / .nii.gz)",
                          file_types=[".nii", ".gz"])
            with gr.Column():
                out_md = gr.Markdown()
                out_img = gr.Image(label="Grad-CAM (axial/sagittal/coronal)")
                out_file = gr.File(label="Grad-CAM NIfTI")
        btn = gr.Button("Run model")
        btn.click(predict_upload, inputs=inp, outputs=[out_md, out_img, out_file])
    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7861,
                      show_error=True)
