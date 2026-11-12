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

CUSTOM_CSS = """
.header-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #fff;
    padding: 24px 32px;
    border-radius: 12px;
    margin-bottom: 16px;
    text-align: center;
}
.header-banner h1 {
    font-size: 1.8rem;
    margin: 0 0 8px 0;
    color: #e0e0e0;
}
.header-banner p {
    font-size: 0.9rem;
    color: #b0b0b0;
    margin: 0;
}
.disclaimer-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 6px;
    margin-top: 12px;
    font-size: 0.85rem;
    color: #856404;
}
.result-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
}
.probability-bar-container {
    background: #e9ecef;
    border-radius: 8px;
    height: 28px;
    overflow: hidden;
    margin: 8px 0;
}
.probability-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: bold;
    font-size: 0.85rem;
    min-width: 40px;
}
.badge-ad {
    background: #dc3545;
    color: #fff;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin: 8px 0;
}
.badge-normal {
    background: #28a745;
    color: #fff;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin: 8px 0;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.metric-item {
    background: #fff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.metric-item .label {
    font-size: 0.75rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-item .value {
    font-size: 1.2rem;
    font-weight: 700;
    color: #212529;
    margin-top: 4px;
}
.region-list {
    background: #fff;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
.region-list li {
    padding: 4px 0;
    border-bottom: 1px solid #f0f0f0;
}
.region-list li:last-child {
    border-bottom: none;
}
.upload-zone {
    border: 2px dashed #adb5bd;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    background: #f8f9fa;
}
.about-section {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 24px;
    margin: 12px 0;
}
.about-section h3 {
    color: #0f3460;
    border-bottom: 2px solid #dee2e6;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
"""


def _get_model():
    from brainvuln.config import (
        checkpoint_identity, resolve_checkpoint, resolve_threshold)
    from brainvuln.mri.models import ResNet18Binary, ResNet18CBAM, ResNet18WithDemographics
    ckpt_path = resolve_checkpoint()  # canonical configured checkpoint
    identity = checkpoint_identity()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Auto-detect model architecture from checkpoint state_dict keys
    state_keys = list(ckpt.get("state_dict", {}).keys())
    widen = ckpt.get("widen_factor", 1.0)

    has_cbam = any("att" in k and ("channel_att" in k or "spatial_att" in k)
                   for k in state_keys)
    has_demographics = any("demographics" in k or "age_proj" in k
                           for k in state_keys)

    if has_cbam:
        model = ResNet18CBAM(widen_factor=widen)
    elif has_demographics:
        model = ResNet18WithDemographics(widen_factor=widen)
    else:
        model = ResNet18Binary(widen_factor=widen)

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


def _brain_3d_plotly(volume: np.ndarray, cam: np.ndarray | None = None):
    """Create an interactive 3D brain visualization using Plotly.

    Uses marching cubes to extract the brain surface from the MRI volume,
    optionally colored by the Grad-CAM heatmap to show model attention.

    Args:
        volume: preprocessed brain volume [1,1,D,H,W] or [D,H,W].
        cam: Grad-CAM activation map [D,H,W] (optional).

    Returns:
        plotly.graph_objects.Figure ready for gr.Plotly component.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    vol = volume[0] if volume.ndim == 4 else volume  # [D,H,W]

    # Downsample for faster rendering (128³ → 32³ or 64³)
    target_size = 48
    orig_shape = vol.shape
    scale_factors = [target_size / s for s in orig_shape]
    import scipy.ndimage
    vol_ds = scipy.ndimage.zoom(vol, scale_factors, order=1)
    if cam is not None:
        cam_ds = scipy.ndimage.zoom(cam, scale_factors, order=1)
    else:
        cam_ds = None

    # Threshold to get brain surface (remove background)
    brain_mask = vol_ds > np.percentile(vol_ds, 15)

    # Use marching cubes via skimage for surface extraction
    try:
        from skimage.measure import marching_cubes
        # Invert so brain tissue is the "surface"
        surface_vol = vol_ds.astype(np.float64)
        threshold = np.percentile(surface_vol[brain_mask], 20)
        verts, faces, normals, _ = marching_cubes(
            surface_vol, level=threshold, allow_degenerate=True)
    except ImportError:
        # Fallback: simple isosurface from non-zero voxels
        verts, faces, normals = _simple_isosurface(brain_mask)

    if len(verts) == 0 or len(faces) == 0:
        # Fallback: scatter plot of brain voxels
        return _brain_3d_fallback(vol_ds, cam_ds)

    # Rescale vertices to original grid coordinates
    verts_scaled = verts / np.array(scale_factors)

    # Color mapping
    if cam_ds is not None and cam_ds.size > 0:
        # Interpolate CAM values onto vertices
        import scipy.ndimage
        cam_values = scipy.ndimage.map_coordinates(
            cam_ds, verts_scaled.T, order=1, mode="nearest")
        colors = cam_values
        color_title = "Grad-CAM (model attention)"
        color_scale = "YlOrRd"
    else:
        # Color by intensity
        intensity_values = scipy.ndimage.map_coordinates(
            vol_ds, verts_scaled.T, order=1, mode="nearest")
        colors = intensity_values
        color_title = "MRI intensity"
        color_scale = "Greys"

    # Create 3D mesh
    fig = go.Figure(data=go.Mesh3d(
        x=verts_scaled[:, 0],
        y=verts_scaled[:, 1],
        z=verts_scaled[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=colors,
        intensitymode="vertex",
        colorscale=color_scale,
        opacity=0.9,
        flatshading=False,
        name="brain",
    ))

    # Add wireframe overlay for structure
    fig.add_trace(go.Mesh3d(
        x=verts_scaled[:, 0],
        y=verts_scaled[:, 1],
        z=verts_scaled[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color="rgba(100,100,100,0.1)",
        opacity=0.1,
        flatshading=True,
        showscale=False,
        name="wireframe",
    ))

    fig.update_layout(
        title={
            "text": "🧠 3D Brain Visualization",
            "font": {"size": 18, "color": "#0f3460"},
        },
        scene=dict(
            xaxis_title="D (slice)",
            yaxis_title="H (row)",
            zaxis_title="W (col)",
            xaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            yaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            zaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=0.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
    )

    return fig


def _simple_isosurface(mask: np.ndarray):
    """Fallback isosurface extraction without skimage."""
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    # Simple surface: just return boundary voxels
    surface = np.zeros_like(mask)
    for d in range(-1, 2):
        for h in range(-1, 2):
            for w in range(-1, 2):
                if d == 0 and h == 0 and w == 0:
                    continue
                shifted = np.zeros_like(mask)
                dd, hh, ww = max(d, 0), max(h, 0), max(w, 0)
                if d >= 0 and h >= 0 and w >= 0:
                    shifted[dd:, hh:, ww:] = mask[:dd or None, :hh or None, :ww or None]
                elif d < 0 and h >= 0 and w >= 0:
                    shifted[:-abs(d), hh:, ww:] = mask[abs(d):, :hh or None, :ww or None]
                elif d >= 0 and h < 0 and w >= 0:
                    shifted[dd:, :-abs(h), ww:] = mask[:dd or None, abs(h):, :ww or None]
                elif d >= 0 and h >= 0 and w < 0:
                    shifted[dd:, hh:, :-abs(w)] = mask[:dd or None, :hh or None, abs(w):]
                elif d < 0 and h < 0 and w >= 0:
                    shifted[:-abs(d), :-abs(h), ww:] = mask[abs(d):, abs(h):, :ww or None]
                elif d < 0 and h >= 0 and w < 0:
                    shifted[:-abs(d), hh:, :-abs(w)] = mask[abs(d):, :hh or None, abs(w):]
                elif d >= 0 and h < 0 and w < 0:
                    shifted[dd:, :-abs(h), :-abs(w)] = mask[:dd or None, abs(h):, abs(w):]
                else:
                    shifted[:-abs(d), :-abs(h), :-abs(w)] = mask[abs(d):, abs(h):, abs(w):]
                surface |= (mask & ~shifted)
    coords = np.where(surface)
    verts = np.column_stack(coords).astype(np.float64)
    return verts, np.empty((0, 3)), np.empty((0, 3))


def _brain_3d_fallback(vol_ds: np.ndarray, cam_ds: np.ndarray | None):
    """Fallback 3D visualization using scatter plot when marching cubes fails."""
    import plotly.graph_objects as go

    # Sample brain voxels (downsample for performance)
    brain_mask = vol_ds > np.percentile(vol_ds, 15)
    coords = np.where(brain_mask)

    # Subsample for performance (max 5000 points)
    max_points = 5000
    if len(coords[0]) > max_points:
        indices = np.random.choice(len(coords[0]), max_points, replace=False)
        coords = (coords[0][indices], coords[1][indices], coords[2][indices])

    if cam_ds is not None:
        colors = cam_ds[coords]
        color_title = "Grad-CAM"
        color_scale = "YlOrRd"
    else:
        colors = vol_ds[coords]
        color_title = "Intensity"
        color_scale = "Greys"

    fig = go.Figure(data=go.Scatter3d(
        x=coords[0], y=coords[1], z=coords[2],
        mode="markers",
        marker=dict(
            size=2,
            color=colors,
            colorscale=color_scale,
            opacity=0.6,
            colorbar=dict(title=color_title),
        ),
    ))

    fig.update_layout(
        title="🧠 3D Brain (voxel scatter)",
        scene=dict(
            xaxis_title="D", yaxis_title="H", zaxis_title="W",
            aspectmode="data",
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


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


def _probability_bar_html(prob: float, threshold: float) -> str:
    """Render an HTML probability bar with color coding."""
    color = "#dc3545" if prob >= threshold else "#28a745"
    pct = int(prob * 100)
    return (
        f'<div class="probability-bar-container">'
        f'<div class="probability-bar-fill" style="width:{max(pct,5)}%;background:{color};">'
        f'{pct}%</div></div>'
    )


def _badge_html(label: str) -> str:
    """Render a classification badge."""
    if "Alzheimer" in label:
        return f'<div class="badge-ad">{label}</div>'
    else:
        return f'<div class="badge-normal">{label}</div>'


def predict_upload(file, use_tta):
    if file is None:
        return (DISCLAIMER, None, None, None, None)
    from brainvuln.mri.preprocess import (
        PREPROCESS_VERSION, preprocess_session)
    from brainvuln.mri.gradcam import gradcam_auto, save_cam_nifti, save_planes_png

    model, ckpt_path, threshold, identity = _get_model()
    sha = identity["sha256"]
    path = Path(file)
    try:
        volume = preprocess_session(path)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"❌ Could not process {path.name}: {exc}"
        return (error_msg, None, None, None, None)
    x = torch.from_numpy(volume[None].astype(np.float32))  # [1,1,D,H,W]

    # Load temperature calibration if available
    temperature = 1.0
    try:
        from brainvuln.config import resolve_temperature
        temperature = resolve_temperature(ckpt_path=str(ckpt_path))
    except (FileNotFoundError, ValueError, TypeError):
        temperature = 1.0

    # Inference: single pass or Test-Time Augmentation
    if use_tta:
        from predict import tta_inference
        prob = float(tta_inference(model, x, num_views=5).item())
    else:
        with torch.no_grad():
            logit = float(model(x).item())
            prob = float(torch.sigmoid(torch.tensor(logit / temperature)).item())

    label = ("Alzheimer's disease (model prediction)" if prob >= threshold
             else "cognitively normal (model prediction)")
    cam = gradcam_auto(model, x)
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    stem = path.stem.replace(".nii", "")
    save_cam_nifti(cam, path, out / f"{stem}_gradcam.nii.gz")
    png = save_planes_png(cam, volume, out / f"{stem}_gradcam.png")
    mri_png = _mri_png(volume, out)

    top = _top_regions(out / f"{stem}_gradcam.nii.gz")
    regions_html = ""
    if top is not None:
        atlas = None
        try:
            import pandas as pd
            atlas = pd.read_csv("data/derived/atlas_dk68_tianS1_info.csv")
            struct = dict(zip(atlas["label"].astype(str),
                              atlas["structure"].astype(str)))
        except Exception:  # noqa: BLE001
            struct = {}
        items = "\n".join(
            f'<li><code>{k}</code> ({struct.get(str(k), "atlas parcel")})'
            f': relevance {v:.3f}</li>'
            for k, v in top.items())
        regions_html = (
            '<div class="region-list">'
            '<strong>Top regions by model-derived relevance</strong> '
            '<em>(atlas-derived, research explanation — not biology)</em>'
            f'<ul>{items}</ul></div>')
    else:
        regions_html = (
            '<div class="region-list">'
            'Regional relevance unavailable (atlas files missing in this '
            'checkout); the CAM volume is still saved.</div>')

    # Build structured result HTML
    tta_note = "" if not use_tta else "<br><em>(Test-Time Augmentation: 6 views averaged)</em>"
    calib_note = "" if temperature == 1.0 else f"<br><em>(Temperature calibrated: T={temperature:.2f})</em>"
    model_arch = type(model).__name__
    result_html = (
        f'<div class="result-card">'
        f'<h3 style="margin-top:0;color:#0f3460;">Prediction Results</h3>'
        f'{_badge_html(label)}'
        f'<div class="metric-grid">'
        f'<div class="metric-item"><div class="label">Probability</div>'
        f'<div class="value">{prob:.1%}</div></div>'
        f'<div class="metric-item"><div class="label">Threshold</div>'
        f'<div class="value">{threshold:.2f}</div></div>'
        f'<div class="metric-item"><div class="label">Model</div>'
        f'<div class="value" style="font-size:0.85rem;">{model_arch}{tta_note}{calib_note}</div></div>'
        f'</div>'
        f'<div style="margin:12px 0;"><strong>Confidence bar:</strong></div>'
        f'{_probability_bar_html(prob, threshold)}'
        f'{regions_html}'
        f'<div style="margin-top:16px;font-size:0.8rem;color:#6c757d;">'
        f'Checkpoint SHA256: <code>{sha}</code> | '
        f'Preprocessing: {PREPROCESS_VERSION}</div>'
        f'</div>'
    )

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

    # Generate 3D brain visualization
    brain_3d_fig = _brain_3d_plotly(volume, cam)

    return (result_html, str(mri_png), str(png), brain_3d_fig,
            str(out / "prediction.json"), str(out / f"{stem}_gradcam.nii.gz"))


def _brain_3d_change_plotly(diff_map: np.ndarray, abs_diff_map: np.ndarray,
                             brain_mask: np.ndarray):
    """Create an interactive 3D visualization of brain changes between two time points.

    Uses marching cubes to extract the brain surface, colored by the difference
    map: red = atrophy (tissue loss), blue = growth (tissue gain), gray = stable.

    Args:
        diff_map: signed difference map [D,H,W] (T2 - T1).
        abs_diff_map: absolute difference map [D,H,W].
        brain_mask: combined brain mask [D,H,W].

    Returns:
        plotly.graph_objects.Figure for gr.Plot component.
    """
    import plotly.graph_objects as go

    # Downsample for rendering
    target_size = 48
    orig_shape = diff_map.shape
    scale_factors = [target_size / s for s in orig_shape]
    import scipy.ndimage
    diff_ds = scipy.ndimage.zoom(diff_map, scale_factors, order=1)
    abs_diff_ds = scipy.ndimage.zoom(abs_diff_map, scale_factors, order=1)
    mask_ds = scipy.ndimage.zoom(brain_mask.astype(np.float32), scale_factors, order=1) > 0.5

    # Extract surface using marching cubes
    try:
        from skimage.measure import marching_cubes
        surface_vol = abs_diff_ds.astype(np.float64)
        threshold = np.percentile(surface_vol[mask_ds], 10) if mask_ds.any() else 0
        verts, faces, normals, _ = marching_cubes(
            surface_vol, level=threshold, allow_degenerate=True)
    except ImportError:
        verts, faces, normals = _simple_isosurface(mask_ds)

    if len(verts) == 0 or len(faces) == 0:
        # Fallback scatter plot
        coords = np.where(mask_ds)
        max_points = 3000
        if len(coords[0]) > max_points:
            indices = np.random.choice(len(coords[0]), max_points, replace=False)
            coords = (coords[0][indices], coords[1][indices], coords[2][indices])

        fig = go.Figure(data=go.Scatter3d(
            x=coords[0], y=coords[1], z=coords[2],
            mode="markers",
            marker=dict(size=2, color="gray", opacity=0.5),
        ))
        fig.update_layout(
            title="🧠 3D Brain Change (scatter fallback)",
            scene=dict(xaxis_title="D", yaxis_title="H", zaxis_title="W", aspectmode="data"),
            height=500, margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    # Rescale vertices to original grid coordinates
    verts_scaled = verts / np.array(scale_factors)

    # Interpolate diff values onto vertices for coloring
    diff_values = scipy.ndimage.map_coordinates(
        diff_ds, verts_scaled.T, order=1, mode="nearest")

    # Create custom colormap: blue (growth) -> gray (stable) -> red (atrophy)
    # Normalize diff values to [-1, 1] range for coloring
    max_abs = np.max(np.abs(diff_values)) if np.max(np.abs(diff_values)) > 0 else 1.0
    normalized = np.clip(diff_values / max_abs, -1, 1)

    # Custom colorscale: blue (growth, +1) -> gray (stable, 0) -> red (atrophy, -1)
    # Plotly expects colorscale from 0 to 1, so map normalized [-1,1] to [0,1]
    intensity = (normalized + 1) / 2.0  # map [-1,1] -> [0,1]

    # Custom colorscale: 0=blue(growth), 0.5=gray(stable), 1=red(atrophy)
    custom_colorscale = [
        [0, "rgb(30,144,255)"],    # blue = growth
        [0.5, "rgb(150,150,150)"], # gray = stable
        [1, "rgb(220,53,69)"],     # red = atrophy
    ]

    fig = go.Figure(data=go.Mesh3d(
        x=verts_scaled[:, 0],
        y=verts_scaled[:, 1],
        z=verts_scaled[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=intensity,
        intensitymode="vertex",
        colorscale=custom_colorscale,
        opacity=0.9,
        flatshading=False,
        name="brain_change",
        colorbar=dict(
            title="Change",
            tickvals=[0, 0.5, 1],
            ticktext=["Growth", "Stable", "Atrophy"],
        ),
    ))

    fig.update_layout(
        title={
            "text": "🧠 3D Brain Change Map (Time 1 → Time 2)",
            "font": {"size": 18, "color": "#0f3460"},
        },
        scene=dict(
            xaxis_title="D (slice)",
            yaxis_title="H (row)",
            zaxis_title="W (col)",
            xaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            yaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            zaxis=dict(showbackground=False, showticklabels=False, showgrid=False),
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333"),
    )

    return fig


def compare_longitudinal(file_t1, file_t2, use_tta):
    """Compare two MRI scans from the same patient at different time points.

    Args:
        file_t1: path to the first (earlier) MRI scan.
        file_t2: path to the second (later) MRI scan.
        use_tta: enable Test-Time Augmentation for inference.

    Returns:
        Tuple of (results_html, change_3d_fig).
    """
    from brainvuln.mri.preprocess import preprocess_session
    from brainvuln.mri.longitudinal import compare_scans

    # Validate inputs
    if file_t1 is None or file_t2 is None:
        error_html = (
            '<div class="result-card">'
            '<h3 style="color:#dc3545;">⚠️ Missing Input</h3>'
            '<p>Please upload both Time 1 (earlier) and Time 2 (later) scans.</p>'
            '</div>')
        return (error_html, None)

    path_t1 = Path(file_t1) if isinstance(file_t1, str) else Path(file_t1.name)
    path_t2 = Path(file_t2) if isinstance(file_t2, str) else Path(file_t2.name)

    # Preprocess both scans
    try:
        volume_t1 = preprocess_session(path_t1)
    except Exception as exc:
        error_html = (
            f'<div class="result-card">'
            f'<h3 style="color:#dc3545;">⚠️ Error processing Time 1 scan</h3>'
            f'<p>{str(exc)}</p></div>')
        return (error_html, None)

    try:
        volume_t2 = preprocess_session(path_t2)
    except Exception as exc:
        error_html = (
            f'<div class="result-card">'
            f'<h3 style="color:#dc3545;">⚠️ Error processing Time 2 scan</h3>'
            f'<p>{str(exc)}</p></div>')
        return (error_html, None)

    # Run model inference on both scans
    model, ckpt_path, threshold, identity = _get_model()

    # Load temperature calibration if available
    temperature = 1.0
    try:
        from brainvuln.config import resolve_temperature
        temperature = resolve_temperature(ckpt_path=str(ckpt_path))
    except (FileNotFoundError, ValueError, TypeError):
        temperature = 1.0

    x_t1 = torch.from_numpy(volume_t1[None].astype(np.float32))
    x_t2 = torch.from_numpy(volume_t2[None].astype(np.float32))

    if use_tta:
        from predict import tta_inference
        prob_t1 = float(tta_inference(model, x_t1, num_views=5).item())
        prob_t2 = float(tta_inference(model, x_t2, num_views=5).item())
    else:
        with torch.no_grad():
            logit_t1 = float(model(x_t1).item())
            logit_t2 = float(model(x_t2).item())
            prob_t1 = float(torch.sigmoid(torch.tensor(logit_t1 / temperature)).item())
            prob_t2 = float(torch.sigmoid(torch.tensor(logit_t2 / temperature)).item())

    # Run longitudinal comparison
    result = compare_scans(
        volume_t1, volume_t2,
        prob_t1=prob_t1, prob_t2=prob_t2,
        threshold=threshold,
    )

    # Build 3D change visualization
    combined_mask = result.brain_mask_t1 | result.brain_mask_t2
    change_3d_fig = _brain_3d_change_plotly(
        result.diff_map, result.abs_diff_map, combined_mask
    )

    # Build results HTML
    vol_change_class = "text-danger" if result.volume_change_pct < 0 else "text-success"
    vol_change_icon = "📉" if result.volume_change_pct < 0 else "📈"
    vol_change_label = "Atrophy (volume loss)" if result.volume_change_pct < 0 else "Volume gain"

    # Prediction change analysis
    pred_change_html = ""
    if result.prob_change is not None:
        if result.prob_change > 0.1:
            pred_status = (
                '<span style="color:#dc3545;font-weight:600;">'
                '⚠️ Significant increase in AD probability</span>')
        elif result.prob_change < -0.1:
            pred_status = (
                '<span style="color:#28a745;font-weight:600;">'
                '✅ Decrease in AD probability</span>')
        else:
            pred_status = (
                '<span style="color:#ffc107;font-weight:600;">'
                '➡️ Stable AD probability</span>')

        label_change = ""
        if result.label_t1 != result.label_t2:
            label_change = (
                f'<br><em>Classification changed from '
                f'<strong>{result.label_t1}</strong> to '
                f'<strong>{result.label_t2}</strong></em>')

        pred_change_html = f'''
        <div class="metric-item">
            <div class="label">Prediction Change</div>
            <div class="value">{pred_status}{label_change}</div>
        </div>
        '''

    # Regional changes table
    regions_html = ""
    if result.regional_changes:
        regions_html = '<h4 style="margin-top:20px;">Top Changing Regions</h4>'
        regions_html += '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        regions_html += (
            '<tr style="background:#0f3460;color:#fff;">'
            '<th style="padding:6px 8px;text-align:left;">Region</th>'
            '<th style="padding:6px 8px;text-align:center;">Change Type</th>'
            '<th style="padding:6px 8px;text-align:center;">Mean Change</th>'
            '<th style="padding:6px 8px;text-align:center;">Max Change</th>'
            '</tr>')

        for region in result.regional_changes[:8]:
            type_color = {
                "atrophy": "#dc3545",
                "growth": "#007bff",
                "stable": "#6c757d",
            }.get(region["change_type"], "#6c757d")

            type_icon = {
                "atrophy": "📉",
                "growth": "📈",
                "stable": "➡️",
            }.get(region["change_type"], "➡️")

            regions_html += (
                f'<tr style="border-bottom:1px solid #dee2e6;">'
                f'<td style="padding:6px 8px;">{region["name"]}</td>'
                f'<td style="padding:6px 8px;text-align:center;color:{type_color};'
                f'font-weight:600;">{type_icon} {region["change_type"].title()}</td>'
                f'<td style="padding:6px 8px;text-align:center;">'
                f'{region["mean_change"]:+.4f}</td>'
                f'<td style="padding:6px 8px;text-align:center;">'
                f'{region["max_abs_change"]:.4f}</td>'
                f'</tr>')
        regions_html += '</table>'

    result_html = f'''
    <div class="result-card">
        <h3 style="margin-top:0;color:#0f3460;">
            📊 Longitudinal Comparison Results
        </h3>
        <p style="font-size:0.85rem;color:#6c757d;">
            Comparing <strong>{path_t1.name}</strong> (Time 1) → 
            <strong>{path_t2.name}</strong> (Time 2)
        </p>

        <div class="metric-grid" style="display:grid;
            grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
            gap:12px; margin:16px 0;">
            <div class="metric-item">
                <div class="label">Brain Volume (Time 1)</div>
                <div class="value">{result.volume_t1:,.0f} mm³</div>
            </div>
            <div class="metric-item">
                <div class="label">Brain Volume (Time 2)</div>
                <div class="value">{result.volume_t2:,.0f} mm³</div>
            </div>
            <div class="metric-item">
                <div class="label">Volume Change</div>
                <div class="value" style="color:{"#dc3545" if result.volume_change_pct < 0 else "#28a745"};
                    font-size:1.3rem;">{vol_change_icon} {result.volume_change_pct:+.2f}%</div>
                <div style="font-size:0.75rem;color:#6c757d;">{vol_change_label}</div>
            </div>
            <div class="metric-item">
                <div class="label">Brain Overlap</div>
                <div class="value">{result.overlap_pct:.1f}%</div>
            </div>
            <div class="metric-item">
                <div class="label">AD Probability (Time 1)</div>
                <div class="value">{result.prob_t1:.1%}</div>
                <div style="font-size:0.75rem;color:#6c757d;">
                    Label: {result.label_t1 or "—"}</div>
            </div>
            <div class="metric-item">
                <div class="label">AD Probability (Time 2)</div>
                <div class="value">{result.prob_t2:.1%}</div>
                <div style="font-size:0.75rem;color:#6c757d;">
                    Label: {result.label_t2 or "—"}</div>
            </div>
            {pred_change_html}
        </div>

        {regions_html}

        <div style="margin-top:16px;font-size:0.8rem;color:#6c757d;">
            <strong>Note:</strong> Volume change reflects differences in brain 
            mask extraction between two independently preprocessed scans. 
            For precise volumetric tracking, scans should be co-registered 
            before comparison. Probability change reflects the model's 
            assessment of Alzheimer's risk at each time point.
        </div>
        <div style="margin-top:8px;font-size:0.8rem;color:#6c757d;">
            Model: {type(model).__name__} | 
            Checkpoint: {identity.get("sha256", "unknown")[:12]}... |
            Threshold: {threshold:.2f}
        </div>
    </div>
    '''

    return (result_html, change_3d_fig)


def predict_batch(files, use_tta):
    """Process multiple MRI files and return a summary table."""
    if files is None or len(files) == 0:
        return "<p>No files uploaded.</p>"

    from brainvuln.mri.preprocess import preprocess_session
    from brainvuln.mri.gradcam import gradcam_auto

    model, ckpt_path, threshold, identity = _get_model()

    # Load temperature calibration if available
    temperature = 1.0
    try:
        from brainvuln.config import resolve_temperature
        temperature = resolve_temperature(ckpt_path=str(ckpt_path))
    except (FileNotFoundError, ValueError, TypeError):
        temperature = 1.0

    results = []

    for f in files:
        path = Path(f) if isinstance(f, str) else Path(f.name)
        try:
            volume = preprocess_session(path)
        except Exception as exc:
            results.append((path.name, "Error", str(exc)[:60], "—", "—"))
            continue

        x = torch.from_numpy(volume[None].astype(np.float32))

        if use_tta:
            from predict import tta_inference
            prob = float(tta_inference(model, x, num_views=5).item())
        else:
            with torch.no_grad():
                logit = float(model(x).item())
                prob = float(torch.sigmoid(torch.tensor(logit / temperature)).item())

        label = "AD" if prob >= threshold else "Normal"
        results.append((
            path.name,
            label,
            f"{prob:.1%}",
            f"{threshold:.2f}",
            type(model).__name__
        ))

    # Build HTML table
    html = (
        '<div class="result-card">'
        '<h3 style="margin-top:0;color:#0f3460;">Batch Results '
        f'({len(results)} scans)</h3>'
        '<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">'
        '<tr style="background:#0f3460;color:#fff;">'
        '<th style="padding:8px;text-align:left;">File</th>'
        '<th style="padding:8px;text-align:center;">Prediction</th>'
        '<th style="padding:8px;text-align:center;">Probability</th>'
        '<th style="padding:8px;text-align:center;">Threshold</th>'
        '<th style="padding:8px;text-align:center;">Model</th>'
        '</tr>'
    )
    for fname, label, prob_str, thr_str, mname in results:
        bg = "#ffeaea" if label == "AD" else "#eaffea"
        html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 8px;">{fname}</td>'
            f'<td style="padding:6px 8px;text-align:center;font-weight:600;">'
            f'{label}</td>'
            f'<td style="padding:6px 8px;text-align:center;">{prob_str}</td>'
            f'<td style="padding:6px 8px;text-align:center;">{thr_str}</td>'
            f'<td style="padding:6px 8px;text-align:center;">{mname}</td>'
            f'</tr>'
        )
    html += '</table></div>'
    return html


def build_ui():
    with gr.Blocks(title="BrainVuln — Alzheimer's MRI Classifier") as ui:
        # Header
        with gr.Row():
            gr.HTML(
                '<div class="header-banner">'
                '<h1>🧠 BrainVuln — Alzheimer\'s MRI Classifier</h1>'
                '<p>Upload a T1-weighted brain MRI scan for AI-powered '
                'Alzheimer\'s disease risk assessment</p>'
                '</div>')

        with gr.Tabs():
            # ─── Upload Tab ───
            with gr.TabItem("📤 Upload & Predict"):
                gr.Markdown(
                    "Upload a **T1-weighted brain MRI** scan (`.nii` or "
                    "`.nii.gz` format). The model will analyze the scan and "
                    "provide a prediction with visual explanations.")

                with gr.Row():
                    with gr.Column(scale=1):
                        inp = gr.File(
                            label="Upload MRI Scan",
                            file_types=[".nii", ".gz", ".img"],
                            type="filepath")
                        tta_checkbox = gr.Checkbox(
                            label="🔥 Use Test-Time Augmentation (more accurate, slower)",
                            value=False,
                            info="Averages 6 augmented views for +1-2% accuracy")
                        btn = gr.Button(
                            "🔬 Run Analysis",
                            variant="primary",
                            size="lg")
                        gr.Markdown(
                            '<div class="disclaimer-box">'
                            '⚠️ ' + DISCLAIMER + '</div>')
                        gr.Markdown(
                            '<div style="margin-top:12px;font-size:0.85rem;'
                            'color:#6c757d;">' + SUPPORTED_INPUT_NOTE +
                            '</div>')

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "### How it works\n"
                            "1. **Upload** your T1-weighted MRI scan\n"
                            "2. **Preprocessing** — the scan is normalized "
                            "to 2mm resolution and 128³ volume\n"
                            "3. **Inference** — a 3D ResNet-18 model "
                            "analyzes the brain structures\n"
                            "4. **Results** — you'll see the prediction, "
                            "confidence, and highlighted regions\n\n"
                            "The entire process takes about 30-60 seconds.")

            # ─── Batch Upload Tab ───
            with gr.TabItem("📦 Batch Upload"):
                gr.Markdown(
                    "Upload **multiple MRI scans** at once for batch analysis. "
                    "All scans are processed sequentially and results are shown "
                    "in a summary table.")

                batch_inp = gr.File(
                    label="Upload Multiple MRI Scans",
                    file_types=[".nii", ".gz", ".img"],
                    type="filepath",
                    file_count="multiple")
                batch_tta = gr.Checkbox(
                    label="🔥 Use Test-Time Augmentation (more accurate, slower)",
                    value=False,
                    info="Averages 6 augmented views for +1-2% accuracy")
                batch_btn = gr.Button(
                    "🔬 Run Batch Analysis",
                    variant="primary",
                    size="lg")
                batch_out = gr.HTML(
                    value='<div style="text-align:center;padding:40px;'
                    'color:#adb5bd;">Upload multiple scans and run batch '
                    'analysis to see results here.</div>')
                gr.Markdown(
                    '<div class="disclaimer-box">'
                    '⚠️ ' + DISCLAIMER + '</div>')

                batch_btn.click(
                    predict_batch,
                    inputs=[batch_inp, batch_tta],
                    outputs=[batch_out])

            # ─── Time Series Tab ───
            with gr.TabItem("📈 Time Series Tracking"):
                gr.Markdown(
                    "### 🧠 Longitudinal Brain Change Tracking\n\n"
                    "Upload **two MRI scans from the same patient** at different "
                    "time points to track brain changes over time. This is useful "
                    "for monitoring Alzheimer's disease progression.\n\n"
                    "**What you'll see:**\n"
                    "- Brain volume change between the two scans\n"
                    "- 3D visualization showing atrophy (red) and growth (blue) regions\n"
                    "- Model prediction comparison (AD probability at each time point)\n"
                    "- Regional analysis of where the most change occurred")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Time 1 (Earlier Scan)")
                        ts_inp_t1 = gr.File(
                            label="Upload Earlier MRI Scan",
                            file_types=[".nii", ".gz", ".img"],
                            type="filepath")
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Time 2 (Later Scan)")
                        ts_inp_t2 = gr.File(
                            label="Upload Later MRI Scan",
                            file_types=[".nii", ".gz", ".img"],
                            type="filepath")

                with gr.Row():
                    ts_tta = gr.Checkbox(
                        label="🔥 Use Test-Time Augmentation (more accurate, slower)",
                        value=False,
                        info="Averages 6 augmented views for +1-2% accuracy")
                    ts_btn = gr.Button(
                        "🔬 Compare Scans",
                        variant="primary",
                        size="lg")

                gr.Markdown(
                    '<div class="disclaimer-box">'
                    '⚠️ ' + DISCLAIMER + '</div>')

                gr.Markdown(
                    "### 📊 Comparison Results\n"
                    "Results will appear here after you upload both scans and click Compare.")

                ts_out_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;'
                    'color:#adb57d;">Upload two scans from the same patient '
                    'at different time points to see comparison results here.</div>')

                gr.Markdown(
                    "### 🧠 3D Brain Change Map\n"
                    "Interactive 3D view showing where the brain has changed. "
                    "**Red = atrophy** (tissue loss), **Blue = growth**, "
                    "**Gray = stable**. Click and drag to rotate, scroll to zoom.")

                ts_out_3d = gr.Plot(
                    label="3D Brain Change Map",
                    elem_id="brain-change-3d-plot")

                ts_btn.click(
                    compare_longitudinal,
                    inputs=[ts_inp_t1, ts_inp_t2, ts_tta],
                    outputs=[ts_out_html, ts_out_3d])

            # ─── Results Tab ───
            with gr.TabItem("📊 Results"):
                out_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;'
                    'color:#adb57d;">Upload and run an analysis to see '
                    'results here.</div>')

                # 3D Brain Visualization
                gr.Markdown(
                    "### 🧠 3D Brain Visualization\n"
                    "Interactive 3D view of the brain with Grad-CAM heatmap "
                    "overlay. **Click and drag to rotate**, scroll to zoom.")
                out_3d = gr.Plot(
                    label="3D Brain",
                    elem_id="brain-3d-plot")

                with gr.Row():
                    with gr.Column():
                        out_mri = gr.Image(
                            label="🔍 MRI Visualization "
                            "(axial / sagittal / coronal)")
                    with gr.Column():
                        out_img = gr.Image(
                            label="🔥 Grad-CAM Heatmap "
                            "(model attention map)")
                with gr.Row():
                    out_json = gr.File(label="📄 Download prediction.json")
                    out_cam = gr.File(
                        label="🧩 Download Grad-CAM NIfTI")

            # ─── About Tab ───
            with gr.TabItem("ℹ️ About"):
                gr.HTML(
                    '<div class="about-section">'
                    '<h3>About BrainVuln</h3>'
                    '<p>BrainVuln is a research prototype for detecting '
                    'Alzheimer\'s disease from T1-weighted brain MRI scans '
                    'using deep learning.</p>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Model Architecture</h3>'
                    '<ul>'
                    '<li><strong>Backbone:</strong> 3D ResNet-18 '
                    '(MONAI framework)</li>'
                    '<li><strong>Parameters:</strong> ~33 million</li>'
                    '<li><strong>Input:</strong> 128×128×128 voxel '
                    'volumes at 2mm resolution</li>'
                    '<li><strong>Output:</strong> Binary classification '
                    '(Alzheimer\'s vs. cognitively normal)</li>'
                    '</ul>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Training Data</h3>'
                    '<ul>'
                    '<li><strong>Dataset:</strong> OASIS-1 (Open '
                    'Access Series of Imaging Studies)</li>'
                    '<li><strong>Task:</strong> Distinguish Alzheimer\'s '
                    'disease from cognitively normal subjects</li>'
                    '<li><strong>Preprocessing:</strong> RAS reorient → '
                    '2mm resample → tight brain crop → 128³ pad/crop → '
                    'brain z-score normalization</li>'
                    '</ul>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Performance Metrics</h3>'
                    '<div class="metric-grid">'
                    '<div class="metric-item">'
                    '<div class="label">Test Accuracy</div>'
                    '<div class="value">66.7%</div></div>'
                    '<div class="metric-item">'
                    '<div class="label">ROC-AUC</div>'
                    '<div class="value">0.817</div></div>'
                    '<div class="metric-item">'
                    '<div class="label">Sensitivity</div>'
                    '<div class="value">86.7%</div></div>'
                    '<div class="metric-item">'
                    '<div class="label">Specificity</div>'
                    '<div class="value">41.7%</div></div>'
                    '</div>'
                    '<p style="margin-top:16px;font-size:0.85rem;'
                    'color:#6c757d;">Metrics measured on a 27-subject '
                    'test set from OASIS-1. Accuracy can be improved '
                    'through retraining with demographics fusion, wider '
                    'models, and ensembling — see '
                    '<code>ACCURACY_IMPROVEMENTS.md</code> for details.</p>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Grad-CAM Explanation</h3>'
                    '<p>Grad-CAM (Gradient-weighted Class Activation '
                    'Mapping) highlights which brain regions the model '
                    'focused on when making its prediction. Warmer colors '
                    'indicate higher model attention.</p>'
                    '<p>This is a <strong>research explanation tool</strong> '
                    '— it shows where the model looked, not necessarily '
                    'where pathology exists.</p>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Limitations</h3>'
                    '<ul>'
                    '<li>Trained on a small single-site research cohort '
                    '(OASIS-1)</li>'
                    '<li>Not validated on external datasets (ADNI, OASIS-3)</li>'
                    '<li>Low specificity (41.7%) — high false positive rate</li>'
                    '<li>Not intended for clinical use</li>'
                    '<li>Results may vary significantly across scanners '
                    'and protocols</li>'
                    '</ul>'
                    '</div>')

                gr.HTML(
                    '<div class="about-section">'
                    '<h3>Technical Details</h3>'
                    '<ul>'
                    '<li><strong>Framework:</strong> PyTorch + MONAI</li>'
                    '<li><strong>Preprocessing:</strong> nibabel, nilearn</li>'
                    '<li><strong>Visualization:</strong> Gradio web UI</li>'
                    '<li><strong>Service API:</strong> FastAPI backend '
                    'available in <code>service/</code></li>'
                    '</ul>'
                    '<p style="margin-top:12px;">'
                    '<strong>Repository:</strong> '
                    '<a href="https://github.com/kanishk-thakur/brain-vault" '
                    'target="_blank">brain-vault</a></p>'
                    '</div>')

                gr.HTML(
                    '<div class="disclaimer-box" style="margin-top:20px;">'
                    '<strong>⚠️ Research Use Only</strong><br>'
                    'This tool is a research prototype. It is NOT a '
                    'clinical diagnostic system. Outputs must NOT inform '
                    'any clinical decision. Consult a qualified healthcare '
                    'professional for medical advice.</div>')

        # Wire up the button
        btn.click(
            predict_upload,
            inputs=[inp, tta_checkbox],
            outputs=[out_html, out_mri, out_img, out_3d, out_json, out_cam])

    return ui


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7861,
                      show_error=True, css=CUSTOM_CSS, theme=gr.themes.Soft())
