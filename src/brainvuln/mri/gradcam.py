"""Explainable AI (Part K): 3D Grad-CAM + occlusion sensitivity.

Both methods produce per-subject relevance volumes on the model's *input*
grid (128³ at 2 mm), saved as NIfTI in the T88 (MNI-affine) space so they
can be parcellated with the same atlas used everywhere else in BrainVuln.

Grad-CAM is a gradient-weighted attention heuristic, **not** causal proof:
regions with high CAM are regions the network used, not regions where
pathology exists (Part AG language rules apply downstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Grad-CAM
# ---------------------------------------------------------------------------


@torch.enable_grad()
def gradcam_3d(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    layer_name: str = "layer4",
) -> np.ndarray:
    """Grad-CAM for a single input volume [1,1,D,H,W]; returns [D,H,W] in [0,1].

    Gradients of the positive-class logit are pooled over the last
    convolutional feature map (``layer4`` of the MONAI ResNet). The CAM is
    upsampled to the input grid and ReLU-activated.
    """
    model.eval()
    feat, _ = model.forward_features(volume)
    feat = feat.detach().requires_grad_(True)  # leaf so backward can fill .grad

    logits = model.fc(model.pool(feat).flatten(1)).squeeze(1)
    score = logits.sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    weights = feat.grad.detach().sum(dim=(2, 3, 4), keepdim=True)  # [1,C,1,1,1]
    cam = F.relu((weights * feat.detach()).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear",
                        align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    mx = float(cam.max())
    return cam / mx if mx > 0 else cam


class GradCAMHook:
    """Alternative hook-based CAM for models without forward_features."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)
        self.model = model

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def cam(self, volume: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(volume)
        score = logits.sum()
        score.backward()
        w = self.grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=volume.shape[2:], mode="trilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        mx = float(cam.max())
        return cam / mx if mx > 0 else cam


# ---------------------------------------------------------------------------
# occlusion sensitivity (independent method)
# ---------------------------------------------------------------------------


@torch.no_grad()
def occlusion_sensitivity(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    patch: int = 32,
    stride: int = 16,
    device: str | torch.device = "cpu",
    batch_size: int = 8,
) -> np.ndarray:
    """Probability-drop occlusion map (independent of gradients).

    Slides a cubic patch of zeros across the volume; the map value at the
    patch centre is the drop in disease probability when that patch is
    hidden. Occluded forwards are BATCHED (chunked) — a naive per-position
    loop takes hours on CPU. The strided drop grid is then trilinearly
    interpolated into a dense map. Patch/stride define the effective
    resolution of this check; Grad-CAM remains the fine-grained method.
    """
    model.eval()
    vol = volume.to(device)
    base = float(torch.sigmoid(model(vol)))
    D, H, W = volume.shape[2:]
    coords = [(d, h, w)
              for d in range(0, D - patch + 1, stride)
              for h in range(0, H - patch + 1, stride)
              for w in range(0, W - patch + 1, stride)]
    drops = np.empty(len(coords), dtype=np.float32)
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        occ = vol.repeat(len(chunk), 1, 1, 1, 1)
        for b, (d, h, w) in enumerate(chunk):
            occ[b, :, d:d + patch, h:h + patch, w:w + patch] = 0.0
        probs = torch.sigmoid(model(occ)).cpu().numpy()
        drops[start:start + len(chunk)] = np.maximum(base - probs, 0.0)

    # splat the drop grid at patch centres, then nearest-fill densely
    grid = np.zeros((D, H, W), dtype=np.float32)
    half = patch // 2
    for (d, h, w), v in zip(coords, drops):
        grid[d + half, h + half, w + half] = v
    import scipy.ndimage as ndi
    mask = grid > 0
    if mask.sum() >= 2:
        _dist, nearest = ndi.distance_transform_edt(~mask, return_indices=True)
        dense = grid[tuple(nearest)]
    else:
        dense = grid
    mx = float(dense.max())
    return dense / mx if mx > 0 else dense


# ---------------------------------------------------------------------------
# NIfTI output + plane visualizations
# ---------------------------------------------------------------------------


def preprocessed_grid_affine(reference_nii_path: str | Path):
    """True affine of the 128³ preprocessed grid, derived from the raw scan.

    ``preprocess_session`` resamples the reference to 2 mm, crops to the
    brain bounding box and centre-pads to 128³. The 128³ voxel indices are
    therefore NOT millimetre coordinates of the reference frame — the grid
    has its own affine. Mirrors that geometry exactly so attribution maps
    can be placed back into the reference (T88) space without a spatial
    shift.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    ref = nib.load(str(reference_nii_path))
    data = np.asanyarray(ref.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    zoom = np.asarray(ref.header.get_zooms()[:3], dtype=float)
    target_affine = ref.affine.copy()
    target_affine[:3, :3] = (target_affine[:3, :3] @ np.diag(1.0 / zoom)) * 2.0
    new_shape = np.maximum(
        1, np.round(np.asarray(data.shape[:3]) * (zoom / 2.0))).astype(int)
    r = resample_from_to(nib.Nifti1Image(data, ref.affine),
                         (new_shape.tolist(), target_affine), order=1, cval=0.0)
    vol2 = np.asanyarray(r.dataobj, dtype=np.float32)
    idx = np.argwhere(vol2 > 0)
    if len(idx) == 0:
        raise ValueError(f"empty reference volume: {reference_nii_path}")
    lo, hi = idx.min(0), idx.max(0) + 1
    aff128 = target_affine.copy()
    off = np.zeros(3, dtype=float)
    for axis in range(3):
        size = int(hi[axis] - lo[axis])
        free = 128 - size
        if free < 0:  # central crop, same rule as preprocess_session
            off[axis] = lo[axis] + (-free) // 2
        else:
            off[axis] = lo[axis] - free // 2
    aff128[:3, 3] = target_affine[:3, 3] + target_affine[:3, :3] @ off
    return aff128


def save_cam_nifti(cam: np.ndarray, reference_nii_path: str | Path,
                   out_path: str | Path) -> Path:
    """Save a 128³ CAM into the reference (T88) space, correctly aligned.

    The CAM's grid carries the *true* affine of the preprocessed 128³ grid
    (see ``preprocessed_grid_affine``); writing the raw voxel indices with an
    identity affine would place the map ~2 cm outside the brain in the
    reference frame and silently corrupt every downstream parcellation.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    ref = nib.load(str(reference_nii_path))
    aff128 = preprocessed_grid_affine(reference_nii_path)
    cam_img = nib.Nifti1Image(cam.astype(np.float32), aff128)
    # resample CAM into the reference (T88) grid for parcellation/overlay
    cam_t88 = resample_from_to(cam_img, (ref.shape[:3], ref.affine),
                               order=1, cval=0.0)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(cam_t88, str(out_path))
    return Path(out_path)


def save_planes_png(cam: np.ndarray, volume: np.ndarray,
                    out_path: str | Path) -> Path:
    """Axial/sagittal/coronal overlays of the CAM on the MRI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = {"axial": (2, cam.shape[2] // 2),
             "sagittal": (0, cam.shape[0] // 2),
             "coronal": (1, cam.shape[1] // 2)}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, (axis, pos)) in zip(axes, views.items()):
        img = np.take(volume[0], pos, axis=axis)
        m = np.take(cam, pos, axis=axis)
        ax.imshow(np.rot90(img), cmap="gray",
                  vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
        if m.max() > 0:
            ax.imshow(np.rot90(m), cmap="jet", alpha=0.4 * (m > 0.05))
        ax.set_title(name)
        ax.axis("off")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return Path(out_path)
