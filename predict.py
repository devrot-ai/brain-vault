#!/usr/bin/env python
"""Single-image inference (Part AA).

    python predict.py --image patient.nii.gz

Prints the prediction in the documented format and saves:

    outputs/patient_gradcam.nii.gz   3D Grad-CAM in the input's space
    outputs/patient_gradcam.png      axial/sagittal/coronal overlay
    outputs/prediction.json          machine-readable record

RESEARCH USE ONLY — this is not a clinical diagnostic system. Probabilities
come from a model trained on a small research cohort (OASIS-1) and must not
be used for any clinical decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _resolve_model_path(arg: str | None) -> Path:
    from brainvuln.config import resolve_checkpoint
    return resolve_checkpoint(arg)


def tta_inference(model, x: torch.Tensor, num_views: int = 5, seed: int = 42) -> torch.Tensor:
    """Test-Time Augmentation: run the model on multiple augmented views and average probabilities.

    Applies mild, biologically plausible augmentations (small rotations, flips, scales,
    intensity shifts) to the input volume and averages the sigmoid probabilities across
    all views. This reduces variance and typically improves accuracy by 1-2% without
    any retraining.

    Args:
        model: trained model (must be in eval mode).
        x: input tensor [1, 1, D, H, W].
        num_views: number of augmented views (excluding the original).
        seed: random seed for reproducibility.

    Returns:
        Averaged probability scalar tensor.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    probs = []

    # View 0: original (no augmentation)
    with torch.no_grad():
        logit = model(x)
        probs.append(torch.sigmoid(logit))

    # Generate augmented views
    for i in range(num_views):
        augmented = _apply_tta_aug(x, seed=seed + i)
        with torch.no_grad():
            logit = model(augmented)
            probs.append(torch.sigmoid(logit))

    # Average probabilities across all views
    avg_prob = torch.mean(torch.stack(probs))
    return avg_prob


def _apply_tta_aug(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply a single mild TTA augmentation to a preprocessed volume.

    Uses the same transform types as training but with reduced strength to keep
    the augmented view biologically plausible.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    vol = x.clone()  # [1, 1, D, H, W]

    # Small random rotation (±5 degrees per axis)
    angle_x = random.uniform(-5, 5) * np.pi / 180
    angle_y = random.uniform(-5, 5) * np.pi / 180
    angle_z = random.uniform(-5, 5) * np.pi / 180

    # Build rotation matrix
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    rot = np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,    sx * cy,                cx * cy                ],
    ])

    # Small random translation (±2 voxels)
    tx = random.uniform(-2, 2)
    ty = random.uniform(-2, 2)
    tz = random.uniform(-2, 2)

    # Small random scale (±3%)
    sc = random.uniform(0.97, 1.03)

    # Combine into affine: rotation * scale + translation
    affine = np.eye(4)
    affine[:3, :3] = rot * sc
    affine[:3, 3] = [tx, ty, tz]

    # Apply affine using grid_sample
    D, H, W = vol.shape[2], vol.shape[3], vol.shape[4]

    # Build sampling grid
    zs = torch.linspace(-1, 1, D).view(-1, 1, 1).expand(D, H, W)
    ys = torch.linspace(-1, 1, H).view(1, -1, 1).expand(D, H, W)
    xs = torch.linspace(-1, 1, W).view(1, 1, -1).expand(D, H, W)
    grid = torch.stack([xs, ys, zs], dim=-1).unsqueeze(0)  # [1, D, H, W, 3]

    # Transform grid by inverse affine
    affine_t = torch.tensor(affine, dtype=torch.float32)
    # Apply affine to homogeneous coordinates
    xyz = grid.reshape(-1, 3)  # [D*H*W, 3]
    ones = torch.ones(xyz.shape[0], 1)
    homog = torch.cat([xyz, ones], dim=1)  # [D*H*W, 4]
    transformed = (homog @ affine_t.T)[:,:3]  # [D*H*W, 3]
    transformed = transformed.reshape(1, D, H, W, 3)

    augmented_vol = torch.nn.functional.grid_sample(
        vol, transformed, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # Mild intensity perturbation (±5% scale, ±0.05 shift)
    if random.random() > 0.5:
        intensity_scale = random.uniform(0.95, 1.05)
        augmented_vol = augmented_vol * intensity_scale

    if random.random() > 0.5:
        intensity_shift = random.uniform(-0.05, 0.05)
        augmented_vol = augmented_vol + intensity_shift

    # Small Gaussian noise
    if random.random() > 0.5:
        noise = torch.randn_like(augmented_vol) * 0.01
        augmented_vol = augmented_vol + noise

    return augmented_vol


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="T1 volume (.nii / .nii.gz)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the frozen validation threshold")
    ap.add_argument("--temperature", default="1.0",
                    help="logit temperature: a number, or 'auto' for the fitted calibration "
                         "artifact (default: 1.0, which pairs with the frozen threshold)")
    ap.add_argument("--tta", action="store_true", default=False,
                    help="enable Test-Time Augmentation (averages 6 views: original + 5 augmented)")
    ap.add_argument("--tta-views", type=int, default=5,
                    help="number of augmented views for TTA (default: 5, total 6 with original)")
    args = ap.parse_args(argv)

    from brainvuln.mri.preprocess import PREPROCESS_VERSION, preprocess_session
    from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti, save_planes_png
    from brainvuln.mri.models import ResNet18Binary, ResNet18CBAM, ResNet18WithDemographics

    ckpt_path = _resolve_model_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Auto-detect model architecture from checkpoint state_dict keys
    state_keys = list(ckpt.get("state_dict", {}).keys())
    widen = ckpt.get("widen_factor", 1.0)

    has_demographics = any("demographics" in k or "age_proj" in k
                           for k in state_keys)
    has_cbam = any("att" in k and ("channel_att" in k or "spatial_att" in k)
                   for k in state_keys)

    if has_cbam:
        model = ResNet18CBAM(widen_factor=widen)
        print("Detected CBAM attention model (ResNet18CBAM)")
    elif has_demographics:
        model = ResNet18WithDemographics(widen_factor=widen)
        print("Detected demographics-fused model (ResNet18WithDemographics)")
    else:
        model = ResNet18Binary(widen_factor=widen)
        print("Detected standard model (ResNet18Binary)")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    from brainvuln.config import checkpoint_identity, resolve_threshold, resolve_temperature
    identity = checkpoint_identity(str(ckpt_path))

    # Temperature: the frozen threshold was selected in UNCALIBRATED probability
    # space, so identity (1.0) is the default and the only mode that is valid
    # with the frozen threshold alone. 'auto' applies the fitted calibration,
    # which changes probability space — pairing it with the uncalibrated
    # operating point silently moves the decision boundary, so require an
    # explicit --threshold in that mode.
    if str(args.temperature).lower() == "auto":
        temperature = resolve_temperature()
        if args.threshold is None:
            print("--temperature auto applies a fitted calibration, which changes probability "
                  "space; the frozen threshold (0.07) was selected in uncalibrated space. "
                  "Pass --threshold explicitly to choose the operating point in calibrated "
                  "space, or omit --temperature to use the frozen operating point.",
                  file=sys.stderr)
            return 2
    else:
        try:
            temperature = float(args.temperature)
        except ValueError:
            print(f"invalid --temperature {args.temperature!r}: use a number or 'auto'",
                  file=sys.stderr)
            return 2
        if not temperature > 0:
            print("--temperature must be > 0", file=sys.stderr)
            return 2

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2
    supported_nifti = image_path.name.endswith(".nii") or image_path.name.endswith(".nii.gz")
    supported_analyze = image_path.suffix.lower() == ".img" and image_path.with_suffix(".hdr").is_file()
    if not (supported_nifti or supported_analyze):
        print(f"unsupported input format: {image_path.name}. BrainVuln takes a 3D NIfTI "
              "volume (.nii / .nii.gz) or an ANALYZE .img/.hdr pair. DICOM series or "
              "JPEG/PNG must be converted first (e.g. dcm2niix).", file=sys.stderr)
        return 2

    # same deterministic preprocessing as training (128³, 2 mm, brain z)
    try:
        volume = preprocess_session(image_path)       # [1, D, H, W]
    except (ValueError, IndexError) as exc:
        print(f"cannot read MRI volume {image_path}: {exc}. Expected a skull-stripped T1 "
              "volume (OASIS T88-style, single 3D volume, nonzero brain voxels).",
              file=sys.stderr)
        return 2
    if not np.isfinite(volume).all():
        print(f"preprocessed volume from {image_path} contains non-finite values",
              file=sys.stderr)
        return 2
    brain_fraction = float((volume[0] != 0).mean())
    x = torch.from_numpy(volume[None].astype(np.float32))  # add batch -> [1,1,D,H,W]

    # Inference: single pass or Test-Time Augmentation
    if args.tta:
        print(f"Running Test-Time Augmentation ({args.tta_views} augmented views + original)...")
        prob_tensor = tta_inference(model, x, num_views=args.tta_views)
        prob = float(prob_tensor.item())
        # Back-calculate logit from probability for reporting
        logit = float(torch.log(torch.tensor(prob / (1 - max(prob, 1e-7)))).item())
        print(f"TTA averaged probability: {prob:.4f}")
    else:
        with torch.no_grad():
            logit = float(model(x).item())
            prob = float(torch.sigmoid(torch.tensor(logit / temperature)).item())

    threshold, threshold_source = resolve_threshold(args.threshold, ckpt_path)
    label = "Alzheimer's disease" if prob >= threshold else "cognitively normal"
    model_name = f"BrainVuln-{type(model).__name__}"

    print("Prediction:")
    print(f"  {label}")
    print("Probability:")
    print(f"  {prob:.2f}")
    print("Threshold:")
    print(f"  {threshold:.2f} ({threshold_source})")
    temperature_source = ("fitted calibration artifact (--temperature auto)"
                          if str(args.temperature).lower() == "auto"
                          else "identity (no calibration applied)")
    print("Temperature:")
    print(f"  {temperature:.3f} ({temperature_source})")
    print("Model:")
    print(f"  {model_name}")
    if args.tta:
        print(f"TTA views:  {args.tta_views + 1} (1 original + {args.tta_views} augmented)")
    print("Checkpoint sha256:")
    print("  " + identity["sha256"])
    print("Preprocessing version:")
    print(f"  {PREPROCESS_VERSION}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem.replace(".nii", "")
    cam = gradcam_3d(model, x)
    save_cam_nifti(cam, image_path, out_dir / f"{stem}_gradcam.nii.gz")
    save_planes_png(cam, volume, out_dir / f"{stem}_gradcam.png")
    (out_dir / "prediction.json").write_text(json.dumps({
        "image": str(image_path),
        "probability": prob,
        "logit": logit,
        "label": label,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "tta_enabled": args.tta,
        "tta_views": args.tta_views + 1 if args.tta else None,
        "input_qc": {
            "format": "NIfTI",
            "preprocessed_shape": list(volume.shape),
            "brain_fraction": brain_fraction,
            "finite": True,
        },
        "model": model_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_identity": identity,
        "checkpoint_sha256": identity["sha256"],
        "disclaimer": ("Research use only. Not a clinical diagnostic system. "
                       "Trained on a small research cohort; no causal or "
                       "clinical claims."),
    }, indent=2))
    print(f"\nsaved {out_dir / (stem + '_gradcam.nii.gz')} and "
          f"{out_dir / (stem + '_gradcam.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
