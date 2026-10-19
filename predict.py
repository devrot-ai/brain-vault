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
    args = ap.parse_args(argv)

    from brainvuln.mri.preprocess import PREPROCESS_VERSION, preprocess_session
    from brainvuln.mri.gradcam import gradcam_3d, save_cam_nifti, save_planes_png
    from brainvuln.mri.models import ResNet18Binary

    ckpt_path = _resolve_model_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
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
