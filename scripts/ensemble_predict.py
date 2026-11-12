#!/usr/bin/env python
"""Ensemble inference: average probabilities across multiple checkpoints.

Trains multiple models with different random seeds and averages their
probabilities during inference. This reduces variance and typically improves
accuracy by 3-5% compared to a single model.

    python scripts/ensemble_predict.py --image patient.nii.gz \
        --checkpoints results/ml/resnet_seed42/checkpoints/best.pt \
                          results/ml/resnet_seed1/checkpoints/best.pt \
                          results/ml/resnet_seed2/checkpoints/best.pt

Or use --sweep-dir to auto-discover checkpoints from a sweep directory:

    python scripts/ensemble_predict.py --image patient.nii.gz \
        --sweep-dir results/ml/

RESEARCH USE ONLY — this is not a clinical diagnostic system.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_model_from_checkpoint(ckpt_path: str):
    """Load a model from a checkpoint file, auto-detecting architecture."""
    from brainvuln.mri.models import ResNet18Binary, ResNet18CBAM, ResNet18WithDemographics

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_keys = list(ckpt.get("state_dict", {}).keys())

    widen_factor = ckpt.get("widen_factor", 1.0)

    # Detect model architecture from state_dict keys
    has_cbam = any("att" in k and ("channel_att" in k or "spatial_att" in k)
                   for k in state_keys)
    has_demographics = any("fc.0." in k or "fc.3." in k for k in state_keys)

    if has_cbam:
        model = ResNet18CBAM(widen_factor=widen_factor)
        print(f"  [Ensemble] {ckpt_path}: ResNet18CBAM (widen={widen_factor})")
    elif has_demographics:
        model = ResNet18WithDemographics(widen_factor=widen_factor)
        print(f"  [Ensemble] {ckpt_path}: ResNet18WithDemographics (widen={widen_factor})")
    else:
        model = ResNet18Binary(widen_factor=widen_factor)
        print(f"  [Ensemble] {ckpt_path}: ResNet18Binary (widen={widen_factor})")

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def ensemble_predict(models, x: torch.Tensor, device: str = "cpu") -> float:
    """Average sigmoid probabilities across an ensemble of models.

    Args:
        models: list of trained models (all in eval mode).
        x: input tensor [1, 1, D, H, W].
        device: computation device.

    Returns:
        Averaged probability as float.
    """
    x = x.to(device)
    probs = []

    with torch.no_grad():
        for model in models:
            model = model.to(device)
            if getattr(model, "needs_demographics", False):
                # For demographics models without demo data, use dummy values
                # (mean age=73, mean sex=0.5 for mixed population)
                age = torch.tensor([73.0], device=device)
                sex = torch.tensor([0.5], device=device)
                logit = model(x, age, sex)
            else:
                logit = model(x)
            prob = torch.sigmoid(logit).item()
            probs.append(prob)

    avg_prob = float(np.mean(probs))
    return avg_prob


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="T1 volume (.nii / .nii.gz)")
    ap.add_argument("--checkpoints", nargs="+", default=[],
                    help="paths to checkpoint files to ensemble")
    ap.add_argument("--sweep-dir", default=None,
                    help="auto-discover checkpoints from results/ml/ subdirectories")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--threshold", type=float, default=0.07,
                    help="decision threshold (default: 0.07 from validation)")
    ap.add_argument("--tta", action="store_true", default=False,
                    help="enable Test-Time Augmentation for each model in ensemble")
    ap.add_argument("--tta-views", type=int, default=5,
                    help="number of augmented views per model for TTA")
    args = ap.parse_args(argv)

    # Collect checkpoints
    checkpoints = list(args.checkpoints)

    if args.sweep_dir:
        sweep = Path(args.sweep_dir)
        if sweep.exists():
            for best_pt in sorted(sweep.glob("*/checkpoints/best.pt")):
                if str(best_pt) not in [str(c) for c in checkpoints]:
                    checkpoints.append(str(best_pt))

    if not checkpoints:
        print("Error: no checkpoints found. Provide --checkpoints or --sweep-dir.",
              file=sys.stderr)
        return 2

    print(f"[Ensemble] Loading {len(checkpoints)} models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for ckpt_path in checkpoints:
        try:
            model = load_model_from_checkpoint(ckpt_path)
            models.append(model)
        except Exception as e:
            print(f"  [Ensemble] Warning: failed to load {ckpt_path}: {e}",
                  file=sys.stderr)

    if not models:
        print("Error: no models loaded successfully.", file=sys.stderr)
        return 2

    print(f"[Ensemble] Loaded {len(models)} models for ensemble inference.")

    # Load and preprocess image
    from brainvuln.mri.preprocess import preprocess_session

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"image not found: {image_path}", file=sys.stderr)
        return 2

    try:
        volume = preprocess_session(image_path)
    except (ValueError, IndexError) as exc:
        print(f"cannot read MRI volume {image_path}: {exc}", file=sys.stderr)
        return 2

    x = torch.from_numpy(volume[None].astype(np.float32))  # [1, 1, D, H, W]

    # Ensemble prediction
    if args.tta:
        print(f"[Ensemble] Running with TTA ({args.tta_views} views per model)...")
        from predict import tta_inference

        all_probs = []
        for model in models:
            prob = tta_inference(model, x, num_views=args.tta_views)
            all_probs.append(prob.item())
        prob = float(np.mean(all_probs))
    else:
        prob = ensemble_predict(models, x, device=str(device))

    # Decision
    label = "Alzheimer's disease" if prob >= args.threshold else "cognitively normal"

    print("\nEnsemble Prediction:")
    print(f"  {label}")
    print(f"Probability: {prob:.4f}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Ensemble size: {len(models)} models")
    if args.tta:
        print(f"TTA: {args.tta_views + 1} views per model")

    # Save prediction JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem.replace(".nii", "")

    (out_dir / "ensemble_prediction.json").write_text(json.dumps({
        "image": str(image_path),
        "probability": prob,
        "label": label,
        "threshold": args.threshold,
        "ensemble_size": len(models),
        "checkpoints": [str(c) for c in checkpoints],
        "tta_enabled": args.tta,
        "tta_views": args.tta_views + 1 if args.tta else None,
        "disclaimer": ("Research use only. Not a clinical diagnostic system. "
                       "Trained on a small research cohort; no causal or "
                       "clinical claims."),
    }, indent=2))

    print(f"\nsaved {out_dir / 'ensemble_prediction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
