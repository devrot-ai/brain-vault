#!/usr/bin/env python
"""Test-time augmentation (TTA) evaluation for a frozen MRI checkpoint.

Averages the positive-class probability over a fixed, label-free set of small
affine perturbations (rotation, translation, scale). The identity transform is
part of the set, so one run reports both the plain model and its TTA variant
on the same subjects.

Scientific rule
---------------
Selection happens on the VALIDATION split only. ``--split val`` writes the
TTA-selected threshold (Youden's J, same rule as the frozen pipeline) and the
per-subject probabilities; ``--split test`` takes that threshold via
``--threshold`` and never re-selects it. The canonical config, the frozen
serving threshold and the canonical checkpoint are not read or written.

Usage
-----
    python scripts/eval_tta.py --run-dir results/ml/resnet_seed42_retrain \\
        --split val
    python scripts/eval_tta.py --run-dir results/ml/resnet_seed42_retrain \\
        --split test --threshold 0.07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from brainvuln.mri.evaluate import classification_metrics, select_threshold  # noqa: E402
from brainvuln.mri.models import ResNet18Binary  # noqa: E402

CACHE = ROOT / "data" / "derived" / "oasis1_128"
SPLITS = ROOT / "data" / "splits"


def _rotation(axis: str, degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _theta(rotation: np.ndarray | None = None, scale: float = 1.0,
           shift_vox: tuple[float, float, float] = (0.0, 0.0, 0.0),
           shape: tuple[int, int, int] = (128, 128, 128)) -> torch.Tensor:
    """Build a grid_sample theta (output->input) for a small affine transform.

    The requested transform is applied to the input; grid_sample needs its
    inverse, hence the transpose of a rotation (and 1/scale).
    """
    mat = np.eye(3) if rotation is None else rotation
    mat = mat.T / scale
    # a shift of d voxels is 2d/N in normalised coordinates (align_corners=False)
    offset = np.array([2.0 * shift_vox[i] / shape[i] for i in range(3)])
    theta = np.concatenate([mat, offset[:, None]], axis=1)
    return torch.from_numpy(theta[None].astype(np.float32))


def build_variants() -> list[tuple[str, torch.Tensor]]:
    return [
        ("identity", _theta()),
        ("rot_z_+4", _theta(rotation=_rotation("z", 4.0))),
        ("rot_z_-4", _theta(rotation=_rotation("z", -4.0))),
        ("rot_x_+4", _theta(rotation=_rotation("x", 4.0))),
        ("rot_y_-4", _theta(rotation=_rotation("y", -4.0))),
        ("shift_y_+2", _theta(shift_vox=(0.0, 2.0, 0.0))),
        ("shift_y_-2", _theta(shift_vox=(0.0, -2.0, 0.0))),
        ("scale_1.03", _theta(scale=1.03)),
    ]


def _apply(volume: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, volume.shape, align_corners=False)
    return F.grid_sample(volume, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)


@torch.no_grad()
def predict_variants(model: torch.nn.Module, volume: torch.Tensor,
                     variants: list[tuple[str, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, theta in variants:
        # cached volumes are already [1, D, H, W]; grid_sample wants a batch
        x = _apply(volume[None], theta)
        out[name] = float(torch.sigmoid(model(x)).item())
    out["tta_mean"] = float(np.mean([v for k, v in out.items() if k != "tta_mean"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing checkpoints/best.pt")
    ap.add_argument("--checkpoint", default=None,
                    help="override checkpoint path (default: <run-dir>/checkpoints/best.pt)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed operating threshold (required for --split test)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <run-dir>/tta)")
    args = ap.parse_args(argv)

    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() \
        else Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.is_file():
        print(f"checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2
    if args.split == "test" and args.threshold is None:
        print("--split test requires --threshold from the validation run", file=sys.stderr)
        return 2

    split = pd.read_csv(SPLITS / f"{args.split}_subjects.csv")
    missing = [s for s in split["session_id"].astype(str)
               if not (CACHE / f"{s}.npy").is_file()]
    if missing:
        print(f"{len(missing)} cached {args.split} volumes missing", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    variants = build_variants()
    rows = []
    for _, r in split.iterrows():
        sid = str(r["session_id"])
        volume = torch.from_numpy(np.load(CACHE / f"{sid}.npy").astype(np.float32))
        probs = predict_variants(model, volume, variants)
        rows.append({"subject": r["subject"], "session_id": sid,
                     "label": int(r["label"]), **probs})
    df = pd.DataFrame(rows)

    labels = df["label"].to_numpy()
    identity = df["identity"].to_numpy()
    tta = df["tta_mean"].to_numpy()

    if args.split == "val":
        threshold = select_threshold(labels, tta, "youden")
    else:
        threshold = float(args.threshold)

    metrics = {
        "identity": classification_metrics(labels, identity, threshold),
        "tta": classification_metrics(labels, tta, threshold),
    }
    per_variant = {
        c: {"roc_auc": float(roc_auc_score(labels, df[c])),
            "mean_probability": float(df[c].mean())}
        for c in df.columns if c not in ("subject", "session_id", "label")
    }

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "tta"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": {"path": str(ckpt_path),
                       "sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest()},
        "split": args.split,
        "n": int(len(df)),
        "threshold": float(threshold),
        "threshold_selected_on": "validation (Youden, TTA probabilities)"
        if args.split == "val" else "fixed from validation run",
        "variants": [v[0] for v in variants],
        "per_variant": per_variant,
        "metrics_identity": metrics["identity"],
        "metrics_tta": metrics["tta"],
        "delta_tta_minus_identity": {
            k: float(metrics["tta"][k] - metrics["identity"][k])
            for k in ("roc_auc", "accuracy", "balanced_accuracy", "sensitivity",
                      "specificity", "brier")
        },
    }
    df.to_csv(out_dir / f"{args.split}_tta_predictions.csv", index=False)
    (out_dir / f"{args.split}_tta_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.split} n={len(df)} threshold={threshold:.4f} -> {out_dir}")
    print(f"  identity : AUC {metrics['identity']['roc_auc']:.3f} "
          f"acc {metrics['identity']['accuracy']:.3f} "
          f"bacc {metrics['identity']['balanced_accuracy']:.3f}")
    print(f"  TTA      : AUC {metrics['tta']['roc_auc']:.3f} "
          f"acc {metrics['tta']['accuracy']:.3f} "
          f"bacc {metrics['tta']['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
