"""Fit temperature scaling on validation logits (calibration)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln.config import resolve_checkpoint, resolve_threshold
from brainvuln.mri.calibration import apply_temperature, fit_temperature, save_artifact
from brainvuln.mri.models import ResNet18Binary
from brainvuln.mri.preprocess import preprocess_session


def load_val_predictions():
    import pandas as pd

    val_csv = Path("data/splits/val_subjects.csv")
    df = pd.read_csv(val_csv)
    ckpt_path = resolve_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    logits, labels, probs = [], [], []
    threshold, _ = resolve_threshold(ckpt_path=ckpt_path)
    for _, row in df.iterrows():
        subject = str(row["subject"])
        # locate the masked_gfc .img under raw/oasis1/extracted/*/*/<subject>/...
        matches = list(Path("data/raw/oasis1").rglob(f"*{subject}*t88_masked_gfc.img"))
        if not matches:
            print(f"no image for {subject}", file=sys.stderr)
            continue
        img = matches[0]
        volume = preprocess_session(img)
        x = torch.from_numpy(volume[None].astype(np.float32))
        with torch.no_grad():
            logit = float(model(x).item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
        logits.append(logit)
        labels.append(int(row["label"]))
        probs.append(prob)
    return np.array(logits), np.array(labels), np.array(probs), threshold


def main():
    logits, labels, probs, threshold = load_val_predictions()
    T = fit_temperature(logits, labels)
    cal_probs = apply_temperature(logits, T)
    print(f"fitted temperature T={T:.4f} on {len(logits)} validation subjects")
    payload = {
        "temperature": T,
        "n_val_subjects": int(len(logits)),
        "threshold_frozen": float(threshold),
        "nll_before": float(
            -(labels * np.log(np.clip(probs, 1e-9, 1 - 1e-9))
              + (1 - labels) * np.log(np.clip(1 - probs, 1e-9, 1 - 1e-9))).mean()
        ),
        "nll_after": float(
            -(labels * np.log(np.clip(cal_probs, 1e-9, 1 - 1e-9))
              + (1 - labels) * np.log(np.clip(1 - cal_probs, 1e-9, 1 - 1e-9))).mean()
        ),
    }
    out = Path("results/ml/evaluation/calibration_fit.json")
    save_artifact(out, T, threshold, method="temperature_fit_on_validation")
    print("saved", out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
