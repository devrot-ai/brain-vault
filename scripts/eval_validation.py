"""Validation-only accuracy gate for the canonical MRI checkpoint.

Evaluates the canonical checkpoint on the VALIDATION split only and checks
whether the model meets a minimum accuracy target. This is the acceptance
gate for "accuracy >= 0.85": it is computed on validation, never on test,
and no threshold selected here is written into the serving path.

Exit codes
----------
0   minimum accuracy met at some validation-selected threshold
1   minimum accuracy NOT met (gate fails; model needs retraining/ensembling)
2   cannot evaluate (missing cache, checkpoint, split, or malformed volume)

Outputs
-------
results/ml/resnet_seed42/predictions/val_predictions.csv
results/ml/evaluation/accuracy_gate.json

The frozen serving threshold and its regression test
(tests/test_threshold_frozen.py) are never touched by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainvuln.config import resolve_checkpoint, resolve_threshold  # noqa: E402
from brainvuln.mri.models import ResNet18Binary  # noqa: E402

DEFAULT_MIN_ACCURACY = 0.85


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _predict(model: torch.nn.Module, arr: np.ndarray) -> tuple[float, float]:
    if arr.ndim == 3:
        arr = arr[None]
    elif arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(
            f"expected cached volume shape [D,H,W] or [1,D,H,W], got {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("volume contains non-finite values")
    x = torch.from_numpy(arr[None].astype(np.float32))
    with torch.no_grad():
        try:
            logit = float(model(x).item())
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"model could not evaluate volume with shape {arr.shape}: {exc}") from exc
    return logit, float(1.0 / (1.0 + np.exp(-logit)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-accuracy", type=float, default=DEFAULT_MIN_ACCURACY,
                    help=f"minimum validation accuracy required (default {DEFAULT_MIN_ACCURACY})")
    ap.add_argument("--split", default="data/splits/val_subjects.csv",
                    help="split CSV with subject,session_id,label columns (validation only)")
    args = ap.parse_args(argv)

    if not 0.0 < args.min_accuracy <= 1.0:
        print("--min-accuracy must be in (0, 1]", file=sys.stderr)
        return 2

    try:
        ckpt_path = resolve_checkpoint()
    except FileNotFoundError as exc:
        print(f"cannot evaluate: {exc}", file=sys.stderr)
        return 2

    split_path = ROOT / args.split
    cache = ROOT / "data/derived/oasis1_128"
    if not split_path.is_file():
        print(f"cannot evaluate: split not found: {split_path}", file=sys.stderr)
        return 2
    if not cache.is_dir():
        print(f"cannot evaluate: preprocessed cache not found: {cache}", file=sys.stderr)
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ResNet18Binary()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val = pd.read_csv(split_path)
    for col in ("subject", "session_id", "label"):
        if col not in val.columns:
            print(f"cannot evaluate: split CSV missing column '{col}'", file=sys.stderr)
            return 2

    rows: list[tuple[str, int, float, float]] = []
    for _, r in val.iterrows():
        npy = cache / f"{r['session_id']}.npy"
        if not npy.is_file():
            print(f"cannot evaluate: missing cached volume {npy}", file=sys.stderr)
            return 2
        try:
            logit, prob = _predict(model, np.load(npy))
        except ValueError as exc:
            print(f"cannot evaluate: {npy}: {exc}", file=sys.stderr)
            return 2
        rows.append((str(r["subject"]), int(r["label"]), logit, prob))

    df = pd.DataFrame(rows, columns=["subject", "label", "logit", "prob"])
    out = ckpt_path.parents[1] / "predictions" / "val_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    frozen, frozen_source = resolve_threshold(ckpt_path=ckpt_path)
    pred_at = lambda t: (df["prob"].values >= t).astype(int)  # noqa: E731
    acc_frozen = float((pred_at(frozen) == df["label"].values).mean())

    # validation-only sweep: every distinct probability is a candidate cut
    cands = np.unique(np.r_[df["prob"].values, 1.0])
    accs = np.array([float((pred_at(t) == df["label"].values).mean()) for t in cands])
    best_i = int(np.argmax(accs))
    best_t, best_acc = float(cands[best_i]), float(accs[best_i])
    hits = cands[accs >= args.min_accuracy]

    n = len(df)
    k_best = int(round(best_acc * n))
    ci_lo, ci_hi = wilson_interval(k_best, n)

    print(f"validation n={n} positives={int(df['label'].sum())} -> {out}")
    print(f"accuracy @ frozen threshold {frozen:.4f} ({frozen_source}): {acc_frozen:.4f}")
    print(f"best validation accuracy: {best_acc:.4f} at threshold {best_t:.4f} "
          f"({k_best}/{n}, 95% CI {ci_lo:.3f}-{ci_hi:.3f})")
    if len(hits):
        print(f"{len(hits)} thresholds reach >= {args.min_accuracy:.2f} validation accuracy "
              f"(range {hits.min():.4f}..{hits.max():.4f}); gate PASS")
    else:
        print(f"NO threshold reaches >= {args.min_accuracy:.2f} validation accuracy; gate FAIL")
        print("The target is not reachable by operating-point choice — it requires "
              "retraining or ensembling with validation-only selection.")

    report = {
        "split": args.split,
        "n": n,
        "n_positives": int(df["label"].sum()),
        "min_accuracy_required": args.min_accuracy,
        "accuracy_at_frozen_threshold": acc_frozen,
        "frozen_threshold": frozen,
        "frozen_threshold_source": frozen_source,
        "best_validation_accuracy": best_acc,
        "best_validation_threshold": best_t,
        "best_accuracy_95ci": [ci_lo, ci_hi],
        "thresholds_meeting_target": [float(t) for t in hits],
        "gate_passed": bool(len(hits)),
        "note": ("Validation-only evaluation. No threshold from this report is applied to "
                 "serving or to the test split; the frozen serving threshold is unchanged."),
    }
    report_path = ROOT / "results" / "ml" / "evaluation" / "accuracy_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"gate report -> {report_path}")
    return 0 if len(hits) else 1


if __name__ == "__main__":
    sys.exit(main())
