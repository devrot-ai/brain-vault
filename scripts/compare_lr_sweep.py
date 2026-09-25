#!/usr/bin/env python
"""Compare LR-probe trajectories against the known 1e-4 baseline.

Decision context (see scripts/sweep_lr.sh header): the production run at
LR 1e-4 underfits from scratch -- train loss plateaus at ~0.59-0.67 for all
epochs and best val_auc (0.8111) lands at epoch 1 and is never exceeded.
A probe "breaks the plateau" when its train loss falls clearly below that
band while val_auc does not degrade. This script only reports; the choice of
the production retraining setting is made by the operator from its table.

Usage:
    python scripts/compare_lr_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "results" / "ml"

# Known baseline: production run at LR 1e-4, full augmentation (archived
# as *_v1 after the diagnosis); train loss plateaued at ~0.59-0.67 for all
# epochs, best val_auc (0.8111) at epoch 1, never exceeded.
BASELINE_DIR = ML / "resnet_lr1e4_fullaug_v1"
BASELINE_BAND = (0.59, 0.67)  # observed train-loss plateau at 1e-4

RUNS = [
    ("v1 baseline lr1e-4 fullaug", BASELINE_DIR),
    ("probe  lr3e-4 fullaug", ML / "resnet_seed42_lr3e4_probe"),
    ("probe  lr1e-3 fullaug", ML / "resnet_seed42_lr1e3_probe"),
    ("probe  lr1e-3 noaug", ML / "resnet_seed42_noaug_lr1e3_noaug_probe"),
    ("V2 PROD lr1e-4 lightaug cos12", ML / "resnet_seed42"),
]


def load_history(run_dir: Path) -> dict | None:
    f = run_dir / "metrics" / "train_history.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def main() -> None:
    print(f"1e-4 train-loss plateau band: {BASELINE_BAND}")
    print(
        f"{'run':<22}{'epochs':>7}{'best_auc':>9}{'best_ep':>8}"
        f"{'min_loss':>9}{'last_loss':>10}  trajectory (ep: loss/val)"
    )
    rows = []
    for name, run_dir in RUNS:
        hist = load_history(run_dir)
        if hist is None:
            print(f"{name:<22}{'-':>7}  (no train_history.json yet)")
            continue
        ep = hist.get("history", [])
        if not ep:
            print(f"{name:<22}{hist.get('epochs_run', 0):>7}  (empty history)")
            continue
        losses = [e["train_loss"] for e in ep]
        aucs = [e["val_auc"] for e in ep]
        traj = " ".join(
            f"{e['epoch']}:{e['train_loss']:.3f}/{e['val_auc']:.3f}" for e in ep
        )
        print(
            f"{name:<22}{len(ep):>7}{hist['best_val_auc']:>9.4f}"
            f"{hist['best_epoch']:>8}{min(losses):>9.3f}{losses[-1]:>10.3f}  {traj}"
        )
        rows.append((name, hist, losses))

    broke = [n for n, _, ls in rows if min(ls) < BASELINE_BAND[0] - 0.03]
    print()
    if not rows:
        print("No probe histories available yet.")
    elif broke:
        print(f"Broke the 1e-4 plateau: {', '.join(broke)}")
        print("Read val_auc alongside: fitting gains are only good if val")
        print("does not degrade (no-aug probe fits but overfits).")
    else:
        print("No run broke the plateau; LR is not the binding constraint.")
        print("Next levers: model capacity/normalization, longer warmup, or")
        print("label noise inspection before any augmentation change.")


if __name__ == "__main__":
    main()
