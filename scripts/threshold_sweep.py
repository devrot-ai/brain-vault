"""Cost-aware operating-threshold sweep (threshold re-tune, Item: threshold)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_predictions(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["probability"]), int(row["diagnosis"] == "AD")))
    return rows


def confusion_at(probs_labels, threshold: float) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for p, y in probs_labels:
        pred = 1 if p >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    bal = 0.5 * (sens + spec)
    return {"sensitivity": sens, "specificity": spec, "accuracy": acc,
            "balanced_accuracy": bal, **cm}


def cost_optimal_threshold(probs_labels, cost_fp: float, cost_fn: float,
                           n_grid: int = 1001):
    best_t, best_cost, best_cm = 0.5, float("inf"), None
    grid = [1e-3 + (1 - 2e-3) * i / (n_grid - 1) for i in range(n_grid)]
    for t in grid:
        cm = confusion_at(probs_labels, t)
        cost = cost_fp * cm["fp"] + cost_fn * cm["fn"]
        if cost < best_cost:
            best_cost, best_t, best_cm = cost, t, cm
    return best_t, best_cost / len(probs_labels), best_cm


def youden_threshold(probs_labels, n_grid: int = 197):
    best_t, best_j = 0.5, -1.0
    grid = [1e-3 + (1 - 2e-3) * i / (n_grid - 1) for i in range(n_grid)]
    for t in grid:
        cm = confusion_at(probs_labels, t)
        sens = cm["tp"] / (cm["tp"] + cm["fn"]) if cm["tp"] + cm["fn"] else 0.0
        spec = cm["tn"] / (cm["tn"] + cm["fp"]) if cm["tn"] + cm["fp"] else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sweep operating threshold on a predictions CSV")
    ap.add_argument("predictions_csv", help="CSV with probability,diagnosis columns")
    ap.add_argument("--cost-fp", type=float, default=1.0)
    ap.add_argument("--cost-fn", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    pl = load_predictions(Path(args.predictions_csv))
    t_youden = youden_threshold(pl)
    cm_youden = confusion_at(pl, t_youden)
    m_youden = metrics(cm_youden)

    t_cost, cost_per_subj, cm_cost = cost_optimal_threshold(pl, args.cost_fp, args.cost_fn)
    m_cost = metrics(cm_cost)

    report = {
        "n": len(pl),
        "cost_fp": args.cost_fp,
        "cost_fn": args.cost_fn,
        "youden": {"threshold": t_youden, **m_youden},
        "cost_optimal": {"threshold": t_cost, "expected_cost_per_subject": cost_per_subj, **m_cost},
        "note": "cost-optimal trades sensitivity vs specificity per explicit FP/FN costs; "
                "frozen validation threshold (0.07) is the reported operating point unless overridden",
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
