"""Evaluation (Parts I/J): metrics, subject-level bootstrap CIs, thresholds.

The test protocol runs ONLY after model selection is frozen; this module
never sees a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# threshold selection — validation only (Part J)
# ---------------------------------------------------------------------------


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, method: str = "youden"
) -> float:
    """Choose the operating threshold on held-out validation data only.

    * ``youden`` maximizes sensitivity + specificity - 1.
    * ``f1`` maximizes F1.
    The chosen value is frozen afterwards and applied unchanged to test.
    """
    thresholds = np.linspace(0.01, 0.99, 197)
    if method == "youden":
        scores = []
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            scores.append(sens + spec - 1)
    elif method == "f1":
        scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                  for t in thresholds]
    else:
        raise ValueError(f"unknown threshold method {method!r}")
    return float(thresholds[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# ---------------------------------------------------------------------------
# subject-level bootstrap (Part I) — resample SUBJECTS, keep all their scans
# ---------------------------------------------------------------------------


def subject_bootstrap_ci(
    y_true,
    y_prob,
    subjects,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
    metrics: tuple = ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity",
                      "specificity", "brier"),
) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=int),
                       "p": np.asarray(y_prob, dtype=float),
                       "s": list(subjects)})
    subject_ids = df["s"].unique()
    by_subject = {s: (g["y"].to_numpy(), g["p"].to_numpy())
                  for s, g in df.groupby("s")}
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        ys = np.concatenate([by_subject[s][0] for s in pick])
        ps = np.concatenate([by_subject[s][1] for s in pick])
        if len(set(ys)) < 2:
            continue
        m = classification_metrics(ys, ps, threshold)
        for k in metrics:
            draws[k].append(m[k])
    out = {}
    for k in metrics:
        arr = np.asarray(draws[k], dtype=float)
        out[k] = {"point": None, "lo": float(np.nanpercentile(arr, 2.5)),
                  "hi": float(np.nanpercentile(arr, 97.5))}
    # point estimates from the true (un-resampled) data
    full = classification_metrics(df["y"], df["p"], threshold)
    for k in metrics:
        out[k]["point"] = full[k]
    return out


# ---------------------------------------------------------------------------
# predictions io
# ---------------------------------------------------------------------------


def save_predictions(
    path: str | Path, session_ids, subject_ids, y_true, y_prob
) -> pd.DataFrame:
    df = pd.DataFrame({
        "session_id": list(session_ids),
        "subject_id": list(subject_ids),
        "label": list(map(int, y_true)),
        "probability": list(map(float, y_prob)),
    })
    df.to_csv(path, index=False)
    return df


@torch.no_grad()
def predict_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, list, list]:
    model.eval()
    ys, ps, sids, subs = [], [], [], []
    for volumes, labels, _ages, _sexes, meta in loader:
        logits = model(volumes.to(device))
        ps.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels.numpy().tolist())
        sids.extend(meta["session_id"])
        subs.extend(meta["subject_id"])
    return np.asarray(ys), np.asarray(ps), sids, subs
