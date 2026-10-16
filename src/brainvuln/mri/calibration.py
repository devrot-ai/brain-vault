"""Calibration and operating-point selection for BrainVuln inference.

The primary model's probabilities are overconfident (calibration slope 0.34);
this module fits a scalar temperature on VALIDATION logits and selects an
operating threshold that minimises an explicit cost, so serving code can
apply both (temperature first, threshold second) without re-tuning on test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def fit_temperature(
    logits: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    lr: float = 0.01,
    n_steps: int = 500,
    eps: float = 1e-8,
) -> float:
    """Fit a scalar temperature T (>=0.5) minimising negative log-likelihood.

    T=1 leaves inputs unchanged; T>1 softens (corrects over-confidence),
    T<1 sharpens. Fitted on validation data only; never re-fit on test.
    """
    z = torch.as_tensor(logits, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(labels, dtype=torch.float64).reshape(-1)
    if z.shape[0] != y.shape[0] or z.shape[0] < 2:
        raise ValueError("need at least 2 matching (logit,label) pairs")
    log_t = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.Adam([log_t], lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = bce(z / (torch.exp(log_t) + eps), y)
        loss.backward()
        opt.step()
    t = float(torch.exp(log_t).clamp(min=0.5).detach())
    return t if np.isfinite(t) else 1.0


def apply_temperature(
    logits: np.ndarray | torch.Tensor, temperature: float = 1.0
) -> np.ndarray:
    temps = max(float(temperature), 1e-6)
    z = np.asarray(logits, dtype=float).reshape(-1) / temps
    return 1.0 / (1.0 + np.exp(-z))


def cost_optimal_threshold(
    probabilities: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    n_grid: int = 1001,
) -> tuple[float, float]:
    """Choose threshold in (0,1) minimising cost_fp*FP + cost_fn*FN (validation).

    Returns (threshold, expected_cost_per_subject).
    """
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels).astype(int).reshape(-1)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and labels must have the same non-zero shape")
    grid = np.linspace(1e-3, 1 - 1e-3, n_grid)
    best_t, best_cost = 0.5, float("inf")
    for t in grid:
        pred = (p >= t).astype(int)
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t, best_cost / max(1, y.size)


def youden_threshold(
    probabilities: np.ndarray | list[float], labels: np.ndarray | list[int]
) -> float:
    """Validation-only Youden's J threshold (legacy primary method)."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels).astype(int).reshape(-1)
    grid = np.linspace(1e-3, 1 - 1e-3, 197)
    best_j, best_t = -1.0, 0.5
    for t in grid:
        pred = (p >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        tn = float(((pred == 0) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def save_artifact(
    path: Path, temperature: float, threshold: float, method: str = "cost_optimal"
) -> None:
    """Persist T + threshold to JSON for serving-layer use."""
    payload = {
        "temperature": float(temperature),
        "threshold": float(threshold),
        "threshold_method": method,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
