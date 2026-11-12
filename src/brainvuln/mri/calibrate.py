#!/usr/bin/env python
"""Temperature scaling for probability calibration (Part Z).

Poorly calibrated probabilities (Brier score 0.192, slope 0.34) mean the
model's confidence doesn't match its accuracy. Temperature scaling fixes this
by dividing logits by a learned scalar T before applying sigmoid:

    calibrated_prob = sigmoid(logit / T)

- T > 1: softens overconfident predictions
- T < 1: sharpens underconfident predictions
- T = 1: no change (identity)

Fitted on validation data only; never touches the test set.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brainvuln.mri.dataset import collate_items


class TemperatureScaler(nn.Module):
    """Learnable temperature scalar for post-hoc calibration.

    Optimizes a single parameter T to minimize NLL loss on validation logits.
    The model weights are frozen; only T is tuned.
    """

    def __init__(self, initial_t: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_t, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / self.temperature
        return scaled

    def get_temperature(self) -> float:
        return float(self.temperature.detach().item())


def collect_logits(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on all batches and collect logits and labels.

    Args:
        model: trained model in eval mode.
        loader: DataLoader for validation set.
        device: torch device string.

    Returns:
        (logits, labels) as numpy arrays.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for volumes, labels, ages, sexes, _meta in loader:
            volumes = volumes.to(device)
            ages = ages.to(device)
            sexes = sexes.to(device)

            if getattr(model, "needs_demographics", False):
                logits = model(volumes, ages, sexes)
            else:
                logits = model(volumes)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0).ravel()
    labels = np.concatenate(all_labels, axis=0).ravel()
    return logits, labels


def fit_temperature(
    model,
    val_loader: DataLoader,
    device: str = "cpu",
    initial_t: float = 1.0,
    lr: float = 0.1,
    patience: int = 50,
    max_iters: int = 1000,
) -> float:
    """Fit temperature T on validation logits to minimize NLL loss.

    The model weights are frozen. Only the temperature scalar is optimized.

    Args:
        model: trained model in eval mode.
        val_loader: DataLoader for validation set.
        device: torch device string.
        initial_t: starting temperature (default 1.0).
        lr: learning rate for temperature optimization.
        patience: early stopping patience.
        max_iters: maximum optimization iterations.

    Returns:
        Optimized temperature value.
    """
    device_obj = torch.device(device)

    # Collect all validation logits
    logits_arr, labels_arr = collect_logits(model, val_loader, device)
    logits_t = torch.tensor(logits_arr, dtype=torch.float32, requires_grad=False)
    labels_t = torch.tensor(labels_arr, dtype=torch.float32, requires_grad=False)

    # Create temperature scaler
    scaler = TemperatureScaler(initial_t)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iters)

    nll = nn.functional.binary_cross_entropy_with_logits

    best_loss = float("inf")
    best_t = initial_t
    bad = 0

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits_t)
        loss = nll(scaled_logits, labels_t)
        loss.backward()
        return loss

    for i in range(max_iters):
        loss = optimizer.step(closure)
        current_t = scaler.get_temperature()

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_t = current_t
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            print(f"  Temperature scaling converged at iteration {i} "
                  f"(T={best_t:.4f}, NLL={best_loss:.6f})", flush=True)
            break

    print(f"  Fitted temperature: {best_t:.4f} (NLL={best_loss:.6f})", flush=True)
    return best_t


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities.

    Args:
        logits: raw model logits (any shape).
        temperature: scalar temperature value.

    Returns:
        Calibrated probabilities in [0, 1].
    """
    scaled = logits / temperature
    probs = 1.0 / (1.0 + np.exp(-scaled))
    return probs


def save_calibration_fit(temperature: float, output_path: str | Path) -> None:
    """Save fitted temperature to JSON for later use in inference.

    Args:
        temperature: fitted temperature value.
        output_path: path to save JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"temperature": round(temperature, 6)}, indent=2))
    print(f"  Saved calibration fit to {output_path} (T={temperature:.4f})", flush=True)
