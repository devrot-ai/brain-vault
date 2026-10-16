"""Model-attribution utilities (Item 5): Integrated Gradients + SmoothGrad."""

from __future__ import annotations

import torch


def integrated_gradients(model, x, baseline=None, n_steps: int = 8):
    """Compute IG attributions for logit(output) w.r.t. input x."""
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(x)
    diff = x - baseline
    attrib = torch.zeros_like(x)
    model.zero_grad()
    for k in range(n_steps):
        alpha = (k + 1) / n_steps
        xi = (baseline + alpha * diff).requires_grad_(True)
        out = model(xi)
        (out.sum()).backward()
        grad = xi.grad
        attrib = attrib + grad
    attrib = diff * (attrib / n_steps)
    return attrib.detach()


def smoothgrad(model, x, sigma: float = 0.15, n_samples: int = 8):
    g_sum = torch.zeros_like(x)
    for _ in range(n_samples):
        noisy = x + sigma * torch.randn_like(x)
        noisy.requires_grad_(True)
        out = model(noisy)
        (out.sum()).backward()
        g_sum = g_sum + noisy.grad
        model.zero_grad()
    return (g_sum / n_samples).detach()
