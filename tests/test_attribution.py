from __future__ import annotations

import torch
import torch.nn as nn
from brainvuln.mri.attribution import integrated_gradients, smoothgrad


class Tiny3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv3d(1, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).flatten(1)
        return self.fc(h).squeeze(1)


def test_ig_runs():
    m = Tiny3D()
    m.eval()
    x = torch.randn(1, 1, 8, 8, 8)
    attr = integrated_gradients(m, x, n_steps=4)
    assert attr.shape == x.shape
    assert torch.isfinite(attr).all()


def test_smoothgrad_runs():
    m = Tiny3D()
    m.eval()
    x = torch.randn(1, 1, 8, 8, 8)
    attr = smoothgrad(m, x, n_samples=3)
    assert attr.shape == x.shape
    assert torch.isfinite(attr).all()
