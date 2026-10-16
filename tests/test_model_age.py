from __future__ import annotations

import torch

from brainvuln.mri.model_age import ResNet18AgeAugmented


def test_forward_smoke():
    m = ResNet18AgeAugmented()
    m.eval()
    x = torch.zeros(1, 1, 16, 16, 16)
    age_sex = torch.tensor([[70.0, 1.0]])
    with torch.no_grad():
        out = m(x, age_sex)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()
