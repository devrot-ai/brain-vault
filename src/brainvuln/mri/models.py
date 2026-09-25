"""Models (Part F/G): 3D ResNet-18 primary, simple 3D CNN + age/sex baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResNet18Binary(nn.Module):
    """MONAI 3D ResNet-18, one logit → sigmoid probability (primary model)."""

    def __init__(self, in_channels: int = 1, spatial_dims: int = 3) -> None:
        from monai.networks.nets import ResNet

        super().__init__()
        self.backbone = ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=(2, 2, 2),
            widen_factor=1.0,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-pool feature map (for Grad-CAM) and post-pool features."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.act(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)          # [B, 512, d, h, w] — Grad-CAM target
        feat = self.pool(x).flatten(1)
        return x, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward_features(x)
        return self.fc(feat).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


class SimpleCNN3D(nn.Module):
    """Plain 3D CNN baseline: 4 conv blocks + FC head."""

    def __init__(self, in_channels: int = 1, width: int = 16) -> None:
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm3d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            )
        self.features = nn.Sequential(
            block(in_channels, width),      # 128 → 64
            block(width, width * 2),        # 64 → 32
            block(width * 2, width * 4),    # 32 → 16
            block(width * 4, width * 8),    # 16 → 8
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(width * 8, 1)
        self.width = width * 8

    def forward_features(self, x: torch.Tensor):
        fmap = self.features(x)
        return fmap, self.pool(fmap).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward_features(x)
        return self.fc(feat).squeeze(1)


class AgeSexBaseline(nn.Module):
    """Logistic regression on age + sex only — the demographic ceiling.

    Demonstrates what a model can achieve without seeing the brain: if the
    imaging models do not clearly beat this, the volumetric signal adds no
    value (Part G).
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        x = torch.stack([age, sex], dim=1)
        return self.linear(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
