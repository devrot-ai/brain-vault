"""Models (Part F/G): 3D ResNet-18 primary, CBAM attention variant, simple 3D CNN + age/sex baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM (Convolutional Block Attention Module).

    Learns to weight feature channels by importance, helping the model focus
    on channels that encode Alzheimer's-relevant patterns (hippocampal atrophy,
    ventricular enlargement) while suppressing irrelevant channels.

    Reference: CBAM, Woo et al., ECCV 2018.
    """

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W]
        avg_out = self.shared_mlp(self.avg_pool(x).flatten(1))
        max_out = self.shared_mlp(self.max_pool(x).flatten(1))
        channel_att = self.sigmoid(avg_out + max_out).view(-1, x.shape[1], 1, 1, 1)
        return x * channel_att


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM adapted for 3D volumes.

    Learns to weight spatial locations by importance, helping the model focus
    on specific brain regions (hippocampus, entorhinal cortex, temporal lobe)
    rather than attending uniformly to the entire volume.

    For 3D volumes, this applies 3D convolution to capture spatial attention
    across all three dimensions simultaneously.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=pad)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W]
        # Average and max pool across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, D, H, W]

        # Concatenate along channel dim: [B, 2, D, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)

        # Apply 3D conv: [B, 1, D, H, W]
        spatial_att = self.sigmoid(self.conv(combined))

        return x * spatial_att


class CBAMBlock(nn.Module):
    """CBAM: Combined Channel and Spatial Attention Module.

    Applies channel attention first, then spatial attention, to refine
    the feature maps. This helps the model learn to focus on
    Alzheimer's-relevant brain regions automatically.
    """

    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class ResNet18Binary(nn.Module):
    """MONAI 3D ResNet-18, one logit → sigmoid probability (primary model)."""

    def __init__(
        self,
        in_channels: int = 1,
        spatial_dims: int = 3,
        widen_factor: float = 1.0,
    ) -> None:
        from monai.networks.nets import ResNet

        super().__init__()
        self.widen_factor = widen_factor
        # widen_factor scales every conv channel count inside MONAI ResNet;
        # 2.0 → ~4× backbone params
        self.backbone = ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=(2, 2, 2),
            widen_factor=widen_factor,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(int(512 * widen_factor), 1)

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
        x = b.layer4(x)          # [B, C, d, h, w] — Grad-CAM target
        feat = self.pool(x).flatten(1)
        return x, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward_features(x)
        return self.fc(feat).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @property
    def needs_demographics(self) -> bool:
        return False


class ResNet18WithDemographics(nn.Module):
    """ResNet-18 backbone + age/sex demographic features fused before the head.

    Concatenates the pooled imaging feature vector with [age, sex] and passes
    through a small MLP. Demographics are known to be predictive of AD risk,
    so fusing them typically adds 3–5% accuracy.
    """

    def __init__(
        self,
        in_channels: int = 1,
        spatial_dims: int = 3,
        widen_factor: float = 1.0,
        demo_dim: int = 2,
        hidden: int = 128,
    ) -> None:
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
            widen_factor=widen_factor,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(int(512 * widen_factor) + demo_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.act(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        feat = self.pool(x).flatten(1)
        return x, feat

    def forward(self, x: torch.Tensor, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward_features(x)
        demo = torch.stack([age, sex], dim=1)
        combined = torch.cat([feat, demo], dim=1)
        return self.fc(combined).squeeze(1)

    def predict_proba(self, x, age, sex) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, age, sex))

    @property
    def needs_demographics(self) -> bool:
        return True


class ResNet18CBAM(nn.Module):
    """ResNet-18 with CBAM attention modules after each layer stage.

    Inserts CBAM (Convolutional Block Attention Module) after layer1, layer2,
    layer3, and layer4 of the ResNet backbone. This teaches the model to
    automatically focus on Alzheimer's-relevant brain regions (hippocampus,
    entorhinal cortex, temporal lobe) while suppressing irrelevant areas.

    Expected improvement: +2-3% accuracy over plain ResNet-18 by learning
    spatial and channel attention patterns specific to AD biomarkers.

    Reference: CBAM, Woo et al., ECCV 2018.
    """

    def __init__(
        self,
        in_channels: int = 1,
        spatial_dims: int = 3,
        widen_factor: float = 1.0,
        att_ratio: int = 16,
        att_kernel: int = 7,
    ) -> None:
        from monai.networks.nets import ResNet

        super().__init__()
        self.widen_factor = widen_factor

        self.backbone = ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            conv1_t_size=7,
            conv1_t_stride=(2, 2, 2),
            widen_factor=widen_factor,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

        # CBAM attention modules after each layer stage
        # Channel counts scale with widen_factor
        c1 = int(64 * widen_factor)
        c2 = int(128 * widen_factor)
        c3 = int(256 * widen_factor)
        c4 = int(512 * widen_factor)

        self.att1 = CBAMBlock(c1, att_ratio, att_kernel)
        self.att2 = CBAMBlock(c2, att_ratio, att_kernel)
        self.att3 = CBAMBlock(c3, att_ratio, att_kernel)
        self.att4 = CBAMBlock(c4, att_ratio, att_kernel)

        self.fc = nn.Linear(int(512 * widen_factor), 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-pool feature map (for Grad-CAM) and post-pool features."""
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.act(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = self.att1(x)  # CBAM after layer1
        x = b.layer2(x)
        x = self.att2(x)  # CBAM after layer2
        x = b.layer3(x)
        x = self.att3(x)  # CBAM after layer3
        x = b.layer4(x)
        x = self.att4(x)  # CBAM after layer4 — Grad-CAM target
        feat = self.pool(x).flatten(1)
        return x, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feat = self.forward_features(x)
        return self.fc(feat).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @property
    def needs_demographics(self) -> bool:
        return False


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
