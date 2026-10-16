"""Age-augmented ResNet variant for confound-attack experimental work (Item 2)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResNet18AgeAugmented(nn.Module):
    def __init__(self) -> None:
        from monai.networks.nets import ResNet

        super().__init__()
        self.backbone = ResNet(
            block="basic",
            layers=(2, 2, 2, 2),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_size=7,
            conv1_t_stride=(2, 2, 2),
            widen_factor=1.0,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.age_head = nn.Sequential(nn.Linear(2, 32), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512 + 32, 1)

    def forward(self, x, age_sex):
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
        aux = self.age_head(age_sex)
        return self.fc(torch.cat([feat, aux], dim=1)).squeeze(1)


def age_sex_normalize(values, mean, std):
    arr = (values - mean) / std
    return arr
