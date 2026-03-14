from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs.repeat(1, 3, 1, 1))

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = True

