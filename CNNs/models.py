from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SmallMashupCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = features.flatten(start_dim=1)
        return self.classifier(features)


class SmallMashupCRNN(nn.Module):
    def __init__(self, num_classes: int, n_mels: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        reduced_mels = max(1, n_mels // 8)
        self.rnn = nn.LSTM(
            input_size=64 * reduced_mels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        batch_size, channels, mel_bins, time_steps = features.shape
        sequence = features.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * mel_bins)
        sequence, _ = self.rnn(sequence)
        pooled = sequence.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs.repeat(1, 3, 1, 1))


def build_model(model_name: str, num_classes: int, n_mels: int, pretrained: bool = False) -> nn.Module:
    if model_name == "cnn":
        return SmallMashupCNN(num_classes=num_classes)
    if model_name == "crnn":
        return SmallMashupCRNN(num_classes=num_classes, n_mels=n_mels)
    if model_name == "efficientnet":
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model: {model_name}")
