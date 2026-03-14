from __future__ import annotations

import torch
from torch import nn


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


def build_model(model_name: str, num_classes: int, n_mels: int) -> nn.Module:
    if model_name == "cnn":
        return SmallMashupCNN(num_classes=num_classes)
    if model_name == "crnn":
        return SmallMashupCRNN(num_classes=num_classes, n_mels=n_mels)
    raise ValueError(f"Unsupported model: {model_name}")
