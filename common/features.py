from __future__ import annotations

import librosa
import numpy as np
import torch
from torch import nn


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        hop_length: int,
        n_fft: int = 2048,
    ) -> None:
        super().__init__()
        mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels).astype(np.float32)
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.register_buffer("mel_filter", torch.from_numpy(mel_filter))
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 3:
            waveforms = waveforms.squeeze(1)

        stft = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel = torch.matmul(self.mel_filter.unsqueeze(0), power)
        log_mel = 10.0 * torch.log10(mel.clamp_min(1e-10))
        log_mel = log_mel - log_mel.amax(dim=(-2, -1), keepdim=True)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return ((log_mel - mean) / std).unsqueeze(1)
