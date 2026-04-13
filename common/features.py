from __future__ import annotations

import torch
from torch import nn
import torchaudio.transforms as T


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_mels: int,
        hop_length: int,
        n_fft: int = 2048,
        specaugment_enabled: bool = False,
        time_mask_param: int = 24,
        freq_mask_param: int = 12,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
    ) -> None:
        super().__init__()
        self.specaugment_enabled = specaugment_enabled
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power")
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def _apply_specaugment(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_freq_masks):
            x = self.freq_mask(x)
        for _ in range(self.num_time_masks):
            x = self.time_mask(x)
        return x

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 3:
            waveforms = waveforms.squeeze(1)

        x = self.melspec(waveforms)     # (B, M, T)
        x = self.to_db(x)

        x = x - x.amax(dim=(-2, -1), keepdim=True)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std

        if self.training and self.specaugment_enabled:
            x = self._apply_specaugment(x)

        return x.unsqueeze(1)