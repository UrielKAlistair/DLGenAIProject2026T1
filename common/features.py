from __future__ import annotations

import torch
from torch import nn

from torchaudio.functional import melscale_fbanks


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
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.specaugment_enabled = specaugment_enabled
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        mel_filter = melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_mels=n_mels,
            sample_rate=sample_rate,
        ).transpose(0, 1)
        self.register_buffer("mel_filter", mel_filter)
        self.register_buffer("window", torch.hann_window(n_fft))

    def _apply_masks(
        self,
        augmented: torch.Tensor,
        batch_index: int,
        axis_size: int,
        mask_param: int,
        num_masks: int,
        axis: int,
    ) -> None:
        max_width = min(mask_param, axis_size)
        if max_width <= 0:
            return

        for _ in range(num_masks):
            width = int(torch.randint(0, max_width + 1, (1,), device=augmented.device).item())
            if width == 0:
                continue

            start = int(torch.randint(0, axis_size - width + 1, (1,), device=augmented.device).item())
            if axis == 1:
                augmented[batch_index, start : start + width, :] = 0.0
            else:
                augmented[batch_index, :, start : start + width] = 0.0

    def _apply_specaugment(self, log_mel: torch.Tensor) -> torch.Tensor:
        augmented = log_mel.clone()
        _, n_mels, n_frames = augmented.shape

        for batch_index in range(augmented.shape[0]):
            self._apply_masks(
                augmented,
                batch_index,
                n_mels,
                self.freq_mask_param,
                self.num_freq_masks,
                axis=1,
            )
            self._apply_masks(
                augmented,
                batch_index,
                n_frames,
                self.time_mask_param,
                self.num_time_masks,
                axis=2,
            )

        return augmented

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
        normalized = (log_mel - mean) / std
        if self.training and self.specaugment_enabled:
            normalized = self._apply_specaugment(normalized)
        return normalized.unsqueeze(1)
