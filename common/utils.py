from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio import load as load_waveform
from torchaudio.functional import resample
from torch.utils.data import Dataset

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0
STEM_NAMES = ("drums", "vocals", "bass", "other")
_AUDIO_CACHE: dict[tuple[str, int], torch.Tensor] = {}
WANDB_MODE = "offline"
WANDB_PROJECT = "21f3002715-t12026"
WANDB_ENTITY = "arvindanuk-indian-institute-of-technology-madras"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_output_dir(output_root: Path, run_name: str | None, default_prefix: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / (run_name or f"{timestamp}_{default_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    cache_key = (str(path), sample_rate)
    waveform = _AUDIO_CACHE.get(cache_key)

    if waveform is None:
        try:
            waveform, source_sr = load_waveform(str(path))
            waveform = waveform.mean(dim=0)
        except (RuntimeError, OSError):
            waveform_np, source_sr = sf.read(str(path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(waveform_np.mean(axis=1))

        if source_sr != sample_rate:
            waveform = resample(waveform.unsqueeze(0), orig_freq=source_sr, new_freq=sample_rate).squeeze(0)

        _AUDIO_CACHE[cache_key] = waveform

    return waveform


def fit_clip(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.numel() == target_length:
        return waveform

    if waveform.numel() < target_length:
        return F.pad(waveform, (0, target_length - waveform.numel()))

    return waveform[:target_length]


class SyntheticMashupDataset(Dataset):
    DEFAULT_STEM_GAIN_DB_RANGE = (-4.0, 4.0)
    DEFAULT_NOISE_COUNT_RANGE = (1, 3)
    DEFAULT_NOISE_SNR_DB_RANGE = (6.0, 18.0)
    DEFAULT_RANDOM_CROP = True

    def __init__(
        self,
        dataset_dir: Path,
        split_name: str,
        sample_indices: list[int],
    ) -> None:
        self.dataset_dir = dataset_dir
        self.split_name = split_name
        self.sample_indices = sample_indices
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Missing synthetic train-data directory: {self.dataset_dir}. "
                "Run `python -m DnG.common.build_train_data` first."
            )
        self.tensor_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.sample_indices)

    def _data_file(self) -> Path:
        preferred_name = "val.pt" if self.split_name == "val" else "data.pt"
        preferred_path = self.dataset_dir / preferred_name
        if preferred_path.exists():
            return preferred_path

        legacy_path = self.dataset_dir / "data.pt"
        if legacy_path.exists():
            return legacy_path

        return preferred_path

    def _load_dataset_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        data_file = self._data_file()
        if not data_file.exists():
            raise FileNotFoundError(
                f"Missing synthetic train-data file: {data_file}. "
                "Run `python -m DnG.common.build_train_data` first."
            )
        payload = torch.load(data_file, map_location="cpu", weights_only=True, mmap=True)
        return payload["waveforms"], payload["labels"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tensor_cache is None:
            self.tensor_cache = self._load_dataset_tensors()
        waveforms, labels = self.tensor_cache
        sample_index = self.sample_indices[index]
        return waveforms[sample_index], labels[sample_index]


def load_dataset(
    train_dir: Path,
    split_name: str,
    *,
    requested_num_samples: int,
) -> SyntheticMashupDataset:
    manifest_path = train_dir / f"{split_name}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing synthetic train-data manifest: {manifest_path}. "
            "Run `python -m DnG.common.build_train_data` first."
        )

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    cached_num_samples = int(manifest["num_samples"])
    cached_sample_rate = int(manifest["sample_rate"])
    cached_clip_seconds = float(manifest["clip_seconds"])
    if requested_num_samples > cached_num_samples:
        raise ValueError(
            f"{split_name} manifest has {cached_num_samples} samples, "
            f"but loader requested {requested_num_samples}."
        )
    if cached_sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{split_name} manifest has sample_rate={cached_sample_rate}, "
            f"but code expects {SAMPLE_RATE}."
        )
    if cached_clip_seconds != CLIP_SECONDS:
        raise ValueError(
            f"{split_name} manifest has clip_seconds={cached_clip_seconds}, "
            f"but code expects {CLIP_SECONDS}."
        )
    dataset_dir = (train_dir / str(manifest["dataset_dir"])).resolve()
    if requested_num_samples == cached_num_samples:
        sample_indices = list(range(cached_num_samples))
    else:
        subset_rng = random.Random(f"{dataset_dir.name}:{split_name}:{requested_num_samples}")
        sample_indices = sorted(subset_rng.sample(range(cached_num_samples), requested_num_samples))
    return SyntheticMashupDataset(
        dataset_dir=dataset_dir,
        split_name=split_name,
        sample_indices=sample_indices,
    )


def load_train_val_datasets(
    train_dir: Path,
    *,
    train_samples: int,
    val_samples: int,
) -> tuple[SyntheticMashupDataset, SyntheticMashupDataset]:
    train_dataset = load_dataset(
        train_dir,
        "train",
        requested_num_samples=train_samples,
    )
    val_dataset = load_dataset(
        train_dir,
        "val",
        requested_num_samples=val_samples,
    )
    return train_dataset, val_dataset


def init_wandb(
    run_name: str | None,
    config: dict[str, object],
    output_dir: Path,
):
    if WANDB_MODE == "disabled":
        return None

    try:
        import wandb
    except ImportError:
        return None

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        name=run_name or output_dir.name,
        config=config,
        dir=str(output_dir),
    )
