from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
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

STEM_NAMES = ("drums", "vocals", "bass", "other")
_CLASS_DEFAULT = object()

@dataclass(frozen=True)
class SongItem:
    genre: str
    song_id: str
    stem_paths: Dict[str, Path]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(output_root: Path, run_name: str | None, default_prefix: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = output_root / (run_name or f"{timestamp}_{default_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def full_train_val(dataset_root: Path, val_ratio: float, seed: int) -> Dict[str, object]:
    if not (dataset_root / "genres_stems").exists():
        raise FileNotFoundError(f"Expected {dataset_root / 'genres_stems'}")

    songs_by_genre = get_song_dict(dataset_root)
    train_split, val_split = train_val_split(songs_by_genre, val_ratio, seed)

    print("Train/val split:")
    for genre in GENRES:
        if genre in train_split or genre in val_split:
            num_train = len(train_split.get(genre, []))
            num_val = len(val_split.get(genre, []))
            print(f"  {genre}: train={num_train}, val={num_val}")

    return train_split, val_split


def get_song_dict(dataset_root: Path) -> Dict[str, List[SongItem]]:
    songs_by_genre: Dict[str, List[SongItem]] = {}
    stems_root = dataset_root / "genres_stems"

    for genre in GENRES:
        genre_dir = stems_root / genre
        if not genre_dir.exists():
            continue

        genre_songs: List[SongItem] = []
        for song_dir in sorted(path for path in genre_dir.iterdir() if path.is_dir()):
            stem_paths: Dict[str, Path] = {}
            for stem_name in STEM_NAMES:
                if stem_name == "other":
                    stem_path = song_dir / "other.wav"
                    if not stem_path.exists():
                        stem_path = song_dir / "others.wav"
                else:
                    stem_path = song_dir / f"{stem_name}.wav"

                if not stem_path.exists():
                    raise FileNotFoundError(f"Missing {stem_name} stem under {song_dir}")

                stem_paths[stem_name] = stem_path

            genre_songs.append(SongItem(genre=genre, song_id=song_dir.name, stem_paths=stem_paths))

        songs_by_genre[genre] = genre_songs

    return songs_by_genre


def get_noise_paths(dataset_root: Path) -> List[Path]:
    noise_root = dataset_root / "ESC-50-master" / "audio"
    if not noise_root.exists():
        return []
    return sorted(noise_root.rglob("*.wav"))


def train_val_split(
    songs_by_genre: Dict[str, List[SongItem]],
    val_ratio: float,
    seed: int,
) -> tuple[Dict[str, List[SongItem]], Dict[str, List[SongItem]]]:
    rng = random.Random(seed)
    train_split: Dict[str, List[SongItem]] = {}
    val_split: Dict[str, List[SongItem]] = {}

    for genre, songs in songs_by_genre.items():
        shuffled = songs[:]
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_split[genre] = shuffled
            val_split[genre] = []
            continue

        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)

        val_split[genre] = shuffled[:val_count]
        train_split[genre] = shuffled[val_count:]

    return train_split, val_split


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    return _load_audio_cached(str(path), sample_rate).clone()


@lru_cache(maxsize=4000)
def _load_audio_cached(path_str: str, sample_rate: int) -> torch.Tensor:
    path = Path(path_str)
    waveform_np, source_sr = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(waveform_np.T).float()
    waveform = waveform.mean(dim=0)

    if source_sr != sample_rate:
        waveform = resample(waveform.unsqueeze(0), orig_freq=source_sr, new_freq=sample_rate).squeeze(0)

    return waveform


def fit_clip(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.numel() == target_length:
        return waveform

    if waveform.numel() < target_length:
        return F.pad(waveform, (0, target_length - waveform.numel()))

    return waveform[:target_length]


def fit_clip_for_inference(waveform: np.ndarray, sample_rate: int, clip_seconds: float) -> np.ndarray:
    target_length = int(sample_rate * clip_seconds)
    if waveform.shape[0] == target_length:
        return waveform

    if waveform.shape[0] < target_length:
        padded = np.zeros(target_length, dtype=np.float32)
        padded[: waveform.shape[0]] = waveform.astype(np.float32)
        return padded

    return waveform[:target_length]


class SyntheticMashupDataset(Dataset):
    DEFAULT_STEM_GAIN_DB_RANGE = (-4.0, 4.0)
    DEFAULT_NOISE_COUNT_RANGE = (1, 3)
    DEFAULT_NOISE_SNR_DB_RANGE = (6.0, 18.0)
    DEFAULT_RANDOM_CROP = True

    def __init__(
        self,
        songs_by_genre: Dict[str, List[SongItem]],
        num_samples: int,
        sample_rate: int,
        clip_seconds: float,
        seed: int,
        noise_paths: List[Path] | None = None,
        stem_gain_db_range: tuple[float, float] | object = _CLASS_DEFAULT,
        noise_count_range: tuple[int, int] | object = _CLASS_DEFAULT,
        noise_snr_db_range: tuple[float, float] | None | object = _CLASS_DEFAULT,
        random_crop: bool | object = _CLASS_DEFAULT,
    ) -> None:
        if stem_gain_db_range is _CLASS_DEFAULT:
            stem_gain_db_range = self.DEFAULT_STEM_GAIN_DB_RANGE
        if noise_count_range is _CLASS_DEFAULT:
            noise_count_range = self.DEFAULT_NOISE_COUNT_RANGE
        if noise_snr_db_range is _CLASS_DEFAULT:
            noise_snr_db_range = self.DEFAULT_NOISE_SNR_DB_RANGE
        if random_crop is _CLASS_DEFAULT:
            random_crop = self.DEFAULT_RANDOM_CROP

        self.songs_by_genre = songs_by_genre
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * clip_seconds)
        self.seed = seed
        self.noise_paths = list(noise_paths or [])
        self.stem_gain_db_range = stem_gain_db_range
        self.noise_count_range = noise_count_range
        self.noise_snr_db_range = noise_snr_db_range
        self.random_crop = random_crop
        self.genre_names = [genre for genre, songs in songs_by_genre.items() if songs]
        if not self.genre_names:
            raise ValueError("SyntheticMashupDataset needs at least one genre with songs.")
        if self.noise_count_range[0] < 0 or self.noise_count_range[0] > self.noise_count_range[1]:
            raise ValueError("noise_count_range must satisfy 0 <= min <= max.")

    def __len__(self) -> int:
        return self.num_samples

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + 1009 * index)

    def _sample_songs(self, songs: List[SongItem], rng: random.Random) -> List[SongItem]:
        if len(songs) >= len(STEM_NAMES):
            return rng.sample(songs, len(STEM_NAMES))
        return [rng.choice(songs) for _ in STEM_NAMES]

    def _fit_training_clip(self, waveform: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if waveform.numel() <= self.target_length:
            return fit_clip(waveform, self.target_length)
        if not self.random_crop:
            return waveform[: self.target_length]

        start_idx = rng.randint(0, waveform.numel() - self.target_length)
        return waveform[start_idx : start_idx + self.target_length]

    def _stem_gain(self, rng: random.Random) -> float:
        min_db, max_db = self.stem_gain_db_range
        if max_db <= min_db:
            return 1.0
        gain_db = rng.uniform(min_db, max_db)
        return float(10.0 ** (gain_db / 20.0))

    def _add_noise_clip_with_snr(
        self,
        mixed: torch.Tensor,
        mix_rms: torch.Tensor,
        noise: torch.Tensor,
        rng: random.Random,
    ) -> torch.Tensor:
        if noise.numel() > self.target_length:
            noise = self._fit_training_clip(noise, rng)

        start_idx = rng.randint(0, self.target_length - noise.numel())
        snr_db = rng.uniform(*self.noise_snr_db_range)
        target_noise_rms = mix_rms / (10.0 ** (snr_db / 20.0))
        noise_rms = noise.pow(2.0).mean().sqrt().clamp_min(1e-6)
        scaled_noise = noise * (target_noise_rms / noise_rms)

        mixed[start_idx : start_idx + noise.numel()] += scaled_noise
        return mixed

    def _add_noise(self, mix: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if not self.noise_paths:
            return mix

        mixed = mix.clone()
        num_noises = rng.randint(*self.noise_count_range)
        if num_noises == 0:
            return mixed

        mix_rms = mix.pow(2.0).mean().sqrt().clamp_min(1e-6)
        for _ in range(num_noises):
            noise = load_audio(rng.choice(self.noise_paths), self.sample_rate)
            if self.noise_snr_db_range is None:
                if noise.numel() > self.target_length:
                    noise = noise[: self.target_length]
                start_idx = rng.randint(0, self.target_length - noise.numel())
                intensity = rng.uniform(0.1, 0.4)
                mixed[start_idx : start_idx + noise.numel()] += noise * intensity
                continue

            mixed = self._add_noise_clip_with_snr(mixed, mix_rms, noise, rng)
        return mixed

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        genre = self.genre_names[index % len(self.genre_names)]
        songs = self.songs_by_genre[genre]
        chosen_songs = self._sample_songs(songs, rng)

        stems: List[torch.Tensor] = []
        for song, stem_name in zip(chosen_songs, STEM_NAMES):
            waveform = load_audio(song.stem_paths[stem_name], self.sample_rate)
            stems.append(self._fit_training_clip(waveform, rng))

        stems = [stem * self._stem_gain(rng) for stem in stems]
        mix = torch.stack(stems, dim=0).sum(dim=0)
        mix = self._add_noise(mix, rng)
        peak = mix.abs().max().clamp_min(1e-6)
        mix = mix / peak
        label = torch.tensor(GENRES.index(genre), dtype=torch.long)
        return mix.unsqueeze(0), label


def init_wandb(
    project: str,
    entity: str | None,
    mode: str,
    run_name: str | None,
    config: Dict[str, object],
    output_dir: Path,
):
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError:
        return None

    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=run_name or output_dir.name,
        config=config,
        dir=str(output_dir),
    )
