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
    def __init__(
        self,
        songs_by_genre: Dict[str, List[SongItem]],
        num_samples: int,
        sample_rate: int,
        clip_seconds: float,
        seed: int,
        noise_paths: List[Path] | None = None,
    ) -> None:
        self.songs_by_genre = songs_by_genre
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.target_length = int(sample_rate * clip_seconds)
        self.seed = seed
        self.noise_paths = list(noise_paths or [])
        self.genre_names = [genre for genre, songs in songs_by_genre.items() if songs]
        if not self.genre_names:
            raise ValueError("SyntheticMashupDataset needs at least one genre with songs.")

    def __len__(self) -> int:
        return self.num_samples

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + 1009 * index)

    def _sample_songs(self, songs: List[SongItem], rng: random.Random) -> List[SongItem]:
        if len(songs) >= len(STEM_NAMES):
            return rng.sample(songs, len(STEM_NAMES))
        return [rng.choice(songs) for _ in STEM_NAMES]

    def _add_noise(self, mix: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if not self.noise_paths:
            return mix

        noise = load_audio(rng.choice(self.noise_paths), self.sample_rate)
        if noise.numel() > self.target_length:
            noise = noise[: self.target_length]

        start_idx = rng.randint(0, self.target_length - noise.numel())
        intensity = rng.uniform(0.1, 0.4)
        mixed = mix.clone()
        mixed[start_idx : start_idx + noise.numel()] += noise * intensity
        return mixed

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        genre = self.genre_names[index % len(self.genre_names)]
        songs = self.songs_by_genre[genre]
        chosen_songs = self._sample_songs(songs, rng)

        stems: List[torch.Tensor] = []
        for song, stem_name in zip(chosen_songs, STEM_NAMES):
            waveform = load_audio(song.stem_paths[stem_name], self.sample_rate)
            waveform = fit_clip(waveform, self.target_length)
            stems.append(waveform)

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
