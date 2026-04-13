from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .utils import (
    CLIP_SECONDS,
    GENRES,
    SAMPLE_RATE,
    STEM_NAMES,
    SyntheticMashupDataset,
    fit_clip,
    load_audio,
    save_json,
    seed_everything,
)

SEED = 42
VAL_RATIO = 0.2
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 400


def discover_songs(dataset_root: Path) -> dict[str, list[dict[str, object]]]:
    songs_by_genre: dict[str, list[dict[str, object]]] = {}
    stems_root = dataset_root / "genres_stems"

    for genre in GENRES:
        genre_dir = stems_root / genre
        if not genre_dir.exists():
            continue

        genre_songs: list[dict[str, object]] = []
        for song_dir in sorted(path for path in genre_dir.iterdir() if path.is_dir()):
            stem_paths = {}
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

            genre_songs.append({"song_id": song_dir.name, "stem_paths": stem_paths})

        songs_by_genre[genre] = genre_songs

    return songs_by_genre


def split_train_val(
    songs_by_genre: dict[str, list[dict[str, object]]],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]]]:
    rng = random.Random(seed)
    train_split: dict[str, list[dict[str, object]]] = {}
    val_split: dict[str, list[dict[str, object]]] = {}

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


def resolve_dataset_dir(
    train_dir: Path,
    *,
    songs_by_genre: dict[str, list[dict[str, object]]],
    num_samples: int,
    seed: int,
    noise_paths: list[Path],
) -> Path:
    signature_payload = {
        "genres": {
            genre: [str(song["song_id"]) for song in songs]
            for genre, songs in sorted(songs_by_genre.items())
        },
        "num_samples": num_samples,
        "sample_rate": SAMPLE_RATE,
        "target_length": int(SAMPLE_RATE * CLIP_SECONDS),
        "seed": seed,
        "noise_paths": [str(path) for path in noise_paths],
        "stem_gain_db_range": SyntheticMashupDataset.DEFAULT_STEM_GAIN_DB_RANGE,
        "noise_count_range": SyntheticMashupDataset.DEFAULT_NOISE_COUNT_RANGE,
        "noise_snr_db_range": SyntheticMashupDataset.DEFAULT_NOISE_SNR_DB_RANGE,
        "random_crop": SyntheticMashupDataset.DEFAULT_RANDOM_CROP,
    }
    digest = hashlib.sha1(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return train_dir / digest


def build_dataset(
    train_dir: Path,
    *,
    songs_by_genre: dict[str, list[dict[str, object]]],
    num_samples: int,
    seed: int,
    noise_paths: list[Path],
    split_name: str,
) -> Path:
    dataset_dir = resolve_dataset_dir(
        train_dir,
        songs_by_genre=songs_by_genre,
        num_samples=num_samples,
        seed=seed,
        noise_paths=noise_paths,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    nonempty_genres = [genre for genre, songs in songs_by_genre.items() if songs]
    if not nonempty_genres:
        raise ValueError("Synthetic train data needs at least one genre with songs.")

    min_noise_count, max_noise_count = SyntheticMashupDataset.DEFAULT_NOISE_COUNT_RANGE
    if min_noise_count < 0 or min_noise_count > max_noise_count:
        raise ValueError("noise_count_range must satisfy 0 <= min <= max.")

    print(f"Building {split_name} data in {dataset_dir}")

    target_length = int(SAMPLE_RATE * CLIP_SECONDS)
    data_file = dataset_dir / "data.pt"
    if not data_file.exists():
        final_waveforms = torch.empty((num_samples, 1, target_length), dtype=torch.float32)
        final_labels = torch.empty((num_samples,), dtype=torch.long)

        progress = tqdm(range(num_samples), desc=f"build {split_name}", unit="sample")
        for sample_index in progress:
            rng = random.Random(seed + 1009 * sample_index)
            genre = nonempty_genres[sample_index % len(nonempty_genres)]
            songs = songs_by_genre[genre]
            chosen_songs = rng.sample(songs, len(STEM_NAMES))

            stems = []
            for song, stem_name in zip(chosen_songs, STEM_NAMES):
                waveform = load_audio(song["stem_paths"][stem_name], SAMPLE_RATE)
                if waveform.numel() <= target_length:
                    waveform = fit_clip(waveform, target_length)
                elif SyntheticMashupDataset.DEFAULT_RANDOM_CROP:
                    start_idx = rng.randint(0, waveform.numel() - target_length)
                    waveform = waveform[start_idx : start_idx + target_length]
                else:
                    waveform = waveform[:target_length]

                min_db, max_db = SyntheticMashupDataset.DEFAULT_STEM_GAIN_DB_RANGE
                if max_db > min_db:
                    gain_db = rng.uniform(min_db, max_db)
                    waveform = waveform * float(10.0 ** (gain_db / 20.0))

                stems.append(waveform)

            mix = torch.stack(stems, dim=0).sum(dim=0)
            if noise_paths and SyntheticMashupDataset.DEFAULT_NOISE_SNR_DB_RANGE is not None:
                mixed = mix.clone()
                num_noises = rng.randint(min_noise_count, max_noise_count)
                mix_rms = mix.pow(2.0).mean().sqrt().clamp_min(1e-6)

                for _ in range(num_noises):
                    noise = load_audio(rng.choice(noise_paths), SAMPLE_RATE)
                    if noise.numel() > target_length:
                        if SyntheticMashupDataset.DEFAULT_RANDOM_CROP:
                            start_idx = rng.randint(0, noise.numel() - target_length)
                            noise = noise[start_idx : start_idx + target_length]
                        else:
                            noise = noise[:target_length]

                    start_idx = rng.randint(0, target_length - noise.numel())
                    snr_db = rng.uniform(*SyntheticMashupDataset.DEFAULT_NOISE_SNR_DB_RANGE)
                    target_noise_rms = mix_rms / (10.0 ** (snr_db / 20.0))
                    noise_rms = noise.pow(2.0).mean().sqrt().clamp_min(1e-6)
                    scaled_noise = noise * (target_noise_rms / noise_rms)
                    mixed[start_idx : start_idx + noise.numel()] += scaled_noise

                mix = mixed

            peak = mix.abs().max().clamp_min(1e-6)
            final_waveforms[sample_index] = (mix / peak).unsqueeze(0)
            final_labels[sample_index] = GENRES.index(genre)

        temp_file = data_file.with_suffix(".tmp")
        torch.save(
            {
                "waveforms": final_waveforms,
                "labels": final_labels,
            },
            temp_file,
        )
        temp_file.replace(data_file)

    save_json(
        train_dir / f"{split_name}.json",
        {
            "split": split_name,
            "dataset_dir": dataset_dir.name,
            "num_samples": num_samples,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": CLIP_SECONDS,
        },
    )
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic train data to disk.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)

    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=VAL_SAMPLES)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    dataset_root = args.dataset_root.expanduser().resolve()
    train_dir = args.train_dir.expanduser().resolve()
    train_dir.mkdir(parents=True, exist_ok=True)

    stems_root = dataset_root / "genres_stems"
    if not stems_root.exists():
        raise FileNotFoundError(f"Expected {stems_root}")

    songs_by_genre = discover_songs(dataset_root)
    train_split, val_split = split_train_val(songs_by_genre, args.val_ratio, SEED)

    print("Train/val split:")
    for genre in GENRES:
        if genre in train_split or genre in val_split:
            num_train = len(train_split.get(genre, []))
            num_val = len(val_split.get(genre, []))
            print(f"  {genre}: train={num_train}, val={num_val}")

    noise_root = dataset_root / "ESC-50-master" / "audio"
    noise_paths = [] if not noise_root.exists() else sorted(noise_root.rglob("*.wav"))

    build_dataset(
        train_dir,
        songs_by_genre=train_split,
        num_samples=args.train_samples,
        seed=SEED,
        noise_paths=noise_paths,
        split_name="train",
    )
    build_dataset(
        train_dir,
        songs_by_genre=val_split,
        num_samples=args.val_samples,
        seed=SEED + 999,
        noise_paths=noise_paths,
        split_name="val",
    )

    print(f"Finished building train data under {train_dir}")


if __name__ == "__main__":
    main()
