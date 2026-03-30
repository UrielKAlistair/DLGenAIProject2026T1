from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torchaudio import transforms
from torchaudio.functional import compute_deltas
from tqdm import tqdm

try:
    from ..common.utils import CLIP_SECONDS, GENRES, SAMPLE_RATE, fit_clip, load_audio
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.utils import CLIP_SECONDS, GENRES, SAMPLE_RATE, fit_clip, load_audio

N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatBoost MFCC inference on test mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def extract_features(waveform: torch.Tensor, mfcc_transform: transforms.MFCC) -> np.ndarray:
    mfcc = mfcc_transform(waveform.unsqueeze(0)).squeeze(0)
    delta = compute_deltas(mfcc.unsqueeze(0)).squeeze(0)
    stacked = torch.cat([mfcc, delta], dim=0)
    features = torch.cat([stacked.mean(dim=1), stacked.std(dim=1)], dim=0)
    return features.numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    mfcc_transform = transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
        },
    )

    test_csv_path = dataset_root / "test.csv"
    predictions = []

    with test_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in tqdm(rows, desc="Inference", unit="file"):
        audio_path = dataset_root / row["filename"]
        waveform = load_audio(audio_path, SAMPLE_RATE)
        waveform = fit_clip(waveform, int(SAMPLE_RATE * CLIP_SECONDS))
        features = extract_features(waveform, mfcc_transform).reshape(1, -1)
        prediction = model.predict(features)
        label_index = int(np.asarray(prediction).reshape(-1)[0])
        predictions.append({"id": row["id"], "genre": GENRES[label_index]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "genre"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved test predictions to {output_path}")


if __name__ == "__main__":
    main()
