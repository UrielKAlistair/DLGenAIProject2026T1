from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

try:
    from ..common.utils import GENRES, fit_clip_for_inference, load_audio
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.utils import GENRES, fit_clip_for_inference, load_audio

SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0
N_MFCC = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatBoost MFCC inference on test mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("DnG/mmfc_trees/outputs/mfcc_catboost_v1/model.pkl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("DnG/mmfc_trees/outputs/mfcc_catboost_v1/submission.csv"),
    )
    return parser.parse_args()

def extract_features(waveform: np.ndarray, sample_rate: int, n_mfcc: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    stacked = np.concatenate([mfcc, delta], axis=0)
    return np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)], axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    test_csv_path = dataset_root / "test.csv"
    predictions = []

    with test_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in tqdm(rows, desc="Inference", unit="file"):
        audio_path = dataset_root / row["filename"]
        waveform = load_audio(audio_path, SAMPLE_RATE).numpy()
        waveform = fit_clip_for_inference(waveform, SAMPLE_RATE, CLIP_SECONDS)
        features = extract_features(waveform, SAMPLE_RATE, N_MFCC).reshape(1, -1)
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
