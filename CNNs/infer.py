from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..common.features import LogMelFrontend
from .models import build_model
from ..common.utils import GENRES, load_audio


def parse_args(fixed_model: str | None) -> argparse.Namespace:
    description = "Run spectrogram-model inference on test mashups."
    if fixed_model is not None:
        description = f"Run {fixed_model.upper()} inference on test mashups."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    if fixed_model is None:
        parser.add_argument("--model", choices=["cnn", "crnn", "efficientnet"], required=True)
    return parser.parse_args()


def fit_clip_for_inference(waveform: np.ndarray, sample_rate: int, clip_seconds: float) -> np.ndarray:
    target_length = int(sample_rate * clip_seconds)
    if waveform.shape[0] == target_length:
        return waveform

    if waveform.shape[0] < target_length:
        padded = np.zeros(target_length, dtype=np.float32)
        padded[: waveform.shape[0]] = waveform.astype(np.float32)
        return padded

    return waveform[:target_length]


def main(fixed_model: str | None = None) -> None:
    args = parse_args(fixed_model)
    model_name = fixed_model or args.model
    dataset_root = args.dataset_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    summary_path = args.summary_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]

    frontend = LogMelFrontend(
        sample_rate=int(config["sample_rate"]),
        n_mels=int(config["n_mels"]),
        hop_length=int(config["hop_length"]),
        n_fft=int(config.get("n_fft", 2048)),
    ).to(device)
    model = build_model(
        model_name,
        num_classes=len(GENRES),
        n_mels=int(config["n_mels"]),
        pretrained=False,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    frontend.eval()
    model.eval()

    test_csv_path = dataset_root / "test.csv"
    with test_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    predictions = []
    for row in tqdm(rows, desc="Inference", unit="file"):
        audio_path = dataset_root / row["filename"]
        waveform = load_audio(audio_path, int(config["sample_rate"])).numpy()
        waveform = fit_clip_for_inference(waveform, int(config["sample_rate"]), float(config["clip_seconds"]))
        waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            inputs = frontend(waveform_tensor)
            logits = model(inputs)
            label_index = int(logits.argmax(dim=1).item())

        predictions.append({"id": row["id"], "genre": GENRES[label_index]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "genre"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved test predictions to {output_path}")


if __name__ == "__main__":
    main()
