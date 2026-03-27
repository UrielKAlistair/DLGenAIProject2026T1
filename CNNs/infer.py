from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from tqdm import tqdm

from ..common.features import LogMelFrontend
from .models import build_model
from ..common.utils import CLIP_SECONDS, GENRES, SAMPLE_RATE, fit_clip, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spectrogram-model inference on test mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", choices=["cnn", "crnn"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
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
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    frontend.eval()
    model.eval()

    test_csv_path = dataset_root / "test.csv"
    with test_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    target_length = int(SAMPLE_RATE * CLIP_SECONDS)
    predictions = []
    for row in tqdm(rows, desc="Inference", unit="file"):
        audio_path = dataset_root / row["filename"]
        waveform = load_audio(audio_path, SAMPLE_RATE)
        waveform_tensor = fit_clip(waveform, target_length).unsqueeze(0).to(device)

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
