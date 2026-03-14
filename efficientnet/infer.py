from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..common.features import LogMelFrontend
from ..common.utils import GENRES, fit_clip_for_inference, load_audio
from .model import EfficientNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EfficientNet inference on test mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()
def main() -> None:
    args = parse_args()
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
    model = EfficientNetClassifier(num_classes=len(GENRES), pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    frontend.eval()
    model.eval()

    with (dataset_root / "test.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    predictions = []
    for row in tqdm(rows, desc="Inference", unit="file"):
        waveform = load_audio(dataset_root / row["filename"], int(config["sample_rate"])).numpy()
        waveform = fit_clip_for_inference(waveform, int(config["sample_rate"]), float(config["clip_seconds"]))
        waveform_tensor = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(frontend(waveform_tensor))
            label_index = int(logits.argmax(dim=1).item())

        predictions.append({"id": row["id"], "genre": GENRES[label_index]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "genre"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved test predictions to {output_path}")
