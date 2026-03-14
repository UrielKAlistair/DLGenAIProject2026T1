from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..common.features import LogMelFrontend
from .models import build_model
from ..common.torch_trainer import train_model
from ..common.utils import (
    GENRES,
    SyntheticMashupDataset,
    full_train_val,
    get_noise_paths,
    make_output_dir,
    seed_everything,
)

SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0
SEED = 42

WANDB_MODE = "online"
WANDB_PROJECT = "21f3002715-t12026"
WANDB_ENTITY = "arvindanuk-indian-institute-of-technology-madras"

VAL_RATIO = 0.2
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 400

N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_WORKERS = min(8, os.cpu_count() or 1)
PREFETCH_FACTOR = 2

MODEL_CONFIGS = {
    "cnn": {
        "default_prefix": "cnn_mel",
        "num_epochs": 12,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "pretrained": False,
    },
    "crnn": {
        "default_prefix": "crnn_mel",
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "pretrained": False,
    },
    "efficientnet": {
        "default_prefix": "efficientnet_b0",
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "pretrained": False,
    },
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spectrogram model on synthetic mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model_name = args.model
    model_config = MODEL_CONFIGS[model_name]

    seed_everything(SEED)
    output_dir = make_output_dir(args.output_root, args.run_name, model_config["default_prefix"])
    dataset_root = args.dataset_root.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split = full_train_val(dataset_root, VAL_RATIO, SEED)
    noise_paths = get_noise_paths(dataset_root)

    train_waveforms = SyntheticMashupDataset(
        songs_by_genre=train_split,
        num_samples=TRAIN_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED,
        noise_paths=noise_paths,
    )
    val_waveforms = SyntheticMashupDataset(
        songs_by_genre=val_split,
        num_samples=VAL_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED + 999,
        noise_paths=noise_paths,
    )

    loader_kwargs = {
        "batch_size": model_config["batch_size"],
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(train_waveforms, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_waveforms, shuffle=False, **loader_kwargs)

    frontend = LogMelFrontend(SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT).to(device)
    model = build_model(
        model_name,
        num_classes=len(GENRES),
        n_mels=N_MELS,
        pretrained=model_config["pretrained"],
    ).to(device)
    config = {
        "model": model_name,
        "val_ratio": VAL_RATIO,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "num_epochs": model_config["num_epochs"],
        "batch_size": model_config["batch_size"],
        "learning_rate": model_config["learning_rate"],
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "num_workers": NUM_WORKERS,
        "pretrained": model_config["pretrained"],
    }
    train_model(
        frontend,
        model,
        train_loader,
        val_loader,
        output_dir,
        config,
        args.run_name,
        device,
        num_epochs=model_config["num_epochs"],
        learning_rate=model_config["learning_rate"],
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        wandb_mode=WANDB_MODE,
    )

    print(f"Saved {model_name.upper()} run to {output_dir}")


if __name__ == "__main__":
    main()
