from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LogMelFrontend, SmallMashupCNN
from ..common.utils import (
    GENRES,
    SyntheticMashupDataset,
    full_train_val,
    init_wandb,
    make_output_dir,
    save_json,
    seed_everything,
)

SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0
SEED = 42

WANDB_MODE = "online"
WANDB_PROJECT = "21f3002715-t12026"
WANDB_ENTITY = "arvindanuk-indian-institute-of-technology-madras"

# Worth Playing with these:
VAL_RATIO = 0.2
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 400
NUM_EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_WORKERS = min(8, os.cpu_count() or 1)
PREFETCH_FACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CNN on synthetic mashup mel spectrograms.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def train_one_epoch(
    frontend: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_items = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"train {epoch}/{NUM_EPOCHS}", leave=False, unit="batch")
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        inputs = frontend(inputs)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * targets.size(0)
        total_items += int(targets.size(0))
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / max(total_items, 1)
    return epoch_loss, metrics(all_targets, all_predictions)


@torch.no_grad()
def val_loss(
    frontend: nn.Module,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"val {epoch}/{NUM_EPOCHS}", leave=False, unit="batch")
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        inputs = frontend(inputs)
        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += float(loss.item()) * targets.size(0)
        total_items += int(targets.size(0))
        predictions = logits.argmax(dim=1)
        all_targets.extend(targets.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = total_loss / max(total_items, 1)
    return epoch_loss, metrics(all_targets, all_predictions)


def train_model(
    frontend: nn.Module,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: dict[str, float | int],
    run_name: str | None,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    wandb_run = init_wandb(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        run_name=run_name,
        config=config,
        output_dir=output_dir,
    )

    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_metrics = train_one_epoch(frontend, model, train_loader, criterion, optimizer, device, epoch)
        val_loss_value, val_metrics = val_loss(frontend, model, val_loader, criterion, device, epoch)
        epoch_seconds = time.time() - epoch_start

        print(
            f"epoch {epoch}/{NUM_EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss_value:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"time={epoch_seconds / 60:.1f}m"
        )

        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "model.pt")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss_value,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "epoch_seconds": epoch_seconds,
                }
            )

    summary = {
        "config": config,
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
    }
    save_json(output_dir / "summary.json", summary)

    if wandb_run is not None and best_val_metrics is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_accuracy"] = best_val_metrics["accuracy"]
        wandb_run.summary["best_val_macro_f1"] = best_val_metrics["macro_f1"]
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    output_dir = make_output_dir(args.output_root, args.run_name, "cnn_mel")
    dataset_root = args.dataset_root.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split = full_train_val(dataset_root, VAL_RATIO, SEED)

    train_waveforms = SyntheticMashupDataset(
        songs_by_genre=train_split,
        num_samples=TRAIN_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED,
    )
    val_waveforms = SyntheticMashupDataset(
        songs_by_genre=val_split,
        num_samples=VAL_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED + 999,
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(train_waveforms, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_waveforms, shuffle=False, **loader_kwargs)

    frontend = LogMelFrontend(SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT).to(device)
    model = SmallMashupCNN(num_classes=len(GENRES)).to(device)
    config = {
        "val_ratio": VAL_RATIO,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "num_workers": NUM_WORKERS,
    }
    train_model(frontend, model, train_loader, val_loader, output_dir, config, args.run_name, device)

    print(f"Saved CNN run to {output_dir}")


if __name__ == "__main__":
    main()
