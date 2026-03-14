from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..common.features import LogMelFrontend
from ..common.torch_trainer import load_checkpoint, train_one_epoch, val_loss
from ..common.utils import (
    GENRES,
    SyntheticMashupDataset,
    full_train_val,
    init_wandb,
    get_noise_paths,
    make_output_dir,
    save_json,
    seed_everything,
)
from .model import EfficientNetClassifier

SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0
SEED = 42

WANDB_MODE = "online"
WANDB_PROJECT = "21f3002715-t12026"
WANDB_ENTITY = "arvindanuk-indian-institute-of-technology-madras"

VAL_RATIO = 0.2
TRAIN_SAMPLES = 1500
VAL_SAMPLES = 400

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
FINETUNE_LEARNING_RATE = 1e-4

N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
NUM_WORKERS = min(8, os.cpu_count() or 1)
PREFETCH_FACTOR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet on synthetic mashup log-mels.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--finetune-learning-rate", type=float, default=FINETUNE_LEARNING_RATE)
    return parser.parse_args()


def train_model(
    frontend: nn.Module,
    model: EfficientNetClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    config: dict[str, float | int | bool | str],
    run_name: str | None,
    device: torch.device,
    *,
    num_epochs: int,
    learning_rate: float,
    resume_from: Path | None,
    unfreeze_epoch: int | None,
    finetune_learning_rate: float | None,
) -> None:
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    wandb_run = init_wandb(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
        run_name=run_name,
        config=config,
        output_dir=output_dir,
    )

    best_val_metrics: dict[str, float] | None = None
    best_epoch = 0
    start_epoch = 1

    if resume_from is not None:
        start_epoch, best_epoch, best_val_metrics = load_checkpoint(
            resume_from,
            model,
            optimizer,
            scheduler,
            device,
        )
        print(f"Resuming from {resume_from}")

    for epoch in range(start_epoch, num_epochs + 1):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            new_learning_rate = finetune_learning_rate or learning_rate
            optimizer = torch.optim.Adam(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=new_learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )
            print(f"Unfroze backbone at epoch {epoch}. Reset optimizer lr to {new_learning_rate:.2e}")

        epoch_start = time.time()
        train_loss_value, train_metrics = train_one_epoch(
            frontend,
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs,
        )
        val_loss_value, val_metrics = val_loss(
            frontend,
            model,
            val_loader,
            criterion,
            device,
            epoch,
            num_epochs,
        )
        scheduler.step(val_metrics["macro_f1"])
        epoch_seconds = time.time() - epoch_start

        print(
            f"epoch {epoch}/{num_epochs} "
            f"train_loss={train_loss_value:.4f} val_loss={val_loss_value:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"time={epoch_seconds / 60:.1f}m"
        )

        if best_val_metrics is None or val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_val_metrics = val_metrics
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "model.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
            },
            output_dir / "checkpoint.pt",
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_value,
                    "val_loss": val_loss_value,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": epoch_seconds,
                }
            )

    summary = {
        "config": config,
        "best_epoch": best_epoch,
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
    output_dir = make_output_dir(args.output_root, args.run_name, "efficientnet_b0")
    dataset_root = args.dataset_root.expanduser().resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split, val_split = full_train_val(dataset_root, VAL_RATIO, SEED)
    noise_paths = get_noise_paths(dataset_root)

    train_waveforms = SyntheticMashupDataset(
        songs_by_genre=train_split,
        num_samples=args.train_samples,
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
    model = EfficientNetClassifier(num_classes=len(GENRES), pretrained=args.pretrained).to(device)
    if args.pretrained and args.freeze_backbone_epochs > 0:
        model.freeze_backbone()

    config = {
        "model": "efficientnet",
        "val_ratio": VAL_RATIO,
        "train_samples": args.train_samples,
        "val_samples": VAL_SAMPLES,
        "num_epochs": args.num_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": args.learning_rate,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "num_workers": NUM_WORKERS,
        "pretrained": args.pretrained,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "finetune_learning_rate": args.finetune_learning_rate,
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
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from.expanduser().resolve() if args.resume_from is not None else None,
        unfreeze_epoch=args.freeze_backbone_epochs + 1 if args.freeze_backbone_epochs > 0 else None,
        finetune_learning_rate=args.finetune_learning_rate,
    )

    print(f"Saved EfficientNet run to {output_dir}")


if __name__ == "__main__":
    main()
