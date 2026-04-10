from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from ..common.features import LogMelFrontend
from ..common.torch_trainer import load_checkpoint, run_epoch
from ..common.utils import (
    CLIP_SECONDS,
    GENRES,
    SAMPLE_RATE,
    SyntheticMashupDataset,
    init_wandb,
    load_train_val_datasets,
    make_output_dir,
    save_json,
    seed_everything,
)
from .model import EfficientNetClassifier

SEED = 42

FINETUNE_LEARNING_RATE = 1e-4

N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
SPECAUGMENT_ENABLED = True
TIME_MASK_PARAM = 24
FREQ_MASK_PARAM = 12
NUM_TIME_MASKS = 2
NUM_FREQ_MASKS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet on synthetic mashup log-mels.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--train-samples", type=int, default=1500)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)

    parser.add_argument("--resume-from", type=Path, default=None)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--finetune-learning-rate", type=float, default=FINETUNE_LEARNING_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    output_dir = make_output_dir(args.output_root, args.run_name, "efficientnet_b0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    resume_from = args.resume_from.expanduser().resolve() if args.resume_from is not None else None
    checkpoint_epoch = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu")
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            checkpoint_epoch = int(checkpoint["epoch"])

    train_dir = args.train_dir.expanduser().resolve()
    train_waveforms, val_waveforms = load_train_val_datasets(
        train_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(train_waveforms, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_waveforms, shuffle=False, **loader_kwargs)

    frontend = LogMelFrontend(
        SAMPLE_RATE,
        N_MELS,
        HOP_LENGTH,
        N_FFT,
        specaugment_enabled=SPECAUGMENT_ENABLED,
        time_mask_param=TIME_MASK_PARAM,
        freq_mask_param=FREQ_MASK_PARAM,
        num_time_masks=NUM_TIME_MASKS,
        num_freq_masks=NUM_FREQ_MASKS,
    ).to(device)
    model = EfficientNetClassifier(num_classes=len(GENRES), pretrained=args.pretrained).to(device)
    if args.pretrained and args.freeze_backbone_epochs > 0:
        if checkpoint_epoch <= args.freeze_backbone_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()

    config = {
        "model": "efficientnet",
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "full_data_finetune_samples": args.train_samples + args.val_samples,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "specaugment_enabled": SPECAUGMENT_ENABLED,
        "time_mask_param": TIME_MASK_PARAM,
        "freq_mask_param": FREQ_MASK_PARAM,
        "num_time_masks": NUM_TIME_MASKS,
        "num_freq_masks": NUM_FREQ_MASKS,
        "synthetic_train_dir": str(train_dir),
        "synthetic_stem_gain_db_range": list(SyntheticMashupDataset.DEFAULT_STEM_GAIN_DB_RANGE),
        "synthetic_noise_count_range": list(SyntheticMashupDataset.DEFAULT_NOISE_COUNT_RANGE),
        "synthetic_noise_snr_db_range": list(SyntheticMashupDataset.DEFAULT_NOISE_SNR_DB_RANGE),
        "synthetic_random_crop": SyntheticMashupDataset.DEFAULT_RANDOM_CROP,
        "pretrained": args.pretrained,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "finetune_learning_rate": args.finetune_learning_rate,
    }
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    wandb_run = init_wandb(
        run_name=args.run_name,
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

    unfreeze_epoch = args.freeze_backbone_epochs + 1 if args.freeze_backbone_epochs > 0 else None

    for epoch in range(start_epoch, args.num_epochs + 1):
        if unfreeze_epoch is not None and epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            new_learning_rate = args.finetune_learning_rate or args.learning_rate
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
        train_loss_value, train_metrics = run_epoch(
            frontend,
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.num_epochs,
            mode="train",
        )
        val_loss_value, val_metrics = run_epoch(
            frontend,
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
            mode="val",
        )
        scheduler.step(val_metrics["macro_f1"])
        epoch_seconds = time.time() - epoch_start

        print(
            f"epoch {epoch}/{args.num_epochs} "
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

    if best_val_metrics is not None:
        model.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))
        full_data_loader = DataLoader(ConcatDataset([train_waveforms, val_waveforms]), shuffle=True, **loader_kwargs)
        full_data_optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=optimizer.param_groups[0]["lr"],
        )
        full_data_loss_value, full_data_metrics = run_epoch(
            frontend,
            model,
            full_data_loader,
            criterion,
            full_data_optimizer,
            device,
            args.num_epochs + 1,
            args.num_epochs + 1,
            mode="train",
        )
        torch.save(model.state_dict(), output_dir / "full_data_model.pt")
        print(
            "Completed final full-data fine-tune epoch "
            f"loss={full_data_loss_value:.4f} macro_f1={full_data_metrics['macro_f1']:.4f}"
        )
        summary["full_data_finetune"] = {
            "model_path": "full_data_model.pt",
            "loss": full_data_loss_value,
            "metrics": full_data_metrics,
        }
        if wandb_run is not None:
            wandb_run.log(
                {
                    "full_data_finetune_loss": full_data_loss_value,
                    "full_data_finetune_accuracy": full_data_metrics["accuracy"],
                    "full_data_finetune_macro_f1": full_data_metrics["macro_f1"],
                }
            )
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_val_accuracy"] = best_val_metrics["accuracy"]
            wandb_run.summary["best_val_macro_f1"] = best_val_metrics["macro_f1"]

    save_json(output_dir / "summary.json", summary)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"Saved EfficientNet run to {output_dir}")


if __name__ == "__main__":
    main()
