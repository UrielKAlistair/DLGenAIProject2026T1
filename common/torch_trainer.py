from __future__ import annotations

import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import init_wandb, save_json


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
    num_epochs: int,
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_items = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"train {epoch}/{num_epochs}", leave=False, unit="batch")
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
    num_epochs: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"val {epoch}/{num_epochs}", leave=False, unit="batch")
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
    *,
    num_epochs: int,
    learning_rate: float,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    wandb_run = init_wandb(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        run_name=run_name,
        config=config,
        output_dir=output_dir,
    )

    best_train_metrics: dict[str, float] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
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
        epoch_seconds = time.time() - epoch_start

        print(
            f"epoch {epoch}/{num_epochs} "
            f"train_loss={train_loss_value:.4f} val_loss={val_loss_value:.4f} "
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
                    "train_loss": train_loss_value,
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
