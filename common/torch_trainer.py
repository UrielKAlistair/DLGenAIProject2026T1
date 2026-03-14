from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def load_checkpoint(
    resume_from: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, int, dict[str, float] | None]:
    checkpoint = torch.load(resume_from, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val_metrics = checkpoint.get("best_val_metrics")
        return start_epoch, best_epoch, best_val_metrics

    model.load_state_dict(checkpoint)
    return 1, 0, None


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
