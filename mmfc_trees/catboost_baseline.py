from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torchaudio import transforms
from torchaudio.functional import compute_deltas
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common.utils import (
    CLIP_SECONDS,
    SAMPLE_RATE,
    SyntheticMashupDataset,
    init_wandb,
    load_train_val_datasets,
    make_output_dir,
    save_json,
    seed_everything,
)

SEED = 42

N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
N_ESTIMATORS = 2000
LEARNING_RATE = 0.03
DEPTH = 5
L2_LEAF_REG = 20.0
RANDOM_STRENGTH = 2.0
BAGGING_TEMPERATURE = 1.0
EARLY_STOPPING_ROUNDS = 100
FEATURE_BATCH_SIZE = 128
NUM_WORKERS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MFCC baseline on synthetic mashups.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-samples", type=int, default=1500)
    parser.add_argument("--val-samples", type=int, default=400)
    return parser.parse_args()


def extract_features(
    dataset: SyntheticMashupDataset,
    mfcc_transform: transforms.MFCC,
    split_name: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader_kwargs = {
        "batch_size": FEATURE_BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }

    loader = DataLoader(dataset, **loader_kwargs)
    feature_batches = []
    label_batches = []

    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc=f"Extract {split_name}", unit="batch"):
            waveforms = waveforms.squeeze(1).to(device, non_blocking=device.type == "cuda")
            mfcc = mfcc_transform(waveforms)
            delta = compute_deltas(mfcc)
            stacked = torch.cat([mfcc, delta], dim=1)
            feature_batch = torch.cat([stacked.mean(dim=2), stacked.std(dim=2)], dim=1)
            feature_batches.append(feature_batch.cpu())
            label_batches.append(labels.cpu())

    features = torch.cat(feature_batches, dim=0).numpy().astype(np.float32)
    target_labels = torch.cat(label_batches, dim=0).numpy().astype(np.int64)
    return features, target_labels


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    output_dir = make_output_dir(args.output_root, args.run_name, "mfcc_catboost")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = args.train_dir.expanduser().resolve()
    train_dataset, val_dataset = load_train_val_datasets(
        train_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        preload_to_ram=False,
    )

    print("Extracting MFCC features...")
    mfcc_transform = transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
        },
    ).to(device)
    x_train, y_train = extract_features(train_dataset, mfcc_transform, "train", device)
    x_val, y_val = extract_features(val_dataset, mfcc_transform, "val", device)

    model_kwargs = {
        "iterations": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "l2_leaf_reg": L2_LEAF_REG,
        "random_strength": RANDOM_STRENGTH,
        "bagging_temperature": BAGGING_TEMPERATURE,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "random_seed": SEED,
        "verbose": 50,
    }
    if device.type == "cuda":
        model_kwargs["task_type"] = "GPU"
        model_kwargs["devices"] = "0"
    else:
        model_kwargs["thread_count"] = -1

    model = CatBoostClassifier(**model_kwargs)
    model.fit(
        x_train,
        y_train,
        eval_set=(x_val, y_val),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    train_predictions = model.predict(x_train)
    val_predictions = model.predict(x_val)
    train_metrics = metrics(y_train, train_predictions)
    val_metrics = metrics(y_val, val_predictions)

    with (output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    # Just Logging beyond here.
    
    config = {
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "iterations": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "depth": DEPTH,
        "l2_leaf_reg": L2_LEAF_REG,
        "random_strength": RANDOM_STRENGTH,
        "bagging_temperature": BAGGING_TEMPERATURE,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "feature_num_workers": NUM_WORKERS,
        "device": device.type,
        "catboost_task_type": model.get_param("task_type") or "CPU",
        "synthetic_train_dir": str(train_dir),
        "synthetic_stem_gain_db_range": list(SyntheticMashupDataset.DEFAULT_STEM_GAIN_DB_RANGE),
        "synthetic_noise_count_range": list(SyntheticMashupDataset.DEFAULT_NOISE_COUNT_RANGE),
        "synthetic_noise_snr_db_range": list(SyntheticMashupDataset.DEFAULT_NOISE_SNR_DB_RANGE),
        "synthetic_random_crop": SyntheticMashupDataset.DEFAULT_RANDOM_CROP,
    }

    summary = {
        "config": config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    save_json(output_dir / "summary.json", summary)

    wandb_run = init_wandb(
        run_name=args.run_name,
        config=config,
        output_dir=output_dir,
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        wandb_run.summary["train_accuracy"] = train_metrics["accuracy"]
        wandb_run.summary["train_macro_f1"] = train_metrics["macro_f1"]
        wandb_run.summary["val_accuracy"] = val_metrics["accuracy"]
        wandb_run.summary["val_macro_f1"] = val_metrics["macro_f1"]
        wandb_run.summary["best_val_macro_f1"] = val_metrics["macro_f1"]
        wandb_run.finish()

    print(f"Saved baseline run to {output_dir}")


if __name__ == "__main__":
    main()
