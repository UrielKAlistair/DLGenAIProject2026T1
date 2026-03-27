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
N_ESTIMATORS = 300


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
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for index in range(len(dataset)):
        waveform, label = dataset[index]
        mfcc = mfcc_transform(waveform).squeeze(0)
        delta = compute_deltas(mfcc.unsqueeze(0)).squeeze(0)
        stacked = torch.cat([mfcc, delta], dim=0)
        feature_vector = torch.cat([stacked.mean(dim=1), stacked.std(dim=1)], dim=0)
        features.append(feature_vector.numpy().astype(np.float32))
        labels.append(int(label.item()))

    return np.stack(features), np.array(labels, dtype=np.int64)


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    output_dir = make_output_dir(args.output_root, args.run_name, "mfcc_catboost")
    train_dir = args.train_dir.expanduser().resolve()
    train_dataset, val_dataset = load_train_val_datasets(
        train_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )

    print("Extracting MFCC features...")
    mfcc_transform = transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)
    x_train, y_train = extract_features(train_dataset, mfcc_transform)
    x_val, y_val = extract_features(val_dataset, mfcc_transform)

    model = CatBoostClassifier(
        iterations=N_ESTIMATORS,
        loss_function="MultiClass",
        random_seed=SEED,
        verbose=False,
        thread_count=-1,
    )
    model.fit(x_train, y_train)

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
