from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from catboost import CatBoostClassifier
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..common.utils import (
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

N_MFCC = 40
N_ESTIMATORS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MFCC baseline on synthetic mashups.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def extract_features(dataset: SyntheticMashupDataset, sample_rate: int, n_mfcc: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []

    for index in range(len(dataset)):
        waveform, label = dataset[index]
        waveform_np = waveform.squeeze(0).numpy()
        mfcc = librosa.feature.mfcc(y=waveform_np, sr=sample_rate, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        stacked = np.concatenate([mfcc, delta], axis=0)
        feature_vector = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)], axis=0).astype(np.float32)
        features.append(feature_vector)
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
    dataset_root = args.dataset_root.expanduser().resolve()

    train_split, val_split = full_train_val(dataset_root, VAL_RATIO, SEED)

    train_dataset = SyntheticMashupDataset(
        songs_by_genre=train_split,
        num_samples=TRAIN_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED,
    )
    val_dataset = SyntheticMashupDataset(
        songs_by_genre=val_split,
        num_samples=VAL_SAMPLES,
        sample_rate=SAMPLE_RATE,
        clip_seconds=CLIP_SECONDS,
        seed=SEED + 999,
    )

    print("Extracting MFCC features...")
    x_train, y_train = extract_features(train_dataset, SAMPLE_RATE, N_MFCC)
    x_val, y_val = extract_features(val_dataset, SAMPLE_RATE, N_MFCC)

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
        "val_ratio": VAL_RATIO,
        "train_samples": TRAIN_SAMPLES,
        "val_samples": VAL_SAMPLES,
    }

    summary = {
        "config": config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    save_json(output_dir / "summary.json", summary)

    wandb_run = init_wandb(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode=WANDB_MODE,
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
