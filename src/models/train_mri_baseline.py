
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.datasets.madnet_dataset import MADNetDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
)
PROCESSED_DIR = DATA_ROOT / "Actual_data" / "processed"

FEATURES_CSV = PROCESSED_DIR / "adni_mri_clinical_features.csv"
MRI_DIR = PROCESSED_DIR / "mri_full_nifti"
OUTPUT_DIR = PROCESSED_DIR / "mri_baseline"

CLASS_NAMES = ["CN", "MCI", "Dementia"]
NUM_CLASSES = len(CLASS_NAMES)

# Downsample the dataset's 128x160x160 tensors inside the model.
# This reduces CPU memory consumption. It is a computational
# compromise and should be evaluated against other input resolutions.
MODEL_INPUT_SHAPE = (64, 80, 80)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Small3DCNN(nn.Module):
    """Compact MRI-only 3D CNN for three-class classification."""

    def __init__(self, dropout: float = 0.35):
        super().__init__()

        
        self.features = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 5, 5)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(24 * 4 * 5 * 5, NUM_CLASSES),
        )


    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        # Input arrives as [B, 1, 128, 160, 160].
        # Reduce resolution before feature extraction.
        if tuple(mri.shape[-3:]) != MODEL_INPUT_SHAPE:
            mri = F.interpolate(
                mri,
                size=MODEL_INPUT_SHAPE,
                mode="trilinear",
                align_corners=False,
            )

        return self.classifier(self.features(mri))


def make_datasets():
    train_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="train",
    )

    shared_stats = {
        "clinical_means": train_ds.clinical_means,
        "clinical_stds": train_ds.clinical_stds,
    }

    val_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="validation",
        **shared_stats,
    )

    test_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="test",
        **shared_stats,
    )

    # Defensive check: no participant should appear in more than one split.
    train_rids = set(train_ds.df["rid"])
    val_rids = set(val_ds.df["rid"])
    test_rids = set(test_ds.df["rid"])

    assert not (train_rids & val_rids), "Train/validation participant overlap"
    assert not (train_rids & test_rids), "Train/test participant overlap"
    assert not (val_rids & test_rids), "Validation/test participant overlap"

    return train_ds, val_ds, test_ds


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator if shuffle else None,
    )


def calculate_class_weights(dataset) -> torch.Tensor:
    labels = dataset.df["target"].map(
        MADNetDataset.LABEL_MAP
    ).astype(int).to_numpy()

    counts = np.bincount(labels, minlength=NUM_CLASSES)

    if np.any(counts == 0):
        raise ValueError(
            f"Training split is missing a class. Class counts: {counts.tolist()}"
        )

    # Inverse-frequency weighting, normalized to mean 1.
    weights = len(labels) / (NUM_CLASSES * counts.astype(np.float64))
    weights = weights / weights.mean()

    print("Training class counts:", dict(zip(CLASS_NAMES, counts.tolist())))
    print("Loss weights:", dict(zip(CLASS_NAMES, weights.round(4).tolist())))

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
    max_batches=None,
):
    model.train(training)

    total_loss = 0.0
    total_examples = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            mri = batch["mri"].to(device)
            labels = batch["label"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(mri)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predictions = logits.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    if total_examples == 0:
        raise RuntimeError("No examples were processed in this epoch.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            all_labels, all_predictions
        ),
        "examples": total_examples,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=float),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a compact MRI-only 3D CNN baseline."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional training batch limit for a quick smoke test.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional validation batch limit for a quick smoke test.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set after training.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive.")

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 68)
    print("M-ADNet MRI-only baseline")
    print("Device:", device)
    print("Model input resolution:", MODEL_INPUT_SHAPE)
    print("Batch size:", args.batch_size)
    print("Output directory:", OUTPUT_DIR)
    print("=" * 68)

    for path in (FEATURES_CSV, MRI_DIR):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    train_ds, val_ds, test_ds = make_datasets()

    print(
        f"Dataset sizes: train={len(train_ds)}, "
        f"validation={len(val_ds)}, test={len(test_ds)}"
    )
    print(
        "Participant counts:",
        f"train={train_ds.df['rid'].nunique()},",
        f"validation={val_ds.df['rid'].nunique()},",
        f"test={test_ds.df['rid'].nunique()}",
    )

    train_loader = make_loader(
        train_ds, args.batch_size, True, args.seed
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.seed
    )
    test_loader = make_loader(
        test_ds, args.batch_size, False, args.seed
    )

    class_weights = calculate_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = Small3DCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            max_batches=args.max_train_batches,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_examples": train_metrics["examples"],
            "val_examples": val_metrics["examples"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "training_history.csv", index=False
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"train F1 {row['train_macro_f1']:.4f} | "
            f"val loss {row['val_loss']:.4f} | "
            f"val macro-F1 {row['val_macro_f1']:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )

        # Maximize validation macro-F1. Strict improvement avoids
        # repeatedly replacing a checkpoint with an equal score.
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())

            checkpoint = {
                "model_state_dict": best_state,
                "model_name": "Small3DCNN",
                "class_names": CLASS_NAMES,
                "label_map": MADNetDataset.LABEL_MAP,
                "model_input_shape": MODEL_INPUT_SHAPE,
                "dataset_input_shape": MADNetDataset.TARGET_SHAPE,
                "epoch": best_epoch,
                "validation_macro_f1": best_val_f1,
                "seed": args.seed,
            }
            torch.save(
                checkpoint,
                OUTPUT_DIR / "best_mri_baseline.pt",
            )
            print("  Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No model checkpoint was produced.")

    save_json(
        OUTPUT_DIR / "run_config.json",
        {
            "seed": args.seed,
            "device": str(device),
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "model_input_shape": MODEL_INPUT_SHAPE,
            "dataset_input_shape": MADNetDataset.TARGET_SHAPE,
            "class_names": CLASS_NAMES,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
            "test_evaluated": args.evaluate_test,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Training time: {(time.time() - start_time) / 60:.1f} minutes")

    if args.evaluate_test:
        # Restore the best validation-selected checkpoint before testing.
        model.load_state_dict(best_state)
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        y_true = test_metrics["labels"]
        y_pred = test_metrics["predictions"]

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))
        )

        save_json(
            OUTPUT_DIR / "test_metrics.json",
            {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "loss": test_metrics["loss"],
                "examples": test_metrics["examples"],
                "classification_report": report,
            },
        )
        pd.DataFrame(
            cm,
            index=[f"true_{name}" for name in CLASS_NAMES],
            columns=[f"pred_{name}" for name in CLASS_NAMES],
        ).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

        print("\nHeld-out test results")
        print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
        print(f"Macro-F1:          {test_metrics['macro_f1']:.4f}")
        print(
            f"Balanced accuracy: "
            f"{test_metrics['balanced_accuracy']:.4f}"
        )
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES)),
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )
        print("Saved test metrics and confusion matrix.")

    print("\nFinished. Outputs are in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
