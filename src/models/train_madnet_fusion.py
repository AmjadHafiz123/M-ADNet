
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


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

DATA_ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
)

PROCESSED_DIR = DATA_ROOT / "Actual_data" / "processed"

FEATURES_CSV = (
    PROCESSED_DIR / "adni_mri_clinical_features.csv"
)

MRI_DIR = PROCESSED_DIR / "mri_full_nifti"

MRI_CHECKPOINT = (
    PROCESSED_DIR
    / "mri_baseline"
    / "best_mri_baseline.pt"
)

OUTPUT_DIR = PROCESSED_DIR / "madnet_fusion"

CLASS_NAMES = ["CN", "MCI", "Dementia"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_INPUT_SHAPE = (64, 80, 80)

RANDOM_STATE = 42


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



class MRIEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1, bias=False),
            nn.GroupNorm(2, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(4, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 24, 3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((4, 5, 5)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(24 * 4 * 5 * 5, NUM_CLASSES),
        )

    def forward(self, mri):

        if tuple(mri.shape[-3:]) != MODEL_INPUT_SHAPE:
            mri = F.interpolate(
                mri,
                size=MODEL_INPUT_SHAPE,
                mode="trilinear",
                align_corners=False,
            )

        features = self.features(mri)
        logits = self.classifier(features)

        return logits


class ClinicalEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, clinical):
        return self.network(clinical)


class MADNetFusion(nn.Module):

    def __init__(self):
        super().__init__()

        self.mri_encoder = MRIEncoder()
        self.clinical_encoder = ClinicalEncoder()

        self.clinical_correction = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(16, NUM_CLASSES),
        )

        # Start with zero clinical correction.
        # This preserves the MRI-only predictions initially.
        final_layer = self.clinical_correction[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, mri, clinical):

        mri_logits = self.mri_encoder(mri)

        clinical_features = self.clinical_encoder(
            clinical
        )

        clinical_adjustment = self.clinical_correction(
            clinical_features
        )

        return mri_logits + clinical_adjustment


# --------------------------------------------------
# Dataset and loaders
# --------------------------------------------------

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

    train_rids = set(train_ds.df["rid"])
    val_rids = set(val_ds.df["rid"])
    test_rids = set(test_ds.df["rid"])

    assert train_rids.isdisjoint(val_rids)
    assert train_rids.isdisjoint(test_rids)
    assert val_rids.isdisjoint(test_rids)

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


# --------------------------------------------------
# MRI checkpoint initialization
# --------------------------------------------------


def initialize_mri_encoder(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    saved_state = checkpoint["model_state_dict"]

    model.mri_encoder.load_state_dict(
        saved_state,
        strict=True,
    )

    print("Loaded complete pretrained MRI model.")
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print(
        "Checkpoint validation macro-F1:",
        checkpoint.get("validation_macro_f1"),
    )

# --------------------------------------------------
# Class weights
# --------------------------------------------------

def calculate_class_weights(dataset):

    labels = dataset.df["target"].map(
        MADNetDataset.LABEL_MAP
    ).astype(int).to_numpy()

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES,
    )

    if np.any(counts == 0):
        raise ValueError(
            f"Training split missing a class: {counts.tolist()}"
        )

    weights = (
        len(labels)
        / (NUM_CLASSES * counts.astype(np.float64))
    )

    weights = weights / weights.mean()

    print(
        "Training class counts:",
        dict(zip(CLASS_NAMES, counts.tolist())),
    )

    print(
        "Class weights:",
        dict(zip(CLASS_NAMES, weights.round(4).tolist())),
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


# --------------------------------------------------
# Training and evaluation epoch
# --------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):

    model.train(training)

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_ids = []
    all_rids = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            mri = batch["mri"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(mri, clinical)

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_examples += batch_size

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.extend(
                labels.detach().cpu().tolist()
            )

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

            all_probabilities.extend(
                probabilities.detach().cpu().tolist()
            )

            all_image_ids.extend(batch["image_id"])
            all_rids.extend(batch["rid"])

    if total_examples == 0:
        raise RuntimeError("No examples processed.")

    return {
        "loss": total_loss / total_examples,

        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),

        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            all_labels,
            all_predictions,
        ),

        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "image_ids": all_image_ids,
        "rids": all_rids,
        "examples": total_examples,
    }


# --------------------------------------------------
# Save utilities
# --------------------------------------------------

def save_json(path, data):

    path.write_text(
        json.dumps(data, indent=2, default=float),
        encoding="utf-8",
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train M-ADNet MRI-clinical fusion model."
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.patience < 1
    ):
        parser.error("Training arguments must be positive.")

    set_seed(args.seed)

    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    print("=" * 68)
    print("M-ADNet multimodal fusion training")
    print("Device:", device)
    print("Output directory:", OUTPUT_DIR)
    print("=" * 68)

    for path in (FEATURES_CSV, MRI_DIR, MRI_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

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
        train_ds,
        args.batch_size,
        True,
        args.seed,
    )

    val_loader = make_loader(
        val_ds,
        args.batch_size,
        False,
        args.seed,
    )

    # Test dataset is constructed and checked for split integrity,
    # but deliberately not evaluated during model selection.

    class_weights = calculate_class_weights(
        train_ds
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    model = MADNetFusion().to(device)

    initialize_mri_encoder(
        model,
        MRI_CHECKPOINT,
    )

    model.eval()

    sample = train_ds[0]

    with torch.no_grad():
        sample_mri = sample["mri"].unsqueeze(0)
        sample_clinical = sample["clinical"].unsqueeze(0)

        fusion_logits = model(sample_mri, sample_clinical)

        mri_logits = model.mri_encoder(sample_mri)

    if not torch.allclose(
        fusion_logits,
        mri_logits,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Fusion initialization does not reproduce MRI predictions."
        )

    print("PASS: Initial fusion predictions match MRI-only predictions.")

    # Use a smaller learning rate for the pretrained MRI encoder.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.mri_encoder.parameters(),
                "lr": args.encoder_learning_rate,
            },
            {
                "params": model.clinical_encoder.parameters(),
                "lr": args.learning_rate,
            },
            {
                "params": model.clinical_correction.parameters(),
                "lr": args.learning_rate,
            },
        ],
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
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            training=False,
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
            "epoch_seconds": round(
                time.time() - epoch_start,
                2,
            ),
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            OUTPUT_DIR / "training_history.csv",
            index=False,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"train F1 {row['train_macro_f1']:.4f} | "
            f"val loss {row['val_loss']:.4f} | "
            f"val F1 {row['val_macro_f1']:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_val_f1:

            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint = {
                "model_state_dict": best_state,
                "model_name": "MADNetFusion",
                "class_names": CLASS_NAMES,
                "label_map": MADNetDataset.LABEL_MAP,
                "model_input_shape": MODEL_INPUT_SHAPE,
                "dataset_input_shape": MADNetDataset.TARGET_SHAPE,
                "epoch": best_epoch,
                "validation_macro_f1": best_val_f1,
                "seed": args.seed,
                "clinical_means": train_ds.clinical_means.to_dict(),
                "clinical_stds": train_ds.clinical_stds.to_dict(),
                "pretrained_mri_checkpoint": str(MRI_CHECKPOINT),
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR / "best_madnet_fusion.pt",
            )

            print("  Saved new best fusion checkpoint.")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No best fusion checkpoint was produced."
        )

    # Restore the best validation-selected model.
    model.load_state_dict(best_state)

    # Recompute validation predictions for the best checkpoint.
    best_val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        optimizer=None,
        device=device,
        training=False,
    )

    y_true = best_val_metrics["labels"]
    y_pred = best_val_metrics["predictions"]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    prediction_df = pd.DataFrame({
        "image_id": best_val_metrics["image_ids"],
        "rid": best_val_metrics["rids"],
        "true_label": [
            CLASS_NAMES[i] for i in y_true
        ],
        "predicted_label": [
            CLASS_NAMES[i] for i in y_pred
        ],
    })

    probabilities = np.asarray(
        best_val_metrics["probabilities"]
    )

    for index, class_name in enumerate(CLASS_NAMES):
        prediction_df[f"prob_{class_name}"] = (
            probabilities[:, index]
        )

    prediction_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    metrics = {
        "best_epoch": best_epoch,
        "validation_macro_f1": float(best_val_f1),
        "validation_accuracy": float(
            best_val_metrics["accuracy"]
        ),
        "validation_balanced_accuracy": float(
            best_val_metrics["balanced_accuracy"]
        ),
        "validation_loss": float(
            best_val_metrics["loss"]
        ),
        "classification_report": report,
        "class_names": CLASS_NAMES,
        "train_visits": len(train_ds),
        "validation_visits": len(val_ds),
        "test_visits": len(test_ds),
        "train_participants": train_ds.df["rid"].nunique(),
        "validation_participants": val_ds.df["rid"].nunique(),
        "test_participants": test_ds.df["rid"].nunique(),
        "test_evaluated": False,
    }

    save_json(
        OUTPUT_DIR / "validation_metrics.json",
        metrics,
    )

    config = {
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "patience": args.patience,
        "model_input_shape": MODEL_INPUT_SHAPE,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "pretrained_mri_checkpoint": str(MRI_CHECKPOINT),
        "test_evaluated": False,
    }

    save_json(
        OUTPUT_DIR / "run_config.json",
        config,
    )

    print("\nBEST FUSION MODEL")
    print("------------------------------")
    print("Best epoch:", best_epoch)
    print("Validation macro-F1:", round(best_val_f1, 4))
    print(
        "Validation accuracy:",
        round(best_val_metrics["accuracy"], 4),
    )
    print(
        "Validation balanced accuracy:",
        round(best_val_metrics["balanced_accuracy"], 4),
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

    print("Confusion matrix:")
    print(cm)

    print(
        "\nTraining time:",
        round((time.time() - start_time) / 60, 2),
        "minutes",
    )

    print("\nTest set was NOT evaluated.")
    print("Outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
