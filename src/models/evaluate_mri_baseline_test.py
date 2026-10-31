
from __future__ import annotations

import json
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


DATA_ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
)
PROCESSED_DIR = DATA_ROOT / "Actual_data" / "processed"

FEATURES_CSV = PROCESSED_DIR / "adni_mri_clinical_features.csv"
MRI_DIR = PROCESSED_DIR / "mri_full_nifti"

CHECKPOINT_PATH = (
    PROCESSED_DIR / "mri_baseline" / "best_mri_baseline.pt"
)

FUSION_PREDICTIONS = (
    PROCESSED_DIR
    / "madnet_fusion"
    / "test_evaluation"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "mri_baseline"
    / "test_evaluation"
)

CLASS_NAMES = ["CN", "MCI", "Dementia"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_INPUT_SHAPE = (64, 80, 80)


class Small3DCNN(nn.Module):
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
        if tuple(mri.shape[-3:]) != MODEL_INPUT_SHAPE:
            mri = F.interpolate(
                mri,
                size=MODEL_INPUT_SHAPE,
                mode="trilinear",
                align_corners=False,
            )

        return self.classifier(self.features(mri))


def save_json(path: Path, data) -> None:
    path.write_text(
        json.dumps(data, indent=2, default=float),
        encoding="utf-8",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (FEATURES_CSV, MRI_DIR, CHECKPOINT_PATH,
                 FUSION_PREDICTIONS):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")

    print("=" * 65)
    print("M-ADNet MRI-only final test evaluation")
    print("=" * 65)

    # Fit preprocessing statistics on the training split only.
    train_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="train",
    )

    shared_stats = {
        "clinical_means": train_ds.clinical_means,
        "clinical_stds": train_ds.clinical_stds,
    }

    test_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="test",
        **shared_stats,
    )

    train_rids = set(train_ds.df["rid"])
    test_rids = set(test_ds.df["rid"])

    assert not (train_rids & test_rids), (
        "Participant leakage between training and test sets."
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = Small3DCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    all_labels = []
    all_predictions = []
    all_image_ids = []
    all_rids = []

    with torch.no_grad():
        for batch in test_loader:
            mri = batch["mri"].to(device)
            labels = batch["label"].to(device)

            logits = model(mri)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_image_ids.extend(batch["image_id"])
            all_rids.extend(
                [int(rid) for rid in batch["rid"]]
            )

    # Calculate test metrics.
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )
    balanced_acc = balanced_accuracy_score(
        all_labels, all_predictions
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    # Save individual predictions.
    predictions_df = pd.DataFrame({
        "image_id": all_image_ids,
        "rid": all_rids,
        "true_label": [CLASS_NAMES[x] for x in all_labels],
        "predicted_label": [
            CLASS_NAMES[x] for x in all_predictions
        ],
        "correct": [
            true == pred
            for true, pred in zip(all_labels, all_predictions)
        ],
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save confusion matrix.
    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    # Save metrics.
    save_json(
        OUTPUT_DIR / "test_metrics.json",
        {
            "checkpoint_epoch": checkpoint.get("epoch"),
            "selected_validation_macro_f1":
                checkpoint.get("validation_macro_f1"),
            "test_visits": len(all_labels),
            "test_participants": len(set(all_rids)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "classification_report": report,
        },
    )

    # Verify that this evaluation uses exactly the same
    # test visits and labels as the fusion evaluation.
    fusion_df = pd.read_csv(FUSION_PREDICTIONS)

    required_columns = {
        "image_id", "true_label"
    }
    if not required_columns.issubset(fusion_df.columns):
        raise ValueError(
            "Fusion predictions file must contain image_id "
            "and true_label columns."
        )

    if predictions_df["image_id"].duplicated().any():
        raise ValueError("Duplicate MRI-only test image IDs.")

    if fusion_df["image_id"].duplicated().any():
        raise ValueError("Duplicate fusion test image IDs.")

    comparison = predictions_df[
        ["image_id", "true_label", "predicted_label"]
    ].merge(
        fusion_df[["image_id", "true_label", "predicted_label"]],
        on="image_id",
        how="outer",
        suffixes=("_mri", "_fusion"),
        indicator=True,
    )

    same_visits = (
        len(comparison) == len(predictions_df)
        and len(comparison) == len(fusion_df)
        and (comparison["_merge"] == "both").all()
    )

    same_labels = (
        same_visits
        and (
            comparison["true_label_mri"]
            == comparison["true_label_fusion"]
        ).all()
    )

    comparison["mri_correct"] = (
        comparison["true_label_mri"]
        == comparison["predicted_label_mri"]
    )
    comparison["fusion_correct"] = (
        comparison["true_label_fusion"]
        == comparison["predicted_label_fusion"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "mri_fusion_test_comparison.csv",
        index=False,
    )

    if not same_visits or not same_labels:
        raise RuntimeError(
            "MRI-only and fusion test sets do not match exactly. "
            "Inspect mri_fusion_test_comparison.csv."
        )

    print("\nTest set verification: PASS")
    print("Same test visits:", same_visits)
    print("Same clinical labels:", same_labels)

    print("\nMRI-only test results")
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Test visits:", len(all_labels))
    print("Test participants:", len(set(all_rids)))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(cm)

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
