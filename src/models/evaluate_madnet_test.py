
from pathlib import Path
import json

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

FEATURES_CSV = PROCESSED_DIR / "adni_mri_clinical_features.csv"
MRI_DIR = PROCESSED_DIR / "mri_full_nifti"

CHECKPOINT_PATH = (
    PROCESSED_DIR
    / "madnet_fusion"
    / "best_madnet_fusion.pt"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "madnet_fusion"
    / "test_evaluation"
)

CLASS_NAMES = ["CN", "MCI", "Dementia"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_INPUT_SHAPE = (64, 80, 80)


# --------------------------------------------------
# Model architecture
# Must match the trained model exactly.
# --------------------------------------------------

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
        return self.classifier(features)


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

    def forward(self, mri, clinical):

        mri_logits = self.mri_encoder(mri)

        clinical_features = self.clinical_encoder(clinical)

        clinical_adjustment = self.clinical_correction(
            clinical_features
        )

        return mri_logits + clinical_adjustment


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

    device = torch.device("cpu")

    print("=" * 65)
    print("M-ADNet FINAL TEST EVALUATION")
    print("=" * 65)

    for path in (FEATURES_CSV, MRI_DIR, CHECKPOINT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    # Load selected checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("model_name") != "MADNetFusion":
        raise ValueError("Unexpected checkpoint model.")

    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class order mismatch.")

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Selected validation macro-F1:",
        checkpoint["validation_macro_f1"],
    )

    # Use the exact clinical statistics saved during training.
    clinical_means = pd.Series(
        checkpoint["clinical_means"]
    )

    clinical_stds = pd.Series(
        checkpoint["clinical_stds"]
    )

    test_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="test",
        clinical_means=clinical_means,
        clinical_stds=clinical_stds,
    )

    # Verify participant-level split integrity.
    all_features = pd.read_csv(FEATURES_CSV)

    train_rids = set(
        all_features.loc[
            all_features["split"] == "train", "rid"
        ].astype(str)
    )

    val_rids = set(
        all_features.loc[
            all_features["split"] == "validation", "rid"
        ].astype(str)
    )

    test_rids = set(
        test_ds.df["rid"].astype(str)
    )

    if train_rids & test_rids:
        raise RuntimeError("Train/test participant leakage detected.")

    if val_rids & test_rids:
        raise RuntimeError("Validation/test participant leakage detected.")

    print("\nTest visits:", len(test_ds))
    print("Test participants:", test_ds.df["rid"].nunique())

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = MADNetFusion().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_image_ids = []
    all_rids = []

    # One forward pass per test visit.
    with torch.inference_mode():

        for batch in loader:

            mri = batch["mri"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)

            logits = model(mri, clinical)

            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

            all_image_ids.extend(batch["image_id"])
            all_rids.extend(batch["rid"])

    if not all_labels:
        raise RuntimeError("Test evaluation produced no predictions.")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

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

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    probabilities = np.asarray(all_probabilities)

    predictions_df = pd.DataFrame({
        "image_id": all_image_ids,
        "rid": all_rids,
        "true_label": [
            CLASS_NAMES[i] for i in all_labels
        ],
        "predicted_label": [
            CLASS_NAMES[i] for i in all_predictions
        ],
        "correct": [
            actual == predicted
            for actual, predicted
            in zip(all_labels, all_predictions)
        ],
    })

    for index, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = (
            probabilities[:, index]
        )

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).to_csv(
        OUTPUT_DIR / "test_confusion_matrix.csv"
    )

    metrics = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selected_validation_macro_f1": float(
            checkpoint["validation_macro_f1"]
        ),
        "test_visits": len(all_labels),
        "test_participants": len(set(map(str, all_rids))),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "clinical_statistics_from_checkpoint": True,
        "test_evaluated": True,
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("Balanced accuracy:", round(balanced_acc, 4))

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

    print("\nSaved outputs:")
    print(OUTPUT_DIR)

    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
