
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_madnet_fusion import (
    MADNetFusion,
    FEATURES_CSV,
    MRI_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT_PATH = OUTPUT_DIR / "best_madnet_fusion.pt"

SHAP_OUTPUT_DIR = OUTPUT_DIR / "shap_validation"

FEATURE_NAMES = [
    "Age",
    "Education",
    "Sex",
]

DEVICE = torch.device("cpu")

MAX_BACKGROUND = 50
RANDOM_SEED = 42


# --------------------------------------------------
# Reconstruct clinical inputs
# --------------------------------------------------

def prepare_clinical_matrix(dataset):

    df = dataset.df

    values = []

    for _, row in df.iterrows():

        row_values = []

        for column in dataset.NUMERIC_COLUMNS:

            value = pd.to_numeric(
                row[column],
                errors="coerce",
            )

            if pd.isna(value):
                value = dataset.clinical_means[column]

            standardized = (
                float(value) - dataset.clinical_means[column]
            ) / dataset.clinical_stds[column]

            row_values.append(float(standardized))

        sex = dataset.SEX_MAP.get(row["sex"], None)

        if sex is None or pd.isna(sex):
            raise ValueError(
                f"Invalid sex value for RID {row['rid']}: "
                f"{row['sex']}"
            )

        row_values.append(float(sex))

        values.append(row_values)

    matrix = np.asarray(values, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(
            f"Unexpected clinical matrix shape: {matrix.shape}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Non-finite clinical values detected."
        )

    return matrix


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    torch.set_num_threads(2)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    SHAP_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        CHECKPOINT_PATH,
        FEATURES_CSV,
        MRI_DIR,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file or directory not found: {path}"
            )

    print("=" * 65)
    print("M-ADNet Clinical SHAP Explainability")
    print("=" * 65)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        "Selected validation macro-F1:",
        checkpoint["validation_macro_f1"],
    )

    # --------------------------------------------------
    # Load datasets and saved training statistics
    # --------------------------------------------------

    train_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="train",
    )

    saved_means = checkpoint["clinical_means"]
    saved_stds = checkpoint["clinical_stds"]

    # Confirm that the checkpoint normalization matches
    # the training dataset used during model development.
    for column in train_ds.NUMERIC_COLUMNS:

        if not np.isclose(
            train_ds.clinical_means[column],
            saved_means[column],
        ):
            raise ValueError(
                f"Training mean mismatch for {column}"
            )

        if not np.isclose(
            train_ds.clinical_stds[column],
            saved_stds[column],
        ):
            raise ValueError(
                f"Training standard deviation mismatch for {column}"
            )

    shared_stats = {
        "clinical_means": saved_means,
        "clinical_stds": saved_stds,
    }

    val_ds = MADNetDataset(
        feature_csv=FEATURES_CSV,
        nifti_dir=MRI_DIR,
        split="validation",
        **shared_stats,
    )

    # Participant split verification.
    train_rids = set(train_ds.df["rid"])
    val_rids = set(val_ds.df["rid"])

    if not train_rids.isdisjoint(val_rids):
        raise RuntimeError(
            "Participant leakage detected between train and validation."
        )

    print("Training visits:", len(train_ds))
    print("Validation visits:", len(val_ds))
    print("Training participants:", len(train_rids))
    print("Validation participants:", len(val_rids))
    print("Participant separation: PASS")

    # --------------------------------------------------
    # Prepare clinical inputs
    # --------------------------------------------------

    background = prepare_clinical_matrix(train_ds)
    validation_values = prepare_clinical_matrix(val_ds)

    rng = np.random.default_rng(RANDOM_SEED)

    background_indices = rng.choice(
        len(background),
        size=min(MAX_BACKGROUND, len(background)),
        replace=False,
    )

    background = background[background_indices]

    print("SHAP background samples:", len(background))
    print(
        "Validation explanation samples:",
        len(validation_values),
    )

    # --------------------------------------------------
    # Load trained fusion model
    # --------------------------------------------------

    model = MADNetFusion().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    # Disable dropout and explain only the clinical correction.
    # The MRI logits are independent of clinical input.
    def predict_clinical_correction(clinical_array):

        clinical_tensor = torch.as_tensor(
            np.asarray(clinical_array, dtype=np.float32),
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():

            clinical_features = model.clinical_encoder(
                clinical_tensor
            )

            adjustment = model.clinical_correction(
                clinical_features
            )

        return adjustment.cpu().numpy()

    # --------------------------------------------------
    # SHAP exact explainer
    # --------------------------------------------------

    print("\nCreating SHAP explainer...")

    masker = shap.maskers.Independent(
        background,
        max_samples=len(background),
    )

    explainer = shap.Explainer(
        predict_clinical_correction,
        masker,
        algorithm="exact",
        feature_names=FEATURE_NAMES,
        output_names=CLASS_NAMES,
    )

    print("Calculating SHAP values...")

    explanation = explainer(
        validation_values
    )

    shap_values = np.asarray(
        explanation.values
    )

    base_values = np.asarray(
        explanation.base_values
    )

    expected_shape = (
        len(validation_values),
        len(FEATURE_NAMES),
        len(CLASS_NAMES),
    )

    if shap_values.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected SHAP shape: {shap_values.shape}; "
            f"expected {expected_shape}"
        )

    print("SHAP values shape:", shap_values.shape)

    # --------------------------------------------------
    # Export per-visit SHAP values
    # --------------------------------------------------

    result = pd.DataFrame({
        "image_id": val_ds.df["image_id"].values,
        "rid": val_ds.df["rid"].values,
        "true_label": val_ds.df["target"].values,
    })

    for class_index, class_name in enumerate(CLASS_NAMES):

        for feature_index, feature_name in enumerate(FEATURE_NAMES):

            column_name = (
                f"shap_{class_name}_{feature_name.lower()}"
            )

            result[column_name] = shap_values[
                :, feature_index, class_index
            ]

    result.to_csv(
        SHAP_OUTPUT_DIR / "clinical_shap_per_visit.csv",
        index=False,
    )

    # --------------------------------------------------
    # Global feature importance
    # --------------------------------------------------

    importance_rows = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        for feature_index, feature_name in enumerate(FEATURE_NAMES):

            mean_absolute_shap = float(
                np.mean(
                    np.abs(
                        shap_values[
                            :, feature_index, class_index
                        ]
                    )
                )
            )

            mean_shap = float(
                np.mean(
                    shap_values[
                        :, feature_index, class_index
                    ]
                )
            )

            importance_rows.append({
                "class": class_name,
                "feature": feature_name,
                "mean_absolute_shap": mean_absolute_shap,
                "mean_shap": mean_shap,
            })

    importance_df = pd.DataFrame(importance_rows)

    importance_df.to_csv(
        SHAP_OUTPUT_DIR / "clinical_shap_global_importance.csv",
        index=False,
    )

    # --------------------------------------------------
    # Class-specific SHAP plots
    # --------------------------------------------------

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_values = shap_values[:, :, class_index]

        # Summary beeswarm plot.
        plt.figure()

        shap.summary_plot(
            class_values,
            validation_values,
            feature_names=FEATURE_NAMES,
            show=False,
        )

        plt.title(
            f"Clinical SHAP Summary - {class_name}"
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_OUTPUT_DIR
            / f"shap_summary_{class_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        # Mean absolute SHAP bar plot.
        class_importance = np.mean(
            np.abs(class_values),
            axis=0,
        )

        order = np.argsort(class_importance)

        plt.figure(figsize=(7, 4))

        plt.barh(
            np.array(FEATURE_NAMES)[order],
            class_importance[order],
        )

        plt.xlabel("Mean absolute SHAP value")
        plt.title(
            f"Clinical Feature Importance - {class_name}"
        )

        plt.tight_layout()

        plt.savefig(
            SHAP_OUTPUT_DIR
            / f"shap_importance_{class_name}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    metadata = {
        "model": "MADNetFusion",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_macro_f1": float(
            checkpoint["validation_macro_f1"]
        ),
        "explained_component": "clinical_correction_logits",
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
        "background_samples": len(background),
        "validation_visits": len(validation_values),
        "validation_participants": len(val_rids),
        "explainer": "SHAP Exact",
        "test_set_used": False,
        "interpretation": (
            "SHAP values explain the clinical correction logits. "
            "They do not represent the full MRI-plus-clinical "
            "logits or causal effects."
        ),
    }

    with open(
        SHAP_OUTPUT_DIR / "shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 65)
    print("CLINICAL SHAP ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nGlobal feature importance:")
    print(
        importance_df.sort_values(
            ["class", "mean_absolute_shap"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nSaved files:")
    for path in sorted(SHAP_OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nOutput directory:")
    print(SHAP_OUTPUT_DIR)


if __name__ == "__main__":
    main()
