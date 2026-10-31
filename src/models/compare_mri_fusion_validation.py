
from pathlib import Path
import pandas as pd

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)

MRI_FILE = (
    ROOT / "mri_baseline" / "validation_analysis"
    / "validation_predictions.csv"
)

FUSION_FILE = (
    ROOT / "madnet_fusion" / "validation_predictions.csv"
)

mri = pd.read_csv(MRI_FILE)
fusion = pd.read_csv(FUSION_FILE)

print("MRI rows:", len(mri))
print("Fusion rows:", len(fusion))

print("\nMRI columns:", mri.columns.tolist())
print("Fusion columns:", fusion.columns.tolist())

merged = mri.merge(
    fusion,
    on="image_id",
    suffixes=("_mri", "_fusion"),
    how="outer",
    indicator=True
)

print("\nMerge status:")
print(merged["_merge"].value_counts())

matched = merged[merged["_merge"] == "both"].copy()

print("\nMatched image IDs:", len(matched))

rid_mismatch = (
    matched["rid_mri"].astype(str)
    != matched["rid_fusion"].astype(str)
)

label_mismatch = (
    matched["true_label_mri"].astype(str)
    != matched["true_label_fusion"].astype(str)
)

print("\nRID mismatches:", rid_mismatch.sum())
print("Ground-truth mismatches:", label_mismatch.sum())

if label_mismatch.any():
    print("\nGround-truth mismatch examples:")
    print(
        matched.loc[
            label_mismatch,
            [
                "image_id",
                "rid_mri",
                "rid_fusion",
                "true_label_mri",
                "true_label_fusion"
            ]
        ].head(30).to_string(index=False)
    )

if rid_mismatch.any():
    print("\nParticipant mismatch examples:")
    print(
        matched.loc[
            rid_mismatch,
            [
                "image_id",
                "rid_mri",
                "rid_fusion"
            ]
        ].head(20).to_string(index=False)
    )

print("\nMRI-only target distribution:")
print(mri["true_label"].value_counts())

print("\nFusion target distribution:")
print(fusion["true_label"].value_counts())

print("\nDiagnostic complete.")
