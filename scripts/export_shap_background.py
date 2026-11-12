"""
Export the same 50-row training-set SHAP background used by the research
analysis. Run from the M-ADNet project root with the project venv activated.
"""
from pathlib import Path
import sys

# Add the M-ADNet project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.datasets.madnet_dataset import MADNetDataset

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CSV = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_mri_clinical_features.csv"
)
MRI_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\mri_full_nifti"
)
CHECKPOINT = ROOT / "checkpoints" / "best_madnet_fusion.pt"
OUTPUT = ROOT / "app" / "assets" / "shap_background.npy"
SEED = 42
MAX_BACKGROUND = 50

def main():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    train_ds = MADNetDataset(
        feature_csv=FEATURE_CSV,
        nifti_dir=MRI_DIR,
        split="train",
        clinical_means=checkpoint["clinical_means"],
        clinical_stds=checkpoint["clinical_stds"],
    )

    rows = []
    for _, row in train_ds.df.iterrows():
        values = []
        for column in train_ds.NUMERIC_COLUMNS:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                value = train_ds.clinical_means[column]
            values.append(
                (float(value) - train_ds.clinical_means[column])
                / train_ds.clinical_stds[column]
            )
        sex = train_ds.SEX_MAP.get(row["sex"], None)
        if sex is None or pd.isna(sex):
            raise ValueError(f"Invalid sex value for RID {row['rid']}: {row['sex']}")
        values.append(float(sex))
        rows.append(values)

    matrix = np.asarray(rows, dtype=np.float32)
    rng = np.random.default_rng(SEED)
    indices = rng.choice(
        len(matrix), size=min(MAX_BACKGROUND, len(matrix)), replace=False
    )
    background = matrix[indices]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, background)
    print("Saved:", OUTPUT)
    print("Shape:", background.shape)
    print("Seed:", SEED)

if __name__ == "__main__":
    main()
