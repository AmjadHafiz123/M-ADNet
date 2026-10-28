
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MADNetDataset(Dataset):
    """Dataset for M-ADNet multimodal Alzheimer's disease classification."""

    LABEL_MAP = {
        "CN": 0,
        "MCI": 1,
        "Dementia": 2,
    }

    SEX_MAP = {
        "Male": 0.0,
        "Female": 1.0,
    }

    NUMERIC_COLUMNS = [
        "age_years_approx",
        "education_years",
    ]

    # Common spatial dimensions for batched 3D MRI input.
    TARGET_SHAPE = (128, 160, 160)

    def __init__(
        self,
        feature_csv,
        nifti_dir,
        split="train",
        clinical_means=None,
        clinical_stds=None,
    ):
        self.feature_csv = Path(feature_csv)
        self.nifti_dir = Path(nifti_dir)
        self.split = split

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        if not self.feature_csv.is_file():
            raise FileNotFoundError(
                f"Feature CSV not found: {self.feature_csv}"
            )

        if not self.nifti_dir.is_dir():
            raise FileNotFoundError(
                f"MRI directory not found: {self.nifti_dir}"
            )

        df = pd.read_csv(self.feature_csv)

        required_columns = {
            "image_id",
            "rid",
            "target",
            "split",
            "sex",
            *self.NUMERIC_COLUMNS,
        }
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Feature CSV is missing columns: {sorted(missing)}"
            )

        if df["image_id"].isna().any():
            raise ValueError("Feature CSV contains missing image_id values.")

        df["image_id"] = df["image_id"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        df["rid"] = df["rid"].astype(str).str.strip()

        if df["image_id"].duplicated().any():
            duplicates = df.loc[
                df["image_id"].duplicated(keep=False), "image_id"
            ].tolist()
            raise ValueError(
                f"Duplicate image IDs detected: {duplicates[:10]}"
            )

        # Check participant-level split isolation across the full table.
        participant_split_counts = df.groupby("rid")["split"].nunique()
        leaking = participant_split_counts[
            participant_split_counts > 1
        ].index.tolist()

        if leaking:
            raise ValueError(
                "Participant leakage across splits detected. "
                f"Example RIDs: {leaking[:10]}"
            )

        # Select the requested split without modifying the original assignments.
        self.df = (
            df.loc[df["split"] == split]
            .copy()
            .reset_index(drop=True)
        )

        if self.df.empty:
            raise ValueError(f"No records found for split={split!r}.")

        unknown_labels = (
            set(self.df["target"].dropna()) - set(self.LABEL_MAP)
        )
        if unknown_labels:
            raise ValueError(f"Unexpected target labels: {unknown_labels}")

        if self.df["target"].isna().any():
            raise ValueError("Missing diagnosis targets found.")

        # Match each MRI by image_id. Never rely on file ordering.
        self.image_paths = []

        for image_id in self.df["image_id"]:
            matches = list(
                self.nifti_dir.glob(f"sub-*_image-{image_id}.nii.gz")
            )

            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one NIfTI for image_id "
                    f"{image_id!r}; found {len(matches)}."
                )

            self.image_paths.append(matches[0])

        # Prepare numeric clinical features.
        numeric = self.df[self.NUMERIC_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )

        # Fit imputation and scaling statistics on training data only.
        if clinical_means is None and clinical_stds is None:
            if split != "train":
                raise ValueError(
                    "Validation/test datasets require means and standard "
                    "deviations fitted on the training dataset."
                )

            self.clinical_means = numeric.mean().fillna(0.0)
            self.clinical_stds = (
                numeric.std(ddof=0)
                .replace(0, 1.0)
                .fillna(1.0)
            )

        elif clinical_means is not None and clinical_stds is not None:
            self.clinical_means = pd.Series(
                clinical_means, dtype=float
            ).reindex(self.NUMERIC_COLUMNS)

            self.clinical_stds = pd.Series(
                clinical_stds, dtype=float
            ).reindex(self.NUMERIC_COLUMNS)

            if self.clinical_means.isna().any():
                raise ValueError("Invalid or missing clinical means.")

            if self.clinical_stds.isna().any():
                raise ValueError("Invalid or missing clinical standard deviations.")

            self.clinical_stds = self.clinical_stds.mask(
                self.clinical_stds == 0, 1.0
            )

        else:
            raise ValueError(
                "Supply both clinical_means and clinical_stds, or neither."
            )

        self.clinical_means = self.clinical_means.astype(float)
        self.clinical_stds = self.clinical_stds.astype(float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = self.image_paths[index]

        # Load the preprocessed 3D MRI.
        volume = np.asarray(
            nib.load(str(image_path)).dataobj,
            dtype=np.float32,
        )

        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D MRI for {row['image_id']}, "
                f"got shape {volume.shape}."
            )

        if not np.isfinite(volume).all():
            raise ValueError(
                f"Non-finite voxel values in {image_path.name}."
            )

        # Add channel dimension: [1, X, Y, Z].
        mri = torch.from_numpy(
            np.ascontiguousarray(volume)
        ).unsqueeze(0)

        # Standardize spatial dimensions for batch collation.
        # Shape becomes [1, 128, 160, 160].
        mri = F.interpolate(
            mri.unsqueeze(0),
            size=self.TARGET_SHAPE,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).contiguous()

        # Impute missing numeric values using training-set means,
        # then standardize using training-set statistics.
        clinical_values = []

        for column in self.NUMERIC_COLUMNS:
            value = pd.to_numeric(row[column], errors="coerce")

            if pd.isna(value):
                value = self.clinical_means[column]

            standardized = (
                float(value) - self.clinical_means[column]
            ) / self.clinical_stds[column]

            clinical_values.append(float(standardized))

        # Sex is encoded categorically: Male=0, Female=1.
        sex = self.SEX_MAP.get(row["sex"], None)

        if sex is None or pd.isna(sex):
            raise ValueError(
                f"Missing or unrecognized sex for RID {row['rid']}."
            )

        clinical_values.append(float(sex))

        clinical = torch.tensor(
            clinical_values,
            dtype=torch.float32,
        )

        label = torch.tensor(
            self.LABEL_MAP[row["target"]],
            dtype=torch.long,
        )

        return {
            "mri": mri,
            "clinical": clinical,
            "label": label,
            "image_id": row["image_id"],
            "rid": str(row["rid"]),
        }
