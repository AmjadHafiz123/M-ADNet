from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.models.train_madnet_fusion import (
    MADNetFusion,
    CLASS_NAMES,
)
from app.preprocessing import preprocess_mri


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_madnet_fusion.pt"
)

TARGET_SHAPE = (128, 160, 160)

SEX_MAP = {
    "Male": 0.0,
    "Female": 1.0,
}


class MADNetPredictor:
    """Inference interface for the trained multimodal M-ADNet model."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Only load trusted checkpoints created by this project.
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model = MADNetFusion().to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

        self.class_names = self.checkpoint.get(
            "class_names", CLASS_NAMES
        )

        if list(self.class_names) != list(CLASS_NAMES):
            raise ValueError(
                "Checkpoint class order does not match model class order."
            )

        self.clinical_means = self.checkpoint["clinical_means"]
        self.clinical_stds = self.checkpoint["clinical_stds"]

        self.epoch = self.checkpoint.get("epoch")
        self.validation_macro_f1 = self.checkpoint.get(
            "validation_macro_f1"
        )

    def prepare_mri(self, nifti_path):
        """Load a preprocessed 3D NIfTI and match dataset spatial resizing."""

        nifti_path = Path(nifti_path)

        if not nifti_path.is_file():
            raise FileNotFoundError(
                f"MRI file not found: {nifti_path}"
            )

        image = nib.load(str(nifti_path))

        volume = np.asarray(
            image.dataobj,
            dtype=np.float32,
        )

        if volume.ndim != 3:
            raise ValueError(
                f"Expected a 3D MRI volume; received shape {volume.shape}."
            )

        if not np.isfinite(volume).all():
            raise ValueError(
                "MRI contains NaN or infinite voxel values."
            )

        if float(volume.max()) == float(volume.min()):
            raise ValueError(
                "MRI has constant voxel intensity and cannot be processed."
            )

        # Same tensor conversion and spatial interpolation as MADNetDataset.
        mri = torch.from_numpy(
            np.ascontiguousarray(volume)
        ).unsqueeze(0)

        mri = F.interpolate(
            mri.unsqueeze(0),
            size=TARGET_SHAPE,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).contiguous()

        return mri.unsqueeze(0)

    def prepare_clinical(self, age, education, sex):
        """Apply training-set imputation, scaling and sex encoding."""

        if sex not in SEX_MAP:
            raise ValueError(
                "Sex must be exactly 'Male' or 'Female'."
            )

        clinical_values = []

        for feature, value in (
            ("age_years_approx", age),
            ("education_years", education),
        ):
            mean = float(self.clinical_means[feature])
            std = float(self.clinical_stds[feature])

            if std == 0:
                std = 1.0

            if value is None:
                value = mean

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{feature} must be a finite number."
                )

            standardized = (value - mean) / std
            clinical_values.append(standardized)

        clinical_values.append(SEX_MAP[sex])

        return torch.tensor(
            [clinical_values],
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def predict(self, nifti_path, age, education, sex):
        """Preprocess an original MRI and predict CN, MCI or Dementia."""

        nifti_path = Path(nifti_path)

        if not nifti_path.is_file():
            raise FileNotFoundError(
                f"MRI file not found: {nifti_path}"
            )

        # Preprocess the original MRI in a temporary directory.
        # The temporary processed file is removed after prediction.
        with tempfile.TemporaryDirectory(prefix="madnet_") as temp_dir:

            processed_path = Path(temp_dir) / "processed_mri.nii.gz"

            preprocess_mri(
                input_path=nifti_path,
                output_path=processed_path,
            )

            mri = self.prepare_mri(processed_path).to(self.device)

            clinical = self.prepare_clinical(
                age=age,
                education=education,
                sex=sex,
            ).to(self.device)

            logits = self.model(mri, clinical)

            probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

            probability_dict = {
                name: float(probabilities[i].item())
                for i, name in enumerate(self.class_names)
            }

            return {
                "prediction": self.class_names[predicted_index],
                "probabilities": probability_dict,
                "checkpoint_epoch": self.epoch,
                "validation_macro_f1": self.validation_macro_f1,
            }


if __name__ == "__main__":
    predictor = MADNetPredictor()

    print("M-ADNet inference engine loaded successfully.")
    print("Device:", predictor.device)
    print("Classes:", predictor.class_names)
    print("Checkpoint epoch:", predictor.epoch)
    print("Validation macro-F1:", predictor.validation_macro_f1)
    print("Original MRI preprocessing: enabled")
