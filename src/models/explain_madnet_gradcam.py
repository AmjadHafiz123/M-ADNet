from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.datasets.madnet_dataset import MADNetDataset

# IMPORTANT:
# Import your exact fusion model class from train_madnet_fusion.py.
# Replace MADNetFusion with the actual class name in your file.
from src.models.train_madnet_fusion import MADNetFusion


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_CSV = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_mri_clinical_features.csv"
)

MRI_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\mri_full_nifti"
)

CHECKPOINT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\madnet_fusion\best_madnet_fusion.pt"
)

OUTPUT_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\madnet_fusion\gradcam_validation"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["CN", "MCI", "Dementia"]



class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.save_activation
        )

    def save_activation(self, module, inputs, output):
        # Clone the activation so the hook does not interfere
        # with the in-place ReLU operation.
        self.activations = output.detach()

        # Register a gradient hook directly on the activation tensor.
        if output.requires_grad:
            output.register_hook(self.save_gradient)

    def save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(self, mri, clinical, target_class):
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        logits = self.model(mri, clinical)

        if self.activations is None:
            raise RuntimeError("Grad-CAM did not capture activations.")

        score = logits[:, target_class].sum()
        score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM did not capture gradients. "
                "Check that the target layer is connected to the output."
            )

        weights = self.gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = (weights * self.activations).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=mri.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return logits.detach(), cam.detach()

    def close(self):
        self.forward_handle.remove()

def save_slice_overlay(volume, cam, output_path, image_id, prediction):
    """
    Save axial, coronal and sagittal overlays.

    Both volume and CAM are in the dataset's resized voxel coordinates.
    These are not native scanner-space overlays.
    """

    volume = volume.astype(np.float32)
    cam = cam.astype(np.float32)

    # Select central slices.
    x_mid = volume.shape[0] // 2
    y_mid = volume.shape[1] // 2
    z_mid = volume.shape[2] // 2

    slices = [
        (
            "Axial",
            volume[:, :, z_mid],
            cam[:, :, z_mid],
        ),
        (
            "Coronal",
            volume[:, y_mid, :],
            cam[:, y_mid, :],
        ),
        (
            "Sagittal",
            volume[x_mid, :, :],
            cam[x_mid, :, :],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, image_slice, cam_slice) in zip(axes, slices):
        ax.imshow(
            np.rot90(image_slice),
            cmap="gray",
        )

        ax.imshow(
            np.rot90(cam_slice),
            cmap="jet",
            alpha=0.40,
            vmin=0,
            vmax=1,
        )

        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"Image {image_id} | Predicted: {prediction}"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # Adjust this key only if your checkpoint uses a different name.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            "Could not find model_state_dict or state_dict in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict, strict=True)

    return checkpoint


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    print("Loaded checkpoint successfully")
    print("Model:", checkpoint["model_name"])
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation macro-F1:", checkpoint["validation_macro_f1"])
    print("Model input shape:", checkpoint["model_input_shape"])
    print("Dataset input shape:", checkpoint["dataset_input_shape"])

    # Training script must use this same model class and constructor.
    model = MADNetFusion().to(DEVICE)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            f"Checkpoint keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Reuse the normalization statistics stored in the checkpoint.
    clinical_means = checkpoint["clinical_means"]
    clinical_stds = checkpoint["clinical_stds"]

    dataset = MADNetDataset(
        feature_csv=FEATURE_CSV,
        nifti_dir=MRI_DIR,
        split="validation",
        clinical_means=clinical_means,
        clinical_stds=clinical_stds,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # The final ReLU is features[14], before AdaptiveAvgPool3d.
    target_layer = model.mri_encoder.features[14]

    gradcam = GradCAM3D(model, target_layer)

    results = []

    try:
        for index, batch in enumerate(loader):
            mri = batch["mri"].to(DEVICE)
            clinical = batch["clinical"].to(DEVICE)
            true_label = int(batch["label"].item())

            image_id = batch["image_id"][0]
            rid = batch["rid"][0]

            # First obtain the predicted class without retaining gradients.
            with torch.no_grad():
                logits = model(mri, clinical)

            probabilities = torch.softmax(logits, dim=1)
            predicted_label = int(logits.argmax(dim=1).item())
            confidence = float(
                probabilities[0, predicted_label].item()
            )

            # Explain the predicted class.
            logits, cam = gradcam.generate(
                mri,
                clinical,
                predicted_label,
            )

            # Load original preprocessed volume and apply the same resizing
            # operation used by MADNetDataset for display.
            image_path = dataset.image_paths[index]

            volume = np.asarray(
                nib.load(str(image_path)).dataobj,
                dtype=np.float32,
            )

            volume_tensor = torch.from_numpy(
                np.ascontiguousarray(volume)
            )[None, None]

            resized_volume = F.interpolate(
                volume_tensor,
                size=dataset.TARGET_SHAPE,
                mode="trilinear",
                align_corners=False,
            )[0, 0].numpy()

            cam_np = cam.cpu().numpy()

            filename = f"{image_id}_gradcam.png"

            save_slice_overlay(
                resized_volume,
                cam_np,
                OUTPUT_DIR / filename,
                image_id,
                CLASS_NAMES[predicted_label],
            )

            results.append({
                "image_id": image_id,
                "rid": rid,
                "true_label": CLASS_NAMES[true_label],
                "predicted_label": CLASS_NAMES[predicted_label],
                "correct": true_label == predicted_label,
                "confidence": confidence,
                "gradcam_file": filename,
            })

            print(
                f"{index + 1}/{len(dataset)} | "
                f"Image: {image_id} | "
                f"True: {CLASS_NAMES[true_label]} | "
                f"Predicted: {CLASS_NAMES[predicted_label]}"
            )

    finally:
        gradcam.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "gradcam_results.csv",
        index=False,
    )

    print("\nGrad-CAM completed.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(results_df["true_label"].value_counts())


if __name__ == "__main__":
    main()