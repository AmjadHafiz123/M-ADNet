from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn.functional as F


FEATURE_NAMES = ["Age", "Education", "Sex"]


class GradCAM3D:
    """3D Grad-CAM for the MRI encoder's final ReLU layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach().clone()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient.detach()

    def generate(self, mri, clinical, target_class):
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        # Do not use torch.inference_mode here: Grad-CAM requires gradients.
        with torch.enable_grad():
            logits = self.model(mri, clinical)
            if self.activations is None:
                raise RuntimeError("Grad-CAM did not capture activations.")
            logits[:, int(target_class)].sum().backward()

        if self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture gradients.")

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=mri.shape[2:], mode="trilinear", align_corners=False
        )[0, 0]

        low, high = cam.min(), cam.max()
        cam = (cam - low) / (high - low + 1e-8)
        return logits.detach(), cam.detach()

    def close(self):
        self.handle.remove()


def gradcam_overlay_figure(mri_tensor, cam_tensor, predicted_class):
    """Create central axial, coronal and sagittal overlays in model-input space."""
    volume = mri_tensor.detach().cpu().numpy()[0, 0].astype(np.float32)
    cam = cam_tensor.detach().cpu().numpy().astype(np.float32)
    x_mid, y_mid, z_mid = (n // 2 for n in volume.shape)

    slices = [
        ("Axial", volume[:, :, z_mid], cam[:, :, z_mid]),
        ("Coronal", volume[:, y_mid, :], cam[:, y_mid, :]),
        ("Sagittal", volume[x_mid, :, :], cam[x_mid, :, :]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (name, image_slice, cam_slice) in zip(axes, slices):
        ax.imshow(np.rot90(image_slice), cmap="gray")
        ax.imshow(np.rot90(cam_slice), cmap="jet", alpha=0.40, vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"Grad-CAM | Target class: {predicted_class}")
    fig.tight_layout()
    return fig


def run_gradcam(model, mri_tensor, clinical_tensor, target_class):
    # The research script used features[14], the final ReLU before adaptive pooling.
    target_layer = model.mri_encoder.features[14]
    explainer = GradCAM3D(model, target_layer)
    try:
        _, cam = explainer.generate(
            mri_tensor.to(next(model.parameters()).device),
            clinical_tensor.to(next(model.parameters()).device),
            target_class,
        )
    finally:
        explainer.close()
    return gradcam_overlay_figure(mri_tensor, cam, target_class), cam.cpu().numpy()


def load_shap_background(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SHAP background not found: {path}. Run scripts/export_shap_background.py."
        )
    background = np.load(path).astype(np.float32)
    if background.ndim != 2 or background.shape[1] != 3:
        raise ValueError(f"Expected SHAP background shape (N, 3), got {background.shape}.")
    if len(background) < 2 or not np.isfinite(background).all():
        raise ValueError("SHAP background must contain at least two finite samples.")
    return background


def run_clinical_shap(model, clinical_tensor, background):
    """
    Explain the clinical correction logits only, matching the research script.
    Output is not an explanation of the full MRI-plus-clinical logits.
    """
    device = next(model.parameters()).device
    background = np.asarray(background, dtype=np.float32)
    clinical_values = clinical_tensor.detach().cpu().numpy().astype(np.float32)

    def predict_correction(clinical_array):
        values = torch.as_tensor(
            np.asarray(clinical_array, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            features = model.clinical_encoder(values)
            correction = model.clinical_correction(features)
        return correction.detach().cpu().numpy()

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(
        predict_correction,
        masker,
        algorithm="exact",
        feature_names=FEATURE_NAMES,
        output_names=list(getattr(model, "class_names", ["CN", "MCI", "Dementia"])),
    )
    explanation = explainer(clinical_values)
    values = np.asarray(explanation.values)

    # SHAP's multi-output shape for one observation is (1, features, classes).
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] != 3:
        raise RuntimeError(f"Unexpected SHAP result shape: {values.shape}")

    return {
        "values": values[0],  # (features, classes)
        "base_values": np.asarray(explanation.base_values),
        "feature_names": FEATURE_NAMES,
        "class_names": ["CN", "MCI", "Dementia"],
        "input_values": clinical_values[0],
    }


def shap_figure(shap_result):
    values = shap_result["values"]
    names = shap_result["feature_names"]
    classes = shap_result["class_names"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for class_index, ax in enumerate(axes):
        class_values = values[:, class_index]
        order = np.argsort(np.abs(class_values))
        ax.barh(np.array(names)[order], class_values[order])
        ax.axvline(0, linewidth=0.8)
        ax.set_title(classes[class_index])
        ax.set_xlabel("SHAP value\n(clinical correction logit)")
    fig.suptitle("Clinical-branch SHAP contributions")
    fig.tight_layout()
    return fig
