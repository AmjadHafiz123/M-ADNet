from pathlib import Path

import numpy as np
import SimpleITK as sitk


TARGET_SPACING = (1.5, 1.5, 1.5)
MIN_FOREGROUND_VOXELS = 1000


def normalize_volume(image: sitk.Image) -> sitk.Image:
    """Apply the same foreground clipping and z-score normalization
    used by the original M-ADNet MRI preprocessing pipeline.
    """
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    foreground = array > 0

    if int(foreground.sum()) < MIN_FOREGROUND_VOXELS:
        raise ValueError(
            "MRI contains fewer than 1000 positive foreground voxels. "
            "Please check the uploaded image."
        )

    foreground_values = array[foreground]

    lower = np.percentile(foreground_values, 0.5)
    upper = np.percentile(foreground_values, 99.5)

    clipped = np.clip(foreground_values, lower, upper)

    mean = float(clipped.mean())
    std = float(clipped.std())

    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("MRI foreground intensity variation is insufficient.")

    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[foreground] = (clipped - mean) / std

    result = sitk.GetImageFromArray(normalized)
    result.CopyInformation(image)

    return result


def resample_to_target_spacing(
    image: sitk.Image,
    target_spacing=TARGET_SPACING
) -> sitk.Image:
    """Resample an image using the original training preprocessing settings."""

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(
            original_size[i] * original_spacing[i] / target_spacing[i]
        ))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(image)


def preprocess_mri(
    input_path: str | Path,
    output_path: str | Path
) -> Path:
    """Preprocess an original structural MRI NIfTI file.

    Processing follows the original M-ADNet training pipeline:
    LPS orientation -> 1.5 mm resampling -> foreground clipping
    -> foreground z-score normalization.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"MRI file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(input_path))

    if image.GetDimension() != 3:
        raise ValueError("Uploaded MRI must be a 3D image.")

    image = sitk.Cast(image, sitk.sitkFloat32)

    if not np.all(np.isfinite(sitk.GetArrayViewFromImage(image))):
        raise ValueError("MRI contains non-finite intensity values.")

    image = sitk.DICOMOrient(image, "LPS")
    image = resample_to_target_spacing(image)
    image = normalize_volume(image)

    sitk.WriteImage(image, str(output_path), True)

    return output_path
