
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = BASE / "Actual_data" / "processed"

INPUT = PROCESSED / "adni_mri_visit_shortlist.csv"
OUTPUT_DIR = PROCESSED / "mri_pilot_nifti"
AUDIT = PROCESSED / "adni_mri_pilot_preprocessing_audit.csv"

TARGET_SPACING = (1.5, 1.5, 1.5)
PILOT_SIZE = 10
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_pilot(df):
    """Prefer different participants and cover all three classes."""
    df = df.copy()
    df["rid"] = df["rid"].astype(str)
    df["target"] = df["target"].astype(str)
    df["series_description"] = (
        df["series_description"].fillna("").astype(str)
    )

    df["is_mprage"] = df["series_description"].str.contains(
        "MPRAGE", case=False, regex=False
    )
    df["is_repeat"] = df["series_description"].str.contains(
        "REPEAT|REPET", case=False, regex=True
    )

    df["dicom_file_count"] = pd.to_numeric(
        df["dicom_file_count"], errors="coerce"
    ).fillna(0)

    df = df[
        df["target"].isin(["CN", "MCI", "Dementia"])
        & df["rid"].ne("")
        & df["series_folder"].notna()
        & df["series_folder"].astype(str).str.strip().ne("")
    ].copy()

    # Rank plausible series first; don't modify the original shortlist.
    df = df.sort_values(
        ["is_mprage", "is_repeat", "dicom_file_count"],
        ascending=[False, True, False],
        kind="stable",
    )

    # Choose participants across classes, avoiding repeated participants.
    quotas = {"CN": 4, "MCI": 3, "Dementia": 3}
    selected = []
    used_rids = set()

    for label, quota in quotas.items():
        for _, row in df[df["target"] == label].iterrows():
            if row["rid"] in used_rids:
                continue

            folder = Path(row["series_folder"])
            if not folder.is_dir():
                continue

            selected.append(row)
            used_rids.add(row["rid"])

            if sum(x["target"] == label for x in selected) >= quota:
                break

    # Fill remaining slots from other eligible participants.
    for _, row in df.iterrows():
        if len(selected) >= PILOT_SIZE:
            break
        if row["rid"] in used_rids:
            continue
        if not Path(row["series_folder"]).is_dir():
            continue

        selected.append(row)
        used_rids.add(row["rid"])

    if not selected:
        raise RuntimeError("No eligible pilot scans found.")

    return pd.DataFrame(selected[:PILOT_SIZE])


def normalize_volume(image):
    """Clip intensities and z-score foreground voxels."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError("Non-finite voxel values found.")

    # Background is commonly zero in these MRI scans.
    foreground = array > 0
    if foreground.sum() < 1000:
        raise ValueError("Too few positive voxels for normalization.")

    values = array[foreground]
    low, high = np.percentile(values, [0.5, 99.5])

    if not np.isfinite(low + high) or high <= low:
        raise ValueError("Invalid intensity percentiles.")

    array = np.clip(array, low, high)
    values = array[foreground]

    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-6:
        raise ValueError("Foreground intensity variance is too small.")

    array[foreground] = (values - mean) / std
    array[~foreground] = 0.0

    normalized = sitk.GetImageFromArray(array)
    normalized.CopyInformation(image)

    return normalized, array, low, high, mean, std


def process_one(row, index):
    folder = Path(row["series_folder"])
    result = {
        "rid": row["rid"],
        "image_id": row.get("image_id", ""),
        "target": row["target"],
        "series_folder": str(folder),
        "series_description": row.get("series_description", ""),
        "status": "FAILED",
        "error": "",
        "output_file": "",
    }

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))

    if not series_ids:
        raise RuntimeError("SimpleITK found no DICOM series.")

    series_id = max(
        series_ids,
        key=lambda sid: len(
            reader.GetGDCMSeriesFileNames(str(folder), sid)
        ),
    )
    files = reader.GetGDCMSeriesFileNames(str(folder), series_id)

    if len(files) < 20:
        raise RuntimeError(
            f"Only {len(files)} files in reconstructed series."
        )

    reader.SetFileNames(files)
    image = reader.Execute()

    if image.GetDimension() != 3:
        raise RuntimeError("Expected a 3D image.")

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Normalize anatomical orientation before resampling.
    image = sitk.DICOMOrient(image, "LPS")

    original_spacing_after_orientation = image.GetSpacing()
    original_size_after_orientation = image.GetSize()

    # Preserve physical field of view as closely as possible.
    new_size = [
        max(
            1,
            int(round(
                original_size_after_orientation[i]
                * original_spacing_after_orientation[i]
                / TARGET_SPACING[i]
            )),
        )
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    resampled = resampler.Execute(image)

    normalized, array, low, high, mean, std = normalize_volume(
        resampled
    )

    output_path = OUTPUT_DIR / (
        f"sub-{row['rid']}_image-{row.get('image_id', index)}.nii.gz"
    )
    sitk.WriteImage(normalized, str(output_path), useCompression=True)

    result.update({
        "status": "PASS",
        "output_file": str(output_path),
        "dicom_files_read": len(files),
        "original_size_xyz": str(original_size),
        "original_spacing_xyz_mm": str(original_spacing),
        "processed_size_xyz": str(normalized.GetSize()),
        "processed_spacing_xyz_mm": str(normalized.GetSpacing()),
        "array_shape_zyx": str(array.shape),
        "intensity_clip_low": float(low),
        "intensity_clip_high": float(high),
        "foreground_mean_before_zscore": float(mean),
        "foreground_std_before_zscore": float(std),
        "normalized_min": float(array.min()),
        "normalized_max": float(array.max()),
        "all_finite": bool(np.isfinite(array).all()),
    })

    return result


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)
    pilot = select_pilot(df)

    print("Selected pilot scans:", len(pilot))
    print(pilot["target"].value_counts().to_string())

    results = []

    for index, (_, row) in enumerate(pilot.iterrows(), start=1):
        print(
            f"\n[{index}/{len(pilot)}] "
            f"RID={row['rid']} target={row['target']} "
            f"image={row.get('image_id', '')}"
        )

        try:
            result = process_one(row, index)
        except Exception as exc:
            result = {
                "rid": row["rid"],
                "image_id": row.get("image_id", ""),
                "target": row["target"],
                "series_folder": row["series_folder"],
                "series_description": row.get("series_description", ""),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "output_file": "",
            }

        results.append(result)
        print(result["status"], result.get("error", ""))

    audit = pd.DataFrame(results)
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    print("\n=== PILOT PREPROCESSING SUMMARY ===")
    print("Scans attempted:", len(audit))
    print(audit["status"].value_counts().to_string())
    print("Audit:", AUDIT)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
