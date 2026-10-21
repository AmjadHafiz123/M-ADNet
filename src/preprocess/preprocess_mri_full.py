
from pathlib import Path
import argparse
import json
import re
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk


BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = BASE / "Actual_data" / "processed"
INPUT = PROCESSED / "adni_mri_final_manifest.csv"

OUTPUT_DIR = PROCESSED / "mri_full_nifti"
AUDIT = PROCESSED / "adni_mri_full_preprocessing_audit.csv"
TARGET_SPACING = (1.5, 1.5, 1.5)

REQUIRED_COLUMNS = [
    "image_id", "rid", "target", "split", "series_folder",
    "series_description", "clinical_date", "mri_date",
]


def safe_id(value):
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    return text.strip("_") or "unknown"


def load_manifest():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Manifest not found: {INPUT}")

    df = pd.read_csv(INPUT, low_memory=False, dtype={"rid": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    df["rid"] = df["rid"].fillna("").astype(str).str.strip()
    df["image_id"] = df["image_id"].fillna("").astype(str).str.strip()
    df["target"] = df["target"].fillna("").astype(str).str.strip()
    df["split"] = df["split"].fillna("").astype(str).str.strip()
    df["series_folder"] = (
        df["series_folder"].fillna("").astype(str).str.strip()
    )

    if df["image_id"].eq("").any():
        raise ValueError("Manifest contains blank image_id values.")
    if df["rid"].eq("").any():
        raise ValueError("Manifest contains blank participant IDs.")
    if df["target"].isin(["CN", "MCI", "Dementia"]).eq(False).any():
        raise ValueError("Unexpected or missing target in manifest.")
    if df["split"].isin(["train", "validation", "test"]).eq(False).any():
        raise ValueError("Unexpected or missing split in manifest.")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values in manifest.")

    # Verify participant-level split consistency without changing assignments.
    split_counts = df.groupby("rid")["split"].nunique()
    if (split_counts > 1).any():
        bad = split_counts[split_counts > 1].index.tolist()[:10]
        raise ValueError(
            f"Participants appear in multiple splits: {bad}"
        )

    return df


def normalize_volume(image):
    array = sitk.GetArrayFromImage(image).astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError("Non-finite voxel values found.")

    # Retain the pilot's foreground heuristic; this is not brain extraction.
    foreground = array > 0
    count = int(foreground.sum())
    if count < 1000:
        raise ValueError(
            f"Too few positive voxels for normalization: {count}"
        )

    values = array[foreground]
    low, high = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(low + high) or high <= low:
        raise ValueError("Invalid intensity percentiles.")

    clipped = np.clip(array, low, high)
    values = clipped[foreground]
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-6:
        raise ValueError("Invalid or near-zero foreground standard deviation.")

    clipped[foreground] = (values - mean) / std
    clipped[~foreground] = 0.0

    normalized = sitk.GetImageFromArray(clipped)
    normalized.CopyInformation(image)

    stats = {
        "foreground_voxel_count": count,
        "intensity_clip_low": float(low),
        "intensity_clip_high": float(high),
        "foreground_mean_before_zscore": mean,
        "foreground_std_before_zscore": std,
        "normalized_min": float(clipped.min()),
        "normalized_max": float(clipped.max()),
        "all_finite": bool(np.isfinite(clipped).all()),
    }
    return normalized, clipped, stats


def reconstruct_series(folder):
    reader = sitk.ImageSeriesReader()
    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or [])

    if not series_ids:
        raise RuntimeError("No DICOM series found in folder.")

    # Do not silently choose among multiple series: their identity must be
    # confirmed rather than inferred solely from the number of files.
    if len(series_ids) != 1:
        details = []
        for sid in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(folder), sid
            )
            details.append({"series_id": sid, "file_count": len(files)})
        raise RuntimeError(
            "Multiple DICOM series found; manual series review required: "
            + json.dumps(details)
        )

    series_id = series_ids[0]
    files = list(
        sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(folder), series_id
        )
    )
    if len(files) < 20:
        raise RuntimeError(
            f"Only {len(files)} DICOM files in series."
        )

    reader.SetFileNames(files)
    image = reader.Execute()
    if image.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D image, got {image.GetDimension()}D.")

    return image, series_id, len(files)


def process_one(row, output_dir):
    folder = Path(str(row["series_folder"]))
    result = {
        "image_id": row["image_id"],
        "rid": row["rid"],
        "target": row["target"],
        "split": row["split"],
        "clinical_date": row.get("clinical_date", ""),
        "mri_date": row.get("mri_date", ""),
        "date_gap_days": row.get("date_gap_days", ""),
        "mri_manifest_group": row.get("mri_manifest_group", ""),
        "label_review": row.get("label_review", ""),
        "series_folder": str(folder),
        "series_description": row.get("series_description", ""),
        "status": "FAILED",
        "error": "",
        "output_file": "",
        "selected_dicom_series_id": "",
        "dicom_files_read": 0,
    }

    if not folder.is_dir():
        raise FileNotFoundError(f"Series folder not found: {folder}")

    image, series_id, file_count = reconstruct_series(folder)
    result["selected_dicom_series_id"] = series_id
    result["dicom_files_read"] = file_count

    result["original_size_xyz"] = str(image.GetSize())
    result["original_spacing_xyz_mm"] = str(image.GetSpacing())
    result["original_origin_xyz"] = str(image.GetOrigin())
    result["original_direction"] = str(image.GetDirection())

    image = sitk.DICOMOrient(image, "LPS")

    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / TARGET_SPACING[i])))
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

    normalized, array, stats = normalize_volume(resampled)

    # A unique image_id prevents collisions between repeated participant visits.
    filename = (
        f"sub-{safe_id(row['rid'])}_image-{safe_id(row['image_id'])}.nii.gz"
    )
    output_path = output_dir / filename
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    sitk.WriteImage(normalized, str(output_path), useCompression=True)

    # Read back the file to catch incomplete/corrupt writes.
    check = sitk.ReadImage(str(output_path))
    if check.GetDimension() != 3 or check.GetSize() != normalized.GetSize():
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Written NIfTI failed geometry read-back check.")

    result.update(stats)
    result.update({
        "status": "PASS",
        "output_file": str(output_path),
        "processed_size_xyz": str(normalized.GetSize()),
        "processed_spacing_xyz_mm": str(normalized.GetSpacing()),
        "array_shape_zyx": str(array.shape),
        "output_size_bytes": output_path.stat().st_size,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the first N manifest rows (0 means all)."
    )
    parser.add_argument(
        "--split", choices=["train", "validation", "test"],
        help="Optional split filter."
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Output directory; defaults to the full-cohort directory."
    )
    parser.add_argument(
        "--audit", default=str(AUDIT),
        help="Audit CSV path."
    )
    args = parser.parse_args()

    df = load_manifest()
    if args.split:
        df = df[df["split"] == args.split].copy()
    df = df.sort_values(["split", "rid", "clinical_date", "image_id"])

    if args.limit > 0:
        df = df.head(args.limit).copy()

    output_dir = Path(args.output_dir)
    audit_path = Path(args.audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    print("Manifest visits selected:", len(df))
    print("Participants selected:", df["rid"].nunique())
    print("Split counts:")
    print(df["split"].value_counts().to_string())
    print("Target counts:")
    print(df["target"].value_counts().to_string())
    print("Output directory:", output_dir)
    print("Audit:", audit_path)

    results = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        print(
            f"\n[{i}/{len(df)}] split={row['split']} "
            f"RID={row['rid']} target={row['target']} "
            f"image={row['image_id']}"
        )
        try:
            result = process_one(row, output_dir)
        except Exception as exc:
            result = {
                "image_id": row.get("image_id", ""),
                "rid": row.get("rid", ""),
                "target": row.get("target", ""),
                "split": row.get("split", ""),
                "clinical_date": row.get("clinical_date", ""),
                "mri_date": row.get("mri_date", ""),
                "series_folder": row.get("series_folder", ""),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
                "output_file": "",
            }

        results.append(result)
        print(result["status"], result.get("error", ""))

        # Save after every visit so an interrupted run retains its audit.
        pd.DataFrame(results).to_csv(
            audit_path, index=False, encoding="utf-8-sig"
        )

    audit = pd.DataFrame(results)
    print("\n=== PREPROCESSING SUMMARY ===")
    print("Attempted:", len(audit))
    print(audit["status"].value_counts().to_string())
    print("Audit:", audit_path)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
