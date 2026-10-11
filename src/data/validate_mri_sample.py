
from pathlib import Path
import csv

import numpy as np
import SimpleITK as sitk

AUDIT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_dicom_series_validated.csv"
)

OUTPUT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_mri_sample_validation.csv"
)

SAMPLE_SIZE = 30
MIN_FILES = 20

if not AUDIT.is_file():
    raise FileNotFoundError(AUDIT)

with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

candidates = []
for row in rows:
    try:
        count = int(row.get("file_count") or 0)
    except ValueError:
        continue

    folder = row.get("series_folder", "").strip()
    description = row.get("series_description", "").strip()

    if folder and Path(folder).is_dir() and count >= MIN_FILES:
        row["_count"] = count
        row["_description"] = description or "Unknown"
        candidates.append(row)

if not candidates:
    raise RuntimeError("No suitable MRI series found.")

# Select the largest series for each description first.
by_description = {}
for row in candidates:
    key = row["_description"]
    if (
        key not in by_description
        or row["_count"] > by_description[key]["_count"]
    ):
        by_description[key] = row

selected = list(by_description.values())

# Add evenly distributed candidates to reach the sample size.
selected_ids = {r["series_folder"] for r in selected}
remaining = [
    r for r in candidates if r["series_folder"] not in selected_ids
]
remaining.sort(key=lambda r: r["series_folder"])

slots = max(0, SAMPLE_SIZE - len(selected))
if slots and remaining:
    indices = np.linspace(
        0, len(remaining) - 1,
        num=min(slots, len(remaining)),
        dtype=int,
    )
    for i in indices:
        selected.append(remaining[int(i)])

selected = selected[:SAMPLE_SIZE]

results = []

for number, row in enumerate(selected, start=1):
    folder = Path(row["series_folder"])
    result = {
        "sample_number": number,
        "series_folder": str(folder),
        "series_description": row["_description"],
        "inventory_file_count": row["_count"],
        "status": "FAILED",
        "error": "",
        "simpleitk_file_count": 0,
        "volume_depth": "",
        "volume_height": "",
        "volume_width": "",
        "spacing_x_mm": "",
        "spacing_y_mm": "",
        "spacing_z_mm": "",
        "intensity_min": "",
        "intensity_max": "",
        "nonzero_fraction": "",
        "finite_values": "",
        "plausible_3d": False,
    }

    try:
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
        result["simpleitk_file_count"] = len(files)

        reader.SetFileNames(files)
        image = reader.Execute()
        volume = sitk.GetArrayFromImage(image)

        result["spacing_x_mm"] = image.GetSpacing()[0]
        result["spacing_y_mm"] = image.GetSpacing()[1]
        result["spacing_z_mm"] = image.GetSpacing()[2]

        if volume.ndim == 3:
            depth, height, width = volume.shape
            result["volume_depth"] = depth
            result["volume_height"] = height
            result["volume_width"] = width

            finite = bool(np.isfinite(volume).all())
            nonzero_fraction = float(np.count_nonzero(volume) / volume.size)

            result["finite_values"] = finite
            result["intensity_min"] = float(np.min(volume))
            result["intensity_max"] = float(np.max(volume))
            result["nonzero_fraction"] = nonzero_fraction

            plausible = (
                min(volume.shape) > 1
                and finite
                and len(files) >= MIN_FILES
                and float(np.max(volume)) > float(np.min(volume))
                and nonzero_fraction > 0.01
            )
            result["plausible_3d"] = plausible
            result["status"] = "PASS" if plausible else "REVIEW"
        else:
            result["error"] = f"Unexpected dimensions: {volume.shape}"
            result["status"] = "REVIEW"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    results.append(result)
    print(
        f"[{number}/{len(selected)}] {result['status']}: "
        f"{row['_description']} | {folder.name} | "
        f"{result['error']}"
    )

fieldnames = list(results[0].keys())
with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

passed = sum(r["status"] == "PASS" for r in results)
review = sum(r["status"] == "REVIEW" for r in results)
failed = sum(r["status"] == "FAILED" for r in results)

print("\n=== SAMPLE VALIDATION SUMMARY ===")
print("Series tested:", len(results))
print("PASS:", passed)
print("REVIEW:", review)
print("FAILED:", failed)
print("Report:", OUTPUT)
