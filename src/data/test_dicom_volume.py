
from pathlib import Path
import csv
import numpy as np
import pydicom
import SimpleITK as sitk

AUDIT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_dicom_series_validated.csv"
)

if not AUDIT.is_file():
    raise FileNotFoundError(AUDIT)

# Choose the series with the most DICOM files.
with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

valid = [
    r for r in rows
    if r.get("series_folder")
    and int(r.get("file_count") or 0) >= 20
    and r.get("series_description", "").strip()
]

if not valid:
    raise RuntimeError("No suitable candidate series found.")

candidate = max(valid, key=lambda r: int(r["file_count"]))
folder = Path(candidate["series_folder"])

print("Candidate series:")
print("  Folder:", folder)
print("  Description:", candidate["series_description"])
print("  DICOM files:", candidate["file_count"])

reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(str(folder))

if not series_ids:
    raise RuntimeError("SimpleITK could not identify a DICOM series.")

print("  Series IDs found by SimpleITK:", len(series_ids))

# Use the series ID associated with the most files.
series_id = max(
    series_ids,
    key=lambda sid: len(
        reader.GetGDCMSeriesFileNames(str(folder), sid)
    ),
)
files = reader.GetGDCMSeriesFileNames(str(folder), series_id)

print("  Files selected by SimpleITK:", len(files))

reader.SetFileNames(files)
image = reader.Execute()

volume = sitk.GetArrayFromImage(image)

print("\n=== RECONSTRUCTED VOLUME ===")
print("Array shape (slices, rows, columns):", volume.shape)
print("Image size (x, y, z):", image.GetSize())
print("Voxel spacing (mm):", image.GetSpacing())
print("Origin (mm):", image.GetOrigin())
print("Direction matrix:", image.GetDirection())
print("Pixel data type:", volume.dtype)
print("All pixel values finite:", bool(np.isfinite(volume).all()))
print("Intensity range:", float(volume.min()), "to", float(volume.max()))

if volume.ndim != 3:
    raise RuntimeError("Expected a 3D volume.")

if min(volume.shape) <= 1:
    raise RuntimeError("Volume has a suspiciously small dimension.")

if not np.isfinite(volume).all():
    raise RuntimeError("Volume contains non-finite pixel values.")

print("\nSUCCESS: A 3D volume was reconstructed for inspection.")
