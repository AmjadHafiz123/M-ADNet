
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

AUDIT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_dicom_series_validated.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\qc_images"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

valid = [
    r for r in rows
    if r.get("series_folder")
    and int(r.get("file_count") or 0) >= 20
    and r.get("series_description", "").strip()
]

candidate = max(valid, key=lambda r: int(r["file_count"]))
folder = Path(candidate["series_folder"])

reader = sitk.ImageSeriesReader()
series_ids = reader.GetGDCMSeriesIDs(str(folder))
if not series_ids:
    raise RuntimeError(f"No DICOM series found in {folder}")

series_id = max(
    series_ids,
    key=lambda sid: len(
        reader.GetGDCMSeriesFileNames(str(folder), sid)
    ),
)
files = reader.GetGDCMSeriesFileNames(str(folder), series_id)
reader.SetFileNames(files)

image = reader.Execute()
volume = sitk.GetArrayFromImage(image)

# Mid-volume slices in the array's three index directions.
z, y, x = (np.array(volume.shape) // 2).tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

views = [
    (volume[z, :, :], "Middle slice: array axis 0"),
    (volume[:, y, :], "Middle slice: array axis 1"),
    (volume[:, :, x], "Middle slice: array axis 2"),
]

for ax, (slice_img, title) in zip(axes, views):
    ax.imshow(slice_img, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")

fig.suptitle(
    f"{candidate['series_description']}\n"
    f"Volume shape: {volume.shape}"
)
fig.tight_layout()

output_path = OUTPUT_DIR / "sample_mri_three_views.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Saved:", output_path)
print("Volume shape:", volume.shape)
print("Voxel spacing:", image.GetSpacing())
