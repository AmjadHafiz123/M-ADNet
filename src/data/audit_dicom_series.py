
from pathlib import Path
from collections import Counter
import csv

import pydicom
from pydicom.errors import InvalidDicomError

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\MRI_extracted\ADNI"
)

OUTPUT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\adni_dicom_series_audit.csv"
)

if not ROOT.exists():
    raise FileNotFoundError(f"MRI directory not found: {ROOT}")

# Each immediate scan folder contains one candidate DICOM series.
series_files = {}

print("Discovering DICOM series folders...")

for path in ROOT.rglob("*.dcm"):
    series_files.setdefault(path.parent, []).append(path)

print(f"Candidate series folders: {len(series_files):,}")
print("Reading metadata from one file per series...")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
errors = Counter()

for i, (folder, files) in enumerate(series_files.items(), start=1):
    try:
        # Read metadata only; do not load image pixels.
        ds = pydicom.dcmread(
            str(files[0]),
            stop_before_pixels=True,
            force=False
        )

        rows.append({
            "participant_folder": folder.parts[-4]
                if len(folder.parts) >= 4 else "",
            "series_folder": str(folder),
            "file_count": len(files),
            "modality": str(getattr(ds, "Modality", "")),
            "series_description": str(
                getattr(ds, "SeriesDescription", "")
            ),
            "series_instance_uid": str(
                getattr(ds, "SeriesInstanceUID", "")
            ),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "series_date": str(getattr(ds, "SeriesDate", "")),
            "rows": str(getattr(ds, "Rows", "")),
            "columns": str(getattr(ds, "Columns", "")),
            "image_type": str(getattr(ds, "ImageType", "")),
            "manufacturer": str(getattr(ds, "Manufacturer", "")),
        })

    except (InvalidDicomError, OSError, ValueError) as exc:
        errors[type(exc).__name__] += 1
        rows.append({
            "participant_folder": "",
            "series_folder": str(folder),
            "file_count": len(files),
            "modality": "",
            "series_description": "",
            "series_instance_uid": "",
            "study_date": "",
            "series_date": "",
            "rows": "",
            "columns": "",
            "image_type": "",
            "manufacturer": "",
        })

    if i % 250 == 0:
        print(f"Processed {i:,}/{len(series_files):,} series")

fieldnames = [
    "participant_folder",
    "series_folder",
    "file_count",
    "modality",
    "series_description",
    "series_instance_uid",
    "study_date",
    "series_date",
    "rows",
    "columns",
    "image_type",
    "manufacturer",
]

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\n=== DICOM SERIES AUDIT ===")
print(f"Candidate series folders: {len(series_files):,}")
print(f"Metadata rows written: {len(rows):,}")
print(f"Metadata read errors: {sum(errors.values()):,}")
print(f"Audit CSV: {OUTPUT}")

print("\nLargest candidate series:")
for row in sorted(
    rows, key=lambda r: r["file_count"], reverse=True
)[:10]:
    print(
        f'{row["file_count"]:>5} files | '
        f'{row["series_description"][:40]:40} | '
        f'{row["participant_folder"]}'
    )
