
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")

AUDIT = (
    BASE / "Actual_data" / "processed"
    / "adni_dicom_series_audit.csv"
)

MANIFEST = (
    BASE / "Actual_data"
    / "ADNI_MPRAGE_search_result_7_18_2026.csv"
)

OUT = BASE / "Actual_data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

for path in (AUDIT, MANIFEST):
    if not path.is_file():
        raise FileNotFoundError(path)

audit = pd.read_csv(AUDIT, dtype=str).fillna("")
manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")

print("=== INPUTS ===")
print(f"Audit series rows: {len(audit):,}")
print(f"Manifest rows: {len(manifest):,}")
print(f"Manifest columns: {list(manifest.columns)}")

# Extract participant and image-series IDs from folder paths.
audit["participant_from_path"] = audit["series_folder"].str.extract(
    r"ADNI[\\/]+([^\\/]+)", expand=False
)
audit["image_id_from_path"] = audit["series_folder"].str.extract(
    r"[\\/](I\d+)$", expand=False
)

print("\n=== SERIES INVENTORY ===")
print("Unique participants:", audit["participant_from_path"].nunique())
print("Unique extracted image IDs:", audit["image_id_from_path"].nunique())
print("Duplicate nonempty series UIDs:",
      audit.loc[audit["series_instance_uid"] != "",
                "series_instance_uid"].duplicated().sum())

print("\nMost common series descriptions:")
print(
    audit["series_description"].replace("", "<missing>")
    .value_counts().head(15).to_string()
)

print("\nSeries file-count distribution:")
counts = pd.to_numeric(audit["file_count"], errors="coerce")
print(counts.describe(percentiles=[.25, .5, .75, .9, .95]).to_string())

print("\nImage dimensions (rows x columns):")
dims = (
    audit["rows"].replace("", "?") + " x " +
    audit["columns"].replace("", "?")
)
print(dims.value_counts().head(10).to_string())

# Find the manifest's image-ID column.
id_col = next(
    (c for c in manifest.columns
     if c.strip().lower() == "image data id"),
    None
)

if id_col is None:
    print("\nCould not find an 'Image Data ID' column; skipping ID match.")
else:
    manifest["image_id_normalized"] = (
        manifest[id_col].str.strip().str.extract(r"(I\d+)", expand=False)
    )
    audit_ids = set(audit["image_id_from_path"].dropna())
    manifest_ids = set(manifest["image_id_normalized"].dropna())

    print("\n=== MANIFEST MATCH ===")
    print(f"Unique manifest image IDs: {len(manifest_ids):,}")
    print(f"Unique DICOM folder image IDs: {len(audit_ids):,}")
    print(f"IDs in both: {len(audit_ids & manifest_ids):,}")
    print(f"Manifest IDs without DICOM folder: {len(manifest_ids - audit_ids):,}")
    print(f"DICOM folder IDs absent from manifest: {len(audit_ids - manifest_ids):,}")

    audit["manifest_id_match"] = audit["image_id_from_path"].isin(
        manifest_ids
    )
    audit.to_csv(
        OUT / "adni_dicom_series_validated.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nSaved: {OUT / 'adni_dicom_series_validated.csv'}")
