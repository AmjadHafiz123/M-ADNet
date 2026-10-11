
from pathlib import Path
import pandas as pd

MRI_FILE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\ADNI_MPRAGE_search_result_7_18_2026.csv"
)

CLINICAL_FILE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed\adni_model_ready_30days.csv"
)

OUTPUT_DIR = MRI_FILE.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load MRI manifest.
mri = pd.read_csv(MRI_FILE, dtype=str)
mri.columns = mri.columns.str.strip()

# Load clinical dataset.
clinical = pd.read_csv(CLINICAL_FILE, dtype={"PTID": str})

print("=" * 65)
print("ADNI MRI MANIFEST AUDIT")
print("=" * 65)

print(f"\nMRI manifest rows: {len(mri):,}")
print(f"Manifest columns: {list(mri.columns)}")

required = ["Image Data ID", "Subject", "Group", "Visit", "Format"]
missing = [col for col in required if col not in mri.columns]

if missing:
    raise ValueError(f"Missing expected MRI columns: {missing}")

# Clean identifiers and metadata.
for col in ["Image Data ID", "Subject", "Group", "Visit", "Format"]:
    mri[col] = mri[col].astype("string").str.strip()

mri["Subject"] = mri["Subject"].str.upper()
mri["Visit"] = mri["Visit"].str.lower()
mri["Group"] = mri["Group"].str.upper()

print("\nMRI format counts:")
print(mri["Format"].value_counts(dropna=False).to_string())

print("\nMRI diagnosis-group counts:")
print(mri["Group"].value_counts(dropna=False).to_string())

print("\nMRI visit counts (top 15):")
print(mri["Visit"].value_counts(dropna=False).head(15).to_string())

print("\nUnique MRI identifiers:")
print("  Image Data IDs:", mri["Image Data ID"].nunique())
print("  Subjects:", mri["Subject"].nunique())

print("\nDuplicate checks:")
print(
    "  Duplicate Image Data IDs:",
    mri["Image Data ID"].duplicated().sum()
)
print(
    "  Duplicate subject + visit pairs:",
    mri.duplicated(["Subject", "Visit"]).sum()
)

# Inspect the manifest's Downloaded column, if present.
if "Downloaded" in mri.columns:
    downloaded = mri["Downloaded"].fillna("").str.strip()

    print("\nDownloaded-column values:")
    print(
        downloaded.replace("", "<blank>")
        .value_counts(dropna=False)
        .to_string()
    )
else:
    print("\nNo Downloaded column found.")

# Match MRI subjects against clinical participant identifiers.
if "PTID" not in clinical.columns:
    raise ValueError(
        "PTID not found in the clinical CSV; inspect its columns."
    )

clinical["PTID"] = clinical["PTID"].astype("string").str.strip().str.upper()

mri_subjects = set(mri["Subject"].dropna()) - {""}
clinical_subjects = set(clinical["PTID"].dropna()) - {""}

overlap = mri_subjects & clinical_subjects

print("\nMRI–clinical participant overlap:")
print(f"  Unique MRI subjects: {len(mri_subjects):,}")
print(f"  Unique clinical subjects: {len(clinical_subjects):,}")
print(f"  Matching subjects: {len(overlap):,}")
print(
    "  MRI subjects without a clinical match:",
    len(mri_subjects - clinical_subjects)
)

# Keep every manifest row, adding whether its subject occurs clinically.
mri["Clinical_Match"] = mri["Subject"].isin(overlap)

mri.to_csv(
    OUTPUT_DIR / "adni_mri_manifest_audited.csv",
    index=False,
)

mri[mri["Clinical_Match"]].to_csv(
    OUTPUT_DIR / "adni_mri_clinical_candidates.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT_DIR / "adni_mri_manifest_audited.csv")
print(OUTPUT_DIR / "adni_mri_clinical_candidates.csv")
print("\nNote: participant overlap does not yet establish visit-level matching.")
