
from pathlib import Path
import warnings

import pandas as pd
import pyreadr

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\extracted"
)
OUTPUT_DIR = DATA_DIR.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "DXSUM": "DXSUM.rda",
    "ADAS": "ADAS.rda",
    "CDR": "CDR.rda",
    "MMSE": "MMSE.rda",
}


def load_dataset(name, filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    objects = pyreadr.read_r(str(path))

    if name not in objects:
        raise ValueError(
            f"Expected object {name}; found {list(objects.keys())}"
        )

    df = objects[name].copy()

    # Consistent join keys across datasets.
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df["VISCODE"] = (
        df["VISCODE"].astype("string").str.strip().str.lower()
    )

    if df.duplicated(["RID", "VISCODE"]).any():
        raise ValueError(
            f"{name} has duplicate RID + VISCODE keys. "
            "Resolve them before merging."
        )

    return df


data = {
    name: load_dataset(name, filename)
    for name, filename in FILES.items()
}

# Diagnosis records define the initial cohort and visit rows.
dx = data["DXSUM"]

columns = [
    "RID", "PTID", "VISCODE", "EXAMDATE", "DIAGNOSIS"
]
columns += [
    c for c in ["DXCONFID", "DXNORM", "DXMCI", "DXAD"]
    if c in dx.columns
]

merged = dx[columns].copy()

# Select relevant scores and retain their assessment dates
# so that visit-level alignment can be checked.
score_specs = {
    "ADAS": {
        "scores": ["TOTSCORE", "TOTAL13"],
        "date": "VISDATE",
    },
    "CDR": {
        "scores": ["CDGLOBAL", "CDRSB"],
        "date": "VISDATE",
    },
    "MMSE": {
        "scores": ["MMSCORE"],
        "date": "VISDATE",
    },
}

for name, spec in score_specs.items():
    df = data[name]

    selected = ["RID", "VISCODE"]
    if "PTID" in df.columns:
        selected.append("PTID")

    date_col = spec["date"]
    if date_col in df.columns:
        selected.append(date_col)

    selected += [
        c for c in spec["scores"] if c in df.columns
    ]

    right = df[selected].copy()

    # Keep the diagnosis table's PTID as the canonical identifier.
    right = right.drop(columns=["PTID"], errors="ignore")

    # Give each assessment date a distinct name.
    if date_col in right.columns:
        right = right.rename(
            columns={date_col: f"{name}_VISDATE"}
        )

    merged = merged.merge(
        right,
        on=["RID", "VISCODE"],
        how="left",
        validate="one_to_one",
        indicator=f"_{name}_merge",
    )

    matched = (merged[f"_{name}_merge"] == "both").sum()
    print(
        f"{name}: matched {matched:,} / {len(merged):,} "
        "diagnosis visits"
    )

    merged = merged.drop(columns=[f"_{name}_merge"])

# Parse dates for inspection; retain rows even when dates are missing.
for col in [
    "EXAMDATE", "ADAS_VISDATE", "CDR_VISDATE", "MMSE_VISDATE"
]:
    if col in merged.columns:
        merged[col] = pd.to_datetime(
            merged[col], errors="coerce"
        )

print("\nDiagnosis distribution:")
print(merged["DIAGNOSIS"].value_counts(dropna=False).to_string())

print("\nClinical score missingness:")
score_columns = [
    c for c in ["TOTSCORE", "TOTAL13", "CDGLOBAL", "CDRSB", "MMSCORE"]
    if c in merged.columns
]
print(merged[score_columns].isna().sum().to_string())

# Report date gaps without automatically rejecting records.
print("\nAssessment date differences from diagnosis date:")
for name in score_specs:
    date_col = f"{name}_VISDATE"
    if date_col not in merged.columns:
        continue

    days = (merged[date_col] - merged["EXAMDATE"]).dt.days
    valid = days.dropna()

    if not valid.empty:
        print(
            f"{name}: {len(valid):,} date pairs; "
            f"same day = {(valid == 0).sum():,}; "
            f"over 30 days apart = {(valid.abs() > 30).sum():,}"
        )

output_file = OUTPUT_DIR / "adni_clinical_merged.csv"
merged.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print(f"Merged rows: {len(merged):,}")
print(f"Unique participants: {merged['RID'].nunique():,}")
print(f"Output: {output_file}")
print("PTDEMOG has intentionally not been merged.")
print("No diagnosis records were intentionally discarded.")
