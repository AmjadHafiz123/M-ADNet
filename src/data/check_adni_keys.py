
from pathlib import Path
import warnings
import pyreadr
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\extracted"
)

FILES = {
    "ADAS": "ADAS.rda",
    "CDR": "CDR.rda",
    "DXSUM": "DXSUM.rda",
    "MMSE": "MMSE.rda",
    "PTDEMOG": "PTDEMOG.rda",
}

data = {}

for name, filename in FILES.items():
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    result = pyreadr.read_r(str(path))
    if name not in result:
        raise ValueError(
            f"Expected object {name} not found in {filename}. "
            f"Available objects: {list(result.keys())}"
        )

    df = result[name].copy()

    # Standardize identifiers for comparison.
    for col in ["RID", "PTID", "VISCODE", "VISCODE2"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    data[name] = df

print("\n" + "=" * 65)
print("1. DATASET SUMMARY")
print("=" * 65)

for name, df in data.items():
    print(f"\n{name}: {len(df):,} rows")

    for col in ["RID", "PTID", "VISCODE", "VISCODE2"]:
        if col in df.columns:
            print(f"  Unique {col}: {df[col].nunique(dropna=True):,}")

print("\n" + "=" * 65)
print("2. DUPLICATE PARTICIPANT-VISIT KEYS")
print("=" * 65)

for name, df in data.items():
    for visit_col in ["VISCODE2", "VISCODE"]:
        if {"RID", visit_col}.issubset(df.columns):
            valid = df.dropna(subset=["RID", visit_col])
            duplicated = valid.duplicated(
                subset=["RID", visit_col], keep=False
            )

            print(
                f"{name}, (RID, {visit_col}): "
                f"{duplicated.sum():,} rows in duplicate groups"
            )

print("\n" + "=" * 65)
print("3. DIAGNOSIS DISTRIBUTION")
print("=" * 65)

dx = data["DXSUM"]

if "DIAGNOSIS" in dx.columns:
    print(dx["DIAGNOSIS"].value_counts(dropna=False).to_string())

print("\n" + "=" * 65)
print("4. KEY CLINICAL SCORE AVAILABILITY")
print("=" * 65)

for name, columns in {
    "ADAS": ["TOTSCORE", "TOTAL13"],
    "CDR": ["CDGLOBAL", "CDRSB"],
    "MMSE": ["MMSCORE"],
}.items():
    df = data[name]

    for col in columns:
        if col in df.columns:
            print(
                f"{name}.{col}: "
                f"{df[col].notna().sum():,} non-missing / "
                f"{len(df):,} rows"
            )

print("\n" + "=" * 65)
print("5. PARTICIPANT OVERLAP")
print("=" * 65)

for name, df in data.items():
    print(
        f"{name}: "
        f"{df['RID'].nunique(dropna=True):,} unique participants"
    )

for name in ["ADAS", "CDR", "MMSE", "PTDEMOG"]:
    overlap = (
        set(dx["RID"].dropna())
        & set(data[name]["RID"].dropna())
    )
    print(f"DXSUM ∩ {name}: {len(overlap):,} participants")

print("\nDiagnostic checks completed.")
