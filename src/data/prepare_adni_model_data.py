
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed"
)

INPUT_FILE = DATA_DIR / "adni_clinical_merged.csv"
OUTPUT_FILE = DATA_DIR / "adni_model_ready_30days.csv"

df = pd.read_csv(INPUT_FILE)

# Parse dates.
date_columns = [
    "EXAMDATE",
    "ADAS_VISDATE",
    "CDR_VISDATE",
    "MMSE_VISDATE",
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Calculate absolute date gaps.
for name in ["ADAS", "CDR", "MMSE"]:
    df[f"{name}_GAP_DAYS"] = (
        df[f"{name}_VISDATE"] - df["EXAMDATE"]
    ).dt.days.abs()

# Require all three scores.
score_columns = ["TOTSCORE", "CDGLOBAL", "MMSCORE"]
has_all_scores = df[score_columns].notna().all(axis=1)

# Require valid dates and all assessments within 30 days.
gap_columns = [
    "ADAS_GAP_DAYS",
    "CDR_GAP_DAYS",
    "MMSE_GAP_DAYS",
]

within_threshold = df[gap_columns].le(30).all(axis=1)

model_df = df.loc[has_all_scores & within_threshold].copy()

# Stable ordering for longitudinal work.
model_df = model_df.sort_values(
    ["RID", "EXAMDATE", "VISCODE"],
    na_position="last",
).reset_index(drop=True)

# Save the derived dataset.
model_df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("ADNI MODEL-READY DATASET")
print("=" * 60)
print(f"Original diagnosis visits: {len(df):,}")
print(f"Selected visits:           {len(model_df):,}")
print(f"Unique participants:        {model_df['RID'].nunique():,}")
print(f"Output file: {OUTPUT_FILE}")

print("\nDiagnosis distribution:")
print(model_df["DIAGNOSIS"].value_counts(dropna=False).to_string())

print("\nMissing values in selected scores:")
print(model_df[score_columns].isna().sum().to_string())
