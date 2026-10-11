
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed"
)
INPUT_FILE = DATA_DIR / "adni_clinical_merged.csv"

df = pd.read_csv(INPUT_FILE)

date_cols = {
    "ADAS": "ADAS_VISDATE",
    "CDR": "CDR_VISDATE",
    "MMSE": "MMSE_VISDATE",
}

df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

print("DATE ALIGNMENT SUMMARY")
print("=" * 75)

summary = []

for name, col in date_cols.items():
    if col not in df.columns:
        continue

    df[col] = pd.to_datetime(df[col], errors="coerce")
    days = (df[col] - df["EXAMDATE"]).dt.days
    valid = days.dropna()

    if valid.empty:
        print(f"\n{name}: no valid date pairs")
        continue

    summary.append({
        "assessment": name,
        "valid_date_pairs": len(valid),
        "same_day": int((valid == 0).sum()),
        "within_7_days": int((valid.abs() <= 7).sum()),
        "within_30_days": int((valid.abs() <= 30).sum()),
        "over_30_days": int((valid.abs() > 30).sum()),
        "median_abs_gap_days": float(valid.abs().median()),
        "90th_percentile_abs_gap_days": float(
            valid.abs().quantile(0.90)
        ),
        "missing_assessment_date": int(df[col].isna().sum()),
    })

    print(f"\n{name}")
    print(f"  Valid date pairs:       {len(valid):,}")
    print(f"  Same day:               {(valid == 0).sum():,}")
    print(f"  Within 7 days:          {(valid.abs() <= 7).sum():,}")
    print(f"  Within 30 days:         {(valid.abs() <= 30).sum():,}")
    print(f"  Over 30 days:           {(valid.abs() > 30).sum():,}")
    print(f"  Median absolute gap:    {valid.abs().median():.1f} days")
    print(
        "  90th percentile gap:   "
        f"{valid.abs().quantile(0.90):.1f} days"
    )
    print(f"  Missing assessment date: {df[col].isna().sum():,}")

if summary:
    pd.DataFrame(summary).to_csv(
        DATA_DIR / "adni_date_alignment_summary.csv",
        index=False,
    )

# Save diagnosis visits with at least one assessment >30 days away.
gap_cols = []

for name, col in date_cols.items():
    if col in df.columns:
        gap_col = f"{name}_GAP_DAYS"
        df[gap_col] = (df[col] - df["EXAMDATE"]).dt.days
        gap_cols.append(gap_col)

if gap_cols:
    far = df[
        df[gap_cols].abs().gt(30).any(axis=1)
    ].copy()

    far.to_csv(
        DATA_DIR / "adni_date_gaps_over_30_days.csv",
        index=False,
    )

    print("\nSaved:")
    print(f"  {DATA_DIR / 'adni_date_alignment_summary.csv'}")
    print(f"  {DATA_DIR / 'adni_date_gaps_over_30_days.csv'}")
    print(f"  Visits with any assessment gap >30 days: {len(far):,}")
