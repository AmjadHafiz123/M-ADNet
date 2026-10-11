
from pathlib import Path
import pandas as pd

path = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed\adni_clinical_merged.csv"
)

df = pd.read_csv(path)

df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

date_cols = {
    "ADAS": "ADAS_VISDATE",
    "CDR": "CDR_VISDATE",
    "MMSE": "MMSE_VISDATE",
}

for name, col in date_cols.items():
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[f"{name}_GAP_DAYS"] = (
        df[col] - df["EXAMDATE"]
    ).dt.days.abs()

print("Diagnosis visits:", len(df))

for threshold in [7, 14, 30]:
    print(f"\nWithin {threshold} days:")
    for name in date_cols:
        gap = df[f"{name}_GAP_DAYS"]
        print(f"  {name}: {(gap <= threshold).sum():,}")

    all_three = df[
        [f"{name}_GAP_DAYS" for name in date_cols]
    ].le(threshold).all(axis=1)

    print(f"  All three assessments aligned: {all_three.sum():,}")

    any_score = df[
        ["TOTSCORE", "CDGLOBAL", "MMSCORE"]
    ].notna().all(axis=1)

    print(
        "  All three scores present and aligned:",
        (all_three & any_score).sum()
    )
