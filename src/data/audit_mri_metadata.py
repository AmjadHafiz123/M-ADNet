
from pathlib import Path

import pandas as pd


DATA_ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
)

MRI_CSV = (
    DATA_ROOT
    / "Actual_data"
    / "ADNI_MPRAGE_search_result_7_18_2026.csv"
)


def main() -> None:
    if not MRI_CSV.exists():
        raise FileNotFoundError(
            f"MRI metadata CSV not found: {MRI_CSV}"
        )

    # Read locally; do not print participant-level records.
    df = pd.read_csv(MRI_CSV)

    print("=== MRI metadata audit ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required_columns = [
        "Subject",
        "Group",
        "Modality",
        "Description",
        "Acq Date",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("Missing expected columns:", missing_columns)
        return

    print("\nMissing-value counts:")
    for col in required_columns:
        print(f"{col}: {int(df[col].isna().sum())}")

    dates = pd.to_datetime(
        df["Acq Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )

    print("\nAcquisition-date validation:")
    print(f"Valid dates: {int(dates.notna().sum())}")
    print(f"Invalid or missing dates: {int(dates.isna().sum())}")

    if dates.notna().any():
        print(
            "Year range:",
            int(dates.dt.year.min()),
            "to",
            int(dates.dt.year.max()),
        )

    print("\nModality counts:")
    print(df["Modality"].value_counts(dropna=False).to_string())

    print("\nGroup counts (metadata labels; not yet validated):")
    print(df["Group"].value_counts(dropna=False).to_string())

    print("\nAudit complete. No participant rows were printed.")


if __name__ == "__main__":
    main()
