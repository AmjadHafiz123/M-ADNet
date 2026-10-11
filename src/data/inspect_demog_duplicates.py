
from pathlib import Path
import pyreadr
import pandas as pd

path = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\extracted\PTDEMOG.rda"
)

df = pyreadr.read_r(str(path))["PTDEMOG"]

keys = ["RID", "VISCODE"]
duplicate_rows = df[
    df.duplicated(subset=keys, keep=False)
].copy()

print("=" * 65)
print("PTDEMOG DUPLICATE INVESTIGATION")
print("=" * 65)

print(f"Total rows: {len(df):,}")
print(f"Rows in duplicate groups: {len(duplicate_rows):,}")

print("\nDuplicate group sizes:")
print(
    duplicate_rows.groupby(keys, dropna=False)
    .size()
    .value_counts()
    .sort_index()
    .to_string()
)

# Fields relevant to participant-level demographics.
demographic_cols = [
    "PTGENDER",
    "PTDOBYY",
    "PTEDUCAT",
    "PTMARRY",
    "PTETHCAT",
    "PTRACCAT",
]

available = [c for c in demographic_cols if c in df.columns]

print("\nConflicting demographic values within duplicate groups:")

if duplicate_rows.empty:
    print("No duplicate groups found.")
else:
    conflicts = []

    for col in available:
        counts = (
            duplicate_rows.groupby(keys, dropna=False)[col]
            .nunique(dropna=True)
        )
        conflicts.append(
            (col, int((counts > 1).sum()))
        )

    for col, count in conflicts:
        print(f"{col}: {count} groups with conflicting values")

    print("\nExample duplicate records:")
    show_cols = keys + [
        c for c in available if c not in keys
    ] + ["PTID", "VISDATE", "ID", "update_stamp"]

    show_cols = [c for c in show_cols if c in df.columns]

    print(
        duplicate_rows[show_cols]
        .sort_values(keys)
        .head(20)
        .to_string(index=False)
    )

print("\nParticipant-level demographic consistency:")

for col in available:
    counts = df.groupby("RID", dropna=True)[col].nunique(
        dropna=True
    )
    print(
        f"{col}: "
        f"{int((counts > 1).sum())} participants with "
        "multiple distinct non-missing values"
    )

print("\nInvestigation complete. No rows were modified.")
