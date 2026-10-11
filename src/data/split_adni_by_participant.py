
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed"
)

INPUT_FILE = DATA_DIR / "adni_model_ready_30days.csv"
df = pd.read_csv(INPUT_FILE)

# Exclude records without a usable participant ID or target label.
df = df.dropna(subset=["RID", "DIAGNOSIS"]).copy()
df["RID"] = df["RID"].astype(int)

# First split: 80% of participants for development, 20% for testing.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)

dev_idx, test_idx = next(
    split1.split(df, groups=df["RID"])
)

dev = df.iloc[dev_idx].copy()
test = df.iloc[test_idx].copy()

# Second split: 20% of development participants for validation.
# Overall target is approximately 64% train, 16% validation, 20% test.
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=43
)

train_idx, val_idx = next(
    split2.split(dev, groups=dev["RID"])
)

train = dev.iloc[train_idx].copy()
val = dev.iloc[val_idx].copy()

splits = {
    "train": train,
    "validation": val,
    "test": test,
}

# Verify that no participant occurs in multiple splits.
for name, data in splits.items():
    print(
        f"{name}: {len(data):,} visits, "
        f"{data['RID'].nunique():,} participants"
    )

sets = {
    name: set(data["RID"].unique())
    for name, data in splits.items()
}

assert sets["train"].isdisjoint(sets["validation"])
assert sets["train"].isdisjoint(sets["test"])
assert sets["validation"].isdisjoint(sets["test"])

print("\nParticipant overlap check: PASSED")

for name, data in splits.items():
    print(f"\n{name.upper()} diagnosis distribution:")
    print(data["DIAGNOSIS"].value_counts().to_string())

    data.to_csv(
        DATA_DIR / f"adni_{name}_30days.csv",
        index=False,
    )

print("\nSaved train, validation, and test CSV files.")
