
from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\processed"
)

INPUT_FILE = DATA_DIR / "adni_model_ready_30days.csv"
df = pd.read_csv(INPUT_FILE)

df["EXAMDATE"] = pd.to_datetime(
    df["EXAMDATE"], errors="coerce"
)

print("VISIT-LEVEL DATA")
print("=" * 60)
print(f"Visits: {len(df):,}")
print(f"Participants: {df['RID'].nunique():,}")

# Number of visits per participant.
visits_per_person = df.groupby("RID").size()

print("\nVisits per participant:")
print(visits_per_person.describe().round(2).to_string())
print(f"Participants with >1 visit: {(visits_per_person > 1).sum():,}")

# Diagnosis changes over time.
diagnosis_counts = df.groupby("RID")["DIAGNOSIS"].nunique()
changing = diagnosis_counts[diagnosis_counts > 1]

print("\nDiagnosis consistency:")
print(f"Participants with one observed diagnosis: {(diagnosis_counts == 1).sum():,}")
print(f"Participants with changing diagnoses: {len(changing):,}")

# Show example trajectories for participants with changing diagnoses.
if not changing.empty:
    example_ids = changing.index[:10]
    examples = (
        df[df["RID"].isin(example_ids)]
        .sort_values(["RID", "EXAMDATE"])
        [["RID", "EXAMDATE", "VISCODE", "DIAGNOSIS"]]
    )

    print("\nExample diagnosis trajectories:")
    print(examples.to_string(index=False))

# Save participant-level visit counts.
visits_per_person.rename("N_VISITS").to_csv(
    DATA_DIR / "adni_visits_per_participant.csv"
)

print("\nSaved participant visit counts.")
