import pandas as pd
from pathlib import Path

from src.models.train_mri_baseline import OUTPUT_DIR

DATA_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)

ANALYSIS_DIR = Path(OUTPUT_DIR) / "validation_analysis"

predictions = pd.read_csv(
    ANALYSIS_DIR / "validation_predictions.csv"
)

features = pd.read_csv(
    DATA_DIR / "adni_mri_clinical_features.csv"
)

# Participants needing closer inspection.
selected_rids = [
    1098, 210, 1190, 23, 130, 303,
    2233, 2263, 2379
]

predictions["rid"] = predictions["rid"].astype(str)
features["rid"] = features["rid"].astype(str)

audit = predictions.merge(
    features,
    on=["image_id", "rid"],
    how="left",
    suffixes=("_prediction", "_feature"),
    validate="one_to_one"
)

audit = audit[
    audit["rid"].isin([str(x) for x in selected_rids])
].copy()

columns = [
    "rid",
    "image_id",
    "true_label",
    "predicted_label",
    "prob_CN",
    "prob_MCI",
    "prob_Dementia",
    "clinical_date",
    "mri_date",
    "date_gap_days",
    "age_years_approx",
    "sex",
    "education_years",
]

available_columns = [
    col for col in columns if col in audit.columns
]

audit = audit[available_columns].sort_values(
    ["rid", "clinical_date"]
)

print("\nVALIDATION PARTICIPANT AUDIT")
print("--------------------------------")
print(audit.to_string(index=False))

output_path = ANALYSIS_DIR / "validation_participant_audit.csv"
audit.to_csv(output_path, index=False)

print("\nSaved:", output_path)