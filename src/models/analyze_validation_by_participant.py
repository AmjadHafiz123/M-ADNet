import pandas as pd
from pathlib import Path

from src.models.train_mri_baseline import OUTPUT_DIR

analysis_dir = Path(OUTPUT_DIR) / "validation_analysis"

predictions = pd.read_csv(
    analysis_dir / "validation_predictions.csv"
)

label_names = {
    0: "CN",
    1: "MCI",
    2: "Dementia",
}

predictions["true_name"] = predictions["true_label"].map(label_names)
predictions["predicted_name"] = predictions["predicted_label"].map(label_names)

predictions["correct"] = (
    predictions["true_label"] == predictions["predicted_label"]
)

# Participant-level summary.
participant_summary = (
    predictions.groupby("rid")
    .agg(
        visits=("image_id", "count"),
        actual_diagnoses=("true_name", lambda x: ", ".join(sorted(set(x)))),
        predicted_diagnoses=("predicted_name", lambda x: ", ".join(sorted(set(x)))),
        correct_visits=("correct", "sum"),
        incorrect_visits=("correct", lambda x: (~x).sum()),
    )
    .reset_index()
)

participant_summary["has_multiple_visits"] = (
    participant_summary["visits"] > 1
)

participant_summary["has_multiple_actual_diagnoses"] = (
    participant_summary["actual_diagnoses"].str.contains(",")
)

participant_summary["repeatedly_misclassified"] = (
    (participant_summary["incorrect_visits"] > 1)
)

print("\nVALIDATION PARTICIPANT SUMMARY")
print("--------------------------------")
print("Participants:", len(participant_summary))
print("Participants with multiple visits:",
      participant_summary["has_multiple_visits"].sum())
print("Participants with multiple actual diagnoses:",
      participant_summary["has_multiple_actual_diagnoses"].sum())
print("Participants with at least one error:",
      (participant_summary["incorrect_visits"] > 0).sum())
print("Participants repeatedly misclassified:",
      participant_summary["repeatedly_misclassified"].sum())

print("\nPARTICIPANTS WITH ERRORS")
print("--------------------------------")

errors = participant_summary[
    participant_summary["incorrect_visits"] > 0
].sort_values(
    ["incorrect_visits", "visits"],
    ascending=False
)

print(errors.to_string(index=False))

print("\nPARTICIPANTS WITH MULTIPLE ACTUAL DIAGNOSES")
print("--------------------------------")

changing_diagnosis = participant_summary[
    participant_summary["has_multiple_actual_diagnoses"]
]

print(changing_diagnosis.to_string(index=False))

# Save report.
participant_summary.to_csv(
    analysis_dir / "validation_participant_summary.csv",
    index=False
)

errors.to_csv(
    analysis_dir / "validation_participant_errors.csv",
    index=False
)

print("\nReports saved in:", analysis_dir)