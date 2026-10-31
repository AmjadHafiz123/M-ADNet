
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)

INPUT_FILE = (
    ROOT
    / "madnet_fusion"
    / "validation_comparison"
    / "mri_fusion_case_comparison.csv"
)

OUTPUT_DIR = (
    ROOT
    / "madnet_fusion"
    / "validation_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = {
    "image_id",
    "rid_mri",
    "true_label",
    "predicted_label_mri",
    "predicted_label_fusion",
    "mri_correct",
    "fusion_correct",
    "outcome",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["rid_mri"] = df["rid_mri"].astype(str)


# --------------------------------------------------
# Participant-level aggregation
# --------------------------------------------------

participant_rows = []

for rid, group in df.groupby("rid_mri"):

    true_labels = sorted(
        group["true_label"].dropna().unique().tolist()
    )

    corrected = int(
        (group["outcome"] == "Fusion corrected MRI error").sum()
    )

    regressed = int(
        (group["outcome"] == "MRI only correct").sum()
    )

    both_correct = int(
        (group["outcome"] == "Both correct").sum()
    )

    both_incorrect = int(
        (group["outcome"] == "Both incorrect").sum()
    )

    mri_correct_visits = int(group["mri_correct"].sum())
    fusion_correct_visits = int(group["fusion_correct"].sum())

    net_change = fusion_correct_visits - mri_correct_visits

    if corrected > 0 and regressed == 0:
        participant_outcome = "Fusion improvement only"

    elif corrected == 0 and regressed > 0:
        participant_outcome = "MRI-only advantage"

    elif corrected > 0 and regressed > 0:
        participant_outcome = "Mixed changes"

    else:
        participant_outcome = "No prediction change"

    participant_rows.append({
        "rid": rid,
        "visit_count": len(group),
        "true_diagnoses": "; ".join(true_labels),
        "multiple_diagnoses": len(true_labels) > 1,
        "mri_correct_visits": mri_correct_visits,
        "fusion_correct_visits": fusion_correct_visits,
        "fusion_corrected_visits": corrected,
        "fusion_regressed_visits": regressed,
        "both_correct_visits": both_correct,
        "both_incorrect_visits": both_incorrect,
        "net_correct_visit_change": net_change,
        "participant_outcome": participant_outcome,
    })


participant_df = pd.DataFrame(participant_rows)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("=" * 65)
print("PARTICIPANT-LEVEL FUSION VALIDATION ANALYSIS")
print("=" * 65)

print("\nValidation visits:", len(df))
print("Unique participants:", len(participant_df))

print(
    "Participants with multiple visits:",
    int((participant_df["visit_count"] > 1).sum())
)

print(
    "Participants with multiple diagnoses:",
    int(participant_df["multiple_diagnoses"].sum())
)

print("\nParticipant outcome categories:")
print(
    participant_df["participant_outcome"]
    .value_counts()
    .to_string()
)

print("\nVisit-level correction and regression totals:")
print(
    "Fusion corrected MRI errors:",
    int(participant_df["fusion_corrected_visits"].sum())
)

print(
    "Fusion regressed MRI predictions:",
    int(participant_df["fusion_regressed_visits"].sum())
)

print(
    "Net additional correct visits:",
    int(participant_df["net_correct_visit_change"].sum())
)


# --------------------------------------------------
# Repeated-visit participant analysis
# --------------------------------------------------

repeated = participant_df[
    participant_df["visit_count"] > 1
].copy()

print("\nRepeated-visit participants:", len(repeated))

if len(repeated):
    print("\nRepeated-visit participant outcomes:")
    print(
        repeated["participant_outcome"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Participants with fusion corrections
# --------------------------------------------------

corrected_participants = participant_df[
    participant_df["fusion_corrected_visits"] > 0
].copy()

regressed_participants = participant_df[
    participant_df["fusion_regressed_visits"] > 0
].copy()

print(
    "\nParticipants with at least one corrected MRI error:",
    len(corrected_participants)
)

print(
    "Participants with at least one regressed MRI prediction:",
    len(regressed_participants)
)


# --------------------------------------------------
# Save detailed results
# --------------------------------------------------

participant_df.to_csv(
    OUTPUT_DIR / "fusion_participant_summary.csv",
    index=False,
)

repeated.to_csv(
    OUTPUT_DIR / "fusion_repeated_visit_participants.csv",
    index=False,
)

corrected_participants.to_csv(
    OUTPUT_DIR / "fusion_improved_participants.csv",
    index=False,
)

regressed_participants.to_csv(
    OUTPUT_DIR / "fusion_regressed_participants.csv",
    index=False,
)


# --------------------------------------------------
# Final output
# --------------------------------------------------

print("\n" + "=" * 65)
print("PARTICIPANT ANALYSIS COMPLETE")
print("=" * 65)

print("Saved files:")
print(OUTPUT_DIR / "fusion_participant_summary.csv")
print(OUTPUT_DIR / "fusion_repeated_visit_participants.csv")
print(OUTPUT_DIR / "fusion_improved_participants.csv")
print(OUTPUT_DIR / "fusion_regressed_participants.csv")

print("\nTest set was NOT evaluated.")
