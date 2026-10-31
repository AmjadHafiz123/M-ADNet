
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)

MRI_FILE = (
    ROOT
    / "mri_baseline"
    / "validation_analysis"
    / "validation_predictions.csv"
)

FUSION_FILE = (
    ROOT
    / "madnet_fusion"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "madnet_fusion"
    / "validation_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["CN", "MCI", "Dementia"]

NUMERIC_LABEL_MAP = {
    0: "CN",
    1: "MCI",
    2: "Dementia",
}


# --------------------------------------------------
# Label normalization
# --------------------------------------------------

def normalize_labels(series, column_name, numeric_allowed=True):
    """Convert numeric or textual diagnosis labels to standard names."""

    normalized = []

    for value in series:
        if pd.isna(value):
            raise ValueError(
                f"Missing value found in {column_name}."
            )

        text_value = str(value).strip()

        if numeric_allowed:
            try:
                numeric_value = int(float(text_value))

                if numeric_value in NUMERIC_LABEL_MAP:
                    normalized.append(
                        NUMERIC_LABEL_MAP[numeric_value]
                    )
                    continue
            except ValueError:
                pass

        label = text_value.upper()

        if label == "CN":
            normalized.append("CN")
        elif label == "MCI":
            normalized.append("MCI")
        elif label in {"DEMENTIA", "AD"}:
            normalized.append("Dementia")
        else:
            raise ValueError(
                f"Unknown label '{value}' in {column_name}."
            )

    return pd.Series(normalized, index=series.index)


# --------------------------------------------------
# Load prediction files
# --------------------------------------------------

def load_predictions(path, model_name):

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "image_id",
        "rid",
        "true_label",
        "predicted_label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name} file is missing columns: {missing}"
        )

    if df["image_id"].isna().any():
        raise ValueError(
            f"{model_name} contains missing image IDs."
        )

    if df["rid"].isna().any():
        raise ValueError(
            f"{model_name} contains missing participant IDs."
        )

    if df["image_id"].duplicated().any():
        duplicates = df.loc[
            df["image_id"].duplicated(),
            "image_id"
        ].tolist()

        raise ValueError(
            f"{model_name} contains duplicate image IDs: "
            f"{duplicates[:10]}"
        )

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["rid"] = df["rid"].astype(str).str.strip()

    df["true_label"] = normalize_labels(
        df["true_label"],
        f"{model_name}.true_label",
    )

    df["predicted_label"] = normalize_labels(
        df["predicted_label"],
        f"{model_name}.predicted_label",
    )

    return df


# --------------------------------------------------
# Load both models
# --------------------------------------------------

print("=" * 65)
print("MRI-ONLY VS MRI-CLINICAL FUSION VALIDATION COMPARISON")
print("=" * 65)

mri = load_predictions(MRI_FILE, "MRI-only")
fusion = load_predictions(FUSION_FILE, "Fusion")

print("\nMRI-only validation visits:", len(mri))
print("Fusion validation visits:", len(fusion))


# --------------------------------------------------
# Match validation cases
# --------------------------------------------------

merged = mri.merge(
    fusion,
    on="image_id",
    suffixes=("_mri", "_fusion"),
    how="outer",
    indicator=True,
    validate="one_to_one",
)

merge_counts = merged["_merge"].value_counts()

print("\nMerge status:")
print(merge_counts)

if not (merged["_merge"] == "both").all():
    raise ValueError(
        "The prediction files do not contain identical validation scans. "
        "Investigate missing or additional image IDs."
    )

merged = merged.drop(columns=["_merge"])

if len(merged) != len(mri) or len(merged) != len(fusion):
    raise ValueError(
        "Validation prediction counts do not match."
    )


# --------------------------------------------------
# Verify participants and ground-truth labels
# --------------------------------------------------

rid_mismatch = (
    merged["rid_mri"].astype(str)
    != merged["rid_fusion"].astype(str)
)

if rid_mismatch.any():
    raise ValueError(
        "Participant IDs differ between MRI-only and fusion predictions."
    )

label_mismatch = (
    merged["true_label_mri"]
    != merged["true_label_fusion"]
)

if label_mismatch.any():
    examples = merged.loc[
        label_mismatch,
        [
            "image_id",
            "rid_mri",
            "true_label_mri",
            "true_label_fusion",
        ],
    ]

    print("\nGround-truth mismatch examples:")
    print(examples.head(20).to_string(index=False))

    raise ValueError(
        "Ground-truth diagnoses differ between prediction files."
    )

merged["true_label"] = merged["true_label_mri"]

print("\nPASS: All validation scans match.")
print("PASS: All participant IDs match.")
print("PASS: All ground-truth diagnoses match.")


# --------------------------------------------------
# Determine prediction correctness
# --------------------------------------------------

merged["mri_correct"] = (
    merged["predicted_label_mri"]
    == merged["true_label"]
)

merged["fusion_correct"] = (
    merged["predicted_label_fusion"]
    == merged["true_label"]
)


def determine_outcome(row):

    if row["mri_correct"] and row["fusion_correct"]:
        return "Both correct"

    if row["mri_correct"] and not row["fusion_correct"]:
        return "MRI only correct"

    if not row["mri_correct"] and row["fusion_correct"]:
        return "Fusion corrected MRI error"

    return "Both incorrect"


merged["outcome"] = merged.apply(
    determine_outcome,
    axis=1,
)


# --------------------------------------------------
# Overall performance
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


mri_metrics = calculate_metrics(
    merged["true_label"],
    merged["predicted_label_mri"],
)

fusion_metrics = calculate_metrics(
    merged["true_label"],
    merged["predicted_label_fusion"],
)

metrics_df = pd.DataFrame(
    [
        {"model": "MRI-only", **mri_metrics},
        {"model": "MRI-clinical fusion", **fusion_metrics},
    ]
)

print("\n" + "=" * 65)
print("OVERALL PERFORMANCE")
print("=" * 65)

print(metrics_df.to_string(index=False))


# --------------------------------------------------
# Prediction transitions
# --------------------------------------------------

print("\n" + "=" * 65)
print("PREDICTION TRANSITIONS")
print("=" * 65)

transition_counts = (
    merged["outcome"]
    .value_counts()
    .reindex(
        [
            "Both correct",
            "Fusion corrected MRI error",
            "MRI only correct",
            "Both incorrect",
        ],
        fill_value=0,
    )
)

print(transition_counts.to_string())

print(
    "\nNet change in correctly classified visits:",
    int(
        merged["fusion_correct"].sum()
        - merged["mri_correct"].sum()
    ),
)


# --------------------------------------------------
# Per-class classification reports
# --------------------------------------------------

print("\n" + "=" * 65)
print("MRI-ONLY CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        merged["true_label"],
        merged["predicted_label_mri"],
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    )
)

print("\n" + "=" * 65)
print("FUSION CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        merged["true_label"],
        merged["predicted_label_fusion"],
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    )
)


# --------------------------------------------------
# Confusion matrices
# --------------------------------------------------

mri_cm = confusion_matrix(
    merged["true_label"],
    merged["predicted_label_mri"],
    labels=LABELS,
)

fusion_cm = confusion_matrix(
    merged["true_label"],
    merged["predicted_label_fusion"],
    labels=LABELS,
)

print("\nMRI-only confusion matrix:")
print(
    pd.DataFrame(
        mri_cm,
        index=[f"Actual_{x}" for x in LABELS],
        columns=[f"Predicted_{x}" for x in LABELS],
    )
)

print("\nFusion confusion matrix:")
print(
    pd.DataFrame(
        fusion_cm,
        index=[f"Actual_{x}" for x in LABELS],
        columns=[f"Predicted_{x}" for x in LABELS],
    )
)


# --------------------------------------------------
# Class-specific transitions
# --------------------------------------------------

print("\n" + "=" * 65)
print("CLASS-SPECIFIC PREDICTION TRANSITIONS")
print("=" * 65)

class_transition_rows = []

for label in LABELS:

    subset = merged[
        merged["true_label"] == label
    ]

    counts = (
        subset["outcome"]
        .value_counts()
    )

    row = {
        "class": label,
        "support": len(subset),
        "both_correct": counts.get("Both correct", 0),
        "fusion_corrected": counts.get(
            "Fusion corrected MRI error", 0
        ),
        "mri_only_correct": counts.get(
            "MRI only correct", 0
        ),
        "both_incorrect": counts.get("Both incorrect", 0),
    }

    class_transition_rows.append(row)

    print(f"\n{label} (n={len(subset)}):")
    print(counts.to_string())


class_transitions = pd.DataFrame(
    class_transition_rows
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

merged.to_csv(
    OUTPUT_DIR / "mri_fusion_case_comparison.csv",
    index=False,
)

merged[
    merged["outcome"] == "Fusion corrected MRI error"
].to_csv(
    OUTPUT_DIR / "fusion_corrected_cases.csv",
    index=False,
)

merged[
    merged["outcome"] == "MRI only correct"
].to_csv(
    OUTPUT_DIR / "fusion_regressed_cases.csv",
    index=False,
)

merged[
    merged["outcome"] == "Both incorrect"
].to_csv(
    OUTPUT_DIR / "both_models_incorrect_cases.csv",
    index=False,
)

metrics_df.to_csv(
    OUTPUT_DIR / "model_performance_comparison.csv",
    index=False,
)

class_transitions.to_csv(
    OUTPUT_DIR / "class_transition_summary.csv",
    index=False,
)

pd.DataFrame(
    mri_cm,
    index=[f"Actual_{x}" for x in LABELS],
    columns=[f"Predicted_{x}" for x in LABELS],
).to_csv(
    OUTPUT_DIR / "mri_confusion_matrix.csv"
)

pd.DataFrame(
    fusion_cm,
    index=[f"Actual_{x}" for x in LABELS],
    columns=[f"Predicted_{x}" for x in LABELS],
).to_csv(
    OUTPUT_DIR / "fusion_confusion_matrix.csv"
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("COMPARISON COMPLETE")
print("=" * 65)

print("Validation visits:", len(merged))
print("Validation participants:", merged["rid_mri"].nunique())

print(
    "MRI-only correct:",
    int(merged["mri_correct"].sum()),
)

print(
    "Fusion correct:",
    int(merged["fusion_correct"].sum()),
)

print(
    "Fusion corrected MRI errors:",
    int(
        (
            merged["outcome"]
            == "Fusion corrected MRI error"
        ).sum()
    ),
)

print(
    "Fusion regressed MRI predictions:",
    int(
        (
            merged["outcome"]
            == "MRI only correct"
        ).sum()
    ),
)

print("\nSaved comparison files to:")
print(OUTPUT_DIR)

print("\nTest set was NOT evaluated.")
