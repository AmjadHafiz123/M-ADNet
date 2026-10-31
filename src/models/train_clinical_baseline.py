
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

DATA_PATH = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data"
    r"\processed\adni_mri_clinical_features.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data"
    r"\processed\clinical_baseline"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "age_years_approx",
    "education_years",
]

CATEGORICAL_FEATURES = [
    "sex",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_NAMES = ["CN", "MCI", "Dementia"]

# Limited, predeclared regularization search.
C_VALUES = [0.01, 0.1, 1.0, 10.0]

RANDOM_STATE = 42


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    "rid",
    "image_id",
    "target",
    "split",
] + FEATURES

missing_columns = [
    c for c in required_columns if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["rid"] = df["rid"].astype(str)
df["image_id"] = df["image_id"].astype(str)
df["split"] = df["split"].astype(str).str.lower().str.strip()

df["target"] = df["target"].astype(str).str.strip()

if not set(df["target"].unique()).issubset(set(CLASS_NAMES)):
    raise ValueError(
        f"Unexpected target labels: {df['target'].unique()}"
    )

# Normalize split naming.
df["split"] = df["split"].replace({
    "val": "validation",
    "valid": "validation",
})

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError("One or more data splits are empty.")

# Participant leakage check.
train_rids = set(train_df["rid"])
val_rids = set(val_df["rid"])
test_rids = set(test_df["rid"])

assert train_rids.isdisjoint(val_rids)
assert train_rids.isdisjoint(test_rids)
assert val_rids.isdisjoint(test_rids)

# Duplicate visit check.
if df["image_id"].duplicated().any():
    raise ValueError("Duplicate image IDs detected.")

print("\nDATASET SUMMARY")
print("------------------------------")
for name, subset in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df),
]:
    print(
        f"{name}: {len(subset)} visits, "
        f"{subset['rid'].nunique()} participants"
    )
    print(subset["target"].value_counts().to_dict())


# --------------------------------------------------
# Prepare features and labels
# --------------------------------------------------

X_train = train_df[FEATURES].copy()
X_val = val_df[FEATURES].copy()

y_train = train_df["target"].copy()
y_val = val_df["target"].copy()

# Numeric preprocessing.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical preprocessing.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])


# --------------------------------------------------
# Train models and select using validation macro-F1
# --------------------------------------------------

results = []
best_model = None
best_c = None
best_score = -1.0

for c_value in C_VALUES:

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(y_val, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        predictions,
    )

    results.append({
        "C": c_value,
        "validation_macro_f1": float(macro_f1),
        "validation_accuracy": float(accuracy),
        "validation_balanced_accuracy": float(
            balanced_accuracy
        ),
    })

    print(
        f"C={c_value:<6} "
        f"Macro-F1={macro_f1:.4f} "
        f"Accuracy={accuracy:.4f} "
        f"Balanced accuracy={balanced_accuracy:.4f}"
    )

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = model
        best_c = c_value


# --------------------------------------------------
# Final validation evaluation
# --------------------------------------------------

val_predictions = best_model.predict(X_val)

val_probabilities = best_model.predict_proba(X_val)

report = classification_report(
    y_val,
    val_predictions,
    labels=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=CLASS_NAMES,
)

print("\nBEST CLINICAL MODEL")
print("------------------------------")
print("Selected C:", best_c)
print("Validation macro-F1:", best_score)

print("\nClassification report:")
print(classification_report(
    y_val,
    val_predictions,
    labels=CLASS_NAMES,
    zero_division=0,
))

print("Confusion matrix:")
print(cm)


# --------------------------------------------------
# Save validation predictions
# --------------------------------------------------

prediction_df = val_df[
    ["image_id", "rid", "target"]
].copy()

prediction_df = prediction_df.rename(
    columns={"target": "true_label"}
)

prediction_df["predicted_label"] = val_predictions

for index, class_name in enumerate(
    best_model.named_steps["classifier"].classes_
):
    prediction_df[f"prob_{class_name}"] = (
        val_probabilities[:, index]
    )

prediction_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(
    OUTPUT_DIR / "validation_confusion_matrix.csv"
)

pd.DataFrame(results).to_csv(
    OUTPUT_DIR / "model_selection_results.csv",
    index=False,
)

# Save the fitted pipeline. It includes train-fitted
# imputation, scaling, encoding and classifier.
joblib.dump(
    best_model,
    OUTPUT_DIR / "best_clinical_baseline.joblib",
)

metrics = {
    "selected_C": best_c,
    "validation_macro_f1": float(best_score),
    "validation_accuracy": float(
        accuracy_score(y_val, val_predictions)
    ),
    "validation_balanced_accuracy": float(
        balanced_accuracy_score(y_val, val_predictions)
    ),
    "classification_report": report,
    "class_names": CLASS_NAMES,
    "train_visits": len(train_df),
    "validation_visits": len(val_df),
    "test_visits": len(test_df),
    "train_participants": len(train_rids),
    "validation_participants": len(val_rids),
    "test_participants": len(test_rids),
}

with open(
    OUTPUT_DIR / "validation_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\nClinical baseline completed.")
print("Outputs:", OUTPUT_DIR)
print("Test set was not evaluated.")
