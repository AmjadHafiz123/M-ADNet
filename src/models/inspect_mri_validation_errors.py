import pandas as pd
from pathlib import Path

from src.models.train_mri_baseline import OUTPUT_DIR

analysis_dir = Path(OUTPUT_DIR) / "validation_analysis"

predictions_path = analysis_dir / "validation_predictions.csv"
predictions = pd.read_csv(predictions_path)

print("Available columns:")
print(predictions.columns.tolist())

# Identify incorrect predictions.
errors = predictions[
    predictions["true_label"] != predictions["predicted_label"]
].copy()

print(f"\nTotal validation cases: {len(predictions)}")
print(f"Misclassified cases: {len(errors)}")

print("\nMisclassification counts:")
print(
    pd.crosstab(
        errors["true_label"],
        errors["predicted_label"]
    )
)

print("\nMisclassified validation cases:")
print(errors.to_string(index=False))

# Save a separate error-analysis file.
output_path = analysis_dir / "validation_misclassified_cases.csv"
errors.to_csv(output_path, index=False)

print(f"\nSaved error analysis to: {output_path}")