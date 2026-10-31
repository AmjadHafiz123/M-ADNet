
from pathlib import Path
import pandas as pd

RESULTS = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\Actual_data\processed\madnet_fusion"
    r"\gradcam_validation\gradcam_results.csv"
)

OUTPUT_DIR = RESULTS.parent

df = pd.read_csv(RESULTS)

print("\nTotal validation visits:", len(df))
print("\nPrediction correctness:")
print(df["correct"].value_counts())

print("\nClassification summary:")
print(
    pd.crosstab(
        df["true_label"],
        df["predicted_label"],
        margins=True,
    )
)

print("\nCorrect predictions by actual class:")
print(
    df.groupby("true_label")["correct"]
    .agg(["sum", "count", "mean"])
)

# Save incorrectly classified cases.
errors = df.loc[~df["correct"]].copy()
errors.to_csv(
    OUTPUT_DIR / "gradcam_misclassified_cases.csv",
    index=False,
)

# Save correctly classified cases.
correct = df.loc[df["correct"]].copy()
correct.to_csv(
    OUTPUT_DIR / "gradcam_correct_cases.csv",
    index=False,
)

print("\nSaved:")
print("gradcam_misclassified_cases.csv")
print("gradcam_correct_cases.csv")
