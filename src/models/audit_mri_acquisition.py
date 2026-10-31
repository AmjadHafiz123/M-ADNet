
import pandas as pd
from pathlib import Path

BASE_DIR = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)

ANALYSIS_DIR = (
    BASE_DIR / "mri_baseline" / "validation_analysis"
)

participant_path = (
    ANALYSIS_DIR / "validation_participant_audit.csv"
)

preprocessing_path = (
    BASE_DIR / "adni_mri_full_preprocessing_audit.csv"
)

dicom_path = (
    BASE_DIR / "adni_dicom_series_validated.csv"
)

participants = pd.read_csv(participant_path)
preprocessing = pd.read_csv(preprocessing_path)
dicom = pd.read_csv(dicom_path)

# Keep only the participants selected for closer inspection.
selected_rids = [
    1098, 210, 1190, 23, 130, 303,
    2233, 2263, 2379
]

participants["rid"] = participants["rid"].astype(str)
participants["image_id"] = participants["image_id"].astype(str)

preprocessing["rid"] = preprocessing["rid"].astype(str)
preprocessing["image_id"] = preprocessing["image_id"].astype(str)

dicom["image_id_from_path"] = (
    dicom["image_id_from_path"].astype(str)
)

participants = participants[
    participants["rid"].isin(map(str, selected_rids))
].copy()

# Select relevant preprocessing metadata.
preprocessing_columns = [
    "image_id",
    "rid",
    "status",
    "error",
    "series_description",
    "dicom_files_read",
    "original_size_xyz",
    "original_spacing_xyz_mm",
    "foreground_voxel_count",
    "intensity_clip_low",
    "intensity_clip_high",
    "foreground_mean_before_zscore",
    "foreground_std_before_zscore",
    "processed_size_xyz",
    "processed_spacing_xyz_mm",
    "output_size_bytes",
]

preprocessing_columns = [
    c for c in preprocessing_columns
    if c in preprocessing.columns
]

# Select relevant DICOM metadata.
dicom_columns = [
    "image_id_from_path",
    "file_count",
    "modality",
    "series_description",
    "study_date",
    "series_date",
    "rows",
    "columns",
    "manufacturer",
    "image_type",
]

dicom_columns = [
    c for c in dicom_columns if c in dicom.columns
]

preprocessing_subset = preprocessing[
    preprocessing_columns
].copy()

dicom_subset = dicom[dicom_columns].copy()

# Check whether the joins are unambiguous.
print("Preprocessing duplicate image IDs:",
      preprocessing_subset["image_id"].duplicated().sum())

print("DICOM duplicate image IDs:",
      dicom_subset["image_id_from_path"].duplicated().sum())

# Join participant records to preprocessing audit.
audit = participants.merge(
    preprocessing_subset,
    on=["image_id", "rid"],
    how="left",
    suffixes=("", "_preprocessing"),
    validate="one_to_one"
)

# Join DICOM inventory by image ID.
audit = audit.merge(
    dicom_subset,
    left_on="image_id",
    right_on="image_id_from_path",
    how="left",
    suffixes=("", "_dicom"),
    validate="many_to_one"
)

# Add simple audit flags.
audit["preprocessing_pass"] = (
    audit["status"] == "PASS"
)

audit["dicom_metadata_found"] = (
    audit["manufacturer"].notna()
)

audit["spacing_recorded"] = (
    audit["original_spacing_xyz_mm"].notna()
)

audit["dicom_file_count_matches"] = (
    audit["dicom_files_read"] == audit["file_count"]
)

# Arrange records by participant and date.
audit = audit.sort_values(
    ["rid", "clinical_date"]
)

print("\nMRI ACQUISITION AUDIT")
print("--------------------------------")
print("Audited visits:", len(audit))
print("Preprocessing PASS:",
      audit["preprocessing_pass"].sum())
print("DICOM metadata found:",
      audit["dicom_metadata_found"].sum())
print("Original spacing recorded:",
      audit["spacing_recorded"].sum())
print("DICOM file count matches:",
      audit["dicom_file_count_matches"].sum())

display_columns = [
    "rid",
    "image_id",
    "true_label",
    "predicted_label",
    "date_gap_days",
    "manufacturer",
    "series_description",
    "file_count",
    "rows",
    "columns",
    "original_size_xyz",
    "original_spacing_xyz_mm",
    "processed_spacing_xyz_mm",
    "status",
    "output_size_bytes",
]

display_columns = [
    c for c in display_columns if c in audit.columns
]

print("\nACQUISITION DETAILS")
print("--------------------------------")
print(audit[display_columns].to_string(index=False))

# Summarize acquisition metadata by prediction correctness.
audit["prediction_correct"] = (
    audit["true_label"] == audit["predicted_label"]
)

print("\nMANUFACTURER BY PREDICTION CORRECTNESS")
print("--------------------------------")
print(pd.crosstab(
    audit["manufacturer"],
    audit["prediction_correct"],
    margins=True
))

print("\nSERIES DESCRIPTION BY PREDICTION CORRECTNESS")
print("--------------------------------")
print(pd.crosstab(
    audit["series_description"],
    audit["prediction_correct"],
    margins=True
))

output_path = ANALYSIS_DIR / "validation_mri_acquisition_audit.csv"
audit.to_csv(output_path, index=False)

print("\nSaved:", output_path)
