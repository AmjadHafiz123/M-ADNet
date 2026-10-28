from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import pyreadr


PROJECT = Path(r"C:\Users\User\Desktop\MSC\Research\App\M-ADNet")
DATA = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = DATA / "Actual_data" / "processed"
DEMOGRAPHICS = DATA / "Clinical" / "extracted" / "PTDEMOG.rda"

MANIFEST = PROCESSED / "adni_mri_processing_manifest.csv"
AUDIT = PROCESSED / "adni_mri_full_preprocessing_audit.csv"
SPLITS = PROCESSED / "adni_mri_participant_split_assignments.csv"

OUTPUT = PROCESSED / "adni_mri_clinical_features.csv"
REPORT = PROCESSED / "adni_mri_clinical_features_quality_report.txt"

EXPECTED_TARGETS = {"CN", "MCI", "Dementia"}
EXPECTED_SPLITS = {"train", "validation", "test"}


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def normalize_rid(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (ValueError, TypeError):
        pass
    return text


def normalize_sex(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    mapping = {
        "M": "Male",
        "MALE": "Male",
        "F": "Female",
        "FEMALE": "Female",
    }
    return mapping.get(text, np.nan)


def parse_birth_month(value):
    """Parse PTDOB when represented as month/year, without inventing a day."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE"}:
        return None

    match = re.fullmatch(r"(\d{1,2})/(\d{4})", text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return (year, month)

    # Accept an ISO-like date only when a year and month are explicit.
    match = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return (year, month)

    return None


def parse_date(series):
    return pd.to_datetime(series, errors="coerce")


def main():
    for path in [DEMOGRAPHICS, MANIFEST, AUDIT, SPLITS]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    audit = pd.read_csv(AUDIT, low_memory=False)
    splits = pd.read_csv(SPLITS, low_memory=False)

    require_columns(
        manifest,
        ["image_id", "rid", "target", "split", "clinical_date"],
        "Processing manifest",
    )
    require_columns(audit, ["image_id", "status"], "Preprocessing audit")
    require_columns(splits, ["rid", "split"], "Participant split assignments")

    for df, label in [(manifest, "manifest"), (audit, "audit")]:
        if df["image_id"].isna().any():
            raise ValueError(f"{label} contains missing image_id values")
        df["image_id"] = df["image_id"].astype(str).str.strip()

    manifest["rid"] = manifest["rid"].map(normalize_rid)
    splits["rid"] = splits["rid"].map(normalize_rid)

    if manifest["image_id"].duplicated().any():
        raise ValueError("Processing manifest has duplicate image_id values")
    if audit["image_id"].duplicated().any():
        raise ValueError("Preprocessing audit has duplicate image_id values")
    if splits["rid"].duplicated().any():
        raise ValueError("Split assignment file has duplicate participant RIDs")

    manifest["target"] = manifest["target"].astype(str).str.strip()
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    splits["split"] = splits["split"].astype(str).str.strip().str.lower()

    unexpected_targets = set(manifest["target"].dropna()) - EXPECTED_TARGETS
    if unexpected_targets:
        raise ValueError(f"Unexpected target labels: {unexpected_targets}")

    unexpected_splits = set(manifest["split"].dropna()) - EXPECTED_SPLITS
    if unexpected_splits:
        raise ValueError(f"Unexpected split labels: {unexpected_splits}")

    # Verify the audit and only retain successfully preprocessed MRI visits.
    audit["status"] = audit["status"].astype(str).str.strip().str.upper()
    audit_status = audit[["image_id", "status"]].rename(
        columns={"status": "preprocessing_status"}
    )
    checked = manifest.merge(
        audit_status, on="image_id", how="left", validate="one_to_one"
    )

    if checked["preprocessing_status"].isna().any():
        raise ValueError("Some manifest visits are missing from the preprocessing audit")

    failed = checked.loc[checked["preprocessing_status"] != "PASS", "image_id"]
    if not failed.empty:
        raise ValueError(
            f"{len(failed)} manifest visits do not have PASS status. "
            "Resolve these before feature preparation."
        )

    # Confirm the existing participant split assignments agree with the manifest.
    split_lookup = splits.set_index("rid")["split"]
    checked["assigned_split"] = checked["rid"].map(split_lookup)

    if checked["assigned_split"].isna().any():
        missing = checked.loc[checked["assigned_split"].isna(), "rid"].unique()
        raise ValueError(
            f"Participants missing from split assignments: {missing[:10].tolist()}"
        )

    mismatch = checked["split"] != checked["assigned_split"]
    if mismatch.any():
        examples = checked.loc[
            mismatch, ["rid", "image_id", "split", "assigned_split"]
        ].head(10)
        raise ValueError(
            "Manifest split differs from participant split assignments:\n"
            + examples.to_string(index=False)
        )

    # Read PTDEMOG without merging diagnosis-derived variables.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rda = pyreadr.read_r(str(DEMOGRAPHICS))

    if not rda:
        raise ValueError("No data frame was found in PTDEMOG.rda")

    demo = next(iter(rda.values())).copy()
    require_columns(
        demo, ["RID", "PTGENDER", "PTEDUCAT", "PTDOB"], "PTDEMOG"
    )
    if "PTID" not in demo.columns:
        demo["PTID"] = np.nan

    demo["rid"] = demo["RID"].map(normalize_rid)
    demo = demo.loc[demo["rid"].notna()].copy()
    demo["sex_normalized"] = demo["PTGENDER"].map(normalize_sex)
    demo["education_numeric"] = pd.to_numeric(
        demo["PTEDUCAT"], errors="coerce"
    )
    demo.loc[
        ~demo["education_numeric"].between(0, 40),
        "education_numeric"
    ] = np.nan
    demo["birth_month"] = demo["PTDOB"].map(parse_birth_month)

    demographic_rows = []
    for rid, group in demo.groupby("rid", sort=False):
        sexes = sorted(group["sex_normalized"].dropna().unique().tolist())
        educ = group["education_numeric"].dropna()
        educ_counts = educ.value_counts()
        distinct_educ = sorted(educ.unique().tolist())

        # Choose education only when a unique mode exists.
        if len(educ_counts) == 0:
            education = np.nan
            education_tie = False
        else:
            max_count = educ_counts.max()
            modes = educ_counts[educ_counts == max_count].index.tolist()
            education_tie = len(modes) != 1
            education = float(modes[0]) if len(modes) == 1 else np.nan

        birth_values = sorted(
            set(v for v in group["birth_month"].tolist() if v is not None)
        )

        demographic_rows.append({
            "rid": rid,
            "sex": sexes[0] if len(sexes) == 1 else np.nan,
            "sex_conflict": len(sexes) > 1,
            "education_years": education,
            "education_conflict": len(distinct_educ) > 1,
            "education_mode_tie": education_tie,
            "birthdate_conflict": len(birth_values) > 1,
            "birth_year": birth_values[0][0] if len(birth_values) == 1 else np.nan,
            "birth_month_number": birth_values[0][1] if len(birth_values) == 1 else np.nan,
            "demographic_record_count": int(len(group)),
        })

    person_demo = pd.DataFrame(demographic_rows)

    # Age is approximate because PTDOB typically records month/year, not day.
    checked["clinical_date_parsed"] = parse_date(checked["clinical_date"])
    checked = checked.merge(
        person_demo, on="rid", how="left", validate="many_to_one"
    )

    age_values = []
    for row in checked.itertuples(index=False):
        visit = row.clinical_date_parsed
        year = row.birth_year
        month = row.birth_month_number

        if pd.isna(visit) or pd.isna(year) or pd.isna(month):
            age_values.append(np.nan)
            continue

        age = visit.year - int(year)
        if visit.month < int(month):
            age -= 1

        # Keep implausible values missing and report them rather than silently using them.
        age_values.append(float(age) if 40 <= age <= 110 else np.nan)

    checked["age_years_approx"] = age_values

    # Only approved predictors plus identifiers/metadata are written.
    output_columns = [
        "image_id",
        "rid",
        "target",
        "split",
        "clinical_date",
        "sex",
        "education_years",
        "age_years_approx",
        "sex_conflict",
        "education_conflict",
        "education_mode_tie",
        "birthdate_conflict",
        "demographic_record_count",
    ]

    # Preserve useful MRI metadata when present.
    for column in ["mri_date", "date_gap_days", "series_folder", "output_file"]:
        if column in checked.columns:
            output_columns.append(column)

    features = checked[output_columns].copy()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)

    target_counts = features["target"].value_counts().to_dict()
    split_counts = features["split"].value_counts().to_dict()
    participants_by_split = (
        features.groupby("split")["rid"].nunique().to_dict()
    )
    duplicate_participants_across_splits = (
        features.groupby("rid")["split"].nunique().gt(1).sum()
    )

    lines = [
        "M-ADNet clinical feature preparation report",
        "===========================================",
        f"Input manifest visits: {len(manifest)}",
        f"Visits with successful preprocessing: {len(features)}",
        f"Unique participants: {features['rid'].nunique()}",
        f"Duplicate image IDs: {features['image_id'].duplicated().sum()}",
        f"Participants assigned to multiple splits: {duplicate_participants_across_splits}",
        "",
        f"Target counts: {target_counts}",
        f"Visit counts by split: {split_counts}",
        f"Unique participants by split: {participants_by_split}",
        "",
        f"Missing sex: {features['sex'].isna().sum()}",
        f"Missing education: {features['education_years'].isna().sum()}",
        f"Missing approximate age: {features['age_years_approx'].isna().sum()}",
        f"Participants with sex conflicts: {person_demo['sex_conflict'].sum()}",
        f"Participants with education conflicts: {person_demo['education_conflict'].sum()}",
        f"Participants with education-mode ties: {person_demo['education_mode_tie'].sum()}",
        f"Participants with birth-date conflicts: {person_demo['birthdate_conflict'].sum()}",
        "",
        "Predictor policy:",
        "- Included: sex, education years, approximate age.",
        "- Excluded: ADAS, CDR, MMSE, diagnosis codes and diagnosis-derived fields.",
        "- Education conflicts use a unique participant-level mode; ties remain missing.",
        "- Conflicting birth month/year values remain unresolved and age is missing.",
        "- Missing values are not imputed in this preparation step.",
        "- Clinical visit diagnosis remains the target; MRI manifest group is not the target.",
        "",
        f"Feature table: {OUTPUT}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nFEATURE PREPARATION COMPLETE")


if __name__ == "__main__":
    main()

