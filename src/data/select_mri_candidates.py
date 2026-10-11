
from pathlib import Path
import re

import pandas as pd

BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = BASE / "Actual_data" / "processed"
CLINICAL_PROCESSED = BASE / "Clinical" / "processed"

INPUT = PROCESSED / "adni_mri_clinical_match_candidates.csv"
CANDIDATE_AUDIT = PROCESSED / "adni_mri_selection_audit.csv"
SHORTLIST = PROCESSED / "adni_mri_visit_shortlist.csv"
SUMMARY = PROCESSED / "adni_mri_selection_summary.txt"

SPLIT_FILES = {
    "train": CLINICAL_PROCESSED / "adni_train_30days.csv",
    "validation": CLINICAL_PROCESSED / "adni_validation_30days.csv",
    "test": CLINICAL_PROCESSED / "adni_test_30days.csv",
}


def normalize_rid(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(r"_S_(\d+)", s, re.IGNORECASE)
    if match:
        return str(int(match.group(1)))

    if re.fullmatch(r"\d+", s):
        return str(int(s))

    return ""


def col_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, aliases):
    lookup = {col_key(c): c for c in df.columns}

    for alias in aliases:
        if col_key(alias) in lookup:
            return lookup[col_key(alias)]

    raise ValueError(
        f"Missing expected column {aliases}; "
        f"available columns: {list(df.columns)}"
    )


def as_bool(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)

    required = [
        "image_id", "rid", "clinical_date", "target",
        "date_gap_days", "within_30_days", "series_folder",
        "series_description", "dicom_file_count",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in candidate table: {missing}")

    df["rid"] = df["rid"].map(normalize_rid)
    df["clinical_date"] = pd.to_datetime(
        df["clinical_date"], errors="coerce"
    )
    df["date_gap_days"] = pd.to_numeric(
        df["date_gap_days"], errors="coerce"
    )
    df["dicom_file_count"] = pd.to_numeric(
        df["dicom_file_count"], errors="coerce"
    ).fillna(0)

    descriptions = df["series_description"].fillna("").astype(str)
    desc_upper = descriptions.str.upper()

    df["has_mprage_description"] = desc_upper.str.contains(
        "MPRAGE", regex=False
    )
    df["has_repeat_description"] = desc_upper.str.contains(
        "REPEAT|REPET", regex=True
    )
    df["has_usable_series_path"] = df["series_folder"].notna() & (
        df["series_folder"].astype(str).str.strip() != ""
    )

    # Keep only date-matched candidates with a recognized target
    # and an existing extracted DICOM series folder.
    eligible = df[
        as_bool(df["within_30_days"])
        & df["rid"].ne("")
        & df["clinical_date"].notna()
        & df["target"].isin(["CN", "MCI", "Dementia"])
        & df["has_usable_series_path"]
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No eligible candidates. Check the matching report and paths."
        )

    # Attach the existing clinical-only participant split when possible.
    split_map = {}

    for split_name, path in SPLIT_FILES.items():
        if not path.is_file():
            print(f"WARNING: split file not found: {path}")
            continue

        split_df = pd.read_csv(path, low_memory=False)
        rid_col = find_column(split_df, ["RID", "Research ID"])
        ids = split_df[rid_col].map(normalize_rid)
        ids = ids[ids != ""]

        for rid in ids:
            if rid in split_map and split_map[rid] != split_name:
                raise RuntimeError(
                    f"Participant {rid} appears in multiple splits."
                )
            split_map[rid] = split_name

    eligible["existing_split"] = eligible["rid"].map(
        split_map
    ).fillna("NOT_IN_EXISTING_SPLIT")

    # MRI manifest groups are not treated as the clinical ground truth.
    group = eligible.get(
        "mri_manifest_group",
        pd.Series("", index=eligible.index)
    ).fillna("").astype(str).str.upper().str.strip()

    eligible["label_review"] = "REVIEW_MAPPING"
    eligible.loc[group.isin(["CN", "NORMAL"]), "label_review"] = (
        eligible.loc[group.isin(["CN", "NORMAL"]), "target"]
        .map(lambda x: "CONSISTENT_CN" if x == "CN" else "REVIEW_CN_DISAGREEMENT")
    )
    eligible.loc[group.isin(["AD"]), "label_review"] = (
        eligible.loc[group.isin(["AD"]), "target"]
        .map(lambda x: "CONSISTENT_AD" if x == "Dementia" else "REVIEW_AD_DISAGREEMENT")
    )
    eligible.loc[group.isin(["EMCI", "LMCI", "MCI"]), "label_review"] = (
        "REVIEW_MCI_MAPPING"
    )

    # Choose one series per participant and clinical assessment date.
    # Rank by smallest date gap, MPRAGE description, non-repeat description,
    # then number of DICOM files. This is a reproducible provisional rule,
    # not a claim that the selected image is clinically optimal.
    eligible["_mprage_rank"] = eligible[
        "has_mprage_description"
    ].astype(int)
    eligible["_repeat_rank"] = (
        ~eligible["has_repeat_description"]
    ).astype(int)

    eligible = eligible.sort_values(
        by=[
            "rid",
            "clinical_date",
            "date_gap_days",
            "_mprage_rank",
            "_repeat_rank",
            "dicom_file_count",
        ],
        ascending=[True, True, True, False, False, False],
        kind="stable",
    ).copy()

    eligible["candidates_for_participant_date"] = (
        eligible.groupby(["rid", "clinical_date"])["image_id"]
        .transform("size")
    )

    eligible["targets_for_participant_date"] = (
        eligible.groupby(["rid", "clinical_date"])["target"]
        .transform("nunique")
    )

    eligible["selection_status"] = "NOT_SELECTED_DUPLICATE"
    first_indices = eligible.groupby(
        ["rid", "clinical_date"], sort=False
    ).head(1).index
    eligible.loc[first_indices, "selection_status"] = "SELECTED_PROVISIONAL"

    eligible["duplicate_target_conflict"] = (
        eligible["targets_for_participant_date"] > 1
    )

    # Record why each candidate ranked as it did.
    eligible["selection_rule"] = (
        "Smallest date gap; then MPRAGE; then non-repeat; "
        "then most DICOM files"
    )

    # Remove internal sorting columns.
    eligible = eligible.drop(
        columns=["_mprage_rank", "_repeat_rank"]
    )

    eligible.to_csv(
        CANDIDATE_AUDIT, index=False, encoding="utf-8-sig"
    )

    shortlist = eligible[
        eligible["selection_status"] == "SELECTED_PROVISIONAL"
    ].copy()

    shortlist.to_csv(
        SHORTLIST, index=False, encoding="utf-8-sig"
    )

    split_counts = shortlist["existing_split"].value_counts(
        dropna=False
    ).to_string()
    class_counts = shortlist["target"].value_counts(
        dropna=False
    ).to_string()

    lines = [
        "M-ADNet provisional MRI visit selection",
        f"30-day eligible MRI candidate rows: {len(eligible)}",
        f"Provisional selected participant-date rows: {len(shortlist)}",
        f"Rows not selected as duplicates: {len(eligible) - len(shortlist)}",
        f"Unique selected participants: {shortlist['rid'].nunique()}",
        " ",
        "Existing participant split assignment:",
        split_counts,
        " ",
        "Selected clinical target counts:",
        class_counts,
        " ",
        f"Selected rows with repeated candidate series: "
        f"{int((shortlist['candidates_for_participant_date'] > 1).sum())}",
        f"Selected rows with target conflicts on same participant-date: "
        f"{int(shortlist['duplicate_target_conflict'].sum())}",
        " ",
        "MRI manifest label review counts:",
        shortlist["label_review"].value_counts(dropna=False).to_string(),
        " ",
        "This is a provisional shortlist, not a final model-ready dataset.",
        "Review target conflicts, scan quality, series identity, and visit alignment.",
        "Participants outside the existing clinical split need a deliberate split policy.",
    ]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nSelection audit:", CANDIDATE_AUDIT)
    print("Provisional shortlist:", SHORTLIST)
    print("Summary:", SUMMARY)


if __name__ == "__main__":
    main()
