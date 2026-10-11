
from pathlib import Path
import re
import sys

import pandas as pd

BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")

MANIFEST = BASE / "Actual_data" / "ADNI_MPRAGE_search_result_7_18_2026.csv"
INVENTORY = (
    BASE / "Actual_data" / "processed"
    / "adni_dicom_series_validated.csv"
)
CLINICAL = (
    BASE / "Clinical" / "processed"
    / "adni_clinical_merged.csv"
)
OUT_DIR = BASE / "Actual_data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "adni_mri_clinical_match_candidates.csv"
SUMMARY = OUT_DIR / "adni_mri_clinical_match_summary.txt"

MATCH_WINDOW_DAYS = 30
REVIEW_WINDOW_DAYS = 90


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def find_column(df, aliases, required=True):
    columns = {key(c): c for c in df.columns}

    for alias in aliases:
        if key(alias) in columns:
            return columns[key(alias)]

    if required:
        raise ValueError(
            f"Could not find any of {aliases}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_rid(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(r"_S_(\d+)", s, flags=re.IGNORECASE)
    if match:
        return str(int(match.group(1)))

    if re.fullmatch(r"\d+", s):
        return str(int(s))

    return ""


def normalize_image_id(value):
    if pd.isna(value):
        return ""

    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]

    match = re.search(r"I(\d+)", s, flags=re.IGNORECASE)
    if match:
        return "I" + match.group(1)

    if re.fullmatch(r"\d+", s):
        return "I" + s

    return ""


def rid_from_path(value):
    if pd.isna(value):
        return ""

    match = re.search(r"_S_(\d+)", str(value), flags=re.IGNORECASE)
    return str(int(match.group(1))) if match else ""


def parse_date(value):
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return pd.NaT

    # DICOM dates commonly use YYYYMMDD.
    if re.fullmatch(r"\d{8}", s):
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def normalize_diagnosis(value):
    if pd.isna(value):
        return ""

    s = re.sub(r"[^a-z0-9]", "", str(value).lower())

    if s in {"cn", "normal", "cognitivelynormal", "normalcognition"}:
        return "CN"

    if s in {"mci", "mildcognitiveimpairment"}:
        return "MCI"

    if s in {
        "dementia", "ad", "alzheimersdisease",
        "alzheimers", "alzheimerdisease"
    }:
        return "Dementia"

    # Do not automatically treat EMCI/LMCI or other labels as MCI.
    return ""


def main():
    manifest = read_csv(MANIFEST)
    inventory = read_csv(INVENTORY)
    clinical = read_csv(CLINICAL)

    print("Input rows:")
    print("  MRI manifest:", len(manifest))
    print("  DICOM inventory:", len(inventory))
    print("  Clinical table:", len(clinical))

    # ---- MRI manifest ----
    m_id = find_column(
        manifest,
        ["Image Data ID", "Image ID", "image_id", "ImageDataID"]
    )
    m_subject = find_column(
        manifest,
        ["Subject", "Subject ID", "PTID", "Participant ID"]
    )
    m_visit = find_column(
        manifest,
        ["Visit", "VISCODE", "Visit Code"],
        required=False
    )
    m_date = find_column(
        manifest,
        ["Acq Date", "Acquisition Date", "Study Date", "AcqDate"],
        required=False
    )
    m_group = find_column(
        manifest,
        ["Group", "Research Group", "Diagnosis"],
        required=False
    )

    mri = pd.DataFrame({
        "image_id": manifest[m_id].map(normalize_image_id),
        "manifest_subject": manifest[m_subject].astype("string"),
        "manifest_visit": (
            manifest[m_visit].astype("string")
            if m_visit else ""
        ),
        "manifest_group": (
            manifest[m_group].astype("string")
            if m_group else ""
        ),
        "manifest_date": (
            manifest[m_date].map(parse_date)
            if m_date else pd.NaT
        ),
    })

    mri["rid"] = mri["manifest_subject"].map(normalize_rid)

    # ---- DICOM inventory ----
    i_id = find_column(
        inventory,
        ["image_id", "image_id_from_path", "Image Data ID"],
        required=False
    )
    i_path = find_column(
        inventory,
        ["series_folder", "Series Folder", "folder_path"],
        required=False
    )
    i_description = find_column(
        inventory,
        ["series_description", "Series Description"],
        required=False
    )
    i_count = find_column(
        inventory,
        ["file_count", "File Count", "number_of_files"],
        required=False
    )
    i_date = find_column(
        inventory,
        [
            "AcquisitionDate", "Acquisition Date",
            "SeriesDate", "Series Date",
            "StudyDate", "Study Date"
        ],
        required=False
    )

    if not i_id and not i_path:
        raise ValueError(
            "Inventory needs an image ID or series folder column."
        )

    inv = pd.DataFrame()
    inv["image_id"] = (
        inventory[i_id].map(normalize_image_id)
        if i_id else ""
    )
    inv["series_folder"] = (
        inventory[i_path].astype("string")
        if i_path else ""
    )
    inv["series_description"] = (
        inventory[i_description].astype("string")
        if i_description else ""
    )
    inv["dicom_file_count"] = (
        pd.to_numeric(inventory[i_count], errors="coerce")
        if i_count else pd.NA
    )
    inv["dicom_date"] = (
        inventory[i_date].map(parse_date)
        if i_date else pd.NaT
    )

    # Derive IDs from paths where inventory IDs are absent.
    if i_path:
        path_ids = inventory[i_path].astype("string").str.extract(
            r"[\\/](I\d+)$", expand=False
        )
        path_ids = path_ids.map(normalize_image_id)
        inv["image_id"] = inv["image_id"].where(
            inv["image_id"] != "", path_ids
        )

        path_rids = inventory[i_path].map(rid_from_path)
    else:
        path_rids = pd.Series([""] * len(inventory))

    inv["path_rid"] = path_rids

    # Avoid an ambiguous many-to-many join.
    inv = inv.drop_duplicates(subset=["image_id"], keep="first")
    mri = mri.merge(inv, on="image_id", how="left", validate="many_to_one")

    mri["rid"] = mri["rid"].where(
        mri["rid"] != "", mri["path_rid"].fillna("")
    )
    mri["mri_date"] = mri["manifest_date"].fillna(mri["dicom_date"])
    mri["inventory_found"] = mri["series_folder"].notna()

    # ---- Clinical visits ----
    c_rid = find_column(clinical, ["RID", "Research ID"])
    c_visit = find_column(
        clinical, ["VISCODE", "Visit Code"], required=False
    )
    c_date = find_column(
        clinical, ["EXAMDATE", "Exam Date", "Assessment Date"]
    )
    c_dx = find_column(
        clinical,
        ["DX", "Current Diagnosis", "Diagnosis", "DXCURREN"],
    )

    clin = pd.DataFrame({
        "rid": clinical[c_rid].map(normalize_rid),
        "clinical_visit": (
            clinical[c_visit].astype("string")
            if c_visit else ""
        ),
        "clinical_date": clinical[c_date].map(parse_date),
        "clinical_diagnosis_raw": clinical[c_dx].astype("string"),
    })

    clin["target"] = clin["clinical_diagnosis_raw"].map(
        normalize_diagnosis
    )

    # This is a matching table, not a feature table.
    # No cognitive test scores or diagnosis codes are added as predictors.
    clin = clin[
        (clin["rid"] != "")
        & clin["clinical_date"].notna()
        & (clin["target"] != "")
    ].copy()

    clin = clin.drop_duplicates(
        subset=["rid", "clinical_visit", "clinical_date", "target"]
    )

    clinical_by_rid = {
        rid: group.sort_values("clinical_date")
        for rid, group in clin.groupby("rid")
    }

    results = []

    for _, scan in mri.iterrows():
        rid = str(scan["rid"]) if pd.notna(scan["rid"]) else ""
        mri_date = scan["mri_date"]

        base = {
            "image_id": scan["image_id"],
            "rid": rid,
            "manifest_subject": scan["manifest_subject"],
            "mri_visit_label": scan["manifest_visit"],
            "mri_manifest_group": scan["manifest_group"],
            "mri_date": mri_date,
            "series_folder": scan["series_folder"],
            "series_description": scan["series_description"],
            "dicom_file_count": scan["dicom_file_count"],
            "inventory_found": bool(scan["inventory_found"]),
        }

        visits = clinical_by_rid.get(rid)

        if not rid or visits is None or visits.empty:
            results.append({
                **base,
                "clinical_visit": "",
                "clinical_date": pd.NaT,
                "clinical_diagnosis_raw": "",
                "target": "",
                "date_gap_days": pd.NA,
                "within_7_days": False,
                "within_30_days": False,
                "within_90_days": False,
                "match_status": "NO_CLINICAL_VISIT",
            })
            continue

        if pd.isna(mri_date):
            results.append({
                **base,
                "clinical_visit": "",
                "clinical_date": pd.NaT,
                "clinical_diagnosis_raw": "",
                "target": "",
                "date_gap_days": pd.NA,
                "within_7_days": False,
                "within_30_days": False,
                "within_90_days": False,
                "match_status": "NO_MRI_DATE",
            })
            continue

        dated = visits.copy()
        dated["gap"] = (
            dated["clinical_date"] - mri_date
        ).abs().dt.days

        # Select nearest eligible clinical visit by date.
        nearest = dated.sort_values(
            ["gap", "clinical_date"]
        ).iloc[0]

        gap = int(nearest["gap"])

        if gap <= 7:
            status = "MATCH_WITHIN_7_DAYS"
        elif gap <= MATCH_WINDOW_DAYS:
            status = "MATCH_WITHIN_30_DAYS"
        elif gap <= REVIEW_WINDOW_DAYS:
            status = "REVIEW_31_TO_90_DAYS"
        else:
            status = "OUTSIDE_90_DAYS"

        results.append({
            **base,
            "clinical_visit": nearest["clinical_visit"],
            "clinical_date": nearest["clinical_date"],
            "clinical_diagnosis_raw": nearest["clinical_diagnosis_raw"],
            "target": nearest["target"],
            "date_gap_days": gap,
            "within_7_days": gap <= 7,
            "within_30_days": gap <= MATCH_WINDOW_DAYS,
            "within_90_days": gap <= REVIEW_WINDOW_DAYS,
            "match_status": status,
        })

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    summary_lines = [
        "M-ADNet MRI-clinical matching candidate report",
        f"MRI manifest rows: {len(mri)}",
        f"DICOM inventory rows: {len(inventory)}",
        f"Clinical visits with recognized target: {len(clin)}",
        f"Unique MRI participants: {mri['rid'].replace('', pd.NA).nunique()}",
        f"MRI rows with inventory series: {int(result['inventory_found'].sum())}",
        f"Matches within 7 days: {int(result['within_7_days'].sum())}",
        f"Matches within 30 days: {int(result['within_30_days'].sum())}",
        f"Matches within 90 days: {int(result['within_90_days'].sum())}",
        "",
        "Match status counts:",
        result["match_status"].value_counts(dropna=False).to_string(),
        "",
        "Target counts among nearest matched visits:",
        result.loc[result["within_30_days"], "target"]
        .value_counts(dropna=False).to_string(),
        "",
        "This is a candidate table, not the final modeling dataset.",
        "Repeated scans and duplicate participant-visit matches require review.",
        "Only clinical diagnosis is used as the target; cognitive scores are not predictors here.",
    ]

    SUMMARY.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n=== MATCHING SUMMARY ===")
    print("\n".join(summary_lines))
    print("\nCandidate table:", OUTPUT)
    print("Summary file:", SUMMARY)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)
