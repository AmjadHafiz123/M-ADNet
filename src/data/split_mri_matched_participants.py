
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = BASE / "Actual_data" / "processed"

INPUT = PROCESSED / "adni_mri_visit_shortlist.csv"
ASSIGNMENTS = PROCESSED / "adni_mri_participant_split_assignments.csv"
SUMMARY = PROCESSED / "adni_mri_split_summary.txt"

OUTPUTS = {
    "train": PROCESSED / "adni_mri_clinical_train.csv",
    "validation": PROCESSED / "adni_mri_clinical_validation.csv",
    "test": PROCESSED / "adni_mri_clinical_test.csv",
}

SEED = 42
CLASS_ORDER = ["CN", "MCI", "Dementia"]


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, low_memory=False)

    required = ["rid", "clinical_date", "target", "image_id", "series_folder"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["rid"] = df["rid"].astype("string").str.strip()
    df["target"] = df["target"].astype("string").str.strip()
    df["clinical_date"] = pd.to_datetime(
        df["clinical_date"], errors="coerce"
    )

    df = df[
        df["rid"].notna()
        & df["rid"].ne("")
        & df["target"].isin(CLASS_ORDER)
        & df["clinical_date"].notna()
        & df["series_folder"].notna()
        & df["series_folder"].astype(str).str.strip().ne("")
    ].copy()

    if df.empty:
        raise RuntimeError("No eligible rows remain.")

    # One participant-level stratum, based on the most frequent
    # diagnosis across their selected visits. This is used ONLY
    # to balance participant assignment, not to replace visit labels.
    participant_rows = []

    for rid, group in df.groupby("rid", sort=True):
        counts = group["target"].value_counts()
        highest = counts.max()
        tied = set(counts[counts == highest].index)
        dominant = next(c for c in CLASS_ORDER if c in tied)

        participant_rows.append({
            "rid": rid,
            "stratification_label": dominant,
            "visit_count": len(group),
        })

    participants = pd.DataFrame(participant_rows)

    class_counts = participants["stratification_label"].value_counts()
    for label in CLASS_ORDER:
        if class_counts.get(label, 0) < 3:
            raise RuntimeError(
                f"Too few participants in stratum {label} "
                "for reliable train/validation/test stratification."
            )

    # Split participants first, not individual visits.
    train_ids, temp_ids = train_test_split(
        participants["rid"].to_numpy(),
        test_size=0.30,
        random_state=SEED,
        stratify=participants["stratification_label"],
    )

    temp = participants[participants["rid"].isin(temp_ids)].copy()

    val_ids, test_ids = train_test_split(
        temp["rid"].to_numpy(),
        test_size=0.50,
        random_state=SEED + 1,
        stratify=temp["stratification_label"],
    )

    split_map = {}
    for rid in train_ids:
        split_map[rid] = "train"
    for rid in val_ids:
        split_map[rid] = "validation"
    for rid in test_ids:
        split_map[rid] = "test"

    participants["split"] = participants["rid"].map(split_map)
    df["split"] = df["rid"].map(split_map)

    if df["split"].isna().any():
        raise RuntimeError("Some visits were not assigned a split.")

    # Hard checks: no participant can occur in more than one split.
    id_sets = {
        name: set(df.loc[df["split"] == name, "rid"])
        for name in OUTPUTS
    }

    assert id_sets["train"].isdisjoint(id_sets["validation"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["validation"].isdisjoint(id_sets["test"])
    assert set().union(*id_sets.values()) == set(df["rid"])

    participants.to_csv(ASSIGNMENTS, index=False, encoding="utf-8-sig")

    for split_name, path in OUTPUTS.items():
        subset = df[df["split"] == split_name].copy()
        subset.to_csv(path, index=False, encoding="utf-8-sig")

    lines = [
        "M-ADNet participant-level MRI/clinical split audit",
        f"Eligible selected visit records: {len(df)}",
        f"Unique participants: {df['rid'].nunique()}",
        f"Random seed: {SEED}",
        "Target split proportions: 70% / 15% / 15%",
        "",
        "Split sizes:",
    ]

    for split_name in OUTPUTS:
        subset = df[df["split"] == split_name]
        lines.extend([
            f"{split_name}: {len(subset)} visit records, "
            f"{subset['rid'].nunique()} participants",
            "Visit-level diagnosis counts:",
            subset["target"].value_counts().reindex(
                CLASS_ORDER, fill_value=0
            ).to_string(),
            "",
        ])

    lines.extend([
        "Participant-level stratification labels:",
        participants["stratification_label"].value_counts()
        .reindex(CLASS_ORDER, fill_value=0).to_string(),
        "",
        "Participant overlap checks: PASS",
        "All eligible participants assigned: PASS",
        "Note: diagnosis may change between visits. Visit-level targets remain unchanged.",
        "Note: these splits are provisional until scan selection and quality control are finalized.",
    ])

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nParticipant assignments:", ASSIGNMENTS)
    print("Summary:", SUMMARY)
    for path in OUTPUTS.values():
        print("Split file:", path)


if __name__ == "__main__":
    main()
