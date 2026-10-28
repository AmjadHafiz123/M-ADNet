
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.madnet_dataset import MADNetDataset


PROJECT = Path(__file__).resolve().parents[2]
DATA = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")
PROCESSED = DATA / "Actual_data" / "processed"

FEATURES = PROCESSED / "adni_mri_clinical_features.csv"
MRI_DIR = PROCESSED / "mri_full_nifti"


def main():
    train_dataset = MADNetDataset(
        feature_csv=FEATURES,
        nifti_dir=MRI_DIR,
        split="train",
    )

    # Pass training-only statistics into validation and test.
    shared_stats = {
        "clinical_means": train_dataset.clinical_means,
        "clinical_stds": train_dataset.clinical_stds,
    }

    validation_dataset = MADNetDataset(
        feature_csv=FEATURES,
        nifti_dir=MRI_DIR,
        split="validation",
        **shared_stats,
    )

    test_dataset = MADNetDataset(
        feature_csv=FEATURES,
        nifti_dir=MRI_DIR,
        split="test",
        **shared_stats,
    )

    sample = train_dataset[0]

    print("Train visits:", len(train_dataset))
    print("Validation visits:", len(validation_dataset))
    print("Test visits:", len(test_dataset))
    print("Sample MRI shape:", tuple(sample["mri"].shape))
    print("Sample clinical shape:", tuple(sample["clinical"].shape))
    print("Sample label:", sample["label"].item())
    print("Sample image ID:", sample["image_id"])
    print("Sample participant:", sample["rid"])

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))
    print("Batch MRI shape:", tuple(batch["mri"].shape))
    print("Batch clinical shape:", tuple(batch["clinical"].shape))
    print("Batch label shape:", tuple(batch["label"].shape))
    print("DATASET TEST PASSED")


if __name__ == "__main__":
    main()
