
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_mri_baseline import Small3DCNN

BASE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)
FEATURES = BASE / "adni_mri_clinical_features.csv"
MRI_DIR = BASE / "mri_full_nifti"
OUTPUT_DIR = BASE / "mri_baseline"


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)

    dataset = MADNetDataset(FEATURES, MRI_DIR, split="train")

    selected = []
    for label in ["CN", "MCI", "Dementia"]:
        indices = dataset.df.index[
            dataset.df["target"] == label
        ].tolist()
        if len(indices) < 4:
            raise RuntimeError(f"Insufficient samples for {label}")
        selected.extend(indices[:4])

    samples = [dataset[i] for i in selected]
    x = torch.stack([sample["mri"] for sample in samples])
    y = torch.tensor(
        [sample["label"] for sample in samples], dtype=torch.long
    )

    # Extract the 24-dimensional representation from the CNN.
    model = Small3DCNN()
    model.eval()

    with torch.no_grad():
        x_small = F.interpolate(
            x,
            size=(64, 80, 80),
            mode="trilinear",
            align_corners=False,
        )
        features = model.features(x_small).flatten(1)

    print("Feature shape:", tuple(features.shape))
    print("Features finite:", bool(torch.isfinite(features).all()))
    print("Feature standard deviation:", features.std().item())

    # Train a linear classifier on fixed, randomly initialized features.
    classifier = nn.Linear(features.shape[1], 3)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=0.01, weight_decay=0
    )
    criterion = nn.CrossEntropyLoss()

    for step in range(1, 501):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(features)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                current_logits = classifier(features)
                current_loss = criterion(current_logits, y).item()
                accuracy = (
                    current_logits.argmax(dim=1) == y
                ).float().mean().item()

            print(
                f"Step {step:03d}/500 | "
                f"loss={current_loss:.4f} | accuracy={accuracy:.3f}"
            )

    with torch.no_grad():
        probabilities = torch.softmax(classifier(features), dim=1)
        predictions = probabilities.argmax(dim=1)

    names = ["CN", "MCI", "Dementia"]
    rows = []

    for i, sample in enumerate(samples):
        rows.append({
            "image_id": sample["image_id"],
            "true_label": names[y[i].item()],
            "predicted_label": names[predictions[i].item()],
            "p_CN": probabilities[i, 0].item(),
            "p_MCI": probabilities[i, 1].item(),
            "p_Dementia": probabilities[i, 2].item(),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "frozen_feature_predictions.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    print(
        "\nFinal accuracy:",
        (predictions == y).float().mean().item()
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
