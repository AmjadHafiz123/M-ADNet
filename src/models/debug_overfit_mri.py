
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_mri_baseline import Small3DCNN


BASE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)
FEATURES = BASE / "adni_mri_clinical_features.csv"
MRI_DIR = BASE / "mri_full_nifti"
OUTPUT = BASE / "mri_baseline" / "overfit_debug.csv"

SEED = 42
SAMPLES_PER_CLASS = 4
STEPS = 100
LEARNING_RATE = 1e-3


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    dataset = MADNetDataset(FEATURES, MRI_DIR, split="train")

    # Select a balanced subset from training data only.
    chosen = []
    for label in ["CN", "MCI", "Dementia"]:
        candidates = dataset.df.index[
            dataset.df["target"] == label
        ].tolist()

        if len(candidates) < SAMPLES_PER_CLASS:
            raise RuntimeError(f"Not enough training examples for {label}")

        chosen.extend(candidates[:SAMPLES_PER_CLASS])

    subset = Subset(dataset, chosen)
    loader = DataLoader(subset, batch_size=2, shuffle=True, num_workers=0)

    model = Small3DCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    print("Selected training examples:", len(subset))
    print("Labels:", dataset.df.iloc[chosen]["target"].value_counts().to_dict())
    print("Starting overfit diagnostic...")

    model.train()

    for step in range(1, STEPS + 1):
        losses = []
        correct = 0
        total = 0

        for batch in loader:
            x = batch["mri"]
            y = batch["label"]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()

        mean_loss = float(np.mean(losses))
        accuracy = correct / total

        history.append({
            "step": step,
            "loss": mean_loss,
            "accuracy": accuracy,
        })

        if step == 1 or step % 10 == 0:
            print(
                f"Step {step:03d}/{STEPS} | "
                f"loss={mean_loss:.4f} | accuracy={accuracy:.3f}"
            )

    model.eval()
    final_rows = []

    with torch.no_grad():
        for idx in chosen:
            sample = dataset[idx]
            probs = torch.softmax(
                model(sample["mri"].unsqueeze(0)), dim=1
            )[0]

            final_rows.append({
                "image_id": sample["image_id"],
                "true_label": ["CN", "MCI", "Dementia"][
                    sample["label"].item()
                ],
                "predicted_label": ["CN", "MCI", "Dementia"][
                    probs.argmax().item()
                ],
                "p_CN": probs[0].item(),
                "p_MCI": probs[1].item(),
                "p_Dementia": probs[2].item(),
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUTPUT, index=False)
    pd.DataFrame(final_rows).to_csv(
        OUTPUT.with_name("overfit_debug_predictions.csv"),
        index=False,
    )

    print("\nFinal predictions on selected training examples:")
    print(pd.DataFrame(final_rows).to_string(index=False))
    print("\nSaved:", OUTPUT)
    print("Saved:", OUTPUT.with_name("overfit_debug_predictions.csv"))


if __name__ == "__main__":
    main()
