
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_mri_baseline import Small3DCNN


BASE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)
FEATURES = BASE / "adni_mri_clinical_features.csv"
MRI_DIR = BASE / "mri_full_nifti"
OUTPUT_DIR = BASE / "mri_baseline"

STEPS = 200
LEARNING_RATE = 1e-4
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)

    dataset = MADNetDataset(FEATURES, MRI_DIR, split="train")

    # Select four examples from each class, training split only.
    selected = []
    for label in ["CN", "MCI", "Dementia"]:
        candidates = dataset.df.index[
            dataset.df["target"] == label
        ].tolist()

        if len(candidates) < 4:
            raise RuntimeError(f"Not enough training examples for {label}")

        selected.extend(candidates[:4])

    samples = [dataset[i] for i in selected]
    x = torch.stack([s["mri"] for s in samples])
    y = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    print("MRI tensor shape:", tuple(x.shape))
    print("Labels:", y.tolist())
    print("Label counts:", torch.bincount(y, minlength=3).tolist())
    print("Input finite:", bool(torch.isfinite(x).all()))
    print("Input min/max:", x.min().item(), x.max().item())
    print("Input std:", x.std().item())

    model = Small3DCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    history = []
    model.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_logits = model(x)
                eval_loss = criterion(eval_logits, y).item()
                accuracy = (
                    eval_logits.argmax(dim=1) == y
                ).float().mean().item()

            model.train()
            history.append({
                "step": step,
                "loss": eval_loss,
                "accuracy": accuracy,
            })
            print(
                f"Step {step:03d}/{STEPS} | "
                f"loss={eval_loss:.4f} | accuracy={accuracy:.3f}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_path = OUTPUT_DIR / "fixed_batch_overfit.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
        predictions = probabilities.argmax(dim=1)

    label_names = ["CN", "MCI", "Dementia"]
    rows = []

    for i, sample in enumerate(samples):
        rows.append({
            "image_id": sample["image_id"],
            "true_label": label_names[y[i].item()],
            "predicted_label": label_names[predictions[i].item()],
            "p_CN": probabilities[i, 0].item(),
            "p_MCI": probabilities[i, 1].item(),
            "p_Dementia": probabilities[i, 2].item(),
        })

    prediction_path = OUTPUT_DIR / "fixed_batch_predictions.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)

    print("\nFinal accuracy:", (predictions == y).float().mean().item())
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSaved:", history_path)
    print("Saved:", prediction_path)


if __name__ == "__main__":
    main()
