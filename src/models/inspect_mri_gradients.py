
from pathlib import Path

import torch
import torch.nn as nn

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_mri_baseline import Small3DCNN

BASE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data\processed"
)
FEATURES = BASE / "adni_mri_clinical_features.csv"
MRI_DIR = BASE / "mri_full_nifti"

def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    dataset = MADNetDataset(FEATURES, MRI_DIR, split="train")
    selected = []

    for label in ["CN", "MCI", "Dementia"]:
        indices = dataset.df.index[
            dataset.df["target"] == label
        ].tolist()
        selected.extend(indices[:4])

    samples = [dataset[i] for i in selected]
    x = torch.stack([s["mri"] for s in samples])
    y = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    model = Small3DCNN()
    criterion = nn.CrossEntropyLoss()

    model.train()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print("Logits shape:", tuple(logits.shape))
    print("Logits finite:", bool(torch.isfinite(logits).all()))

    print("\nPer-layer gradient diagnostics:")
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            print(f"{name:40s} grad=None")
            continue

        grad = parameter.grad
        print(
            f"{name:40s} "
            f"grad_norm={grad.norm().item():.8e} "
            f"grad_finite={bool(torch.isfinite(grad).all())}"
        )

    print("\nInitial probabilities by sample:")
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)

    for sample, probability, target in zip(samples, probs, y):
        print(
            f"{sample['image_id']:>10s} "
            f"target={target.item()} "
            f"probs={probability.tolist()}"
        )

if __name__ == "__main__":
    main()
