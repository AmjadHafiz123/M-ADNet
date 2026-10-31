
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.madnet_dataset import MADNetDataset
from src.models.train_mri_baseline import (
    FEATURES_CSV,
    MRI_DIR,
    Small3DCNN,
)

torch.manual_seed(42)
torch.set_num_threads(4)

dataset = MADNetDataset(
    feature_csv=FEATURES_CSV,
    nifti_dir=MRI_DIR,
    split="train",
)

selected = []

for class_name in ("CN", "MCI", "Dementia"):
    indices = [
        i for i, label in enumerate(dataset.df["target"].tolist())
        if label == class_name
    ]
    selected.extend(indices[:4])

if len(selected) != 12:
    raise RuntimeError("Could not select 12 examples.")

loader = DataLoader(
    Subset(dataset, selected),
    batch_size=12,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))
mri = batch["mri"]
labels = batch["label"]

print("MRI shape:", tuple(mri.shape))
print("Labels:", labels.tolist())

model = Small3DCNN(dropout=0.0)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.0,
)
criterion = nn.CrossEntropyLoss()

for step in range(1, 301):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(mri)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    if step == 1 or step % 25 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()

        print(
            f"Step {step:03d} | "
            f"loss={loss.item():.4f} | "
            f"accuracy={accuracy:.3f}"
        )

model.eval()

with torch.no_grad():
    logits = model(mri)
    predictions = logits.argmax(dim=1)

print("\nFinal predictions:", predictions.tolist())
print("True labels:      ", labels.tolist())
print(
    "Final accuracy:",
    (predictions == labels).float().mean().item(),
)
