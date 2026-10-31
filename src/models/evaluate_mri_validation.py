
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader

from src.models.train_mri_baseline import (
    OUTPUT_DIR,
    Small3DCNN,
    make_datasets,
    CLASS_NAMES,
)

def main():
    torch.set_num_threads(4)

    train_ds, val_ds, _ = make_datasets()

    loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    checkpoint_path = OUTPUT_DIR / "best_mri_baseline.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = Small3DCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []
    image_ids = []
    rids = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["mri"])
            probs = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            y_true.extend(batch["label"].tolist())
            y_pred.extend(predictions.tolist())
            probabilities.extend(probs.tolist())
            image_ids.extend(batch["image_id"])
            rids.extend(batch["rid"])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    results = {
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_macro_f1": f1_score(
            y_true, y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "classification_report": report,
        "true_class_counts": {
            name: int(sum(v == i for v in y_true))
            for i, name in enumerate(CLASS_NAMES)
        },
        "predicted_class_counts": {
            name: int(sum(v == i for v in y_pred))
            for i, name in enumerate(CLASS_NAMES)
        },
    }

    output_dir = OUTPUT_DIR / "validation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in CLASS_NAMES],
        columns=[f"pred_{n}" for n in CLASS_NAMES],
    ).to_csv(output_dir / "validation_confusion_matrix.csv")

    prediction_df = pd.DataFrame({
        "image_id": image_ids,
        "rid": rids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_CN": np.array(probabilities)[:, 0],
        "prob_MCI": np.array(probabilities)[:, 1],
        "prob_Dementia": np.array(probabilities)[:, 2],
    })

    prediction_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation accuracy:", results["accuracy"])
    print("Validation macro-F1:", results["validation_macro_f1"])
    print("Balanced accuracy:", results["balanced_accuracy"])

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(cm)

    print("\nTrue class counts:", results["true_class_counts"])
    print("Predicted class counts:", results["predicted_class_counts"])

    print("\nSaved validation analysis to:", output_dir)


if __name__ == "__main__":
    main()
