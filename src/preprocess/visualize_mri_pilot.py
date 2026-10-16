
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
)
PROCESSED = BASE / "Actual_data" / "processed"

AUDIT = PROCESSED / "adni_mri_pilot_preprocessing_audit.csv"
OUTPUT_DIR = PROCESSED / "mri_pilot_qc_images"


def main():
    if not AUDIT.is_file():
        raise FileNotFoundError(AUDIT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT, low_memory=False)
    passed = audit[audit["status"] == "PASS"]

    results = []

    for _, row in passed.iterrows():
        path = Path(row["output_file"])
        rid = row["rid"]
        target = row["target"]

        try:
            if not path.is_file():
                raise FileNotFoundError(path)

            image = sitk.ReadImage(str(path))
            if image.GetDimension() != 3:
                raise ValueError("Expected a 3D image")

            # Reorient using physical image geometry.
            image = sitk.DICOMOrient(image, "LPS")
            volume = sitk.GetArrayFromImage(image)

            if not np.isfinite(volume).all():
                raise ValueError("Non-finite voxel values")

            # After LPS reorientation, array axes are Z, Y, X.
            # These are central slices in the three anatomical planes.
            z, y, x = (n // 2 for n in volume.shape)

            views = [
                (volume[z, :, :], "Axial"),
                (volume[:, y, :], "Coronal"),
                (volume[:, :, x], "Sagittal"),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            for ax, (slice_2d, label) in zip(axes, views):
                ax.imshow(
                    np.rot90(slice_2d),
                    cmap="gray",
                    vmin=-2,
                    vmax=2,
                    interpolation="nearest",
                )
                ax.set_title(label)
                ax.axis("off")

            fig.suptitle(
                f"RID {rid} | {target} | LPS reoriented",
                fontsize=11,
            )
            fig.tight_layout()

            output = OUTPUT_DIR / f"sub-{rid}_physical_qc.png"
            fig.savefig(output, dpi=140, bbox_inches="tight")
            plt.close(fig)

            results.append({
                "rid": rid,
                "target": target,
                "status": "PASS",
                "direction": str(image.GetDirection()),
                "spacing": str(image.GetSpacing()),
                "shape_zyx": str(volume.shape),
                "image": str(output),
                "error": "",
            })
            print("Saved:", output)

        except Exception as exc:
            results.append({
                "rid": rid,
                "target": target,
                "status": "FAILED",
                "direction": "",
                "spacing": "",
                "shape_zyx": "",
                "image": "",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"FAILED RID={rid}: {exc}")

    report = OUTPUT_DIR / "physical_orientation_audit.csv"
    pd.DataFrame(results).to_csv(report, index=False)

    print("\nVisualization status:")
    print(pd.DataFrame(results)["status"].value_counts().to_string())
    print("Images:", OUTPUT_DIR)
    print("Audit:", report)


if __name__ == "__main__":
    main()
