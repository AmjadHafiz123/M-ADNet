
from pathlib import Path
from collections import Counter

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Actual_data"
)

SKIP_DIRS = {"processed", "__pycache__"}

if not ROOT.exists():
    raise FileNotFoundError(f"Folder not found: {ROOT}")

extension_counts = Counter()
examples = Counter()
top_level = []
image_count = 0
total_bytes = 0

print("MRI DATA DIRECTORY")
print("=" * 65)
print(f"Root: {ROOT}")

for item in ROOT.iterdir():
    if item.name not in SKIP_DIRS:
        kind = "DIR " if item.is_dir() else "FILE"
        top_level.append((kind, item.name))

print("\nTop-level contents:")
for kind, name in sorted(top_level):
    print(f"  {kind}  {name}")

print("\nScanning files recursively...")

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if any(part in SKIP_DIRS for part in path.parts):
        continue

    suffix = path.suffix.lower()

    if path.name.lower().endswith(".nii.gz"):
        kind = ".nii.gz"
    elif suffix:
        kind = suffix
    else:
        kind = "<no extension>"

    extension_counts[kind] += 1
    total_bytes += path.stat().st_size

    if kind in {".dcm", ".ima", ".nii", ".nii.gz"}:
        image_count += 1
        if examples[kind] < 5:
            print(f"  Image example: {path}")
            examples[kind] += 1

print("\nFile counts by extension:")
for ext, count in extension_counts.most_common(30):
    print(f"  {ext}: {count:,}")

print(f"\nPotential image files: {image_count:,}")
print(f"Total scanned file size: {total_bytes / (1024**3):.2f} GiB")
print("\nNote: some DICOM files have no extension.")
