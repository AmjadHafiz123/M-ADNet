
from pathlib import Path
from collections import Counter

ROOT = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\MRI_extracted"
)

if not ROOT.exists():
    raise FileNotFoundError(ROOT)

extensions = Counter()
dicom_candidates = 0
total_files = 0
total_bytes = 0
examples = []

print(f"Scanning: {ROOT}")
print("This may take several minutes...")

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    total_files += 1
    size = path.stat().st_size
    total_bytes += size

    suffix = path.suffix.lower()
    extensions[suffix if suffix else "<no extension>"] += 1

    if suffix in {".dcm", ".ima"}:
        dicom_candidates += 1
        if len(examples) < 10:
            examples.append(str(path))

print("\n=== MRI EXTRACTION AUDIT ===")
print(f"Total files: {total_files:,}")
print(f"Potential DICOM files (.dcm/.ima): {dicom_candidates:,}")
print(f"Total file size: {total_bytes / 1024**3:.2f} GiB")

print("\nMost common extensions:")
for ext, count in extensions.most_common(15):
    print(f"  {ext}: {count:,}")

print("\nExample image paths:")
for path in examples:
    print(f"  {path}")
