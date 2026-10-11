import tarfile
from pathlib import Path
import pyreadr

archive = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\ADNIMERGE2.tar.gz")
output = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\Clinical\extracted")

targets = {"DXSUM.rda", "PTDEMOG.rda", "MMSE.rda", "ADAS.rda", "CDR.rda"}

with tarfile.open(archive, "r:gz") as tar:
    for member in tar.getmembers():
        if Path(member.name).name in targets:
            source = tar.extractfile(member)
            if source is None:
                continue

            destination = output / Path(member.name).name
            destination.write_bytes(source.read())
            print(f"Extracted: {destination.name}")

print("\n" + "=" * 70)
print("DATASET INSPECTION")
print("=" * 70)

for filename in sorted(targets):
    filepath = output / filename
    if not filepath.exists():
        print(f"\nMISSING: {filename}")
        continue

    print(f"\n{'=' * 60}")
    print(f"FILE: {filename}")

    try:
        result = pyreadr.read_r(str(filepath))

        for object_name, df in result.items():
            print(f"\nR object: {object_name}")
            print(f"Shape: {df.shape}")
            print("Columns:")
            print(list(df.columns))
            print("\nFirst 3 rows:")
            print(df.head(3).to_string(index=False))

    except Exception as exc:
        print(f"ERROR: {exc}")
