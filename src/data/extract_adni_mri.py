
from pathlib import Path
from zipfile import ZipFile

archive = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\ADNI_MPRAGE_search_result.zip"
)

destination = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI\MRI_extracted"
)

if not archive.exists():
    raise FileNotFoundError(archive)

if destination.exists() and any(destination.iterdir()):
    raise RuntimeError(
        f"Destination is not empty; refusing to overwrite: {destination}"
    )

destination.mkdir(parents=True, exist_ok=True)

with ZipFile(archive) as z:
    bad_file = z.testzip()
    if bad_file is not None:
        raise RuntimeError(f"Corrupt ZIP member: {bad_file}")

    z.extractall(destination)

print(f"Extraction complete: {destination}")
