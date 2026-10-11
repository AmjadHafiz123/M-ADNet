
from pathlib import Path
from zipfile import ZipFile
from collections import Counter

archive_path = Path(
    r"C:\Users\User\Desktop\MSC\Research\Data\ADNI"
    r"\ADNI_MPRAGE_search_result.zip"
)

with ZipFile(archive_path) as z:
    files = [f for f in z.infolist() if not f.is_dir()]
    total_bytes = sum(f.file_size for f in files)

    print(f"Files: {len(files):,}")
    print(f"Compressed archive: {archive_path.stat().st_size / 1024**3:.2f} GiB")
    print(f"Uncompressed contents: {total_bytes / 1024**3:.2f} GiB")

    folders = Counter(
        f.filename.split("/")[1]
        for f in files
        if f.filename.startswith("ADNI/")
        and len(f.filename.split("/")) > 2
    )
    print(f"Participant folders: {len(folders):,}")
