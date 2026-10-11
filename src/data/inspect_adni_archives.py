
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import tarfile

ROOT = Path(r"C:\Users\User\Desktop\MSC\Research\Data\ADNI")

FILES = [
    ROOT / "ADNI_MPRAGE_search_result.csv",
    ROOT / "ADNI_MPRAGE_search_result.zip",
    ROOT / "ADNI_MPRAGE_search_result_IDA_Metadata.zip",
    ROOT / "ADNIMERGE2.tar.gz",
]

for path in FILES:
    print("\n" + "=" * 75)
    print(f"FILE: {path.name}")

    if not path.exists():
        print("STATUS: NOT FOUND")
        continue

    print(f"Size: {path.stat().st_size / (1024**3):.3f} GiB")

    if path.suffix.lower() == ".zip":
        try:
            with ZipFile(path) as archive:
                members = archive.infolist()
                print(f"Archive entries: {len(members):,}")

                for item in members[:30]:
                    print(
                        f"  {item.filename} "
                        f"({item.file_size / (1024**2):.2f} MiB)"
                    )

                if len(members) > 30:
                    print(f"  ... {len(members) - 30:,} more entries")

                image_entries = [
                    item.filename for item in members
                    if item.filename.lower().endswith(
                        (".dcm", ".ima", ".nii", ".nii.gz")
                    )
                ]
                print(f"Potential image entries: {len(image_entries):,}")

        except BadZipFile:
            print("ERROR: File is not a valid ZIP archive.")

    elif path.name.lower().endswith(".tar.gz"):
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = archive.getmembers()
                print(f"Archive entries: {len(members):,}")
                for item in members[:15]:
                    print(f"  {item.name}")
        except tarfile.TarError as exc:
            print(f"ERROR: {exc}")

    elif path.suffix.lower() == ".csv":
        print("CSV metadata file; not an MRI image archive.")
