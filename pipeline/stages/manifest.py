"""Build deterministic, checksum-verified manifests for raw BraTS files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

SUPPORTED_SUFFIXES = {".nii", ".gz", ".csv"}
MODALITIES = {"t1", "t1ce", "t2", "flair", "seg"}
SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+?)_(?P<modality>t1ce|t1|t2|flair|seg)\.nii$")
SEGMENTATION_VARIANT_PATTERN = re.compile(r"^(?P<subject>.+?)_Segm\.nii$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(path: Path) -> tuple[str | None, str | None]:
    if path.suffix.lower() == ".csv":
        return None, "csv"
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    match = SUBJECT_PATTERN.match(name)
    if match:
        return match.group("subject"), match.group("modality")
    variant = SEGMENTATION_VARIANT_PATTERN.match(name)
    if variant:
        # Some BraTS 2020 masks use a patient-style filename inside the
        # canonical subject directory; the directory is the reliable ID.
        return path.parent.name, "seg"
    return None, "nii"


def build_manifest(root: Path) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {root}")

    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".download-complete.json":
            continue
        if not (
            path.name.endswith(".nii") or path.name.endswith(".nii.gz") or path.suffix == ".csv"
        ):
            continue
        subject, file_type = _classify(path)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "subject_id": subject,
                "modality": file_type if file_type in MODALITIES else None,
                "file_type": file_type,
            }
        )

    subjects = sorted({entry["subject_id"] for entry in files if entry["subject_id"]})
    by_type: dict[str, int] = {}
    for entry in files:
        file_type = str(entry["file_type"])
        by_type[file_type] = by_type.get(file_type, 0) + 1

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "file_count": len(files),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in files),
        "subject_count": len(subjects),
        "subjects": subjects,
        "counts_by_type": dict(sorted(by_type.items())),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in manifest if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
