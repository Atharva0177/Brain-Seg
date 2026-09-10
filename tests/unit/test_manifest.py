import hashlib
from pathlib import Path

from pipeline.stages.manifest import build_manifest


def test_manifest_contains_checksums_and_subject_metadata(tmp_path: Path) -> None:
    subject_dir = tmp_path / "BraTS2020_TrainingData" / "BraTS20_Training_001"
    subject_dir.mkdir(parents=True)
    volume = subject_dir / "BraTS20_Training_001_flair.nii"
    payload = b"synthetic volume"
    volume.write_bytes(payload)
    (subject_dir / "BraTS20_Training_001_seg.nii").write_bytes(b"mask")
    (tmp_path / "name_mapping.csv").write_text("subject,mapped\n", encoding="utf-8")
    (tmp_path / ".download-complete.json").write_text("{}", encoding="utf-8")

    manifest = build_manifest(tmp_path)

    assert manifest["file_count"] == 3
    assert manifest["subject_count"] == 1
    flair = next(entry for entry in manifest["files"] if entry["modality"] == "flair")
    assert flair["subject_id"] == "BraTS20_Training_001"
    assert flair["sha256"] == hashlib.sha256(payload).hexdigest()
    assert ".download-complete.json" not in str(manifest["files"])


def test_manifest_recognizes_braTS_segmentation_filename_variant(tmp_path: Path) -> None:
    subject_dir = tmp_path / "BraTS20_Training_355"
    subject_dir.mkdir()
    path = subject_dir / "W39_1998.09.19_Segm.nii"
    path.write_bytes(b"mask")

    manifest = build_manifest(tmp_path)

    assert manifest["files"][0]["subject_id"] == "BraTS20_Training_355"
    assert manifest["files"][0]["modality"] == "seg"
