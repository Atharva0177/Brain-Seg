from pipeline.data.splits import SplitConfig, create_subject_splits
from pipeline.data.validate_splits import validate_splits


def _manifest() -> dict[str, object]:
    files = []
    for index in range(369):
        subject = f"BraTS20_Training_{index:03d}"
        files.append(
            {
                "path": f"Training/{subject}/{subject}_seg.nii",
                "subject_id": subject,
                "file_type": "seg",
                "sha256": f"{index:064d}",
            }
        )
    return {"file_count": 369, "total_size_bytes": 369, "files": files}


def test_valid_split_manifest_passes_validation() -> None:
    manifest = _manifest()
    split_manifest = create_subject_splits(manifest, SplitConfig(seed=42))
    result = validate_splits(manifest, split_manifest)
    assert result["status"] == "passed"


def test_duplicate_subject_fails_validation() -> None:
    manifest = _manifest()
    split_manifest = create_subject_splits(manifest)
    duplicate = split_manifest["splits"]["train"][0]
    split_manifest["splits"]["test"].append(duplicate)
    result = validate_splits(manifest, split_manifest)
    assert result["status"] == "failed"
    assert "train_test_disjoint" in result["failures"]
