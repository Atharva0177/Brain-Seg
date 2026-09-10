from pipeline.data.splits import SplitConfig, create_subject_splits, labeled_training_subjects


def _manifest(subject_count: int = 20) -> dict[str, object]:
    files = []
    for index in range(subject_count):
        subject = f"BraTS20_Training_{index:03d}"
        files.append(
            {
                "path": f"Training/{subject}/{subject}_seg.nii",
                "subject_id": subject,
                "file_type": "seg",
                "sha256": f"{index:064d}",
            }
        )
    return {"file_count": subject_count, "total_size_bytes": subject_count, "files": files}


def test_labeled_subjects_excludes_unlabeled_validation_cohort() -> None:
    manifest = _manifest()
    manifest["files"].append(
        {
            "path": "Validation/BraTS20_Validation_001/BraTS20_Validation_001_seg.nii",
            "subject_id": "BraTS20_Validation_001",
            "file_type": "seg",
        }
    )
    assert len(labeled_training_subjects(manifest)) == 20


def test_split_requires_production_labeled_subject_count() -> None:
    try:
        create_subject_splits(_manifest())
    except ValueError as error:
        assert "Expected 369" in str(error)
    else:
        raise AssertionError("Expected production subject-count validation to fail")


def test_split_config_requires_unit_sum() -> None:
    try:
        create_subject_splits(_manifest(), SplitConfig(train_fraction=0.8))
    except ValueError as error:
        assert "sum to 1.0" in str(error)
    else:
        raise AssertionError("Expected invalid fraction validation to fail")
