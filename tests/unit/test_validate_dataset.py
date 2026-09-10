import pytest

from pipeline.stages.validate_dataset import DatasetExpectations, validate_manifest


def _manifest() -> dict[str, object]:
    files = []
    for cohort, count in (("Training", 5), ("Validation", 4)):
        for index in range(count):
            subject = f"BraTS20_{cohort}_{index:03d}"
            for modality in ("t1", "t1ce", "t2", "flair"):
                files.append(
                    {
                        "path": f"{cohort}/{subject}/{subject}_{modality}.nii",
                        "subject_id": subject,
                        "file_type": modality,
                    }
                )
            if cohort == "Training":
                files.append(
                    {
                        "path": f"{cohort}/{subject}/{subject}_seg.nii",
                        "subject_id": subject,
                        "file_type": "seg",
                    }
                )
    files.extend({"path": f"metadata-{index}.csv", "file_type": "csv"} for index in range(4))
    return {"files": files}


def test_relaxed_fixture_validation_passes() -> None:
    expectations = DatasetExpectations(
        training_subjects=5,
        validation_subjects=4,
        training_nifti_files=25,
        validation_nifti_files=16,
        csv_files=4,
    )
    result = validate_manifest(_manifest(), expectations, strict=True)
    assert result["status"] == "passed"


def test_strict_validation_rejects_wrong_composition() -> None:
    with pytest.raises(ValueError, match="training_subjects"):
        validate_manifest(_manifest(), strict=True)
