import numpy as np
import pytest

from pipeline.data.subjects import composite_regions, validate_labels


def test_validate_labels_accepts_brats_labels() -> None:
    labels = validate_labels(np.array([[0, 1, 2, 4]], dtype=np.uint8))
    assert labels.dtype == np.uint8


def test_validate_labels_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unexpected labels"):
        validate_labels(np.array([0, 3], dtype=np.uint8), subject_id="subject")


def test_composite_regions_follow_brats_definitions() -> None:
    labels = np.array([0, 1, 2, 4], dtype=np.uint8)
    regions = composite_regions(labels)
    assert regions["whole_tumor"].tolist() == [False, True, True, True]
    assert regions["tumor_core"].tolist() == [False, True, False, True]
    assert regions["enhancing_tumor"].tolist() == [False, False, False, True]
