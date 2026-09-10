from pathlib import Path

import numpy as np

from pipeline.data.cache import cache_key, load_cached_subject, save_cached_subject
from pipeline.data.preprocess import CropBounds, PreprocessedSubject


def _subject() -> PreprocessedSubject:
    return PreprocessedSubject(
        subject_id="subject",
        images=np.ones((4, 2, 3, 4), dtype=np.float32),
        labels=np.zeros((2, 3, 4), dtype=np.uint8),
        crop_bounds=CropBounds((1, 2, 3), (3, 5, 7)),
        original_shape=(10, 11, 12),
        affine=np.eye(4),
        spacing=(1.0, 1.0, 1.0),
        modality_statistics={"t1": {"mean": 1.0}},
    )


def test_cache_key_changes_with_configuration() -> None:
    manifest = {"file_count": 1, "total_size_bytes": 1, "files": []}
    assert cache_key(manifest, "subject", 8) != cache_key(manifest, "subject", 16)


def test_cached_subject_round_trip(tmp_path: Path) -> None:
    subject = _subject()
    cache_path = tmp_path / "subject.npz"
    metadata_path = tmp_path / "subject.json"
    save_cached_subject(subject, cache_path, metadata_path)
    restored = load_cached_subject(cache_path, metadata_path)
    assert np.array_equal(restored.images, subject.images)
    assert np.array_equal(restored.labels, subject.labels)
    assert restored.crop_bounds == subject.crop_bounds
