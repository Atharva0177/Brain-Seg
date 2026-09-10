import numpy as np

from pipeline.data.preprocess import calculate_crop_bounds, normalize_and_crop
from pipeline.data.subjects import LoadedSubject


def _subject() -> LoadedSubject:
    images = np.zeros((4, 10, 12, 14), dtype=np.float32)
    images[:, 3:7, 4:9, 5:11] = np.arange(4 * 4 * 5 * 6, dtype=np.float32).reshape(4, 4, 5, 6)
    labels = np.zeros((10, 12, 14), dtype=np.uint8)
    labels[4:6, 5:8, 7:9] = 4
    return LoadedSubject("subject", images, labels, np.eye(4), (1.0, 1.0, 1.0))


def test_crop_bounds_clamp_to_volume() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 1, 3] = True
    bounds = calculate_crop_bounds(mask, margin=2)
    assert bounds.starts == (0, 0, 1)
    assert bounds.stops == (3, 4, 4)


def test_normalize_and_crop_preserves_alignment_and_foreground_statistics() -> None:
    result = normalize_and_crop(_subject(), margin=1)
    assert result.images.shape[1:] == result.labels.shape
    assert result.images.shape[1:] == (6, 7, 8)
    assert np.allclose(
        result.images[:, np.any(result.images != 0, axis=0)].mean(axis=1), 0, atol=1e-5
    )
