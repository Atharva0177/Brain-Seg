import numpy as np
import pytest

from pipeline.data.patches import sample_patch, sample_patches


def test_patch_sampler_returns_aligned_patch_with_foreground_center() -> None:
    images = np.ones((4, 10, 10, 10), dtype=np.float32)
    labels = np.zeros((10, 10, 10), dtype=np.uint8)
    labels[5, 5, 5] = 4
    sample = sample_patch(
        images,
        labels,
        patch_size=(4, 4, 4),
        foreground_probability=1.0,
        rng=np.random.default_rng(1),
    )
    assert sample.image.shape == (4, 4, 4, 4)
    assert sample.label.shape == (4, 4, 4)
    assert sample.center_type == "foreground"
    assert sample.tumor_fraction > 0


def test_patch_sampler_pads_small_volume() -> None:
    images = np.ones((4, 3, 4, 5), dtype=np.float32)
    labels = np.zeros((3, 4, 5), dtype=np.uint8)
    sample = sample_patch(images, labels, patch_size=(8, 8, 8), rng=np.random.default_rng(1))
    assert sample.image.shape == (4, 8, 8, 8)
    assert sample.label.shape == (8, 8, 8)


def test_patch_sampler_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="aligned"):
        sample_patch(np.zeros((4, 3, 3, 3)), np.zeros((3, 3, 2)))


def test_patch_sampling_is_reproducible() -> None:
    images = np.ones((4, 16, 16, 16), dtype=np.float32)
    labels = np.zeros((16, 16, 16), dtype=np.uint8)
    labels[4:8, 4:8, 4:8] = 1
    first = sample_patches(images, labels, 10, patch_size=(8, 8, 8), seed=42)
    second = sample_patches(images, labels, 10, patch_size=(8, 8, 8), seed=42)
    assert [sample.start for sample in first] == [sample.start for sample in second]
