"""Foreground-biased 3D patch sampling for cached subjects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CenterType = Literal["foreground", "random"]


@dataclass(frozen=True)
class PatchSample:
    image: np.ndarray
    label: np.ndarray
    start: tuple[int, int, int]
    center_type: CenterType
    tumor_fraction: float


def _pad_to_patch(
    images: np.ndarray, labels: np.ndarray, patch_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    spatial_shape = labels.shape
    padding = tuple((0, max(0, size - current)) for current, size in zip(spatial_shape, patch_size))
    if any(after for _, after in padding):
        images = np.pad(images, ((0, 0), *padding), mode="constant")
        labels = np.pad(labels, padding, mode="constant")
    return images, labels, padding


def _random_start(
    shape: tuple[int, int, int], patch_size: tuple[int, int, int], rng: np.random.Generator
) -> tuple[int, int, int]:
    return tuple(
        int(rng.integers(0, max(1, current - size + 1)))
        for current, size in zip(shape, patch_size)
    )


def _centered_start(
    center: tuple[int, int, int], shape: tuple[int, int, int], patch_size: tuple[int, int, int]
) -> tuple[int, int, int]:
    starts = []
    for coordinate, current, size in zip(center, shape, patch_size):
        start = coordinate - size // 2
        starts.append(min(max(start, 0), max(0, current - size)))
    return tuple(starts)


def sample_patch(
    images: np.ndarray,
    labels: np.ndarray,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    foreground_probability: float = 2 / 3,
    rng: np.random.Generator | None = None,
) -> PatchSample:
    if images.ndim != 4 or labels.ndim != 3 or images.shape[1:] != labels.shape:
        raise ValueError("Expected images (channels, x, y, z) aligned with labels (x, y, z).")
    if any(size <= 0 for size in patch_size):
        raise ValueError("Patch dimensions must be positive.")
    if not 0 <= foreground_probability <= 1:
        raise ValueError("foreground_probability must be between 0 and 1.")

    generator = rng or np.random.default_rng()
    padded_images, padded_labels, _ = _pad_to_patch(images, labels, patch_size)
    shape = padded_labels.shape
    foreground_coordinates = np.argwhere(padded_labels > 0)
    use_foreground = bool(foreground_coordinates.size) and generator.random() < foreground_probability

    if use_foreground:
        center = tuple(int(value) for value in foreground_coordinates[generator.integers(len(foreground_coordinates))])
        start = _centered_start(center, shape, patch_size)
        center_type: CenterType = "foreground"
    else:
        start = _random_start(shape, patch_size, generator)
        center_type = "random"

    slices = tuple(slice(begin, begin + size) for begin, size in zip(start, patch_size))
    image_patch = padded_images[(slice(None), *slices)]
    label_patch = padded_labels[slices]
    return PatchSample(
        image=image_patch,
        label=label_patch,
        start=start,
        center_type=center_type,
        tumor_fraction=float(np.count_nonzero(label_patch) / label_patch.size),
    )


def sample_patches(
    images: np.ndarray,
    labels: np.ndarray,
    count: int,
    patch_size: tuple[int, int, int] = (128, 128, 128),
    foreground_probability: float = 2 / 3,
    seed: int = 42,
) -> list[PatchSample]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    rng = np.random.default_rng(seed)
    return [
        sample_patch(images, labels, patch_size, foreground_probability, rng)
        for _ in range(count)
    ]
