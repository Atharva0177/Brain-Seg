"""Memory-bounded normalization and foreground cropping for loaded subjects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .subjects import LoadedSubject


@dataclass(frozen=True)
class CropBounds:
    starts: tuple[int, int, int]
    stops: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(stop - start for start, stop in zip(self.starts, self.stops))


@dataclass
class PreprocessedSubject:
    subject_id: str
    images: np.ndarray
    labels: np.ndarray
    crop_bounds: CropBounds
    original_shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]
    modality_statistics: dict[str, dict[str, float]]


def foreground_mask(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (channels, x, y, z), got {images.shape}")
    return np.any(np.abs(images) > 0, axis=0)


def calculate_crop_bounds(mask: np.ndarray, margin: int = 8) -> CropBounds:
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("Cannot crop an empty foreground mask.")
    coordinates = np.where(mask)
    starts = tuple(max(0, int(values.min()) - margin) for values in coordinates)
    stops = tuple(
        min(mask.shape[axis], int(values.max()) + margin + 1)
        for axis, values in enumerate(coordinates)
    )
    return CropBounds(starts=starts, stops=stops)


def _crop(array: np.ndarray, bounds: CropBounds) -> np.ndarray:
    slices = tuple(slice(start, stop) for start, stop in zip(bounds.starts, bounds.stops))
    if array.ndim == 4:
        return array[(slice(None), *slices)]
    return array[slices]


def normalize_and_crop(subject: LoadedSubject, margin: int = 8) -> PreprocessedSubject:
    images = np.asarray(subject.images, dtype=np.float32)
    labels = np.asarray(subject.labels)
    mask = foreground_mask(images)
    statistics: dict[str, dict[str, float]] = {}
    normalized = np.empty_like(images)

    for index, modality in enumerate(("t1", "t1ce", "t2", "flair")):
        values = images[index][mask]
        mean = float(values.mean())
        standard_deviation = float(values.std())
        scale = standard_deviation if standard_deviation > 1e-6 else 1.0
        normalized[index] = (images[index] - mean) / scale
        normalized[index][~mask] = 0.0
        statistics[modality] = {"mean": mean, "std": standard_deviation, "scale": scale}

    bounds = calculate_crop_bounds(mask, margin=margin)
    return PreprocessedSubject(
        subject_id=subject.subject_id,
        images=_crop(normalized, bounds),
        labels=_crop(labels, bounds),
        crop_bounds=bounds,
        original_shape=tuple(int(value) for value in labels.shape),
        affine=subject.affine,
        spacing=subject.spacing,
        modality_statistics=statistics,
    )


def preprocessing_summary(subject: PreprocessedSubject) -> dict[str, Any]:
    return {
        "subject_id": subject.subject_id,
        "original_shape": subject.original_shape,
        "cropped_shape": subject.images.shape[1:],
        "crop_starts": subject.crop_bounds.starts,
        "crop_stops": subject.crop_bounds.stops,
        "foreground_voxels": int(np.count_nonzero(foreground_mask(subject.images))),
        "label_voxels": int(np.count_nonzero(subject.labels)),
        "modality_statistics": subject.modality_statistics,
    }
