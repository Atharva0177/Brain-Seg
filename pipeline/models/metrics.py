"""Segmentation metrics for BraTS composite regions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .regions import REGION_NAMES, labels_to_regions


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape.")
    intersection = np.count_nonzero(prediction & target)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target must have the same shape.")
    prediction_exists = bool(prediction.any())
    target_exists = bool(target.any())
    if not prediction_exists and not target_exists:
        return 0.0
    if not prediction_exists or not target_exists:
        return float("inf")

    prediction_surface = _surface(prediction)
    target_surface = _surface(target)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    prediction_distance = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)
    distances = np.concatenate(
        [target_distance[prediction_surface], prediction_distance[target_surface]]
    )
    return float(np.percentile(distances, 95))


def region_metrics(
    prediction_regions: Mapping[str, np.ndarray],
    target_regions: Mapping[str, np.ndarray],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for region in REGION_NAMES:
        if region not in prediction_regions or region not in target_regions:
            raise ValueError(f"Missing region: {region}")
        prediction = prediction_regions[region]
        target = target_regions[region]
        result[region] = {
            "dice": dice_score(prediction, target),
            "hd95": hd95(prediction, target, spacing),
        }
    return result


def label_metrics(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    return region_metrics(
        labels_to_regions(prediction_labels), labels_to_regions(target_labels), spacing
    )


def aggregate_metrics(
    subject_metrics: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    if not subject_metrics:
        raise ValueError("Cannot aggregate an empty metric sequence.")
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for region in REGION_NAMES:
        aggregate[region] = {}
        for metric in ("dice", "hd95"):
            values = np.asarray([result[region][metric] for result in subject_metrics], dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                aggregate[region][metric] = {"mean": float("nan"), "std": float("nan")}
                continue
            aggregate[region][metric] = {
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "valid_count": int(finite_values.size),
                "non_finite_count": int(values.size - finite_values.size),
            }
    return aggregate
