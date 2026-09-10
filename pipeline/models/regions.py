"""BraTS composite region conversion for labels and model predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

REGION_NAMES = ("whole_tumor", "tumor_core", "enhancing_tumor")
LABEL_VALUES = (0, 1, 2, 4)


def labels_to_class_indices(labels: np.ndarray) -> np.ndarray:
    """Map stored BraTS labels 0/1/2/4 to contiguous model classes 0/1/2/3."""

    labels = _validate_numpy_labels(labels)
    mapping = np.zeros(5, dtype=np.uint8)
    mapping[0], mapping[1], mapping[2], mapping[4] = 0, 1, 2, 3
    return mapping[labels]


def class_indices_to_labels(class_indices: np.ndarray) -> np.ndarray:
    """Map contiguous model classes 0/1/2/3 back to stored BraTS labels."""

    indices = np.asarray(class_indices)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Model class indices must be integers.")
    unexpected = set(np.unique(indices).tolist()) - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(f"Unexpected model class indices: {sorted(unexpected)}")
    return np.asarray((0, 1, 2, 4), dtype=np.uint8)[indices]


def _validate_numpy_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("BraTS label conversion requires integer class labels.")
    unexpected = set(np.unique(labels).tolist()) - set(LABEL_VALUES)
    if unexpected:
        raise ValueError(f"Unexpected BraTS labels: {sorted(unexpected)}")
    return labels


def labels_to_regions(labels: np.ndarray) -> dict[str, np.ndarray]:
    labels = _validate_numpy_labels(labels)
    return {
        "whole_tumor": labels > 0,
        "tumor_core": np.isin(labels, (1, 4)),
        "enhancing_tumor": labels == 4,
    }


def logits_to_labels(logits: Tensor) -> Tensor:
    if logits.ndim < 2 or logits.shape[1] != 4:
        raise ValueError(f"Expected logits with four class channels, got {tuple(logits.shape)}")
    class_indices = logits.argmax(dim=1)
    # Network channels are background, NCR/NET, edema, enhancing tumor.
    mapping = torch.tensor((0, 1, 2, 4), device=logits.device, dtype=torch.long)
    return mapping[class_indices]


def labels_to_class_indices_torch(labels: Tensor) -> Tensor:
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("BraTS labels must be integer tensors.")
    unexpected = set(torch.unique(labels).detach().cpu().tolist()) - set(LABEL_VALUES)
    if unexpected:
        raise ValueError(f"Unexpected BraTS labels: {sorted(unexpected)}")
    mapping = torch.tensor((0, 1, 2, 0, 3), device=labels.device, dtype=torch.long)
    return mapping[labels.long()]


def labels_to_regions_torch(labels: Tensor) -> dict[str, Tensor]:
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("BraTS label conversion requires integer torch labels.")
    unexpected = set(torch.unique(labels).detach().cpu().tolist()) - set(LABEL_VALUES)
    if unexpected:
        raise ValueError(f"Unexpected BraTS labels: {sorted(unexpected)}")
    return {
        "whole_tumor": labels > 0,
        "tumor_core": (labels == 1) | (labels == 4),
        "enhancing_tumor": labels == 4,
    }


def stack_regions(regions: Mapping[str, np.ndarray | Tensor]) -> np.ndarray | Tensor:
    missing = [name for name in REGION_NAMES if name not in regions]
    if missing:
        raise ValueError(f"Missing regions: {missing}")
    values = [regions[name] for name in REGION_NAMES]
    if isinstance(values[0], torch.Tensor):
        return torch.stack([value.bool() for value in values], dim=1)
    return np.stack([np.asarray(value, dtype=bool) for value in values], axis=0)
