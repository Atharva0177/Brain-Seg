import numpy as np
import pytest
import torch

from pipeline.models.regions import (
    class_indices_to_labels,
    labels_to_class_indices,
    labels_to_regions,
    labels_to_regions_torch,
    logits_to_labels,
    stack_regions,
)


def test_numpy_region_definitions() -> None:
    labels = np.array([0, 1, 2, 4], dtype=np.uint8)
    regions = labels_to_regions(labels)
    assert regions["whole_tumor"].tolist() == [False, True, True, True]
    assert regions["tumor_core"].tolist() == [False, True, False, True]
    assert regions["enhancing_tumor"].tolist() == [False, False, False, True]


def test_logits_map_class_three_to_brats_label_four() -> None:
    logits = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])
    assert logits_to_labels(logits).item() == 4


def test_model_class_indices_round_trip_to_brats_labels() -> None:
    labels = np.array([0, 1, 2, 4], dtype=np.uint8)
    assert np.array_equal(class_indices_to_labels(labels_to_class_indices(labels)), labels)


def test_torch_stack_has_region_channel_dimension() -> None:
    labels = torch.tensor([[[0, 1], [2, 4]]])
    stacked = stack_regions(labels_to_regions_torch(labels))
    assert tuple(stacked.shape) == (1, 3, 2, 2)


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        labels_to_regions(np.array([3], dtype=np.uint8))
