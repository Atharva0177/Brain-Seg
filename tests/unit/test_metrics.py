import numpy as np
import pytest

from pipeline.models.metrics import aggregate_metrics, dice_score, hd95, label_metrics


def test_dice_perfect_and_disjoint_masks() -> None:
    first = np.array([True, False, True])
    second = np.array([True, False, True])
    assert dice_score(first, second) == pytest.approx(1.0)
    assert dice_score(first, ~second) == pytest.approx(0.0)


def test_hd95_perfect_and_missing_region_behavior() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    assert hd95(mask, mask) == pytest.approx(0.0)
    assert np.isinf(hd95(mask, np.zeros_like(mask)))


def test_label_metrics_and_aggregation() -> None:
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 4
    result = label_metrics(labels, labels)
    assert all(metrics["dice"] == pytest.approx(1.0) for metrics in result.values())
    aggregate = aggregate_metrics([result, result])
    assert aggregate["enhancing_tumor"]["dice"]["mean"] == pytest.approx(1.0)
    assert aggregate["enhancing_tumor"]["dice"]["std"] == pytest.approx(0.0)
