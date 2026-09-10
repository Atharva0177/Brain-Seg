import numpy as np
import torch

from pipeline.models.sanity import SanityConfig, run_sanity_gate


def test_sanity_result_contains_history() -> None:
    images = np.random.default_rng(1).normal(size=(4, 8, 8, 8)).astype(np.float32)
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 2:6, 2:6] = 4
    result = run_sanity_gate(images, labels, torch.device("cpu"), SanityConfig(epochs=1))
    assert result["status"] in {"passed", "failed"}
    assert len(result["history"]) == 1
