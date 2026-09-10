import json
from pathlib import Path

import numpy as np
import torch

from pipeline.data.dataset import CachedSubjectDataset


def test_dataset_is_lazy_and_returns_tensors(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    split_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps({"files": []}), encoding="utf-8")
    split_path.write_text(json.dumps({"splits": {"train": ["subject"]}}), encoding="utf-8")

    class Subject:
        subject_id = "subject"
        images = np.zeros((4, 2, 2, 2), dtype=np.float32)
        labels = np.zeros((2, 2, 2), dtype=np.uint8)
        affine = np.eye(4)
        spacing = (1.0, 1.0, 1.0)
        crop_bounds = None

    monkeypatch.setattr(
        "pipeline.data.dataset.preprocess_subject_cached",
        lambda *args, **kwargs: (Subject(), "hit"),
    )
    dataset = CachedSubjectDataset(manifest_path, split_path, "train", tmp_path / "cache")
    assert len(dataset) == 1
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (4, 2, 2, 2)
    assert sample["label"].dtype == torch.int64
    assert sample["cache_status"] == "hit"
