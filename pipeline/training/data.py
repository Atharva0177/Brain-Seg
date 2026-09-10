"""Worker-friendly cached patch dataset for full training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.data.cache import preprocess_subject_cached
from pipeline.data.patches import sample_patch
from pipeline.models.regions import labels_to_class_indices_torch


class CachedPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sample one fixed-size patch per subject without loading full data into RAM."""

    def __init__(self, manifest, subject_ids, cache_root: Path, patch_size: int, seed: int):
        self.manifest = manifest
        self.subject_ids = subject_ids
        self.cache_root = cache_root
        self.patch_size = patch_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject, _ = preprocess_subject_cached(
            self.manifest, self.subject_ids[index], self.cache_root
        )
        patch = sample_patch(
            subject.images,
            subject.labels,
            (self.patch_size,) * 3,
            foreground_probability=2 / 3,
            rng=np.random.default_rng(self.seed + index),
        )
        image = torch.from_numpy(patch.image.copy()).float()
        target = labels_to_class_indices_torch(torch.from_numpy(patch.label.copy()).long())
        return image, target


def load_training_subject_ids(split_path: Path) -> list[str]:
    return list(json.loads(split_path.read_text(encoding="utf-8"))["splits"]["train"])
