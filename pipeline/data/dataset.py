"""Lazy PyTorch/MONAI dataset adapters backed by the subject cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from .cache import preprocess_subject_cached


class CachedSubjectDataset(Dataset[dict[str, Any]]):
    """Load one preprocessed subject per index, never the full corpus at once."""

    def __init__(
        self,
        manifest_path: Path,
        split_path: Path,
        split_name: str,
        cache_root: Path = Path("data/cache"),
        margin: int = 8,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
        if split_name not in split_manifest.get("splits", {}):
            raise ValueError(f"Unknown split: {split_name}")
        self.subject_ids = list(split_manifest["splits"][split_name])
        self.cache_root = cache_root
        self.margin = margin
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id = self.subject_ids[index]
        subject, cache_status = preprocess_subject_cached(
            self.manifest, subject_id, self.cache_root, margin=self.margin
        )
        sample: dict[str, Any] = {
            "image": torch.from_numpy(subject.images.copy()),
            "label": torch.from_numpy(subject.labels.copy()).long(),
            "subject_id": subject.subject_id,
            "affine": torch.from_numpy(subject.affine.copy()).double(),
            "spacing": subject.spacing,
            "crop_bounds": subject.crop_bounds,
            "cache_status": cache_status,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_monai_persistent_dataset(
    items: list[dict[str, Any]],
    transform: Any,
    cache_dir: Path,
) -> Any:
    """Build MONAI's persistent transform cache when MONAI is available."""

    try:
        from monai.data import PersistentDataset
    except ImportError as error:
        raise RuntimeError("MONAI is required for PersistentDataset integration.") from error
    return PersistentDataset(data=items, transform=transform, cache_dir=str(cache_dir))
