"""Compressed, content-addressed cache for preprocessed BraTS subjects."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import CropBounds, PreprocessedSubject, normalize_and_crop
from .splits import _manifest_identity
from .subjects import load_subject


CACHE_SCHEMA_VERSION = 1


def cache_key(manifest: dict[str, Any], subject_id: str, margin: int) -> str:
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_manifest_id": _manifest_identity(manifest),
        "subject_id": subject_id,
        "crop_margin": margin,
        "normalization": "foreground-zscore-v1",
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    directory = cache_root / key[:2]
    return directory / f"{key}.npz", directory / f"{key}.json"


def save_cached_subject(subject: PreprocessedSubject, cache_path: Path, metadata_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=cache_path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(temporary, images=subject.images, labels=subject.labels)
    os.replace(temporary_path, cache_path)

    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "subject_id": subject.subject_id,
        "crop_bounds": asdict(subject.crop_bounds),
        "original_shape": subject.original_shape,
        "affine": subject.affine.tolist(),
        "spacing": subject.spacing,
        "modality_statistics": subject.modality_statistics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_cached_subject(cache_path: Path, metadata_path: Path) -> PreprocessedSubject:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(cache_path) as cached:
        images = cached["images"]
        labels = cached["labels"]
    bounds = CropBounds(
        starts=tuple(metadata["crop_bounds"]["starts"]),
        stops=tuple(metadata["crop_bounds"]["stops"]),
    )
    return PreprocessedSubject(
        subject_id=metadata["subject_id"],
        images=images,
        labels=labels,
        crop_bounds=bounds,
        original_shape=tuple(metadata["original_shape"]),
        affine=np.asarray(metadata["affine"]),
        spacing=tuple(metadata["spacing"]),
        modality_statistics=metadata["modality_statistics"],
    )


def preprocess_subject_cached(
    manifest: dict[str, Any], subject_id: str, cache_root: Path, margin: int = 8
) -> tuple[PreprocessedSubject, str]:
    key = cache_key(manifest, subject_id, margin)
    cache_path, metadata_path = _cache_paths(cache_root, key)
    if cache_path.exists() and metadata_path.exists():
        return load_cached_subject(cache_path, metadata_path), "hit"

    processed = normalize_and_crop(load_subject(manifest, subject_id), margin=margin)
    save_cached_subject(processed, cache_path, metadata_path)
    return processed, "miss"
