"""Multimodal BraTS subject indexing, loading, and label conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODALITIES = ("t1", "t1ce", "t2", "flair")
ALLOWED_LABELS = frozenset({0, 1, 2, 4})


@dataclass(frozen=True)
class SubjectFiles:
    subject_id: str
    modalities: dict[str, Path]
    segmentation: Path


@dataclass
class LoadedSubject:
    subject_id: str
    images: np.ndarray
    labels: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]


def _load_nibabel() -> Any:
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError(
            "nibabel is required for subject loading; activate the brainseg environment."
        ) from error
    return nib


def subject_files_from_manifest(
    manifest: dict[str, Any], subject_id: str
) -> SubjectFiles:
    entries = [
        entry
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and entry.get("subject_id") == subject_id
    ]
    if not entries:
        raise KeyError(f"Subject not found in manifest: {subject_id}")

    root = Path(str(manifest["root"]))
    modalities = {
        str(entry["modality"]): root / str(entry["path"])
        for entry in entries
        if entry.get("modality") in MODALITIES
    }
    segmentation_entries = [entry for entry in entries if entry.get("modality") == "seg"]
    missing = [modality for modality in MODALITIES if modality not in modalities]
    if missing or len(segmentation_entries) != 1:
        raise ValueError(
            f"Subject {subject_id} must have one file for each modality and one mask; "
            f"missing={missing}, masks={len(segmentation_entries)}"
        )
    return SubjectFiles(
        subject_id=subject_id,
        modalities=modalities,
        segmentation=root / str(segmentation_entries[0]["path"]),
    )


def validate_labels(labels: np.ndarray, *, subject_id: str = "unknown") -> np.ndarray:
    integer_labels = np.asarray(labels)
    if not np.issubdtype(integer_labels.dtype, np.integer):
        if not np.all(np.equal(integer_labels, np.round(integer_labels))):
            raise ValueError(f"Non-integer segmentation labels found for {subject_id}")
        integer_labels = integer_labels.astype(np.int16)
    unique = set(np.unique(integer_labels).tolist())
    unexpected = unique - ALLOWED_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels for {subject_id}: {sorted(unexpected)}")
    return integer_labels.astype(np.uint8, copy=False)


def composite_regions(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Return binary Whole Tumor, Tumor Core, and Enhancing Tumor masks."""

    labels = validate_labels(labels)
    return {
        "whole_tumor": labels > 0,
        "tumor_core": np.isin(labels, (1, 4)),
        "enhancing_tumor": labels == 4,
    }


def load_subject(manifest: dict[str, Any], subject_id: str) -> LoadedSubject:
    files = subject_files_from_manifest(manifest, subject_id)
    nib = _load_nibabel()
    images: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    reference_affine: np.ndarray | None = None
    reference_spacing: tuple[float, float, float] | None = None

    for modality in MODALITIES:
        image = nib.load(str(files.modalities[modality]))
        data = np.asarray(image.get_fdata(dtype=np.float32))
        shape = tuple(int(value) for value in data.shape[:3])
        affine = np.asarray(image.affine)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        if reference_shape is None:
            reference_shape, reference_affine, reference_spacing = shape, affine, spacing
        elif shape != reference_shape or not np.allclose(affine, reference_affine):
            raise ValueError(f"Geometry mismatch in {subject_id} modality {modality}")
        images.append(data)

    segmentation = nib.load(str(files.segmentation))
    labels = validate_labels(
        np.asarray(segmentation.get_fdata()), subject_id=subject_id
    )
    if tuple(labels.shape[:3]) != reference_shape or not np.allclose(
        segmentation.affine, reference_affine
    ):
        raise ValueError(f"Geometry mismatch between images and mask for {subject_id}")

    return LoadedSubject(
        subject_id=subject_id,
        images=np.stack(images, axis=0),
        labels=labels,
        affine=reference_affine,
        spacing=reference_spacing,
    )
