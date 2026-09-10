"""Verify that sampled NIfTI files can be opened and have valid geometry."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_nifti(path: Path) -> Any:
    try:
        import nibabel as nib
    except ImportError as error:
        raise RuntimeError(
            "nibabel is required for NIfTI verification; install project dependencies first."
        ) from error
    return nib.load(str(path))


def _spacing(image: Any) -> tuple[float, ...]:
    header = image.header
    return tuple(float(value) for value in header.get_zooms()[:3])


def verify_volume(path: Path) -> dict[str, object]:
    image = _load_nifti(path)
    shape = tuple(int(value) for value in image.shape[:3])
    affine = image.affine.tolist() if hasattr(image.affine, "tolist") else image.affine
    spacing = _spacing(image)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Invalid volume shape for {path}: {shape}")
    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing for {path}: {spacing}")
    return {
        "path": str(path),
        "shape": shape,
        "spacing": spacing,
        "affine": affine,
    }


def verify_samples(root: Path, sample_count: int = 5, seed: int = 42) -> dict[str, object]:
    paths = sorted(root.rglob("*.nii")) + sorted(root.rglob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"No NIfTI files found under {root}")
    randomizer = random.Random(seed)
    selected = randomizer.sample(paths, min(sample_count, len(paths)))
    results = [verify_volume(path) for path in selected]
    return {
        "status": "passed",
        "root": str(root.resolve()),
        "sample_count": len(results),
        "seed": seed,
        "volumes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/volume-verification.json"))
    args = parser.parse_args()
    result = verify_samples(args.root, args.sample_count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
