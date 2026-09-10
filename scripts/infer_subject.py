"""Run canonical BrainSeg inference on a four-modality NIfTI subject folder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.inference import InferenceService
from pipeline.data.preprocess import calculate_crop_bounds, foreground_mask

MODALITIES = ("t1", "t1ce", "t2", "flair")


def find_modality(folder: Path, modality: str) -> Path | None:
    exact = folder / f"{folder.name}_{modality}.nii"
    if exact.exists():
        return exact
    matches = sorted(folder.glob(f"*_{modality}.nii"))
    return matches[0] if matches else None


def load_and_preprocess(
    folder: Path,
) -> tuple[np.ndarray, np.ndarray, object, tuple[int, int, int]]:
    paths = {modality: find_modality(folder, modality) for modality in MODALITIES}
    missing = [modality for modality, path in paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            f"Missing modalities in {folder}: {', '.join(missing)}. "
            "Expected *_t1.nii, *_t1ce.nii, *_t2.nii, and *_flair.nii."
        )

    images = []
    reference = None
    for modality in MODALITIES:
        image = nib.load(str(paths[modality]))
        data = image.get_fdata(dtype=np.float32)
        if reference is None:
            reference = image
        elif data.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            raise ValueError(f"Geometry mismatch in modality {modality}.")
        images.append(data)

    stacked = np.stack(images, axis=0)
    mask = foreground_mask(stacked)
    bounds = calculate_crop_bounds(mask, margin=8)
    slices = tuple(slice(start, stop) for start, stop in zip(bounds.starts, bounds.stops))
    normalized = np.empty_like(stacked)
    for index in range(stacked.shape[0]):
        values = stacked[index][mask]
        mean = values.mean()
        std = values.std()
        normalized[index] = (stacked[index] - mean) / (std if std > 1e-6 else 1.0)
        normalized[index][~mask] = 0
    return normalized[(slice(None), *slices)], reference, bounds.starts, stacked.shape[1:]


def save_overlay(images: np.ndarray, labels: np.ndarray, output: Path) -> None:
    index = labels.shape[2] // 2
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for axis, image, name in zip(axes[:4], images, MODALITIES):
        axis.imshow(image[:, :, index], cmap="gray")
        axis.set_title(name.upper())
        axis.axis("off")
    axes[4].imshow(images[0, :, :, index], cmap="gray")
    axes[4].imshow(
        np.ma.masked_where(labels[:, :, index] == 0, labels[:, :, index]),
        cmap="nipy_spectral",
        alpha=0.55,
        vmin=0,
        vmax=4,
    )
    axes[4].set_title("Prediction")
    axes[4].axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder containing four modality NIfTI files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--tracking-uri", default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI"))
    args = parser.parse_args()
    if not args.input.is_dir():
        raise ValueError("Input must be a subject folder containing all four MRI modalities.")

    images, reference, starts, original_shape = load_and_preprocess(args.input)
    service = InferenceService(tracking_uri=args.tracking_uri)
    started = time.perf_counter()
    result = service.predict(images)
    labels = result["segmentation"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / f"{args.input.name}_segmentation.nii.gz"
    full_labels = np.zeros(original_shape, dtype=np.uint8)
    stops = tuple(start + size for start, size in zip(starts, labels.shape))
    full_labels[tuple(slice(start, stop) for start, stop in zip(starts, stops))] = labels
    nib.save(nib.Nifti1Image(full_labels, reference.affine, reference.header), str(output_image))
    overlay = output_dir / f"{args.input.name}_overlay.png"
    save_overlay(images, labels, overlay)
    summary = {
        "status": "completed",
        "input_folder": str(args.input.resolve()),
        "segmentation": str(output_image),
        "overlay": str(overlay),
        "shape": list(full_labels.shape),
        "labels_present": sorted(int(value) for value in np.unique(full_labels)),
        "model_name": result["model_name"],
        "model_version": result["model_version"],
        "device": result["device"],
        "latency_ms": result["latency_ms"],
        "total_seconds": (time.perf_counter() - started) * 1000,
    }
    (output_dir / f"{args.input.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
