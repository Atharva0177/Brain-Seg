"""Generate charts, distributions, and labeled sample images for BraTS data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models.regions import labels_to_regions

MODALITIES = ("t1", "t1ce", "t2", "flair")
LABEL_COLORS = {0: "black", 1: "#4ade80", 2: "#facc15", 4: "#f87171"}


def load_subject(manifest: dict, subject_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    root = Path(manifest["root"])
    entries = [entry for entry in manifest["files"] if entry.get("subject_id") == subject_id]
    images = {}
    label = None
    for entry in entries:
        path = root / entry["path"]
        modality = entry.get("modality")
        if modality in MODALITIES:
            images[modality] = nib.load(str(path)).get_fdata(dtype=np.float32)
        elif modality == "seg":
            label = nib.load(str(path)).get_fdata().astype(np.uint8)
    if set(images) != set(MODALITIES) or label is None:
        raise ValueError(f"Incomplete subject: {subject_id}")
    return images, label


def save_split_chart(split: dict, output: Path) -> None:
    names = ["train", "validation", "test"]
    values = [len(split["splits"][name]) for name in names]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(names, values, color=["#86efac", "#facc15", "#60a5fa"])
    axis.set_title("Subject-level split composition")
    axis.set_ylabel("Subjects")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_label_chart(label_counts: dict[str, int], output: Path) -> None:
    labels = list(label_counts)
    values = list(label_counts.values())
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, values, color=[LABEL_COLORS[int(label)] for label in labels])
    axis.set_title("Aggregated sampled label voxels")
    axis.set_ylabel("Voxels")
    axis.set_xlabel("BraTS label")
    axis.set_yscale("log")
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value * 1.1, f"{value:,}", ha="center", fontsize=8
        )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_intensity_chart(values: dict[str, list[float]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, modality in zip(axes.flat, MODALITIES):
        axis.hist(values[modality], bins=50, color="#86efac", alpha=0.85)
        axis.set_title(modality.upper())
        axis.set_xlabel("Nonzero intensity")
        axis.set_ylabel("Count")
    fig.suptitle("Sampled modality intensity distributions")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_montage(
    images: dict[str, np.ndarray], labels: np.ndarray, subject_id: str, output: Path
) -> None:
    z = labels.shape[2] // 2
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for index, modality in enumerate(MODALITIES):
        image = images[modality][:, :, z]
        axes[0, index].imshow(image, cmap="gray")
        axes[0, index].set_title(modality.upper())
        axes[0, index].axis("off")
        axes[1, index].imshow(image, cmap="gray")
        axes[1, index].imshow(
            np.ma.masked_where(labels[:, :, z] == 0, labels[:, :, z]),
            cmap="nipy_spectral",
            alpha=0.48,
            vmin=0,
            vmax=4,
        )
        axes[1, index].set_title(f"{modality.upper()} + label")
        axes[1, index].axis("off")
    fig.suptitle(f"{subject_id} axial center slice")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset-visualization"))
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.splits.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    selected = sorted(
        rng.sample(split["splits"]["train"], min(args.subjects, len(split["splits"]["train"])))
    )

    label_counts = {"0": 0, "1": 0, "2": 0, "4": 0}
    intensity_values = {modality: [] for modality in MODALITIES}
    summaries = []
    for subject_id in selected:
        images, labels = load_subject(manifest, subject_id)
        for value in (0, 1, 2, 4):
            label_counts[str(value)] += int(np.count_nonzero(labels == value))
        for modality in MODALITIES:
            nonzero = images[modality][images[modality] != 0]
            if nonzero.size:
                sample = nonzero[:: max(1, nonzero.size // 5000)]
                intensity_values[modality].extend(sample.tolist())
        regions = labels_to_regions(labels)
        summaries.append(
            {
                "subject_id": subject_id,
                "shape": list(labels.shape),
                "spacing": [
                    float(value)
                    for value in nib.load(
                        str(
                            Path(manifest["root"])
                            / next(
                                entry["path"]
                                for entry in manifest["files"]
                                if entry.get("subject_id") == subject_id
                            )
                        )
                    ).header.get_zooms()[:3]
                ],
                "label_voxels": {name: int(mask.sum()) for name, mask in regions.items()},
            }
        )
        save_montage(images, labels, subject_id, output / f"sample-{subject_id}.png")

    save_split_chart(split, output / "split-composition.png")
    save_label_chart(label_counts, output / "label-distribution.png")
    save_intensity_chart(intensity_values, output / "intensity-distributions.png")
    payload = {
        "seed": args.seed,
        "subjects": selected,
        "label_counts": label_counts,
        "summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = f"""# Dataset Visualization Report\n\nGenerated from `{args.manifest}` with seed `{args.seed}`.\n\n## Samples\n\n{chr(10).join(f"- `{subject}`: `sample-{subject}.png`" for subject in selected)}\n\n## Charts\n\n- `split-composition.png`\n- `label-distribution.png`\n- `intensity-distributions.png`\n\n## Data\n\n- Sampled subjects: {len(selected)}\n- Total manifest files: {manifest["file_count"]}\n- Total subjects in manifest: {manifest["subject_count"]}\n"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"status": "completed", "output_dir": str(output), "subjects": selected}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
