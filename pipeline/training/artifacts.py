"""Generate visual and uncertainty artifacts from trained predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from pipeline.data.cache import preprocess_subject_cached
from pipeline.models.inference import sliding_window_logits
from pipeline.models.regions import logits_to_labels
from pipeline.models.training import load_checkpoint
from pipeline.training.full import build_model


def _prediction_for_subject(manifest, selected, checkpoint, subject_id, cache_root, device):
    subject, _ = preprocess_subject_cached(manifest, subject_id, cache_root)
    model = build_model(selected).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    inputs = torch.from_numpy(subject.images).float().unsqueeze(0)
    logits = sliding_window_logits(
        model, inputs, window_size=(int(selected["patch_size"]),) * 3, device=device
    )
    return subject, logits_to_labels(logits).squeeze(0).cpu().numpy()


def save_overlay(subject, prediction: np.ndarray, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    index = subject.labels.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(subject.images[0, :, :, index], cmap="gray")
    axes[0].set_title("T1")
    axes[1].imshow(subject.labels[:, :, index], cmap="nipy_spectral", vmin=0, vmax=4)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction[:, :, index], cmap="nipy_spectral", vmin=0, vmax=4)
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def generate_failure_gallery(
    manifest: dict[str, Any],
    evaluation: dict[str, Any],
    selected: dict[str, Any],
    checkpoint: Path,
    *,
    cache_root: Path,
    output_dir: Path,
    device: torch.device,
    count: int = 5,
) -> dict[str, Any]:
    ranked = sorted(
        evaluation["subject_results"],
        key=lambda entry: np.mean([metric["dice"] for metric in entry["metrics"].values()]),
    )
    selected_subjects = [entry["subject_id"] for entry in ranked[:count]]
    for subject_id in selected_subjects:
        subject, prediction = _prediction_for_subject(
            manifest, selected, checkpoint, subject_id, cache_root, device
        )
        save_overlay(
            subject,
            prediction,
            output_dir / f"{subject_id}.png",
            f"BrainSeg failure case: {subject_id}",
        )
    result = {"status": "completed", "subjects": selected_subjects, "count": len(selected_subjects)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gallery.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.inference_mode()
def uncertainty_for_subject(
    manifest: dict[str, Any],
    selected: dict[str, Any],
    checkpoint: Path,
    subject_id: str,
    *,
    cache_root: Path,
    device: torch.device,
    passes: int = 8,
) -> dict[str, Any]:
    subject, _ = preprocess_subject_cached(manifest, subject_id, cache_root)
    model = build_model(selected).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.train()
    enable_dropout(model)
    inputs = torch.from_numpy(subject.images).float().unsqueeze(0)
    samples = []
    for _ in range(passes):
        logits = sliding_window_logits(
            model, inputs, window_size=(int(selected["patch_size"]),) * 3, device=device, amp=True
        )
        samples.append(logits.softmax(dim=1).cpu())
    probabilities = torch.stack(samples, dim=0)
    variance = probabilities.var(dim=0).mean(dim=1).squeeze(0).numpy()
    entropy = (
        -(probabilities.mean(dim=0) * probabilities.mean(dim=0).clamp_min(1e-8).log())
        .sum(dim=1)
        .squeeze(0)
        .numpy()
    )
    return {
        "subject_id": subject_id,
        "passes": passes,
        "variance_mean": float(variance.mean()),
        "variance_max": float(variance.max()),
        "entropy_mean": float(entropy.mean()),
        "entropy_max": float(entropy.max()),
    }


@torch.inference_mode()
def tta_uncertainty_for_subject(
    manifest: dict[str, Any],
    selected: dict[str, Any],
    checkpoint: Path,
    subject_id: str,
    *,
    cache_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Estimate uncertainty from identity and spatial-flip TTA predictions."""

    subject, _ = preprocess_subject_cached(manifest, subject_id, cache_root)
    model = build_model(selected).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    inputs = torch.from_numpy(subject.images).float().unsqueeze(0)
    transforms = ((), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4))
    probabilities = []
    for dimensions in transforms:
        augmented = torch.flip(inputs, dims=dimensions) if dimensions else inputs
        logits = sliding_window_logits(
            model,
            augmented,
            window_size=(int(selected["patch_size"]),) * 3,
            device=device,
            amp=True,
        )
        prediction = logits.softmax(dim=1)
        if dimensions:
            prediction = torch.flip(prediction, dims=dimensions)
        probabilities.append(prediction.cpu())
    stacked = torch.stack(probabilities, dim=0)
    mean_probability = stacked.mean(dim=0)
    variance = stacked.var(dim=0).mean(dim=1).squeeze(0).numpy()
    entropy = (
        -(mean_probability * mean_probability.clamp_min(1e-8).log()).sum(dim=1).squeeze(0).numpy()
    )
    return {
        "subject_id": subject_id,
        "tta_count": len(transforms),
        "variance_mean": float(variance.mean()),
        "variance_max": float(variance.max()),
        "entropy_mean": float(entropy.mean()),
        "entropy_max": float(entropy.max()),
    }
