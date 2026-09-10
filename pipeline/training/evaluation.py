"""Full-volume evaluation, uncertainty, and failure-artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline.data.cache import preprocess_subject_cached
from pipeline.models.inference import sliding_window_logits
from pipeline.models.metrics import aggregate_metrics, label_metrics
from pipeline.models.regions import logits_to_labels
from pipeline.models.training import load_checkpoint
from pipeline.training.full import build_model


def evaluate_checkpoint(
    manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    selected: dict[str, Any],
    checkpoint: Path,
    *,
    cache_root: Path,
    output_dir: Path,
    device_name: str = "cuda",
    max_subjects: int | None = None,
) -> dict[str, Any]:
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = build_model(selected).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    subjects = split_manifest["splits"]["test"][:max_subjects]
    subject_results = []
    failures = []
    for subject_id in subjects:
        subject, _ = preprocess_subject_cached(manifest, subject_id, cache_root)
        inputs = torch.from_numpy(subject.images).float().unsqueeze(0)
        logits = sliding_window_logits(
            model, inputs, window_size=(int(selected["patch_size"]),) * 3, device=device
        )
        prediction = logits_to_labels(logits).squeeze(0).cpu().numpy()
        metrics = label_metrics(prediction, subject.labels, subject.spacing)
        subject_results.append({"subject_id": subject_id, "metrics": metrics})
        score = float(np.mean([value["dice"] for value in metrics.values()]))
        if score < 0.5:
            failures.append({"subject_id": subject_id, "composite_dice": score})
    result = {
        "status": "completed",
        "subjects_evaluated": len(subject_results),
        "subject_results": subject_results,
        "aggregate": aggregate_metrics([entry["metrics"] for entry in subject_results]),
        "failure_cases": failures,
        "non_finite_hd95_cases": [
            {
                "subject_id": entry["subject_id"],
                "regions": [
                    region
                    for region, metrics in entry["metrics"].items()
                    if not np.isfinite(metrics["hd95"])
                ],
            }
            for entry in subject_results
            if any(not np.isfinite(metrics["hd95"]) for metrics in entry["metrics"].values())
        ],
        "device": str(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def mc_dropout_uncertainty(logits_samples: torch.Tensor) -> torch.Tensor:
    """Return per-voxel predictive variance across stochastic logits/probabilities."""

    probabilities = logits_samples.softmax(dim=2)
    return probabilities.var(dim=0).mean(dim=1)
