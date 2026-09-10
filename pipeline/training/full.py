"""Bounded full-training workflow for the selected Phase 3 configuration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from app.tracking.mlflow import TrackingConfig, start_run
from pipeline.data.cache import preprocess_subject_cached
from pipeline.models.inference import sliding_window_logits
from pipeline.models.losses import DiceCrossEntropyLoss
from pipeline.models.regions import logits_to_labels
from pipeline.models.training import (
    GradientAccumulator,
    autocast_context,
    make_grad_scaler,
    peak_memory_bytes,
    reset_peak_memory,
    save_checkpoint,
)
from pipeline.models.unet3d import UNet3D, UNetConfig
from pipeline.training.data import CachedPatchDataset


def build_model(parameters: dict[str, Any]) -> torch.nn.Module:
    channels = tuple(parameters.get("channels", (4, 8, 16)))
    config = UNetConfig(channels=channels, num_res_units=1)
    if parameters.get("architecture") == "attention_unet":
        from pipeline.models.attention_unet3d import AttentionUNet3D

        return AttentionUNet3D(config)
    return UNet3D(config)


def train_full(
    manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    selected: dict[str, Any],
    *,
    cache_root: Path,
    output_dir: Path,
    tracking: TrackingConfig,
    epochs: int = 10,
    device_name: str = "cuda",
    max_subjects: int | None = None,
    num_workers: int = 2,
    batch_size: int = 1,
    compile_model: bool = False,
    channels: tuple[int, ...] | None = None,
    patch_size: int | None = None,
) -> dict[str, Any]:
    selected = dict(selected)
    selected.setdefault(
        "cross_entropy_weight",
        1.0 - float(selected["dice_weight"]),
    )
    if channels is not None:
        selected["channels"] = tuple(channels)
    if patch_size is not None:
        selected["patch_size"] = patch_size
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(selected).to(device)
    compile_active = False
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        compile_active = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(selected["learning_rate"]))
    scaler = make_grad_scaler(device)
    accumulator = GradientAccumulator(optimizer, scaler, accumulation_steps=1)
    criterion = DiceCrossEntropyLoss(
        dice_weight=float(selected["dice_weight"]),
        cross_entropy_weight=float(selected["cross_entropy_weight"]),
    )
    train_ids = split_manifest["splits"]["train"]
    if max_subjects is not None:
        if max_subjects < 1:
            raise ValueError("max_subjects must be positive when provided.")
        train_ids = train_ids[:max_subjects]
    history: list[dict[str, float]] = []
    split_ids = {
        "train": train_ids,
        "validation": list(split_manifest["splits"]["validation"]),
        "test": list(split_manifest["splits"]["test"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    with start_run(
        tracking,
        run_name="full-training",
        tags={"purpose": "full-training", "phase": "BRATS-035"},
        parameters={**selected, "epochs": epochs, "device": str(device)},
    ) as run:
        mlflow.set_tag("torch_compile_requested", str(compile_model).lower())
        epoch_iterator = tqdm(
            range(epochs),
            desc="full training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iterator:
            model.train()
            reset_peak_memory(device)
            epoch_start = time.monotonic()
            losses: list[float] = []
            dataset = CachedPatchDataset(
                manifest, train_ids, cache_root, int(selected["patch_size"]), seed=epoch
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            total_subjects = len(loader)
            subject_iterator = tqdm(
                enumerate(loader, start=1),
                total=total_subjects,
                desc=f"epoch {epoch + 1}/{epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for _subject_index, (inputs, targets) in subject_iterator:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                try:
                    with autocast_context(device):
                        loss = criterion(model(inputs), targets)
                except Exception as error:
                    if not compile_active or "triton" not in str(error).lower():
                        raise
                    print(
                        "torch.compile is unavailable in this environment because Triton "
                        "is missing; falling back to eager CUDA execution.",
                        flush=True,
                    )
                    model = build_model(selected).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=float(selected["learning_rate"])
                    )
                    accumulator = GradientAccumulator(optimizer, scaler, accumulation_steps=1)
                    compile_active = False
                    mlflow.set_tag("torch_compile_active", "false")
                    with autocast_context(device):
                        loss = criterion(model(inputs), targets)
                if compile_active:
                    mlflow.set_tag("torch_compile_active", "true")
                accumulator.backward(loss)
                losses.append(float(loss.detach().cpu()))
                subject_iterator.set_postfix(loss=f"{losses[-1]:.5f}")
            validation_loss = _split_patch_loss(
                model,
                criterion,
                manifest,
                split_ids["validation"],
                cache_root,
                int(selected["patch_size"]),
                device,
                epoch,
            )
            epoch_duration = time.monotonic() - epoch_start
            metrics = {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "duration_seconds": epoch_duration,
                "peak_memory_bytes": float(peak_memory_bytes(device)),
                "subjects": float(len(train_ids)),
                "samples_per_second": float(len(train_ids) / max(epoch_duration, 1e-6)),
                "validation_loss": validation_loss,
            }
            history.append(metrics)
            mlflow.log_metrics({key: value for key, value in metrics.items()}, step=epoch + 1)
            epoch_iterator.set_postfix(
                loss=f"{metrics['loss']:.5f}",
                memory_mb=f"{metrics['peak_memory_bytes'] / 1024**2:.0f}",
            )

        checkpoint = output_dir / "best-model.pt"
        save_checkpoint(checkpoint, model, optimizer, scaler, epoch=epochs, metrics=history[-1])
        mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
        result = {
            "status": "completed",
            "run_id": run.info.run_id,
            "checkpoint": str(checkpoint),
            "selected": selected,
            "history": history,
            "device": str(device),
            "channels": selected.get("channels", (4, 8, 16)),
            "patch_size": selected["patch_size"],
        }
        result_path = output_dir / "training-result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(result_path), artifact_path="diagnostics")
        _save_training_curves(history, output_dir / "training-curves.png")
        _save_split_previews(
            model,
            manifest,
            split_ids,
            cache_root,
            output_dir / "prediction-previews",
            int(selected["patch_size"]),
            device,
        )
        mlflow.log_artifact(str(output_dir / "training-curves.png"), artifact_path="plots")
        mlflow.log_artifacts(str(output_dir / "prediction-previews"), artifact_path="previews")
    return result


@torch.no_grad()
def _split_patch_loss(
    model, criterion, manifest, subject_ids, cache_root, patch_size, device, seed
):
    if not subject_ids:
        return 0.0
    model.eval()
    losses = []
    dataset = CachedPatchDataset(manifest, subject_ids, cache_root, patch_size, seed=seed)
    for inputs, targets in dataset:
        inputs = inputs.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        losses.append(float(criterion(model(inputs), targets).detach().cpu()))
    model.train()
    return float(np.mean(losses))


def _save_training_curves(history, output):
    epochs = [point["epoch"] for point in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [point["loss"] for point in history], label="train")
    axes[0].plot(epochs, [point["validation_loss"] for point in history], label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [point["duration_seconds"] for point in history])
    axes[1].set_title("Epoch duration (s)")
    axes[2].plot(epochs, [point["peak_memory_bytes"] / 1024**3 for point in history])
    axes[2].set_title("Peak GPU memory (GiB)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_split_previews(model, manifest, split_ids, cache_root, output_dir, patch_size, device):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, subject_ids in split_ids.items():
        if not subject_ids:
            continue
        subject, _ = preprocess_subject_cached(manifest, subject_ids[0], cache_root)
        logits = sliding_window_logits(
            model,
            torch.from_numpy(subject.images).float().unsqueeze(0),
            (patch_size,) * 3,
            device=device,
        )
        prediction = logits_to_labels(logits).squeeze(0).cpu().numpy()
        z = subject.labels.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(subject.images[0, :, :, z], cmap="gray")
        axes[0].set_title("T1")
        axes[1].imshow(subject.labels[:, :, z], cmap="nipy_spectral", vmin=0, vmax=4)
        axes[1].set_title("Ground truth")
        axes[2].imshow(prediction[:, :, z], cmap="nipy_spectral", vmin=0, vmax=4)
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{split_name}: {subject_ids[0]}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{split_name}-{subject_ids[0]}.png", dpi=150)
        plt.close(fig)
