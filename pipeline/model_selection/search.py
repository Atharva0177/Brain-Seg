"""Bounded Optuna model selection with MLflow trial tracking."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from torch import nn

from app.tracking.mlflow import TrackingConfig, start_run
from pipeline.data.cache import preprocess_subject_cached
from pipeline.data.patches import sample_patch
from pipeline.models.gate import require_sanity_artifact
from pipeline.models.losses import DiceCrossEntropyLoss
from pipeline.models.unet3d import UNet3D, UNetConfig


@dataclass(frozen=True)
class SearchConfig:
    trial_count: int = 3
    trial_epochs: int = 2
    max_gpu_seconds: float = 900.0
    seed: int = 42
    patch_sizes: tuple[int, ...] = (32, 64)
    architectures: tuple[str, ...] = ("unet", "attention_unet")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.trial_count < 1 or self.trial_epochs < 1:
            raise ValueError("trial_count and trial_epochs must be positive.")
        if self.max_gpu_seconds <= 0:
            raise ValueError("max_gpu_seconds must be positive.")
        if not self.patch_sizes or any(size <= 0 for size in self.patch_sizes):
            raise ValueError("patch_sizes must contain positive values.")


def suggest_trial_config(trial: optuna.Trial, config: SearchConfig) -> dict[str, Any]:
    architecture = trial.suggest_categorical("architecture", list(config.architectures))
    patch_size = trial.suggest_categorical("patch_size", list(config.patch_sizes))
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dice_weight = trial.suggest_float("dice_weight", 0.25, 0.75)
    return {
        "architecture": architecture,
        "patch_size": patch_size,
        "learning_rate": learning_rate,
        "dice_weight": dice_weight,
        "cross_entropy_weight": 1.0 - dice_weight,
    }


def _model(parameters: dict[str, Any]) -> nn.Module:
    config = UNetConfig(channels=(4, 8, 16), num_res_units=1)
    if parameters["architecture"] == "attention_unet":
        from pipeline.models.attention_unet3d import AttentionUNet3D

        return AttentionUNet3D(config)
    return UNet3D(config)


def _trial_train(
    images: np.ndarray,
    labels: np.ndarray,
    parameters: dict[str, Any],
    epochs: int,
    device: torch.device,
) -> tuple[float, float]:
    model = _model(parameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"])
    criterion = DiceCrossEntropyLoss(
        dice_weight=parameters["dice_weight"],
        cross_entropy_weight=parameters["cross_entropy_weight"],
    )
    patch = sample_patch(
        images,
        labels,
        (parameters["patch_size"],) * 3,
        foreground_probability=2 / 3,
        rng=np.random.default_rng(42),
    )
    inputs = torch.from_numpy(patch.image).float().unsqueeze(0).to(device)
    targets = torch.from_numpy(patch.label).long().unsqueeze(0).to(device)
    from pipeline.models.regions import labels_to_class_indices_torch

    targets = labels_to_class_indices_torch(targets)
    best_dice = 0.0
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        prediction = logits.argmax(dim=1)
        dices = []
        for label in (1, 2, 3):
            pred = prediction == label
            truth = targets == label
            denominator = int(pred.sum() + truth.sum())
            dices.append(1.0 if denominator == 0 else float(2 * (pred & truth).sum() / denominator))
        best_dice = max(best_dice, sum(dices) / len(dices))
    return final_loss, best_dice


def run_search(
    manifest: dict[str, Any],
    sanity_artifact: Path,
    *,
    cache_root: Path,
    tracking: TrackingConfig,
    config: SearchConfig = SearchConfig(),
    subject_id: str = "BraTS20_Training_001",
    output: Path = Path("artifacts/model-selection.json"),
) -> dict[str, Any]:
    config.validate()
    require_sanity_artifact(sanity_artifact)
    device = torch.device(config.device)
    subject, _ = preprocess_subject_cached(manifest, subject_id, cache_root)
    started = time.monotonic()
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trials: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        if time.monotonic() - started >= config.max_gpu_seconds:
            raise optuna.exceptions.StudyStop()  # type: ignore[attr-defined]
        parameters = suggest_trial_config(trial, config)
        trial_start = time.monotonic()
        with start_run(
            tracking,
            run_name=f"optuna-trial-{trial.number}",
            tags={"purpose": "model-selection", "trial": str(trial.number)},
            parameters=parameters,
        ) as run:
            loss, dice = _trial_train(
                subject.images, subject.labels, parameters, config.trial_epochs, device
            )
            duration = time.monotonic() - trial_start
            mlflow_metrics = {
                "final_loss": loss,
                "composite_dice": dice,
                "duration_seconds": duration,
            }
            import mlflow

            mlflow.log_metrics(mlflow_metrics)
            trial.set_user_attr("mlflow_run_id", run.info.run_id)
            trial.set_user_attr("duration_seconds", duration)
            trials.append(
                {
                    "number": trial.number,
                    "params": parameters,
                    "metrics": mlflow_metrics,
                    "run_id": run.info.run_id,
                }
            )
        return dice

    study.optimize(objective, n_trials=config.trial_count, catch=(RuntimeError,))
    best = study.best_trial
    result = {
        "status": "completed",
        "config": asdict(config),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "run_id": best.user_attrs.get("mlflow_run_id"),
        },
        "trials": trials,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
