"""Reusable resource-aware training utilities for 3D segmentation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingConfig:
    amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None

    def validate(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")


def autocast_context(device: torch.device, enabled: bool = True):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def make_grad_scaler(device: torch.device, enabled: bool = True):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


class GradientAccumulator:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        accumulation_steps: int = 1,
        gradient_clip_norm: float | None = None,
    ) -> None:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1.")
        self.optimizer = optimizer
        self.scaler = scaler
        self.accumulation_steps = accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.step_index = 0

    def backward(self, loss: Tensor) -> bool:
        self.scaler.scale(loss / self.accumulation_steps).backward()
        self.step_index += 1
        if self.step_index % self.accumulation_steps != 0:
            return False
        self.scaler.unscale_(self.optimizer)
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], self.gradient_clip_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return True

    def flush(self) -> bool:
        if self.step_index % self.accumulation_steps == 0:
            return False
        self.scaler.unscale_(self.optimizer)
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]["params"], self.gradient_clip_norm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return True


def peak_memory_bytes(device: torch.device) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    fd, temporary_name = tempfile.mkstemp(suffix=".pt", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
