"""Automated overfit-single-batch sanity gate."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .losses import DiceCrossEntropyLoss
from .regions import labels_to_class_indices_torch
from .unet3d import UNet3D, UNetConfig


@dataclass(frozen=True)
class SanityConfig:
    epochs: int = 40
    learning_rate: float = 1e-3
    min_dice: float = 0.85
    max_final_loss_ratio: float = 0.30
    seed: int = 42


def _foreground_dice(prediction: Tensor, target: Tensor) -> float:
    values = []
    for label in (1, 2, 3):
        predicted = prediction == label
        expected = target == label
        denominator = int(predicted.sum() + expected.sum())
        values.append(
            1.0 if denominator == 0 else float(2 * (predicted & expected).sum() / denominator)
        )
    return sum(values) / len(values)


def run_sanity_gate(
    images: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    config: SanityConfig = SanityConfig(),
) -> dict[str, Any]:
    if config.epochs < 1:
        raise ValueError("epochs must be positive.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = UNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = DiceCrossEntropyLoss()
    input_tensor = torch.from_numpy(images).float().unsqueeze(0).to(device)
    target_tensor = labels_to_class_indices_torch(
        torch.from_numpy(labels).long().unsqueeze(0).to(device)
    )
    history: list[dict[str, float]] = []
    initial_loss: float | None = None
    best_state: dict[str, Tensor] | None = None
    best_dice = float("-inf")

    model.train()
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dice = _foreground_dice(logits.argmax(dim=1), target_tensor)
        loss_value = float(loss.detach().cpu())
        initial_loss = loss_value if initial_loss is None else initial_loss
        history.append({"epoch": epoch + 1, "loss": loss_value, "foreground_dice": dice})
        if dice > best_dice:
            best_dice = dice
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Sanity gate did not produce a best model state.")
    model.load_state_dict(best_state)
    final = max(history, key=lambda entry: entry["foreground_dice"])
    passed = (
        final["foreground_dice"] >= config.min_dice
        and final["loss"] <= initial_loss * config.max_final_loss_ratio
    )
    return {
        "status": "passed" if passed else "failed",
        "device": str(device),
        "seed": config.seed,
        "config": config.__dict__,
        "initial_loss": initial_loss,
        "final": final,
        "history": history,
    }


def write_sanity_result(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
