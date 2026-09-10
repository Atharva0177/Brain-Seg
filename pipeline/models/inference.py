"""Sliding-window 3D inference with Gaussian-weighted blending."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .training import autocast_context


@dataclass(frozen=True)
class Window:
    starts: tuple[int, int, int]
    stops: tuple[int, int, int]


def _positions(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, step))
    final = length - window
    if positions[-1] != final:
        positions.append(final)
    return positions


def generate_windows(
    spatial_shape: tuple[int, int, int], window_size: tuple[int, int, int], overlap: float
) -> list[Window]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    if any(size <= 0 for size in window_size):
        raise ValueError("window_size values must be positive.")
    steps = tuple(max(1, int(size * (1 - overlap))) for size in window_size)
    return [
        Window((x, y, z), (x + window_size[0], y + window_size[1], z + window_size[2]))
        for x in _positions(spatial_shape[0], window_size[0], steps[0])
        for y in _positions(spatial_shape[1], window_size[1], steps[1])
        for z in _positions(spatial_shape[2], window_size[2], steps[2])
    ]


def gaussian_importance_map(
    window_size: tuple[int, int, int], sigma_scale: float = 0.125
) -> Tensor:
    if any(size <= 0 for size in window_size):
        raise ValueError("window_size values must be positive.")
    axes = [torch.arange(size, dtype=torch.float32) for size in window_size]
    grids = torch.meshgrid(*axes, indexing="ij")
    importance = torch.ones(window_size, dtype=torch.float32)
    for grid, size in zip(grids, window_size):
        center = (size - 1) / 2
        sigma = max(size * sigma_scale, 1e-6)
        importance *= torch.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return importance.clamp_min(1e-6)


def _pad_input(
    inputs: Tensor, window_size: tuple[int, int, int]
) -> tuple[Tensor, tuple[int, int, int]]:
    spatial = inputs.shape[-3:]
    padding_after = tuple(max(0, size - current) for current, size in zip(spatial, window_size))
    if any(padding_after):
        inputs = torch.nn.functional.pad(
            inputs,
            (0, padding_after[2], 0, padding_after[1], 0, padding_after[0]),
        )
    return inputs, padding_after


@torch.inference_mode()
def sliding_window_logits(
    model: nn.Module,
    inputs: Tensor,
    window_size: tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    device: torch.device | None = None,
    amp: bool = True,
    window_batch_size: int = 1,
) -> Tensor:
    """Return blended logits with shape (N, classes, X, Y, Z)."""

    if inputs.ndim != 5:
        raise ValueError("Expected input shape (N, C, X, Y, Z).")
    if inputs.shape[0] != 1:
        raise ValueError("Sliding-window inference currently accepts batch size 1.")
    if window_batch_size < 1:
        raise ValueError("window_batch_size must be positive.")
    if device is not None:
        target_device = device
    else:
        first_parameter = next(model.parameters(), None)
        target_device = first_parameter.device if first_parameter is not None else inputs.device
    inputs = inputs.to(target_device)
    original_shape = tuple(int(value) for value in inputs.shape[-3:])
    padded_inputs, _ = _pad_input(inputs, window_size)
    padded_shape = tuple(int(value) for value in padded_inputs.shape[-3:])
    windows = generate_windows(padded_shape, window_size, overlap)
    importance = gaussian_importance_map(window_size).to(target_device)
    output: Tensor | None = None
    weights = torch.zeros((1, 1, *padded_shape), device=target_device)

    model_was_training = model.training
    model.eval()
    try:
        for offset in range(0, len(windows), window_batch_size):
            batch_windows = windows[offset : offset + window_batch_size]
            patches = torch.cat(
                [
                    padded_inputs[
                        :,
                        :,
                        window.starts[0] : window.stops[0],
                        window.starts[1] : window.stops[1],
                        window.starts[2] : window.stops[2],
                    ]
                    for window in batch_windows
                ],
                dim=0,
            )
            with autocast_context(target_device, enabled=amp):
                predictions = model(patches)
            if output is None:
                output = torch.zeros((1, predictions.shape[1], *padded_shape), device=target_device)
            for index, window in enumerate(batch_windows):
                weighted = predictions[index : index + 1] * importance
                output[
                    :,
                    :,
                    window.starts[0] : window.stops[0],
                    window.starts[1] : window.stops[1],
                    window.starts[2] : window.stops[2],
                ] += weighted
                weights[
                    :,
                    :,
                    window.starts[0] : window.stops[0],
                    window.starts[1] : window.stops[1],
                    window.starts[2] : window.stops[2],
                ] += importance
    finally:
        model.train(model_was_training)

    if output is None:
        raise RuntimeError("No inference windows were generated.")
    output = output / weights.clamp_min(1e-6)
    return output[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]
