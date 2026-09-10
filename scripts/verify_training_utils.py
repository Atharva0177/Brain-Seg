"""Run a CUDA AMP/accumulation/memory/checkpoint micro-test."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models.training import (
    GradientAccumulator,
    autocast_context,
    make_grad_scaler,
    peak_memory_bytes,
    reset_peak_memory,
)
from pipeline.models.unet3d import UNet3D, UNetConfig


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(UNetConfig(channels=(4, 8, 16), num_res_units=1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = make_grad_scaler(device)
    accumulator = GradientAccumulator(optimizer, scaler, accumulation_steps=2)
    reset_peak_memory(device)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(2):
        inputs = torch.randn(1, 4, 32, 32, 32, device=device)
        targets = torch.randint(0, 4, (1, 32, 32, 32), device=device)
        with autocast_context(device):
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        accumulator.backward(loss)
    print(
        json.dumps(
            {
                "device": str(device),
                "loss": float(loss.detach().cpu()),
                "peak_memory_bytes": peak_memory_bytes(device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
