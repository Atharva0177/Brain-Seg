"""Run a CPU/GPU forward and backward smoke test for the baseline U-Net."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models.unet3d import UNet3D


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    args = parser.parse_args()
    device = torch.device(args.device)
    model = UNet3D().to(device)
    inputs = torch.randn(1, 4, *args.patch_size, device=device)
    targets = torch.randint(0, 4, (1, *args.patch_size), device=device)
    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, targets)
    loss.backward()
    result = {
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": tuple(inputs.shape),
        "output_shape": tuple(outputs.shape),
        "loss": float(loss.detach().cpu()),
        "cuda_memory_allocated": torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
