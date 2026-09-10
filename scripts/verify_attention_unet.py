"""Run an Attention U-Net forward/backward smoke test."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.models.attention_unet3d import AttentionUNet3D


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(64, 64, 64))
    args = parser.parse_args()
    device = torch.device(args.device)
    model = AttentionUNet3D().to(device)
    inputs = torch.randn(1, 4, *args.patch_size, device=device)
    outputs = model(inputs)
    loss = outputs.square().mean()
    loss.backward()
    print(
        json.dumps(
            {
                "device": str(device),
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "input_shape": tuple(inputs.shape),
                "output_shape": tuple(outputs.shape),
                "loss": float(loss.detach().cpu()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
