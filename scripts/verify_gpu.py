"""Verify the target NVIDIA/PyTorch environment before model work begins."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


def nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def main() -> int:
    report: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": nvidia_smi(),
        "torch_installed": False,
        "cuda_available": False,
        "status": "blocked",
    }

    try:
        import torch
    except ImportError:
        report["reason"] = "PyTorch is not installed in the active Python environment."
    else:
        report["torch_installed"] = True
        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["device_name"] = torch.cuda.get_device_name(0)
            report["device_capability"] = list(torch.cuda.get_device_capability(0))
            report["status"] = "passed"
        else:
            report["reason"] = "PyTorch is installed but CUDA is unavailable."

    output = Path("artifacts") / "hardware" / "gpu-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
