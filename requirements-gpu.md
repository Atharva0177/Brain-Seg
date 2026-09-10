# GPU Installation Notes

## Conda Setup

```powershell
conda create -n brainseg python=3.12 -y
conda activate brainseg
```

Install the CUDA-compatible PyTorch build using the official PyTorch selector. Then install the remaining project dependencies:

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` includes `torch` for completeness, but an already-installed compatible CUDA build should be retained rather than replaced by an arbitrary default wheel. If pip resolves a different PyTorch build, reinstall the selected CUDA build and rerun the verification script.

After authentication and dataset selection, run the downloader from the repository root:

```powershell
conda activate brainseg
$env:KAGGLE_DATASET = "owner/dataset-slug"
python scripts/download_data.py
```

The Kaggle CLI progress bar is passed through directly to the terminal. The temporary archive/extraction workspace is created beside `data/raw` on the selected output drive, not under the Windows system temp directory. Use `--force` only when a fresh download is deliberately required.

The project declares PyTorch in `pyproject.toml`, but the correct PyTorch wheel must match the installed NVIDIA driver and CUDA support on the target machine.

Before installing the project dependencies:

1. Confirm the driver and GPU with `nvidia-smi`.
2. Confirm the supported CUDA wheel using the official PyTorch installation selector.
3. Install the matching PyTorch build before installing the remaining project dependencies.
4. Run `python scripts/verify_gpu.py` and record `artifacts/hardware/gpu-verification.json` as the `BRATS-003` proof-of-work artifact.

Do not assume that a generic PyPI PyTorch wheel is an `sm_120`/Blackwell-compatible build. A CPU fallback must be treated as a verification failure for the target training environment, not as a silent success.
