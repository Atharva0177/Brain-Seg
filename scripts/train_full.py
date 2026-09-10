"""Train the selected model configuration and log the run to MLflow."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tracking.mlflow import TrackingConfig
from pipeline.training.full import train_full


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full-training"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit subjects for a smoke test; omit for the full training split.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile for the model."
    )
    parser.add_argument(
        "--architecture",
        choices=("attention_unet", "unet"),
        default=None,
        help="Override the selected architecture for a separate training run.",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "high-memory"),
        default="default",
        help="high-memory uses 112^3 patches, batch size 4, and channels 32 64 128 256.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Override model channel widths, e.g. --channels 16 32 64 128.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Override the selected patch size; record this as a new training configuration.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tracking-uri", default=os.getenv("BRAINSEG_MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    parser.add_argument(
        "--no-system-metrics",
        action="store_true",
        help="Disable MLflow host/GPU system metrics logging.",
    )
    args = parser.parse_args()
    if args.profile == "high-memory":
        args.channels = args.channels or (32, 64, 128, 256)
        args.patch_size = args.patch_size or 112
        if args.batch_size == 1:
            args.batch_size = 4
    if args.architecture:
        args.channels = args.channels or (
            (32, 64, 128, 256) if args.profile == "high-memory" else (16, 32, 64, 128)
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    if args.selection.is_file():
        selection = json.loads(args.selection.read_text(encoding="utf-8"))["best_trial"]["params"]
    else:
        primary_path = Path("artifacts/primary-model.json")
        if not primary_path.is_file():
            raise FileNotFoundError(
                "Selection artifact is missing and artifacts/primary-model.json is unavailable."
            )
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        selection = {
            "architecture": primary["architecture"],
            "channels": primary["channels"],
            "patch_size": primary["patch_size"],
            "learning_rate": 0.0002051338263087451,
            "dice_weight": 0.3279972601681013,
            "cross_entropy_weight": 0.6720027398318986,
        }
    if args.architecture:
        selection["architecture"] = args.architecture
    if args.channels:
        selection["channels"] = tuple(args.channels)
    if args.patch_size:
        selection["patch_size"] = args.patch_size
    result = train_full(
        manifest,
        splits,
        selection,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        tracking=TrackingConfig(
            tracking_uri=args.tracking_uri,
            log_system_metrics=not args.no_system_metrics,
        ),
        epochs=args.epochs,
        device_name=args.device,
        max_subjects=args.max_subjects,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        compile_model=args.compile,
        channels=tuple(args.channels) if args.channels else None,
        patch_size=args.patch_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "checkpoint": result["checkpoint"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
