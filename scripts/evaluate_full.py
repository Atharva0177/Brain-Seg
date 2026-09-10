"""Evaluate a trained checkpoint on the held-out split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.training.evaluation import evaluate_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("splits", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    training_result_path = args.checkpoint.parent / "training-result.json"
    if training_result_path.is_file():
        selected = json.loads(training_result_path.read_text(encoding="utf-8"))["selected"]
        print(f"Using trained checkpoint configuration from {training_result_path}")
    elif args.selection.is_file():
        selected = json.loads(args.selection.read_text(encoding="utf-8"))["best_trial"]["params"]
    else:
        raise FileNotFoundError(
            "Neither the checkpoint-adjacent training-result.json nor the selection artifact exists."
        )
    result = evaluate_checkpoint(
        manifest,
        splits,
        selected,
        args.checkpoint,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        device_name=args.device,
        max_subjects=args.max_subjects,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "subjects_evaluated": result["subjects_evaluated"],
                "aggregate": result["aggregate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
